# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""This module contains the implementation of the google_video_gen tool."""

import functools
import json
import logging
import math
import os
import re
import time
import wave
from typing import Any, Callable, Dict, Optional, Tuple

import anthropic
import requests
from aea_cli_ipfs.ipfs_utils import IPFSTool
from google import genai  # type: ignore[import-not-found]
from google.genai import errors as genai_errors  # type: ignore[import-not-found]
from google.genai import types  # type: ignore[import-not-found]
from moviepy.audio.AudioClip import AudioClip, concatenate_audioclips
from moviepy.audio.fx.audio_fadeout import audio_fadeout
from moviepy.editor import AudioFileClip, ColorClip, CompositeAudioClip, VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

# Define MechResponse type alias matching the other tools
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

SYSTEM_INSTRUCTION_CONTENT = (
    'Based on the USER INPUT: "{0}", please provide a short voiceover script and a choice of voice for the voiceover. Format your response as plain text with two lines: "Voiceover: [script]" and "Voice: [voice_name]". Each field should contain the respective content as described below.\n\n'
    '- For the "Voiceover": Use the user input to create a short script no longer than 10 seconds long (around 15-30 words), which has a mindblowing insight. The script should be in direct speech format suitable for text-to-speech AI without any stage directions. Do not just repeat the user input\n\n'
    "- For the \"Voice\": Choose what type of voice would suit the video from the following available Google TTS voices: 'achernar', 'achird', 'algenib', 'algieba', 'alnilam', 'aoede', 'autonoe', 'callirrhoe', 'charon', 'despina', 'enceladus', 'erinome', 'fenrir', 'gacrux', 'iapetus', 'kore', 'laomedeia', 'leda', 'orus', 'puck', 'pulcherrima', 'rasalgethi', 'sadachbia', 'sadaltager', 'schedar', 'sulafat', 'umbriel', 'vindemiatrix', 'zephyr', 'zubenelgenubi'. Only output the name, nothing else. \n\n"
    "Please structure your response in the following plain text format:\n\n"
    "Voiceover: [Insert narrative script here]\n"
    "Voice: [Insert name of voice here]\n\n"
    "EXAMPLE USER INPUT:\n"
    'This is an example desired response for the user input: "Elon Musk flying to mars"\n\n'
    "Voiceover: In the vast expanse of space, Elon Musk propels towards Mars, not just traversing distance, but also the boundaries of human ambition. This journey symbolizes a leap into the unknown, igniting dreams of interplanetary existence.\n"
    "Voice: Kore"
)


# Define allowed tools for this module
ALLOWED_TOOLS = [
    "google_video_gen",
]
HTTP_BAD_REQUEST = 400
HTTP_TOO_MANY_REQUESTS = 429
GEMINI_SERVICE = "gemini_api_key"


class InvalidRequestError(ValueError):
    """The request cannot be served as sent: the fault is in the buyer's input.

    The mech matches this class name to tell a client-side error apart from
    a tool or provider failure. Raise it only for faults in the request.
    """


# Prices are USD per million tokens from ai.google.dev/gemini-api/docs/pricing.
# The script model thinks by default and thinking is billed at the output
# rate. TTS output is audio tokens, 25 per second of speech; its rates are the
# introductory ones through 2026-12-31 and the listed ones from 2027-01-01.
SCRIPT_MODEL = "gemini-3.5-flash-lite"
SCRIPT_INPUT_USD_PER_M = 0.30
SCRIPT_OUTPUT_USD_PER_M = 2.50
TTS_MODEL = "gemini-3.8-flash-lite-tts"
TTS_RATES_UNTIL_2026 = (0.50, 6.00)
TTS_RATES_FROM_2027 = (1.00, 12.00)
TTS_RATE_CHANGE_TS = 1_798_761_600  # 2027-01-01T00:00:00Z


def tts_rates(now: float) -> Tuple[float, float]:
    """USD per million input and output TTS tokens that apply at ``now``.

    :param now: a Unix timestamp.
    :return: ``(input_per_m, output_per_m)``.
    """
    return TTS_RATES_FROM_2027 if now >= TTS_RATE_CHANGE_TS else TTS_RATES_UNTIL_2026


def record_api_cost(
    counter_callback: Optional[Callable[..., Any]], model: str, cost_usd: float
) -> None:
    """Report a token-free charge as ``call_cost`` to the mech cost callback; never fails the answer."""
    if counter_callback is None or cost_usd <= 0:
        return
    try:
        counter_callback(
            input_tokens=0,
            output_tokens=0,
            model=model,
            token_counter=lambda text, model: 0,
            call_cost=cost_usd,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Could not report the {model} cost to the mech: {exc}")


def _usage_cost_usd(response: Any, input_per_m: float, output_per_m: float) -> float:
    """USD for a Gemini response from its usage metadata at per-million-token prices.

    :param response: a ``generate_content`` response.
    :param input_per_m: USD per million prompt tokens.
    :param output_per_m: USD per million output (and thinking) tokens.
    :return: the charge, 0 when the response carries no usage.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0.0
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    # Thinking tokens are billed at the output rate.
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) + int(
        getattr(usage, "thoughts_token_count", 0) or 0
    )
    return prompt_tokens * input_per_m / 1e6 + output_tokens * output_per_m / 1e6


def with_key_rotation(func: Callable) -> Callable[..., MechResponse]:
    """Decorator for handling API key rotation and retries."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponse:
        api_keys = kwargs["api_keys"]
        # Ensure api_keys object has the expected methods
        if (
            not hasattr(api_keys, "max_retries")
            or not hasattr(api_keys, "rotate")
            or not hasattr(api_keys, "get")
        ):
            raise TypeError(
                "api_keys object does not have required methods (max_retries, rotate, get)"
            )

        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Execute the function with retries."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except anthropic.RateLimitError as e:
                return rotate_and_retry(e, ("anthropic",))
            except genai_errors.APIError as e:
                if e.code != HTTP_TOO_MANY_REQUESTS:
                    raise
                return rotate_and_retry(e, (GEMINI_SERVICE,))

        def rotate_and_retry(
            error: Exception, services: Tuple[str, ...]
        ) -> MechResponse:
            """Rotate the services with keys left and retry.

            With no key left for any of them, the error is raised again.

            :param error: the rate-limit error.
            :param services: the services the error is for.
            :return: the retry's response.
            """
            to_rotate = [s for s in services if retries_left.get(s, 0) > 0]
            if not to_rotate:
                raise error
            for service in to_rotate:
                retries_left[service] -= 1
                print(f"Rate limit error for {service}. Rotating key.")
                api_keys.rotate(service)
            return execute()

        return execute()

    return wrapper


def _is_request_rejection(error: genai_errors.ClientError) -> bool:
    """Whether Gemini rejected the request itself, not the operator's key or account.

    Gemini also answers 400 for an invalid API key and for a failed
    precondition (unsupported region, billing), which are not request faults.

    :param error: the Google GenAI client error.
    :return: True for a 400 INVALID_ARGUMENT that is not about the API key.
    """
    if error.code != HTTP_BAD_REQUEST or error.status != "INVALID_ARGUMENT":
        return False
    return "API_KEY_INVALID" not in str(error.details) and "API key" not in str(
        error.message or ""
    )


def _validate_inputs(tool: Optional[str], api_key: Optional[str]) -> None:
    """Validate tool and API key.

    :param tool: the requested tool name.
    :param api_key: the Gemini API key.
    :raises InvalidRequestError: if the tool is not supported.
    :raises ValueError: if the key is missing.
    """
    if tool not in ALLOWED_TOOLS:
        raise InvalidRequestError(f"Tool {tool} is not supported by this agent.")
    if not api_key:
        raise ValueError("Google API key (GEMINI_API_KEY) not provided.")


def download_file(url: str, local_filename: str) -> str:
    """Utility function to download a file from a URL to a local path."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def wave_file(
    filename: str,
    pcm: bytes,
    channels: int = 1,
    rate: int = 24000,
    sample_width: int = 2,
) -> None:
    """Write PCM audio data to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def _generate_text_with_gemini_flash(
    client: genai.Client,
    prompt: str,
    model_name: str = SCRIPT_MODEL,
    counter_callback: Any = None,
) -> str:
    """Generates text content using the Google GenAI Flash model."""
    config = types.GenerateContentConfig(max_output_tokens=500, temperature=0.1)
    print("requesting google TTS with prompt: ", prompt)

    response = client.models.generate_content(
        model=model_name, contents=prompt, config=config
    )
    text = response.text
    print("Raw response from _generate_text_with_gemini_flash: \n", text)
    assert text is not None, "Response text is None"
    record_api_cost(
        counter_callback,
        model_name,
        _usage_cost_usd(response, SCRIPT_INPUT_USD_PER_M, SCRIPT_OUTPUT_USD_PER_M),
    )
    return text


def get_audio_prompts(
    user_input: str, google_client: genai.Client, counter_callback: Any = None
) -> Dict[str, Any]:
    """Construct the message that includes the user input and uses Google GenAI Flash for generation."""
    # Combine system instruction and user input into a single prompt for `contents`
    full_prompt = (
        SYSTEM_INSTRUCTION_CONTENT.format(user_input)
        + "\nNow, generate the audio prompts based on the user input."
    )

    try:
        try:
            raw_content = _generate_text_with_gemini_flash(
                google_client,
                prompt=full_prompt,
                counter_callback=counter_callback,
            )
        except genai_errors.ClientError as e:
            if not _is_request_rejection(e):
                raise
            raise InvalidRequestError(
                f"The model provider rejected the request: {e.message}"
            ) from e
        print("Raw content from _generate_text_with_gemini_flash: \n", raw_content)
        content = raw_content.strip()
        print("Stripped content: \n", content)

        # Parse the plain text content to extract voiceover script and voice
        voiceover_script_match = re.search(r"Voiceover: (.*)", content, re.IGNORECASE)
        voice_match = re.search(r"Voice: (.*)", content, re.IGNORECASE)

        voiceover_script = (
            voiceover_script_match.group(1).strip() if voiceover_script_match else ""
        )
        voice = (
            voice_match.group(1).strip() if voice_match else "Kore"
        )  # Default to Kore if not found

        ALLOWED_VOICES = [
            "achernar",
            "achird",
            "algenib",
            "algieba",
            "alnilam",
            "aoede",
            "autonoe",
            "callirrhoe",
            "charon",
            "despina",
            "enceladus",
            "erinome",
            "fenrir",
            "gacrux",
            "iapetus",
            "kore",
            "laomedeia",
            "leda",
            "orus",
            "puck",
            "pulcherrima",
            "rasalgethi",
            "sadachbia",
            "sadaltager",
            "schedar",
            "sulafat",
            "umbriel",
            "vindemiatrix",
            "zephyr",
            "zubenelgenubi",
        ]
        if voice not in ALLOWED_VOICES:
            logging.warning(
                f"Requested voice '{voice}' is not allowed. Defaulting to 'Kore'."
            )
            voice = "Kore"

        if not voiceover_script:
            raise ValueError("Could not extract voiceover script from model response.")

        json_object = {"voiceover_script": voiceover_script, "voice": voice}

        print("Audio prompts from Google GenAI Flash for TTS: ", json_object)
        return json_object

    except Exception as e:
        logging.error(
            "Failed to get a response from Google GenAI Flash for audio prompts: %s", e
        )
        raise


def get_audio_duration(audio_file_path: str) -> int:
    """Get the duration of an audio file in seconds, rounded up."""
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"The audio file {audio_file_path} does not exist.")

    with AudioFileClip(audio_file_path) as audio_clip:
        duration_seconds = math.ceil(audio_clip.duration)

    print(f"The audio duration is {duration_seconds}")
    return duration_seconds


def compose_final_video(  # pylint: disable=too-many-positional-arguments,too-many-locals
    video_path: str,
    voiceover_path: str,
    file_prefix: str,
    voiceover_volume: float = 1.0,
    soundtrack_volume: float = 0.2,
    fadeout_duration: float = 1.0,
) -> str:
    """Compose the final video."""
    video_clip = VideoFileClip(video_path)

    # Audio Clips
    voiceover_clip = AudioFileClip(voiceover_path)
    voiceover_clip = voiceover_clip.volumex(voiceover_volume)

    def make_silence(_t: float) -> list:  # noqa: E731
        """Return silent audio frame."""
        return [0]

    soundtrack_clip = (
        AudioClip(make_frame=make_silence, duration=video_clip.duration)
        .set_fps(44100)
        .volumex(soundtrack_volume)
    )

    # Determine the longest duration
    print(f"Original Video Duration: {video_clip.duration}")
    print(f"Original Voiceover Duration: {voiceover_clip.duration}")
    print(f"Original Soundtrack Duration: {soundtrack_clip.duration}")

    longest_duration = max(
        video_clip.duration, voiceover_clip.duration, soundtrack_clip.duration
    )

    # Extend video clip with a black frame if needed
    if video_clip.duration < longest_duration:
        black_clip = ColorClip(
            size=video_clip.size,
            color=(0, 0, 0),
            duration=longest_duration - video_clip.duration,
        )
        video_clip = concatenate_videoclips([video_clip, black_clip])
        print(f"Extended Video Duration: {video_clip.duration}")

    # Extend audio clips with silence if needed
    if voiceover_clip.duration < longest_duration:
        silence_duration = longest_duration - voiceover_clip.duration
        silence_clip = AudioClip(
            make_frame=make_silence, duration=silence_duration
        ).set_fps(44100)
        voiceover_clip = concatenate_audioclips([voiceover_clip, silence_clip])
        print(f"Extended Voiceover Duration: {voiceover_clip.duration}")
    if soundtrack_clip.duration < longest_duration:
        silence_duration = longest_duration - soundtrack_clip.duration
        silence_clip = AudioClip(
            make_frame=make_silence, duration=silence_duration
        ).set_fps(44100)
        soundtrack_clip = concatenate_audioclips([soundtrack_clip, silence_clip])
        print(f"Extended Soundtrack Duration: {soundtrack_clip.duration}")

    # Apply fade out to the soundtrack
    soundtrack_clip = soundtrack_clip.fx(audio_fadeout, fadeout_duration)

    final_audio = CompositeAudioClip([soundtrack_clip, voiceover_clip])

    final_video = video_clip.set_audio(final_audio)

    filename = f"{file_prefix}.mp4"
    final_video.write_videofile(filename, codec="libx264", audio_codec="aac", fps=24)

    return filename


def _generate_audio_with_gemini_tts(
    client: genai.Client, text: str, voice_name: str, counter_callback: Any = None
) -> bytes:
    """Generates audio data using the Google GenAI TTS API."""
    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
        ),
    )
    candidates = response.candidates
    assert candidates is not None, "Response candidates is None"
    content = candidates[0].content
    assert content is not None, "Candidate content is None"
    parts = content.parts
    assert parts is not None, "Content parts is None"
    inline_data = parts[0].inline_data
    assert inline_data is not None, "Inline data is None"
    data = inline_data.data
    assert data is not None, "Inline data bytes is None"
    record_api_cost(
        counter_callback,
        TTS_MODEL,
        _usage_cost_usd(response, *tts_rates(time.time())),
    )
    return data


@with_key_rotation
def run(  # pylint: disable=too-many-locals
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Runs the Google video generation task using genai.Client and adds audio."""
    prompt = kwargs["prompt"]
    api_keys = kwargs["api_keys"]
    google_api_key = api_keys.get(GEMINI_SERVICE, None)
    tool = kwargs.get("tool")
    counter_callback = kwargs.get("counter_callback", None)
    model_name = TTS_MODEL

    _validate_inputs(tool, google_api_key)

    # Initialize clients
    google_client = genai.Client(api_key=google_api_key)

    file_prefix = prompt[:20].replace(" ", "_").replace("-", "_").lower()

    temp_files = []
    try:
        # Step 1: Video generation (pending implementation)

        # Step 2: Get audio prompts from OpenAI
        try:
            audio_prompts = get_audio_prompts(
                user_input=prompt,
                google_client=google_client,
                counter_callback=counter_callback,
            )
            voiceover_script = audio_prompts["voiceover_script"]
            voice_choice = audio_prompts["voice"]
            print(f"Voiceover Script: {voiceover_script}")
            print(f"Voice Choice: {voice_choice}")

        except Exception as e:
            print(f"Error getting audio prompts: {e}")
            raise

        # Step 3: Process voiceover using Google TTS
        try:
            voice_name = voice_choice  # Use the voice chosen by the model
            audio_bytes = _generate_audio_with_gemini_tts(
                google_client, voiceover_script, voice_name, counter_callback
            )
            voiceover_filename = f"{file_prefix}_voiceover.wav"
            wave_file(voiceover_filename, audio_bytes)
            voiceover_path = voiceover_filename  # Path is the filename itself now
            temp_files.append(voiceover_path)
        except Exception as e:
            print(f"Error processing voiceover: {e}")
            raise

        # Step 4 & 5: Video merging and IPFS upload
        # Video generation step is pending implementation.
        ipfs_tool = IPFSTool()
        _, audio_hash, _ = ipfs_tool.add(voiceover_path, wrap_with_directory=False)

        result_data = {"audio_hash": audio_hash, "prompt": prompt, "model": model_name}
        return json.dumps(result_data), prompt, None, counter_callback
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
