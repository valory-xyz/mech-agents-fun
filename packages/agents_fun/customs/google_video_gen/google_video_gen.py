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
from google.api_core import (
    exceptions as google_exceptions,  # type: ignore[import-not-found]
)
from google.genai import types  # type: ignore[import-not-found]
from moviepy.audio.AudioClip import AudioClip, concatenate_audioclips
from moviepy.audio.fx.audio_fadeout import audio_fadeout
from moviepy.editor import AudioFileClip, ColorClip, CompositeAudioClip, VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

# Define MechResponse type alias matching the other tools
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

SYSTEM_INSTRUCTION_CONTENT = (
    f'Based on the USER INPUT: "{0}", please provide a short voiceover script and a choice of voice for the voiceover. Format your response as plain text with two lines: "Voiceover: [script]" and "Voice: [voice_name]". Each field should contain the respective content as described below.\n\n'
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
            error_msg = "api_keys object does not have required methods (max_retries, rotate, get)"
            prompt_val = kwargs.get("prompt", "N/A")
            callback_val = kwargs.get("counter_callback", None)
            return error_msg, prompt_val, None, callback_val, None  # Return 5 elements

        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Execute the function with retries."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except (
                anthropic.RateLimitError,
                google_exceptions.ResourceExhausted,
                google_exceptions.TooManyRequests,
            ) as e:
                service = "google_api_key"
                if isinstance(e, anthropic.RateLimitError):
                    service = "anthropic"
                if retries_left.get(service, 0) <= 0:
                    print(f"No retries left for service: {service}")
                    raise e

                retries_left[service] -= 1
                print(
                    f"Rate limit error for {service}. Retries left: {retries_left[service]}. Rotating key."
                )
                api_keys.rotate(service)
                return execute()
            except (
                google_exceptions.GoogleAPIError
            ) as e:  # Specific catch for other GoogleAPIErrors
                # If not a 500 error, or no code attribute, re-raise immediately
                if not hasattr(e, "code") or e.code != 500:  # pylint: disable=no-member
                    raise e
                service = "google_api_key"
                # If no retries left for this service, raise.
                if retries_left.get(service, 0) <= 0:
                    raise e

                # Retries are available, proceed with retry logic.
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"An unexpected error occurred: {e}")
                error_response = str(e)
                prompt_value = kwargs.get(
                    "prompt", "Prompt not available in error context"
                )
                callback_value = kwargs.get("counter_callback", None)
                return error_response, prompt_value, None, callback_value, api_keys

        return execute()

    return wrapper


def _validate_inputs(
    tool: Optional[str], api_key: Optional[str], prompt: str, counter_callback: Any
) -> Optional[Tuple[str, str, None, Any]]:
    """Validate tool and API key."""
    if tool not in ALLOWED_TOOLS:
        return (
            f"Tool {tool} is not supported by this agent.",
            prompt,
            None,
            counter_callback,
        )

    if not api_key:
        return (
            "Google API key (GEMINI_API_KEY) not provided.",
            prompt,
            None,
            counter_callback,
        )
    return None


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
    model_name: str = "gemini-2.0-flash",
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
    return text


def get_audio_prompts(user_input: str, google_client: genai.Client) -> Dict[str, Any]:
    """Construct the message that includes the user input and uses Google GenAI Flash for generation."""
    # Combine system instruction and user input into a single prompt for `contents`
    full_prompt = (
        SYSTEM_INSTRUCTION_CONTENT.format(user_input)
        + "\nNow, generate the audio prompts based on the user input."
    )

    try:
        raw_content = _generate_text_with_gemini_flash(
            google_client,
            prompt=full_prompt,
        )
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


def _generate_video_from_google_api(
    client: genai.Client, prompt: str, model_name: str, counter_callback: Any
) -> Tuple[Optional[bytes], Optional[Tuple[str, str, None, Any]]]:
    """Generates video data using the Google API and handles initial response validation."""
    operation = client.models.generate_videos(
        model=model_name,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            person_generation="allow_adult",
            aspect_ratio="16:9",
        ),
    )

    print("Waiting for video generation operation to complete...")
    while not operation.done:
        time.sleep(20)
        print(
            "Waiting for video generation operation to complete... will check again in 20 seconds"
        )
        operation = client.operations.get(operation)

    result = operation.result
    print(f"Video generation operation result: {result}")
    if result is None or not result.generated_videos:
        return None, (
            "No video data found in the response (generated_videos is empty).",
            prompt,
            None,
            counter_callback,
        )

    video_file = result.generated_videos[0].video
    assert video_file is not None, "Video file is None"

    # Download the video content
    print(f"Downloading video file from URI: {video_file.uri}")
    client.files.download(file=video_file)
    video_bytes = video_file.video_bytes

    return video_bytes, None


def _save_video_and_upload_to_ipfs(
    video_data: bytes, prompt: str, model_name: str, counter_callback: Any
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, str]:
    """Saves the video data to a temporary file, uploads to IPFS, and returns the path."""
    temp_video_path = f"temp_generated_video_{os.getpid()}.mp4"
    try:
        with open(temp_video_path, "wb") as f:
            f.write(video_data)

        ipfs_tool = IPFSTool()
        _, video_hash, _ = ipfs_tool.add(temp_video_path, wrap_with_directory=False)

        result_data = {"video_hash": video_hash, "prompt": prompt, "model": model_name}
        # The `temp_video_path` is returned to be used for video merging.
        return json.dumps(result_data), prompt, None, counter_callback, temp_video_path
    except FileNotFoundError:
        return (
            "IPFS tool not found or not configured correctly.",
            prompt,
            None,
            counter_callback,
            "",  # Return empty string for temp_video_path on error
        )


def _generate_audio_with_gemini_tts(
    client: genai.Client, text: str, voice_name: str
) -> bytes:
    """Generates audio data using the Google GenAI TTS API."""
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
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
    return data


@with_key_rotation
def run(  # pylint: disable=too-many-locals
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Runs the Google video generation task using genai.Client and adds audio."""
    prompt = kwargs["prompt"]
    api_keys = kwargs["api_keys"]
    google_api_key = api_keys.get("gemini_api_key", None)
    tool = kwargs.get("tool")
    counter_callback = kwargs.get("counter_callback", None)
    model_name = "veo-2.0-generate-001"

    # Initialize clients
    google_client = genai.Client(api_key=google_api_key)

    file_prefix = prompt[:20].replace(" ", "_").replace("-", "_").lower()

    validation_error = _validate_inputs(tool, google_api_key, prompt, counter_callback)
    if validation_error:
        return validation_error

    temp_files = []
    try:
        # Step 1: Video generation (pending implementation)

        # Step 2: Get audio prompts from OpenAI
        try:
            audio_prompts = get_audio_prompts(
                user_input=prompt, google_client=google_client
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
                google_client, voiceover_script, voice_name
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

    except google_exceptions.GoogleAPIError as e:
        print(f"Google API error: {e}")
        return f"Google API error: {e}", prompt, None, counter_callback
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"An unexpected error occurred: {e}")
        return f"An error occurred: {e}", prompt, None, counter_callback
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
