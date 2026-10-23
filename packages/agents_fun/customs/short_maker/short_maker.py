# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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
"""This module contains the implementation of the short_maker tool."""

import base64
import functools
import json
import logging
import math
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import anthropic
import openai
import requests
from aea_cli_ipfs.ipfs_utils import IPFSTool
from googleapiclient.errors import HttpError as GoogleApiClientHttpError
from moviepy.audio.AudioClip import AudioClip, concatenate_audioclips
from moviepy.audio.fx.audio_fadeout import audio_fadeout
from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip
from moviepy.video.VideoClip import ColorClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from openai import OpenAI
from replicate import Client as ReplicateClient

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

client: Optional[OpenAI] = None

ALLOWED_TOOLS = [
    "short_maker",
]


class InvalidRequestError(ValueError):
    """The request cannot be served as sent: the fault is in the buyer's input.

    The mech matches this class name to tell a client-side error apart from
    a tool or provider failure. Raise it only for faults in the request.
    """


AUDIO_PROMPT_KEYS = ("voiceover_script", "soundtrack_prompt", "voice")

REPLICATE_MUSIC_GEN = (
    "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb"
)

REPLICATE_MINIMAX_TTS = "minimax/speech-02-turbo"
# Image-to-video at 480p, 81 frames at 16 fps.
REPLICATE_WAN_I2V = "wan-video/wan-2.2-i2v-fast"

# gpt-6-luna is priced by the mech per token, so its usage is reported as
# tokens. The image model is reported as a charge from its usage at the list
# prices below (developers.openai.com/api/docs/pricing, USD per million tokens:
# text input, image output).
TEXT_MODEL = "gpt-6-luna"
IMAGE_MODEL = "gpt-image-2.5-flare"
IMAGE_INPUT_USD_PER_M = 5.0
IMAGE_OUTPUT_USD_PER_M = 30.0
# Replicate prices: Wan 2.2 i2v fast per 480p video (replicate.com/blog/wan-22);
# MiniMax speech-02-turbo per 1,000 input tokens, one token per character
# (its Replicate model page); MusicGen is a community model billed by A100
# time (replicate.com/pricing), so its charge is the prediction's
# ``predict_time`` at that rate.
WAN_COST_PER_VIDEO_USD = 0.05
MINIMAX_TTS_USD_PER_1K_TOKENS = 0.06
A100_USD_PER_SECOND = 0.0014


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


def record_chat_usage(
    counter_callback: Optional[Callable[..., Any]], model: str, response: Any
) -> None:
    """Report a chat completion's token usage to the mech cost callback.

    The mech prices ``model`` from its own table, so only the counts travel.

    :param counter_callback: the mech cost callback, or None outside the mech.
    :param model: the chat model that was called.
    :param response: the chat completion response carrying ``usage``.
    """
    usage = getattr(response, "usage", None)
    if counter_callback is None or usage is None:
        return
    try:
        counter_callback(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model=model,
            token_counter=lambda text, model: 0,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Could not report the {model} usage to the mech: {exc}")


def with_key_rotation(func: Callable) -> Callable[..., MechResponse]:
    """Wrap function with API key rotation logic."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponse:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except anthropic.RateLimitError as e:
                # try with a new key again
                service = "anthropic"
                if retries_left[service] <= 0:
                    raise e
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except openai.RateLimitError as e:
                # rotate only the services the keychain has keys left for
                services = [
                    service
                    for service in ("openai", "openrouter")
                    if retries_left.get(service, 0) > 0
                ]
                if not services:
                    raise e
                for service in services:
                    retries_left[service] -= 1
                    api_keys.rotate(service)
                return execute()
            except GoogleApiClientHttpError as e:
                # try with a new key again
                rate_limit_exceeded_code = 429
                if e.status_code != rate_limit_exceeded_code:
                    raise e
                service = "google_api_key"
                if retries_left[service] <= 0:
                    raise e
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()

        mech_response = execute()
        return mech_response

    return wrapper


def download_file(url: str, local_filepath: str) -> str:
    """Utility function to download a file from a URL to a local path."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filepath


def get_audio_prompts(
    user_input: str, engine: str = TEXT_MODEL, counter_callback: Any = None
) -> Dict[str, Any]:
    """Construct the message that includes the user input"""
    message = {
        "role": "system",
        "content": f'Based on the USER INPUT: "{user_input}", please provide a short voiceover script, a prompt for generating a short soundtrack to go with the script, as well as a choice of voice for the voiceover. Format your response as a JSON object with two fields: "voiceover_script" and "soundtrack_prompt". Each field should contain the respective content as described below.\n\n'
        '- For the "voiceover_script": Use the user input to create a short script no longer than 10 seconds long (around 15-30 words), which has a mindblowing insight. The script should be in direct speech format suitable for text-to-speech AI without any stage directions. Do not just repeat the user input\n\n'
        '- For the "soundtrack_prompt": Devise a prompt that would guide an AI to generate a soundtrack that captures the mood implied by the user input.\n\n'
        "- For the \"voice\": Choose what type of voice would suit the video. If the video deals with emotional or technological themes output 'English_CalmWoman'. If it deals with deep questions, output 'English_ManWithDeepVoice'. If it deals with humourous themes, output 'English_Comedian'. Otherwise, output 'English_Graceful_Lady'. These are the names of the voices for the model. Only output the name, nothing else. \n\n"
        "Please structure your response in the following JSON format:\n\n"
        "{\n"
        '  "voiceover_script": "[Insert narrative script here]",\n'
        '  "soundtrack_prompt": "[Insert soundtrack prompt here]",\n'
        '  "voice": "[Insert name of voice here]",\n'
        "}\n\n"
        "EXAMPLE USER INPUT:\n"
        'This is an example desired response for the user input: "Elon Musk flying to mars"\n\n'
        "{\n"
        '  "voiceover_script": "In the vast expanse of space, Elon Musk propels towards Mars, not just traversing distance, but also the boundaries of human ambition. This journey symbolizes a leap into the unknown, igniting dreams of interplanetary existence.",\n'
        '  "soundtrack_prompt": "Create an awe-inspiring and futuristic soundtrack that combines elements of space-themed ambience, such as soft electronic tones and ethereal sounds, with a hint of suspense to reflect the groundbreaking venture of flying to Mars.",\n'
        '  "voice": "freeman",\n'
        "}\n\n",
    }
    try:
        # Send the message to the chat completions endpoint
        assert client is not None, "OpenAI client not initialized"
        response = client.chat.completions.create(
            model=engine, messages=[message]  # type: ignore[list-item]
        )

        record_chat_usage(counter_callback, engine, response)
        # Parse the JSON content from the response
        content = response.choices[0].message.content
        assert content is not None, "Response content is None"

        # Load the content as a JSON object to ensure proper JSON formatting
        json_object = json.loads(content)

        print(f"Audio Prompts: {json_object}")
        return json_object

    except Exception as e:
        logging.error("Failed to get a response from OpenAI: %s", e)
        raise


def get_shot_prompts(
    user_input: str,
    voiceover_length: float,
    engine: str = TEXT_MODEL,
    counter_callback: Any = None,
) -> Dict[str, Any]:
    """Send a prompt to OpenAI API and return video shot prompts as JSON."""

    # Calculate the number of shots needed
    number_of_shots = math.ceil(voiceover_length / 2.5)

    # Construct the message that includes the user input
    message = {
        "role": "system",
        "content": f'Based on the USER INPUT: "{user_input}", please provide {number_of_shots} prompts for generating shots in a video clip, starting directly with the type of shot or the subject, without using verbs at the beginning. Format your response as a JSON object with the following fields: "shot1_prompt", "shot2_prompt", "shot3_prompt", etc. Each field should contain a description of a shot based on the user input.\n\n'
        "Please structure your response in the following JSON format:\n\n"
        "{\n",
    }

    # Add placeholders for each shot
    for i in range(1, number_of_shots + 1):
        message[
            "content"
        ] += f'  "shot{i}_prompt": "[Insert prompt for shot{i} here]",\n'

    # Close the JSON structure
    message["content"] = message["content"].strip(",\n") + "\n}"

    # Send the message to the chat completions endpoint
    assert client is not None, "OpenAI client not initialized"
    response = client.chat.completions.create(
        model=engine, messages=[message]  # type: ignore[list-item]
    )
    record_chat_usage(counter_callback, engine, response)

    # Parse the JSON content from the response
    content = response.choices[0].message.content
    assert content is not None, "Response content is None"

    # Load the content as a JSON object to ensure proper JSON formatting
    json_object = json.loads(content)

    print(f"Shot Prompts: {json_object}")
    return json_object


def process_voiceover(
    replicate_client: ReplicateClient,
    text: str,
    voice: str,
    counter_callback: Any = None,
) -> str:
    """Narrate the provided text"""
    url = replicate_client.run(
        REPLICATE_MINIMAX_TTS,
        input={
            "text": text,
            "pitch": 0,
            "speed": 1,
            "volume": 1,
            "bitrate": 128000,
            "channel": "mono",
            "emotion": "auto",
            "voice_id": voice,
            "sample_rate": 32000,
            "audio_format": "mp3",
            "language_boost": "English",
            "subtitle_enable": False,
            "english_normalization": False,
        },
    )
    record_api_cost(
        counter_callback,
        REPLICATE_MINIMAX_TTS,
        len(text) / 1000 * MINIMAX_TTS_USD_PER_1K_TOKENS,
    )
    return url


def process_soundtrack(
    replicate_client: ReplicateClient,
    prompt: str,
    duration: int = 10,
    counter_callback: Any = None,
) -> str:
    """Get a soundtrack for the provided prompt, charged by the compute time it used."""
    prediction = replicate_client.predictions.create(
        version=REPLICATE_MUSIC_GEN.split(":", 1)[1],
        input={
            "seed": 3442726813,
            "top_k": 250,
            "top_p": 0,
            "prompt": prompt,
            "duration": duration + 1,
            "temperature": 1,
            "continuation": False,
            "model_version": "large",
            "output_format": "wav",
            "continuation_end": 9,
            "continuation_start": 7,
            "normalization_strategy": "peak",
            "classifier_free_guidance": 3,
        },
    )
    prediction.wait()
    if prediction.status != "succeeded":
        raise RuntimeError(
            f"MusicGen prediction {prediction.status}: {prediction.error}"
        )
    predict_time = float((prediction.metrics or {}).get("predict_time") or 0.0)
    record_api_cost(
        counter_callback, REPLICATE_MUSIC_GEN, predict_time * A100_USD_PER_SECOND
    )
    return str(prediction.output)


def generate_images(
    shot_prompts: Dict[str, Any],
    save_dir: str,
    engine: str = IMAGE_MODEL,
    counter_callback: Any = None,
) -> Dict[str, str]:
    """Generate one image per shot prompt and save each as a PNG under ``save_dir``.

    A shot that fails or comes back without an image is left out.

    :param shot_prompts: shot name to prompt.
    :param save_dir: directory the PNG files are written to.
    :param engine: the OpenAI image model.
    :param counter_callback: the mech cost callback, or None outside the mech.
    :return: shot name to image path for the shots that were generated.
    :raises ValueError: when no shot produced an image.
    """
    generated_images: Dict[str, str] = {}
    assert client is not None, "OpenAI client not initialized"
    for shot_number, prompt in shot_prompts.items():
        try:
            response = client.images.generate(
                model=engine,
                prompt=prompt,
                size="1024x1024",
                quality="medium",
                n=1,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                record_api_cost(
                    counter_callback,
                    engine,
                    int(getattr(usage, "input_tokens", 0) or 0)
                    * IMAGE_INPUT_USD_PER_M
                    / 1e6
                    + int(getattr(usage, "output_tokens", 0) or 0)
                    * IMAGE_OUTPUT_USD_PER_M
                    / 1e6,
                )
            encoded = response.data[0].b64_json if response.data else None
            if not encoded:
                print(f"Skipping {shot_number}, no image data in the response")
                continue
            image_path = os.path.join(save_dir, f"{shot_number}.png")
            with open(image_path, "wb") as handle:
                handle.write(base64.b64decode(encoded))
        except (openai.OpenAIError, OSError, ValueError, LookupError, TypeError) as e:
            print(f"Skipping {shot_number}, no image generated: {e}")
            continue
        generated_images[shot_number] = image_path
        print(f"Generated image for {shot_number}: {image_path}")

    if not generated_images:
        raise ValueError("No shot image could be generated.")
    return generated_images


def process_first_shots(
    replicate_client: ReplicateClient,
    image_path: str,
    prompt: str,
    counter_callback: Any = None,
) -> Optional[str]:
    """Animate one shot image into a short clip with Wan 2.2 on Replicate.

    :param replicate_client: the Replicate client.
    :param image_path: the PNG to animate.
    :param prompt: the shot description that guides the motion.
    :param counter_callback: the mech cost callback, or None outside the mech.
    :return: the clip URL, or None when the call failed.
    """
    if not image_path or not os.path.exists(image_path):
        print(f"Missing shot image: {image_path}")
        return None

    try:
        with open(image_path, "rb") as image:
            video_url = replicate_client.run(
                REPLICATE_WAN_I2V,
                input={
                    "image": image,
                    "prompt": prompt,
                    "resolution": "480p",
                    "num_frames": 81,
                    "frames_per_second": 16,
                    "go_fast": True,
                },
            )
        record_api_cost(counter_callback, REPLICATE_WAN_I2V, WAN_COST_PER_VIDEO_USD)
        print(f"video_url: {video_url}")
        return str(video_url) if video_url else None
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error processing video shot: {e}")
        return None


def get_audio_duration(audio_file_path: str) -> int:
    """Get the duration of an audio file in seconds, rounded up."""
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"The audio file {audio_file_path} does not exist.")

    with AudioFileClip(audio_file_path) as audio_clip:
        duration_seconds = math.ceil(audio_clip.duration)

    print(f"The audio duration is {duration_seconds}")
    return duration_seconds


def make_silence(_t: float) -> List[int]:
    """Generate silence"""
    return [0]


def compose_final_video(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    video_shots: List,
    voiceover_path: str,
    soundtrack_path: str,
    voiceover_volume: float = 1.0,
    soundtrack_volume: float = 0.2,
    fadeout_duration: float = 1.0,
    save_dir: str = ".",
) -> str:
    """Compose the final video."""
    clips = [VideoFileClip(shot) for shot in video_shots]
    final_video_clip = concatenate_videoclips(clips)

    # Audio Clips
    voiceover_clip = AudioFileClip(voiceover_path)
    voiceover_clip = voiceover_clip.volumex(voiceover_volume)
    soundtrack_clip = AudioFileClip(soundtrack_path)
    soundtrack_clip = soundtrack_clip.volumex(soundtrack_volume)

    # Determine the longest duration
    print(f"Original Video Duration: {final_video_clip.duration}")
    print(f"Original Voiceover Duration: {voiceover_clip.duration}")
    print(f"Original Soundtrack Duration: {soundtrack_clip.duration}")

    longest_duration = max(
        final_video_clip.duration, voiceover_clip.duration, soundtrack_clip.duration
    )

    # Extend video clip with a black frame if needed
    if final_video_clip.duration < longest_duration:
        black_clip = ColorClip(
            size=final_video_clip.size,
            color=(0, 0, 0),
            duration=longest_duration - final_video_clip.duration,
        )
        final_video_clip = concatenate_videoclips([final_video_clip, black_clip])
        print(f"Extended Video Duration: {final_video_clip.duration}")

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

    final_video = final_video_clip.set_audio(final_audio)

    final_video_path = os.path.join(save_dir, "final.mp4")
    final_video.write_videofile(
        final_video_path, codec="libx264", audio_codec="aac", fps=24
    )

    return final_video_path


def cleanup_tempdir(tmpdir: str) -> None:
    """Remove the temporary directory and its contents."""
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
        print(f"Cleaned up temporary directory: {tmpdir}")


@with_key_rotation
def run(  # pylint: disable=too-many-locals
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    user_input = kwargs["prompt"]
    openai_key = kwargs["api_keys"]["openai"]
    counter_callback = kwargs.get("counter_callback", None)
    tool = kwargs.get("tool")
    if tool not in ALLOWED_TOOLS:
        raise InvalidRequestError(f"Tool {tool} is not supported by this agent.")

    # Initialize OpenAI client with the provided key
    global client  # pylint: disable=global-statement
    client = OpenAI(api_key=openai_key)

    replicate_key = kwargs["api_keys"]["replicate"]
    client_replicate = ReplicateClient(replicate_key)

    tmpdir = tempfile.mkdtemp(prefix="short_maker_")
    # Step 2: Get audio prompts
    print(f"Getting audio prompts for input: {user_input}")
    audio_prompts = get_audio_prompts(user_input, counter_callback=counter_callback)
    missing = [
        key
        for key in AUDIO_PROMPT_KEYS
        if not isinstance(audio_prompts.get(key), str) or not audio_prompts[key]
    ]
    if missing:
        raise ValueError(f"Audio prompts are missing {', '.join(missing)}.")

    # Step 3: Process voiceover
    print("Processing voiceover...")
    voiceover_script = audio_prompts["voiceover_script"]
    voice_choice = audio_prompts["voice"]
    voiceover = process_voiceover(
        client_replicate, voiceover_script, voice_choice, counter_callback
    )

    # Download voiceover and get duration
    print("Downloading voiceover...")
    voiceover_path = os.path.join(tmpdir, "voiceover.mp3")
    voiceover_path = download_file(voiceover, voiceover_path)
    voiceover_length = get_audio_duration(voiceover_path)

    # Step 5: Get shot prompts
    print("Getting shot prompts...")
    shot_prompts = get_shot_prompts(
        user_input, voiceover_length, counter_callback=counter_callback
    )

    # Step 6: Generate one image per shot
    print("Generating shot images...")
    image_paths = generate_images(
        shot_prompts, tmpdir, counter_callback=counter_callback
    )
    first_shot_path = list(image_paths.values())[0]

    # Steps 7 & 8: Process video shots
    video_urls = [
        url
        for url in (
            process_first_shots(
                client_replicate,
                path,
                str(shot_prompts.get(shot, "")),
                counter_callback,
            )
            for shot, path in image_paths.items()
        )
        if url is not None
    ]
    if not video_urls:
        raise ValueError("No video shot could be generated.")

    # Step 9: Process soundtrack
    soundtrack_prompt = audio_prompts["soundtrack_prompt"]
    soundtrack = process_soundtrack(
        client_replicate, soundtrack_prompt, voiceover_length, counter_callback
    )

    # Download all video shots and soundtrack
    print("Downloading video shots and soundtrack...")
    video_files = [
        download_file(url, os.path.join(tmpdir, f"shot_{i}.mp4"))
        for i, url in enumerate(video_urls)
    ]

    soundtrack_path = os.path.join(tmpdir, "soundtrack.mp3")
    soundtrack_path = download_file(soundtrack, soundtrack_path)

    # Step 10: Compose final video
    print(
        f"Composing final video with parameters: {video_files=}, {voiceover_path=}, {soundtrack_path=}"
    )
    final_video_path = compose_final_video(
        video_files, voiceover_path, soundtrack_path, save_dir=tmpdir
    )
    print(f"Final video composed at: {final_video_path}")

    ipfs_tool = IPFSTool()
    _, video_hash_, _ = ipfs_tool.add(final_video_path, wrap_with_directory=False)
    image_hash_ = ipfs_tool.client.add(  # pylint: disable=unexpected-keyword-arg
        first_shot_path, cid_version=1, wrap_with_directory=False
    )["Hash"]

    cleanup_tempdir(tmpdir)

    print(f"Stored the output on: {video_hash_}")

    body = {
        "video": video_hash_,
        "image": image_hash_,
        "prompt": user_input,
    }
    # The mech records the last model the callback saw; the script model is
    # the one that produced the content, the rest are services it paid.
    if counter_callback is not None and hasattr(counter_callback, "actual_model"):
        counter_callback.actual_model = TEXT_MODEL

    # response text, original prompt, metadata, callback
    return json.dumps(body), user_input, None, counter_callback
