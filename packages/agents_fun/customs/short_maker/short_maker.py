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


ALLOWED_TOOLS = [
    "short_maker",
]

REPLICATE_MUSIC_GEN = (
    "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb"
)

REPLICATE_MINIMAX_TTS = "minimax/speech-02-turbo"
REPLICATE_STABILITY_STABLE_VIDEO_DIFFUSION = "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"


def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
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
                # try with a new key again
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
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
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


def download_file(url: str, local_filepath: str):
    """Utility function to download a file from a URL to a local path."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filepath


def get_audio_prompts(user_input: str, engine: str = "gpt-3.5-turbo") -> Dict[str, Any]:
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
        response = client.chat.completions.create(model=engine, messages=[message])

        # Parse the JSON content from the response
        content = response.choices[0].message.content

        # Load the content as a JSON object to ensure proper JSON formatting
        json_object = json.loads(content)

        print(f"Audio Prompts: {json_object}")
        return json_object

    except Exception as e:
        logging.error("Failed to get a response from OpenAI: %s", e)
        raise


def get_shot_prompts(
    user_input: str, voiceover_length: float, engine: str = "gpt-3.5-turbo"
):
    """
    Sends a prompt to the OpenAI API and returns the response as a JSON object for video shot prompts, without verbs at the start.
    """

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

    try:
        # Send the message to the chat completions endpoint
        response = client.chat.completions.create(model=engine, messages=[message])

        # Parse the JSON content from the response
        content = response.choices[0].message.content

        # Load the content as a JSON object to ensure proper JSON formatting
        json_object = json.loads(content)

        print(f"Shot Prompts: {json_object}")
        return json_object
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_voiceover(client: ReplicateClient, text: str, voice: str) -> str:
    """Narrate the provided text"""
    url = client.run(
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
    return url


def process_soundtrack(client: ReplicateClient, prompt: str, duration: int = 10) -> str:
    """Get a soundtrack for the provided prompt."""
    url = client.run(
        REPLICATE_MUSIC_GEN,
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
    return url


def generate_images_with_dalle(
    shot_prompts: Dict,
    api_key: str,
    engine: str = "dall-e-3",
    number_of_images: int = 1,
):
    """
    uses the shot prompts to generate images from dalle. These images will be used to generate videos with stability ai
    """
    generated_images = {}
    for shot_number, prompt in shot_prompts.items():
        try:
            json_params = dict(
                model=engine,
                prompt=prompt,
                size="1024x1024",
                n=number_of_images,
            )
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Stability-Client-ID": "mechs-tool",
                },
                json=json_params,
            )
            response.raise_for_status()
            body = response.json()
            if body["data"] and len(body["data"]) > 0:
                image_url = body["data"][0]["url"]
                generated_images[shot_number] = image_url
            else:
                generated_images[shot_number] = "No image data in response"

            print(f"Generated image for {shot_number}: {generated_images[shot_number]}")

        except Exception as e:
            print(f"Error generating image for {shot_number}: {e}")
            generated_images[shot_number] = str(e)

    return generated_images


def process_first_shots(client: ReplicateClient, shot_url: str) -> str:
    """
    Processes the first video shots using the Replicate API based on the given video prompt.
    """
    if not shot_url or not shot_url.startswith("http"):
        print(f"Invalid URL: {shot_url}")
        return None

    try:
        video_url = client.run(
            REPLICATE_STABILITY_STABLE_VIDEO_DIFFUSION,
            input={
                "cond_aug": 0.02,
                "decoding_t": 14,
                "input_image": shot_url,
                "video_length": "25_frames_with_svd_xt",
                "sizing_strategy": "maintain_aspect_ratio",
                "motion_bucket_id": 127,
                "frames_per_second": 10,
            },
        )
        print(f"video_url: {video_url}")
        return video_url
    except Exception as e:
        print(f"Error processing video shot: {e}")
        return None


def get_audio_duration(audio_file_path: str):
    """
    Get the duration of an audio file in seconds, rounded up to the nearest second.

    Args:
    audio_file_path (str): The file path to the audio file.

    Returns:
    int: The duration of the audio file in seconds, rounded up.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"The audio file {audio_file_path} does not exist.")

    with AudioFileClip(audio_file_path) as audio_clip:
        duration_seconds = math.ceil(audio_clip.duration)

    print(f"The audio duration is {duration_seconds}")
    return duration_seconds


def make_silence(t: float) -> List[int]:
    """Generate silence"""
    return [0]


def compose_final_video(
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
    voiceover_clip = AudioFileClip(voiceover_path).volumex(voiceover_volume)
    soundtrack_clip = AudioFileClip(soundtrack_path).volumex(soundtrack_volume)

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


def cleanup_tempdir(tmpdir: str):
    """Remove the temporary directory and its contents."""
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
        print(f"Cleaned up temporary directory: {tmpdir}")


@with_key_rotation
def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    user_input = kwargs["prompt"]
    openai_key = kwargs["api_keys"]["openai"]
    counter_callback = kwargs.get("counter_callback", None)
    tool = kwargs.get("tool")
    if tool not in ALLOWED_TOOLS:
        return (
            f"Tool {tool} is not supported by this agent.",
            user_input,
            None,
            counter_callback,
        )

    # Initialize OpenAI client with the provided key
    global client
    client = OpenAI(api_key=openai_key)

    replicate_key = kwargs["api_keys"]["replicate"]
    client_replicate = ReplicateClient(replicate_key)

    tmpdir = tempfile.mkdtemp(prefix="short_maker_")
    # Step 2: Get audio prompts
    print(f"Getting audio prompts for input: {user_input}")
    audio_prompts = get_audio_prompts(user_input)

    # Step 3: Process voiceover
    print("Processing voiceover...")
    voiceover_script = audio_prompts["voiceover_script"]
    voice_choice = audio_prompts["voice"]
    voiceover = process_voiceover(client_replicate, voiceover_script, voice_choice)

    # Download voiceover and get duration
    print("Downloading voiceover...")
    voiceover_path = os.path.join(tmpdir, "voiceover.mp3")
    voiceover_path = download_file(voiceover, voiceover_path)
    voiceover_length = get_audio_duration(voiceover_path)

    # Step 5: Get shot prompts
    print("Getting shot prompts...")
    shot_prompts = get_shot_prompts(user_input, voiceover_length)

    # Step 6: Generate images with DALL-E
    print("Generating images with DALL-E...")
    image_links = generate_images_with_dalle(shot_prompts, openai_key)
    image = list(image_links.values())[0]
    first_shot_path = os.path.join(tmpdir, "first_shot")
    first_shot_path = download_file(image, first_shot_path)

    # Steps 7 & 8: Process video shots
    video_urls = [
        process_first_shots(client_replicate, url) for url in image_links.values()
    ]

    # Step 9: Process soundtrack
    soundtrack_prompt = audio_prompts["soundtrack_prompt"]
    soundtrack = process_soundtrack(
        client_replicate, soundtrack_prompt, voiceover_length
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
    image_hash_ = ipfs_tool.client.add(
        first_shot_path, cid_version=1, wrap_with_directory=False
    )["Hash"]

    cleanup_tempdir(tmpdir)

    print(f"Stored the output on: {video_hash_}")

    body = {
        "video": video_hash_,
        "image": image_hash_,
        "prompt": user_input,
    }

    # response text, original prompt, metadata, callback
    return json.dumps(body), user_input, None, counter_callback
