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
"""This module contains the implementation of the google_image_gen tool based on a working snippet."""

import functools
import json
import os
from io import BytesIO
from typing import Any, Callable, Dict, Optional, Tuple

import anthropic
import openai
from PIL import Image  # type: ignore[import-not-found]
from aea_cli_ipfs.ipfs_utils import IPFSTool
from google import genai  # type: ignore[import-not-found]
from google.api_core import (
    exceptions as google_exceptions,  # type: ignore[import-not-found]
)
from google.genai import types  # type: ignore[import-not-found]

# Define MechResponse type alias matching the other tools
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

DEFAULT_MODEL = "imagen-4.0-generate-001"

# Define allowed tools for this module
ALLOWED_TOOLS = [
    "google_image_gen",
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
                openai.RateLimitError,
                google_exceptions.ResourceExhausted,
                google_exceptions.TooManyRequests,
            ) as e:
                service = "google_api_key"
                if isinstance(e, anthropic.RateLimitError):
                    service = "anthropic"
                elif isinstance(e, openai.RateLimitError):
                    if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                        raise e
                    retries_left["openai"] -= 1
                    retries_left["openrouter"] -= 1
                    api_keys.rotate("openai")
                    api_keys.rotate("openrouter")
                    return execute()

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


def _generate_image_from_google_api(
    client: genai.Client, prompt: str, model_name: Optional[str]
) -> bytes:
    """Generate image data using the Google API.

    :param client: the Google GenAI client.
    :param prompt: the image prompt.
    :param model_name: the Imagen model to use.
    :return: the generated image bytes.
    :raises ValueError: if the response carries no image bytes.
    """
    response = client.models.generate_images(
        model=model_name,  # type: ignore[arg-type]
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1),
    )

    if not response.generated_images:
        raise ValueError(
            "No image data found in the response (generated_images is empty)."
        )

    first_generated_image = response.generated_images[0]
    image_bytes = getattr(
        getattr(first_generated_image, "image", None), "image_bytes", None
    )
    if image_bytes is None:
        raise ValueError("Image data structure is not as expected.")
    return image_bytes


def _save_image_and_upload_to_ipfs(
    image_data: bytes, prompt: str, model_name: Optional[str], counter_callback: Any
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Saves the image data to a temporary file, uploads to IPFS, and cleans up."""
    temp_image_path = f"temp_generated_image_{os.getpid()}.png"
    try:
        image = Image.open(BytesIO(image_data), formats=("PNG", "JPEG", "WEBP"))
        image.save(temp_image_path)

        ipfs_tool = IPFSTool()
        _, image_hash, _ = ipfs_tool.add(temp_image_path, wrap_with_directory=False)

        result_data = {"image_hash": image_hash, "prompt": prompt, "model": model_name}
        return json.dumps(result_data), prompt, None, counter_callback
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@with_key_rotation
def run(**kwargs: Any) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Runs the Google image generation task using genai.Client."""
    prompt = kwargs["prompt"]
    api_keys = kwargs["api_keys"]
    api_key = api_keys.get("gemini_api_key", None)
    tool = kwargs.get("tool")
    counter_callback = kwargs.get("counter_callback", None)
    model_name = kwargs.get("model") or DEFAULT_MODEL

    validation_error = _validate_inputs(tool, api_key, prompt, counter_callback)
    if validation_error:
        return validation_error

    client = genai.Client(api_key=api_key)
    image_data = _generate_image_from_google_api(client, prompt, model_name)
    return _save_image_and_upload_to_ipfs(
        image_data, prompt, model_name, counter_callback
    )
