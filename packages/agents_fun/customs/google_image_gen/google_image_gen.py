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
from PIL import Image
from aea_cli_ipfs.ipfs_utils import IPFSTool
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types


# Define MechResponse type alias matching the other tools
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

# Define allowed tools for this module
ALLOWED_TOOLS = [
    "google_image_gen",
]


def with_key_rotation(func: Callable):
    """Decorator for handling API key rotation and retries."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
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
                if not hasattr(e, "code") or e.code != 500:
                    raise e
                service = "google_api_key"
                # If no retries left for this service, raise.
                if retries_left.get(service, 0) <= 0:
                    raise e

                # Retries are available, proceed with retry logic.
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except Exception as e:
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
    tool: str, api_key: Optional[str], prompt: str, counter_callback: Any
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
    client: genai.Client, prompt: str, model_name: str, counter_callback: Any
) -> Tuple[Optional[bytes], Optional[Tuple[str, str, None, Any]]]:
    """Generates image data using the Google API and handles initial response validation."""
    response = client.models.generate_images(
        model=model_name,
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1),
    )

    if not response.generated_images:
        return None, (
            "No image data found in the response (generated_images is empty).",
            prompt,
            None,
            counter_callback,
        )

    first_generated_image = response.generated_images[0]

    if not hasattr(first_generated_image, "image") or not hasattr(
        first_generated_image.image, "image_bytes"
    ):
        return None, (
            "Image data structure is not as expected.",
            prompt,
            None,
            counter_callback,
        )
    return first_generated_image.image.image_bytes, None


def _save_image_and_upload_to_ipfs(
    image_data: bytes, prompt: str, model_name: str, counter_callback: Any
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Saves the image data to a temporary file, uploads to IPFS, and cleans up."""
    temp_image_path = f"temp_generated_image_{os.getpid()}.png"
    try:
        image = Image.open(BytesIO(image_data))
        image.save(temp_image_path)

        ipfs_tool = IPFSTool()
        _, image_hash, _ = ipfs_tool.add(temp_image_path, wrap_with_directory=False)

        result_data = {"image_hash": image_hash, "prompt": prompt, "model": model_name}
        return json.dumps(result_data), prompt, None, counter_callback
    except FileNotFoundError:
        return (
            "IPFS tool not found or not configured correctly.",
            prompt,
            None,
            counter_callback,
        )
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@with_key_rotation
def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Runs the Google image generation task using genai.Client."""
    prompt = kwargs["prompt"]
    api_keys = kwargs["api_keys"]
    api_key = api_keys.get("gemini_api_key", None)
    tool = kwargs.get("tool")
    counter_callback = kwargs.get("counter_callback", None)
    model_name = kwargs.get("model", "imagen-4.0-generate-001")

    validation_error = _validate_inputs(tool, api_key, prompt, counter_callback)
    if validation_error:
        return validation_error

    try:
        client = genai.Client(api_key=api_key)

        image_data, error_response = _generate_image_from_google_api(
            client, prompt, model_name, counter_callback
        )
        if error_response:
            return error_response

        if (
            image_data is None
        ):  # Should not happen if error_response is None, but as a safeguard
            return (
                "Failed to generate image data without specific error.",
                prompt,
                None,
                counter_callback,
            )

        return _save_image_and_upload_to_ipfs(
            image_data, prompt, model_name, counter_callback
        )

    except google_exceptions.GoogleAPIError as e:
        print(f"Google API error: {e}")
        return f"Google API error: {e}", prompt, None, counter_callback
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"An error occurred: {e}", prompt, None, counter_callback
