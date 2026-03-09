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
"""This module contains the implementation of the recraft_image_gen tool based on the Recraft v3 API."""

import functools
import json
import os
from io import BytesIO
from typing import Any, Callable, Dict, Optional, Tuple

import requests
from PIL import Image
from aea_cli_ipfs.ipfs_utils import IPFSTool
from openai import OpenAI

# Define MechResponse type alias matching the other tools
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

# Define allowed tools for this module
ALLOWED_TOOLS = [
    "recraft-image-gen",
]


def _validate_api_keys(api_keys: Any) -> Optional[str]:
    required_methods = ["max_retries", "rotate", "get"]
    if not all(hasattr(api_keys, method) for method in required_methods):
        return (
            "api_keys object does not have required methods (max_retries, rotate, get)"
        )
    return None


def _create_error_response(error_msg: str, **kwargs: Any) -> MechResponse:
    prompt_val = kwargs.get("prompt", "N/A")
    callback_val = kwargs.get("counter_callback", None)
    return error_msg, prompt_val, None, callback_val, None


def _handle_rate_limit_error(
    e: Exception, service: str, retries_left: Dict[str, int], api_keys: Any
) -> None:
    if retries_left.get(service, 0) <= 0:
        print(f"No retries left for service: {service}")
        raise e
    retries_left[service] -= 1
    print(
        f"Rate limit error for {service}. Retries left: {retries_left[service]}. Rotating key."
    )
    api_keys.rotate(service)


def with_key_rotation(func: Callable) -> Callable[..., MechResponse]:
    """Wrap function with API key rotation logic."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponse:
        api_keys = kwargs["api_keys"]
        validation_error = _validate_api_keys(api_keys)
        if validation_error:
            return _create_error_response(validation_error, **kwargs)
        _ = api_keys.max_retries()

        def execute() -> MechResponse:
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
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


def _validate_inputs(tool: Optional[str], api_key: Optional[str]) -> Optional[str]:
    if tool not in ALLOWED_TOOLS:
        return f"Tool {tool} is not supported by this agent."
    if not api_key:
        return "Recraft API key (RECRAFT_API_KEY) not provided."
    return None


def _generate_content(prompt: str, style: str, api_key: str) -> Any:
    """Generate image content using the Recraft API."""
    client = OpenAI(
        base_url="https://external.api.recraft.ai/v1",
        api_key=api_key,
    )
    return client.images.generate(prompt=prompt, style=style)  # type: ignore[arg-type]


def _extract_image_url_from_response(response: Any) -> Optional[str]:
    # Assuming response.data[0].url is the image URL
    if hasattr(response, "data") and response.data:
        return response.data[0].url
    return None


def _download_image(image_url: str) -> Optional[bytes]:
    """Download an image from a URL."""
    try:
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Failed to download image: {e}")
        return None


def _process_image_and_upload(image_data: bytes) -> Tuple[Optional[str], Optional[str]]:
    image = Image.open(BytesIO(image_data))
    temp_image_path = f"temp_generated_image_{os.getpid()}.png"
    try:
        image.save(temp_image_path)
        ipfs_tool = IPFSTool()
        _, image_hash, _ = ipfs_tool.add(temp_image_path)
        return image_hash, None
    except FileNotFoundError:
        return None, "IPFS tool not found or not configured correctly."
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


def _prepare_result(image_hash: str, prompt: str, style: str) -> str:
    result_data = {"image_hash": image_hash, "prompt": prompt, "style": style}
    return json.dumps(result_data)


@with_key_rotation
def run(**kwargs: Any) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the recraft image generation tool."""
    prompt = kwargs["prompt"]
    style = kwargs.get("style", "realistic_image")
    api_key = kwargs["api_keys"].get("recraft_api_key")
    tool = kwargs.get("tool")
    counter_callback = kwargs.get("counter_callback", None)

    validation_error = _validate_inputs(tool, api_key)
    if validation_error:
        return validation_error, prompt, None, counter_callback

    try:
        response = _generate_content(prompt, style, api_key)
        print("[DEBUG] Raw response from Recraft API:", response)
        image_url = _extract_image_url_from_response(response)
        if image_url is None:
            return (
                "No image URL found in response.data[0].url.",
                prompt,
                None,
                counter_callback,
            )
        image_bytes = _download_image(image_url)
        if image_bytes is None:
            return "Failed to download image from URL.", prompt, None, counter_callback
        image_hash, upload_error = _process_image_and_upload(image_bytes)
        if upload_error:
            return upload_error, prompt, None, counter_callback
        assert image_hash is not None
        result = _prepare_result(image_hash, prompt, style)
        return result, prompt, None, counter_callback
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"An unexpected error occurred: {e}")
        return f"An error occurred: {e}", prompt, None, counter_callback
