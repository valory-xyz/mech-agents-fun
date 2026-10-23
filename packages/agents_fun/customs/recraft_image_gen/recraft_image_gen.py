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
from PIL import Image  # type: ignore[import-not-found]
from aea_cli_ipfs.ipfs_utils import IPFSTool
from openai import BadRequestError, OpenAI

# Define MechResponse type alias matching the other tools
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

# Define allowed tools for this module
ALLOWED_TOOLS = [
    "recraft-image-gen",
]


class InvalidRequestError(ValueError):
    """The request cannot be served as sent: the fault is in the buyer's input.

    The mech matches this class name to tell a client-side error apart from
    a tool or provider failure. Raise it only for faults in the request.
    """


def _validate_api_keys(api_keys: Any) -> None:
    required_methods = ["max_retries", "rotate", "get"]
    if not all(hasattr(api_keys, method) for method in required_methods):
        raise TypeError(
            "api_keys object does not have required methods (max_retries, rotate, get)"
        )


def with_key_rotation(func: Callable) -> Callable[..., MechResponse]:
    """Append the keychain to the result. Recraft has a single key, so nothing rotates."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponse:
        api_keys = kwargs["api_keys"]
        _validate_api_keys(api_keys)
        result = func(*args, **kwargs)
        return result + (api_keys,)

    return wrapper


def _validate_inputs(tool: Optional[str], api_key: Optional[str]) -> None:
    if tool not in ALLOWED_TOOLS:
        raise InvalidRequestError(f"Tool {tool} is not supported by this agent.")
    if not api_key:
        raise ValueError("Recraft API key (RECRAFT_API_KEY) not provided.")


def _generate_content(prompt: str, style: str, api_key: str) -> Any:
    """Generate image content using the Recraft API.

    :param prompt: the image prompt.
    :param style: the Recraft style.
    :param api_key: the Recraft API key.
    :return: the Recraft images response.
    :raises InvalidRequestError: if Recraft rejects the request as invalid.
    """
    client = OpenAI(
        base_url="https://external.api.recraft.ai/v1",
        api_key=api_key,
    )
    try:
        return client.images.generate(prompt=prompt, style=style)  # type: ignore[arg-type]
    except BadRequestError as e:
        raise InvalidRequestError(
            f"The image provider rejected the request: {e.message}"
        ) from e


def _extract_image_url_from_response(response: Any) -> Optional[str]:
    # Assuming response.data[0].url is the image URL
    if hasattr(response, "data") and response.data:
        return response.data[0].url
    return None


def _download_image(image_url: str) -> bytes:
    """Download an image from a URL."""
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    return resp.content


def _process_image_and_upload(image_data: bytes) -> str:
    image = Image.open(BytesIO(image_data), formats=("PNG", "JPEG", "WEBP"))
    temp_image_path = f"temp_generated_image_{os.getpid()}.png"
    try:
        image.save(temp_image_path)
        ipfs_tool = IPFSTool()
        _, image_hash, _ = ipfs_tool.add(temp_image_path)
        return image_hash
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

    _validate_inputs(tool, api_key)

    response = _generate_content(prompt, style, api_key)
    print("[DEBUG] Raw response from Recraft API:", response)
    image_url = _extract_image_url_from_response(response)
    if image_url is None:
        raise ValueError("No image URL found in response.data[0].url.")
    image_bytes = _download_image(image_url)
    image_hash = _process_image_and_upload(image_bytes)
    result = _prepare_result(image_hash, prompt, style)
    return result, prompt, None, counter_callback
