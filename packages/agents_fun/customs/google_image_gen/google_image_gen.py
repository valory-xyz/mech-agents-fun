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

import base64
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

# Define MechResponse type alias matching the other tools
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

# Served through the Interactions API; must match ``default_model`` in component.yaml.
DEFAULT_MODEL = "gemini-nano-banana-2.1"
HTTP_BAD_REQUEST = 400
HTTP_TOO_MANY_REQUESTS = 429
GEMINI_SERVICE = "gemini_api_key"
IMAGE_SIZE = "1K"
# USD per generated image by model and size, ai.google.dev/gemini-api/docs/pricing.
# A request is served only for a model listed here at IMAGE_SIZE, so every
# generated image has a price to report.
IMAGE_COST_USD = {
    "gemini-nano-banana-2.1": {"1K": 0.0336, "2K": 0.0504, "4K": 0.113},
    "gemini-3.1-flash-lite-image": {"1K": 0.0336},
    "gemini-3.1-flash-image": {"0.5K": 0.045, "1K": 0.067, "2K": 0.101, "4K": 0.151},
    "gemini-3-pro-image": {"1K": 0.134, "2K": 0.134, "4K": 0.24},
}
# USD per million tokens by model, same page: (input, text and thinking output,
# image output). The per-image price above is one image's output tokens at the
# image rate.
TOKEN_COST_USD_PER_MILLION = {
    "gemini-nano-banana-2.1": (1.50, 7.50, 30.00),
    "gemini-3.1-flash-lite-image": (0.25, 1.50, 30.00),
    "gemini-3.1-flash-image": (0.50, 3.00, 60.00),
    "gemini-3-pro-image": (2.00, 12.00, 120.00),
}
IMAGE_MODALITY = "image"


def image_cost_usd(model_name: Optional[str], size: str) -> float:
    """The list price of one image from ``model_name`` at ``size``, 0 when unknown.

    :param model_name: the image model the request named.
    :param size: the requested image size.
    :return: the USD price, or 0 for a model or size without a listed price.
    """
    return IMAGE_COST_USD.get(str(model_name), {}).get(size, 0.0)


def _token_count(value: Any) -> int:
    """A usage field as a token count, 0 when the provider left it out.

    :param value: the field as the SDK returned it.
    :return: a non-negative whole number of tokens.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def token_cost_usd(model_name: Optional[str], usage: Any) -> float:
    """The charge for the prompt and the non-image output of one interaction.

    Input tokens are charged at the model's input rate. Output tokens of any
    modality but the image, and thinking tokens, are charged at its text rate.
    The image output is charged by ``image_output_cost_usd``, so output
    without a per-modality breakdown is not charged a second time here.

    :param model_name: the image model the request named.
    :param usage: the interaction's ``usage``, or None when it carries none.
    :return: the USD charge, 0 without usage or for a model without rates.
    """
    rates = TOKEN_COST_USD_PER_MILLION.get(str(model_name))
    if usage is None or rates is None:
        return 0.0
    input_rate, text_rate, _ = rates
    text_tokens = _output_tokens(usage, image=False) + _token_count(
        getattr(usage, "total_thought_tokens", None)
    )
    input_tokens = _token_count(getattr(usage, "total_input_tokens", None))
    return (input_tokens * input_rate + text_tokens * text_rate) / 1_000_000


def _output_tokens(usage: Any, image: bool) -> int:
    """Output tokens of the image modality, or of every other modality.

    :param usage: the interaction's ``usage``.
    :param image: True for the image modality, False for all the others.
    :return: the token count, 0 without a per-modality breakdown.
    """
    by_modality = getattr(usage, "output_tokens_by_modality", None) or []
    return sum(
        _token_count(getattr(entry, "tokens", None))
        for entry in by_modality
        if (getattr(entry, "modality", None) == IMAGE_MODALITY) is image
    )


def image_output_cost_usd(model_name: Optional[str], size: str, usage: Any) -> float:
    """The charge for the image output of one interaction.

    One image is charged its listed price. An interaction whose image output
    tokens come to more than that, as with several images, is charged those
    tokens at the model's image rate.

    :param model_name: the image model the request named.
    :param size: the requested image size.
    :param usage: the interaction's ``usage``, or None when it carries none.
    :return: the USD charge, never below the price of one image.
    """
    one_image = image_cost_usd(model_name, size)
    rates = TOKEN_COST_USD_PER_MILLION.get(str(model_name))
    if usage is None or rates is None:
        return one_image
    by_tokens = _output_tokens(usage, image=True) * rates[2] / 1_000_000
    return max(one_image, by_tokens)


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


# Define allowed tools for this module
ALLOWED_TOOLS = [
    "google_image_gen",
]


class InvalidRequestError(ValueError):
    """The request cannot be served as sent: the fault is in the buyer's input.

    The mech matches this class name to tell a client-side error apart from
    a tool or provider failure. Raise it only for faults in the request.
    """


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
            except openai.RateLimitError as e:
                return rotate_and_retry(e, ("openai", "openrouter"))
            except Exception as e:  # pylint: disable=broad-exception-caught
                if _http_status(e) != HTTP_TOO_MANY_REQUESTS:
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


def _http_status(error: Exception) -> Optional[int]:
    """The HTTP status a Gemini SDK error carries, None for any other error.

    Interactions API errors carry ``status_code``; ``genai.errors.APIError``
    carries ``code``.

    :param error: the raised error.
    :return: the HTTP status, or None.
    """
    for attr in ("status_code", "code"):
        status = getattr(error, attr, None)
        if isinstance(status, int):
            return status
    return None


def _is_request_rejection(error: Exception) -> bool:
    """Whether Gemini rejected the request itself, not the operator's key or account.

    Gemini also answers 400 for an invalid API key and for a failed
    precondition (unsupported region, billing), which are not request faults,
    so only a 400 that names INVALID_ARGUMENT and not the API key counts.

    :param error: the raised error.
    :return: True for a 400 INVALID_ARGUMENT that is not about the API key.
    """
    if _http_status(error) != HTTP_BAD_REQUEST:
        return False
    detail = f"{error} {getattr(error, 'body', '')} {getattr(error, 'status', '')}"
    if "INVALID_ARGUMENT" not in detail.upper().replace("-", "_"):
        return False
    return "API_KEY_INVALID" not in detail and "API key" not in detail


def _validate_inputs(
    tool: Optional[str], model_name: Optional[str], api_key: Optional[str]
) -> None:
    """Validate tool, model and API key.

    :param tool: the requested tool name.
    :param model_name: the image model the request named.
    :param api_key: the Gemini API key.
    :raises InvalidRequestError: if the tool is not supported, or the model
        has no listed price at the served image size.
    :raises ValueError: if the key is missing.
    """
    if tool not in ALLOWED_TOOLS:
        raise InvalidRequestError(f"Tool {tool} is not supported by this agent.")
    if image_cost_usd(model_name, IMAGE_SIZE) <= 0:
        raise InvalidRequestError(f"Model {model_name} is not supported.")
    if not api_key:
        raise ValueError("Google API key (GEMINI_API_KEY) not provided.")


def _generate_image_from_google_api(
    client: genai.Client, prompt: str, model_name: Optional[str], counter_callback: Any
) -> bytes:
    """Generate one image through the Interactions API and report its charge.

    :param client: the Google GenAI client.
    :param prompt: the image prompt.
    :param model_name: the image model the request named.
    :param counter_callback: the mech cost callback, or None outside the mech.
    :return: the generated image bytes.
    :raises InvalidRequestError: if the provider rejects the request as invalid.
    :raises ValueError: if the response carries no image.
    """
    try:
        interaction = client.interactions.create(
            model=model_name,
            input=prompt,
            response_format={"type": "image", "image_size": IMAGE_SIZE},
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        if not _is_request_rejection(e):
            raise
        raise InvalidRequestError(
            f"The image provider rejected the request: {e}"
        ) from e

    output_image = getattr(interaction, "output_image", None)
    encoded = getattr(output_image, "data", None) if output_image else None
    if not encoded:
        raise ValueError("No image data found in the response (output_image is empty).")
    # The image output is billed per image; the prompt and any text or
    # thinking output are billed per token on top of it.
    usage = getattr(interaction, "usage", None)
    record_api_cost(
        counter_callback,
        str(model_name),
        image_output_cost_usd(model_name, IMAGE_SIZE, usage)
        + token_cost_usd(model_name, usage),
    )
    return base64.b64decode(encoded)


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
    api_key = api_keys.get(GEMINI_SERVICE, None)
    tool = kwargs.get("tool")
    counter_callback = kwargs.get("counter_callback", None)
    model_name = kwargs.get("model") or DEFAULT_MODEL

    _validate_inputs(tool, model_name, api_key)

    client = genai.Client(api_key=api_key)
    image_data = _generate_image_from_google_api(
        client, prompt, model_name, counter_callback
    )
    return _save_image_and_upload_to_ipfs(
        image_data, prompt, model_name, counter_callback
    )
