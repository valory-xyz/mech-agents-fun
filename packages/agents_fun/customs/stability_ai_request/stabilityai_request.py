# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""Contains the job definitions"""

import functools
import json
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import anthropic
import openai
import requests
from googleapiclient.errors import HttpError as GoogleApiClientHttpError
from tiktoken import encoding_for_model

DEFAULT_STABILITYAI_SETTINGS = {
    "cfg_scale": 7,
    "weight": 0.5,
    "clip_guidance_preset": "FAST_BLUE",
    "height": 512,
    "width": 512,
    "samples": 1,
    "steps": 30,
}
PREFIX = "stabilityai-"
# Credits per image for SDXL 1.0 on the v1 endpoint at the default settings,
# and USD per credit (docs.stability.ai/pricing). Larger sizes and more steps
# cost more; the default is what this tool reports.
SDXL_CREDITS_PER_IMAGE = 0.9
USD_PER_CREDIT = 0.01
ENGINES = {
    "picture": [
        "stable-diffusion-xl-1024-v1-0",
    ]
}
ENGINE_SIZE_CHART = {
    "stable-diffusion-xl-1024-v1-0": [
        {"height": 1024, "width": 1024},
        {"height": 1152, "width": 896},
        {"height": 896, "width": 1152},
        {"height": 1216, "width": 832},
        {"height": 1344, "width": 768},
        {"height": 768, "width": 1344},
        {"height": 1536, "width": 640},
        {"height": 640, "width": 1536},
    ],
}

ALLOWED_TOOLS = [PREFIX + value for value in ENGINES["picture"]]
HTTP_BAD_REQUEST = 400
HTTP_TOO_MANY_REQUESTS = 429
STABILITY_SERVICE = "stabilityai"


class InvalidRequestError(ValueError):
    """The request cannot be served as sent: the fault is in the buyer's input.

    The mech matches this class name to tell a client-side error apart from
    a tool or provider failure. Raise it only for faults in the request.
    """


MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]


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
                return rotate_and_retry(e, ("anthropic",))
            except openai.RateLimitError as e:
                return rotate_and_retry(e, ("openai", "openrouter"))
            except GoogleApiClientHttpError as e:
                if e.status_code != HTTP_TOO_MANY_REQUESTS:
                    raise e
                return rotate_and_retry(e, ("google_api_key",))
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status != HTTP_TOO_MANY_REQUESTS:
                    raise e
                return rotate_and_retry(e, (STABILITY_SERVICE,))

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
                api_keys.rotate(service)
            return execute()

        mech_response = execute()
        return mech_response

    return wrapper


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


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


class FinishReason(Enum):
    """The finish reasons of the API."""

    SUCCESS = 0
    CONTENT_FILTERED = 1
    ERROR = 2


@with_key_rotation
def run(  # pylint: disable=too-many-locals
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""

    api_key = kwargs["api_keys"][STABILITY_SERVICE]
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    counter_callback = kwargs.get("counter_callback", None)

    if tool not in ALLOWED_TOOLS:
        raise InvalidRequestError(f"Tool {tool} is not in the list of supported tools.")

    # Place content moderation request here if needed
    engine = tool.replace(PREFIX, "")
    cfg_scale = kwargs.get("cfg_scale", DEFAULT_STABILITYAI_SETTINGS["cfg_scale"])
    weight = kwargs.get("weight", DEFAULT_STABILITYAI_SETTINGS["weight"])
    clip_guidance_preset = kwargs.get(
        "clip_guidance_preset", DEFAULT_STABILITYAI_SETTINGS["clip_guidance_preset"]
    )

    # Handle different engine types
    if engine == "stable-diffusion-xl-1024-v1-0":
        height = kwargs.get(
            "height", ENGINE_SIZE_CHART[engine][0]["height"]  # type: ignore[index]
        )  # Access first size as default
        width = kwargs.get("width", ENGINE_SIZE_CHART[engine][0]["width"])  # type: ignore[index]
    else:
        height = kwargs.get("height", ENGINE_SIZE_CHART[engine]["height"])  # type: ignore[call-overload]
        width = kwargs.get("width", ENGINE_SIZE_CHART[engine]["width"])  # type: ignore[call-overload]

    samples = kwargs.get("samples", DEFAULT_STABILITYAI_SETTINGS["samples"])
    steps = kwargs.get("steps", DEFAULT_STABILITYAI_SETTINGS["steps"])

    optional_params = {}
    for optional in ("sampler", "seed", "style_preset", "extras"):
        value = kwargs.get(optional, None)
        if value is not None:
            optional_params[optional] = value

    json_params = {
        "text_prompts": [{"text": prompt, "weight": weight}],
        "cfg_scale": cfg_scale,
        "clip_guidance_preset": clip_guidance_preset,
        "height": height,
        "width": width,
        "samples": samples,
        "steps": steps,
    }
    json_params.update(optional_params)

    # request stabilityai api
    response = requests.post(
        f"https://api.stability.ai/v1/generation/{engine}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Stability-Client-ID": "mechs-tool",
        },
        json=json_params,
        timeout=30,
    )
    if response.status_code == HTTP_BAD_REQUEST:
        raise InvalidRequestError(
            f"Stability AI rejected the request: {response.text[:500]}"
        )
    if response.status_code != 200:
        raise requests.HTTPError(
            f"Non-200 response ({response.status_code}): {response.text}",
            response=response,
        )
    record_api_cost(
        counter_callback,
        engine,
        SDXL_CREDITS_PER_IMAGE * USD_PER_CREDIT * int(samples),
    )
    return json.dumps(response.json()), None, None, counter_callback
