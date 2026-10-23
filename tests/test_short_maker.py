# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""Unit tests for short_maker failure handling."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from packages.agents_fun.customs.short_maker import short_maker as module

from tests.tool_test_helpers import (
    FakeKeyChain,
    assert_invalid_request_contract,
    openai_auth_error,
    openai_rate_limit_error,
)

TOOL = "short_maker"
PROMPT = "a city waking up"
OPENAI_KEY = "sk-test"
VOICEOVER_SECONDS = 5
SHOT_PROMPTS = {"shot1_prompt": "wide shot of a skyline"}
RETRY_RESULT = ("retry answer", PROMPT, None, None)


@pytest.fixture
def openai_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the OpenAI and Replicate client boundaries."""
    client = MagicMock()
    monkeypatch.setattr(module, "OpenAI", MagicMock(return_value=client))
    monkeypatch.setattr(module, "client", client)
    monkeypatch.setattr(module, "ReplicateClient", MagicMock())
    return client


@pytest.fixture
def chat(openai_client: MagicMock) -> MagicMock:
    """The chat completions call of the patched OpenAI client."""
    return openai_client.chat.completions.create


@pytest.fixture
def images(openai_client: MagicMock) -> MagicMock:
    """The image generation call of the patched OpenAI client."""
    return openai_client.images.generate


def _api_keys() -> FakeKeyChain:
    """Build a keychain with one OpenAI retry."""
    return FakeKeyChain(
        {"openai": OPENAI_KEY, "openrouter": "", "replicate": "r8-test"},
        {"openai": 1, "openrouter": 1},
    )


def _run(api_keys: FakeKeyChain, tool: str = TOOL) -> tuple:
    """Call the tool entry point."""
    return module.run(
        prompt=PROMPT, tool=tool, api_keys=api_keys, counter_callback=None
    )


def test_run_raises_on_provider_error(chat: MagicMock) -> None:
    """A provider error that is not a rate limit propagates out of run."""
    chat.side_effect = openai_auth_error()
    api_keys = _api_keys()
    with pytest.raises(type(chat.side_effect)):
        _run(api_keys)
    assert api_keys.rotated == []


def test_run_rotates_openai_key_on_rate_limit(chat: MagicMock) -> None:
    """A rate limit rotates the OpenAI keys before the retry runs."""
    chat.side_effect = [openai_rate_limit_error(), openai_auth_error()]
    api_keys = _api_keys()
    with pytest.raises(type(openai_auth_error())):
        _run(api_keys)
    assert api_keys.rotated == ["openai", "openrouter"]
    assert chat.call_count == 2


def test_wrapper_returns_retry_result_after_rate_limit() -> None:
    """The wrapper returns the retry's answer with the keychain appended."""
    func = MagicMock(side_effect=[openai_rate_limit_error(), RETRY_RESULT])
    api_keys = _api_keys()
    result = module.with_key_rotation(func)(api_keys=api_keys)
    assert result == RETRY_RESULT + (api_keys,)


def test_get_shot_prompts_raises_on_provider_error(chat: MagicMock) -> None:
    """A failed shot prompt call raises instead of returning None."""
    chat.side_effect = openai_auth_error()
    with pytest.raises(type(chat.side_effect)):
        module.get_shot_prompts(PROMPT, VOICEOVER_SECONDS)


IMAGE_BYTES = b"png-bytes"
TWO_SHOT_PROMPTS = {"shot1_prompt": "wide shot", "shot2_prompt": "close up"}


def _image_response(data: Any = IMAGE_BYTES) -> SimpleNamespace:
    """Build an image API answer carrying one base64 image, or ``data`` as given."""
    encoded = base64.b64encode(data).decode() if isinstance(data, bytes) else data
    return SimpleNamespace(data=[SimpleNamespace(b64_json=encoded)], usage=None)


def _image_timeout() -> openai.APITimeoutError:
    """Build the error the OpenAI SDK raises when the image call times out."""
    return openai.APITimeoutError(
        request=httpx.Request("POST", "https://api.openai.com/v1/images/generations")
    )


FAILED_SHOT_RESPONSES = [
    SimpleNamespace(data=[], usage=None),
    SimpleNamespace(data=None, usage=None),
    _image_response(None),
    _image_response(""),
    _image_response("not base64!"),
]


@pytest.mark.parametrize("failed", FAILED_SHOT_RESPONSES)
def test_generate_images_skips_a_failed_shot(
    images: MagicMock, failed: SimpleNamespace, tmp_path: Path
) -> None:
    """A failed or empty shot is left out and the other shots are kept."""
    images.side_effect = [failed, _image_response()]
    paths = module.generate_images(TWO_SHOT_PROMPTS, str(tmp_path))
    assert list(paths) == ["shot2_prompt"]
    assert Path(paths["shot2_prompt"]).read_bytes() == IMAGE_BYTES


def test_generate_images_skips_a_request_error(
    images: MagicMock, tmp_path: Path
) -> None:
    """A shot whose request fails outright is left out."""
    images.side_effect = [_image_timeout(), _image_response()]
    paths = module.generate_images(TWO_SHOT_PROMPTS, str(tmp_path))
    assert list(paths) == ["shot2_prompt"]


def test_generate_images_raises_on_an_unexpected_error(
    images: MagicMock, tmp_path: Path
) -> None:
    """An error outside the provider and data failures is a tool fault and propagates."""
    images.side_effect = RuntimeError("bug")
    with pytest.raises(RuntimeError, match="bug"):
        module.generate_images(TWO_SHOT_PROMPTS, str(tmp_path))


@pytest.mark.parametrize("failed", FAILED_SHOT_RESPONSES)
def test_generate_images_raises_when_no_shot_has_an_image(
    images: MagicMock, failed: SimpleNamespace, tmp_path: Path
) -> None:
    """With every shot failed there is nothing to build a video from."""
    images.return_value = failed
    with pytest.raises(ValueError, match="No shot image"):
        module.generate_images(TWO_SHOT_PROMPTS, str(tmp_path))
    assert images.call_count == len(TWO_SHOT_PROMPTS)


COMPLETE_AUDIO_PROMPTS = {
    "voiceover_script": "Cities wake before we do.",
    "soundtrack_prompt": "soft morning ambience",
    "voice": "English_Graceful_Lady",
}


def _chat_answer(content: dict) -> MagicMock:
    """Build a chat completion whose message is the JSON of content."""
    answer = MagicMock()
    answer.choices[0].message.content = json.dumps(content)
    return answer


@pytest.mark.parametrize(
    "missing_key", ["voiceover_script", "soundtrack_prompt", "voice"]
)
@pytest.mark.parametrize("bad_value", [None, "", 7])
def test_incomplete_audio_prompts_raise_before_paid_steps(
    chat: MagicMock, images: MagicMock, missing_key: str, bad_value: object
) -> None:
    """A missing or empty audio prompt key fails before any Replicate or image call."""
    prompts = dict(COMPLETE_AUDIO_PROMPTS)
    if bad_value is None:
        del prompts[missing_key]
    else:
        prompts[missing_key] = bad_value
    chat.return_value = _chat_answer(prompts)
    with pytest.raises(ValueError, match=missing_key):
        _run(_api_keys())
    module.ReplicateClient.return_value.run.assert_not_called()
    images.assert_not_called()


def test_invalid_request_error_matches_the_mech_contract() -> None:
    """The mech recognises a client-side error by this exact class name."""
    assert_invalid_request_contract(module.InvalidRequestError)


@pytest.mark.parametrize("tool", ["other_tool", ""])
def test_unsupported_tool_is_an_invalid_request(chat: MagicMock, tool: str) -> None:
    """An unsupported tool is a fault in the request; nothing is called."""
    with pytest.raises(module.InvalidRequestError, match="is not supported"):
        _run(_api_keys(), tool=tool)
    chat.assert_not_called()


def test_rate_limit_rotates_only_services_in_the_keychain(chat: MagicMock) -> None:
    """A keychain without openrouter rotates openai alone, not a KeyError."""
    chat.side_effect = [openai_rate_limit_error(), openai_auth_error()]
    api_keys = FakeKeyChain(
        {"openai": OPENAI_KEY, "replicate": "r8-test"}, {"openai": 1}
    )
    with pytest.raises(type(openai_auth_error())):
        _run(api_keys)
    assert api_keys.rotated == ["openai"]
