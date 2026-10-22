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

from unittest.mock import MagicMock

import pytest
import requests

from packages.agents_fun.customs.short_maker import short_maker as module

from tests.tool_test_helpers import (
    FakeKeyChain,
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
def chat(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the OpenAI and Replicate client boundaries."""
    client = MagicMock()
    monkeypatch.setattr(module, "OpenAI", MagicMock(return_value=client))
    monkeypatch.setattr(module, "client", client)
    monkeypatch.setattr(module, "ReplicateClient", MagicMock())
    return client.chat.completions.create


@pytest.fixture
def post(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the HTTP boundary to the DALL-E endpoint."""
    mock_post = MagicMock()
    monkeypatch.setattr(module.requests, "post", mock_post)
    return mock_post


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


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (
            MagicMock(
                raise_for_status=MagicMock(side_effect=requests.HTTPError("500"))
            ),
            requests.HTTPError,
        ),
        (MagicMock(json=lambda: {"data": []}), ValueError),
        (MagicMock(json=lambda: {}), ValueError),
    ],
)
def test_generate_images_raises_on_failed_shot(
    post: MagicMock, response: MagicMock, error_type: type
) -> None:
    """A failed or empty DALL-E answer raises instead of storing error text."""
    post.return_value = response
    with pytest.raises(error_type):
        module.generate_images_with_dalle(SHOT_PROMPTS, OPENAI_KEY)


def test_unsupported_tool_raises_before_any_provider_call(chat: MagicMock) -> None:
    """An unsupported tool raises instead of answering with a message."""
    with pytest.raises(ValueError, match="other_tool is not supported"):
        _run(_api_keys(), tool="other_tool")
    chat.assert_not_called()
