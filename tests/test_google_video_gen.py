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

"""Unit tests for google_video_gen failure handling."""

import json
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import pytest
from google.api_core import exceptions as google_exceptions

from packages.agents_fun.customs.google_video_gen import google_video_gen as module

from tests.tool_test_helpers import FakeKeyChain

TOOL = "google_video_gen"
PROMPT = "ocean waves at night"
GEMINI_KEY = "gemini-test"
AUDIO_HASH = "bafy-audio"
GOOGLE_SERVICE = "google_api_key"
SCRIPT_RESPONSE_TEXT = "Voiceover: The sea keeps time.\nVoice: kore"
PCM_SILENCE = b"\x00\x00" * 16


def _script_response() -> MagicMock:
    """Build the text response that carries the voiceover script."""
    return MagicMock(text=SCRIPT_RESPONSE_TEXT)


def _tts_response() -> MagicMock:
    """Build the TTS response that carries PCM audio."""
    part = MagicMock(inline_data=MagicMock(data=PCM_SILENCE))
    return MagicMock(candidates=[MagicMock(content=MagicMock(parts=[part]))])


@pytest.fixture
def genai_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    """Patch the Google GenAI client and IPFS, the two network boundaries."""
    monkeypatch.chdir(tmp_path)
    client = MagicMock()
    monkeypatch.setattr(module.genai, "Client", MagicMock(return_value=client))
    ipfs = MagicMock()
    ipfs.add.return_value = (None, AUDIO_HASH, None)
    monkeypatch.setattr(module, "IPFSTool", MagicMock(return_value=ipfs))
    return client


def _run(api_keys: FakeKeyChain, tool: str = TOOL) -> tuple:
    """Call the tool entry point."""
    return module.run(
        prompt=PROMPT, tool=tool, api_keys=api_keys, counter_callback=None
    )


@pytest.mark.parametrize(
    "error",
    [
        google_exceptions.PermissionDenied("key rejected"),
        ConnectionError("network unreachable"),
    ],
)
def test_run_raises_on_provider_error(
    genai_client: MagicMock, error: Exception
) -> None:
    """A provider error that is not a rate limit propagates out of run."""
    genai_client.models.generate_content.side_effect = error
    with pytest.raises(type(error)):
        _run(FakeKeyChain({"gemini_api_key": GEMINI_KEY}))


def test_run_raises_when_tts_returns_no_audio(genai_client: MagicMock) -> None:
    """A TTS response without audio is a failed run, not an answer."""
    genai_client.models.generate_content.side_effect = [
        _script_response(),
        MagicMock(candidates=None),
    ]
    with pytest.raises(AssertionError, match="candidates is None"):
        _run(FakeKeyChain({"gemini_api_key": GEMINI_KEY}))


def test_rate_limit_rotates_key_and_returns_retry_result(
    genai_client: MagicMock,
) -> None:
    """A rate limit rotates the Google key and the retry's answer is returned."""
    genai_client.models.generate_content.side_effect = [
        google_exceptions.ResourceExhausted("quota"),
        _script_response(),
        _tts_response(),
    ]
    api_keys = FakeKeyChain({"gemini_api_key": GEMINI_KEY}, {GOOGLE_SERVICE: 1})

    result = _run(api_keys)

    assert api_keys.rotated == [GOOGLE_SERVICE]
    assert json.loads(result[0])["audio_hash"] == AUDIO_HASH
    assert result[-1] is api_keys


def test_rate_limit_reraises_when_no_retries_left(genai_client: MagicMock) -> None:
    """With the retry budget spent, the rate limit propagates."""
    genai_client.models.generate_content.side_effect = (
        google_exceptions.ResourceExhausted("quota")
    )
    api_keys = FakeKeyChain({"gemini_api_key": GEMINI_KEY}, {GOOGLE_SERVICE: 0})
    with pytest.raises(google_exceptions.ResourceExhausted):
        _run(api_keys)
    assert api_keys.rotated == []


@pytest.mark.parametrize(
    ("tool", "keys", "message"),
    [
        ("other_tool", {"gemini_api_key": GEMINI_KEY}, "not supported"),
        (TOOL, {"gemini_api_key": ""}, "GEMINI_API_KEY"),
        (TOOL, {}, "GEMINI_API_KEY"),
    ],
)
def test_invalid_request_raises_before_any_provider_call(
    genai_client: MagicMock, tool: str, keys: Dict[str, str], message: str
) -> None:
    """An unsupported tool or a missing key raises before any paid call."""
    with pytest.raises(ValueError, match=message):
        _run(FakeKeyChain(keys), tool=tool)
    genai_client.models.generate_content.assert_not_called()


def test_keychain_without_rotation_methods_raises(genai_client: MagicMock) -> None:
    """A plain dict in place of the mech KeyChain raises instead of answering."""
    with pytest.raises(TypeError, match="required methods"):
        module.run(prompt=PROMPT, tool=TOOL, api_keys={"gemini_api_key": GEMINI_KEY})
    genai_client.models.generate_content.assert_not_called()
