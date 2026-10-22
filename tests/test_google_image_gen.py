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

"""Unit tests for google_image_gen failure handling."""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from google.api_core import exceptions as google_exceptions

from packages.agents_fun.customs.google_image_gen import google_image_gen as module

from tests.tool_test_helpers import FakeKeyChain

TOOL = "google_image_gen"
PROMPT = "a lighthouse at dawn"
GEMINI_KEY = "gemini-test"
IMAGE_HASH = "bafy-image"
GOOGLE_SERVICE = "google_api_key"


def _png_bytes() -> bytes:
    """Return a tiny valid PNG."""
    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    return buffer.getvalue()


def _image_response() -> MagicMock:
    """Build a generate_images response carrying one image."""
    return MagicMock(
        generated_images=[MagicMock(image=MagicMock(image_bytes=_png_bytes()))]
    )


@pytest.fixture
def genai_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    """Patch the Google GenAI client and IPFS, the two network boundaries."""
    monkeypatch.chdir(tmp_path)
    client = MagicMock()
    monkeypatch.setattr(module.genai, "Client", MagicMock(return_value=client))
    ipfs = MagicMock()
    ipfs.add.return_value = (None, IMAGE_HASH, None)
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
        google_exceptions.InternalServerError("backend down"),
        ConnectionError("network unreachable"),
    ],
)
def test_run_raises_on_provider_error(
    genai_client: MagicMock, error: Exception
) -> None:
    """A provider error that is not a rate limit propagates out of run."""
    genai_client.models.generate_images.side_effect = error
    with pytest.raises(type(error)):
        _run(FakeKeyChain({"gemini_api_key": GEMINI_KEY}))


def test_run_raises_when_response_has_no_images(genai_client: MagicMock) -> None:
    """An empty generation result is a failed run, not an answer."""
    genai_client.models.generate_images.return_value = MagicMock(generated_images=[])
    with pytest.raises(ValueError, match="generated_images is empty"):
        _run(FakeKeyChain({"gemini_api_key": GEMINI_KEY}))


def test_run_raises_when_ipfs_upload_fails(genai_client: MagicMock) -> None:
    """A storage failure after generation propagates out of run."""
    genai_client.models.generate_images.return_value = _image_response()
    module.IPFSTool.return_value.add.side_effect = FileNotFoundError("ipfs missing")
    with pytest.raises(FileNotFoundError):
        _run(FakeKeyChain({"gemini_api_key": GEMINI_KEY}))


def test_rate_limit_rotates_key_and_returns_retry_result(
    genai_client: MagicMock,
) -> None:
    """A rate limit rotates the Google key and the retry's answer is returned."""
    genai_client.models.generate_images.side_effect = [
        google_exceptions.TooManyRequests("slow down"),
        _image_response(),
    ]
    api_keys = FakeKeyChain({"gemini_api_key": GEMINI_KEY}, {GOOGLE_SERVICE: 1})

    result = _run(api_keys)

    assert api_keys.rotated == [GOOGLE_SERVICE]
    assert json.loads(result[0])["image_hash"] == IMAGE_HASH
    assert result[-1] is api_keys


def test_rate_limit_reraises_when_no_retries_left(genai_client: MagicMock) -> None:
    """With the retry budget spent, the rate limit propagates."""
    genai_client.models.generate_images.side_effect = google_exceptions.TooManyRequests(
        "slow down"
    )
    api_keys = FakeKeyChain({"gemini_api_key": GEMINI_KEY}, {GOOGLE_SERVICE: 0})
    with pytest.raises(google_exceptions.TooManyRequests):
        _run(api_keys)
    assert api_keys.rotated == []


def test_unsupported_tool_returns_message_without_calling_provider(
    genai_client: MagicMock,
) -> None:
    """Bad buyer input is answered with a message before any paid call."""
    result = _run(FakeKeyChain({"gemini_api_key": GEMINI_KEY}), tool="other_tool")
    assert "not supported" in result[0]
    genai_client.models.generate_images.assert_not_called()
