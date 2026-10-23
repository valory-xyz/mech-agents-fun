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

"""Unit tests for recraft_image_gen failure handling."""

import json
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Optional
from unittest.mock import MagicMock

import pytest
import requests
from PIL import Image

from packages.agents_fun.customs.recraft_image_gen import recraft_image_gen as module

from tests.tool_test_helpers import (
    FakeKeyChain,
    assert_invalid_request_contract,
    assert_not_invalid_request,
    openai_auth_error,
    openai_bad_request_error,
    openai_rate_limit_error,
    openai_server_error,
)

TOOL = "recraft-image-gen"
PROMPT = "a red bicycle"
RECRAFT_KEY = "recraft-test"
IMAGE_URL = "https://img.recraft.test/1.png"
IMAGE_HASH = "bafy-recraft"
KEY_VALUE = RECRAFT_KEY


def _png_bytes() -> bytes:
    """Return a tiny valid PNG."""
    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def boundaries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    """Patch the Recraft (OpenAI-compatible) client, HTTP download and IPFS."""
    monkeypatch.chdir(tmp_path)
    client = MagicMock()
    client.images.generate.return_value = MagicMock(data=[MagicMock(url=IMAGE_URL)])
    monkeypatch.setattr(module, "OpenAI", MagicMock(return_value=client))
    download = MagicMock(content=_png_bytes())
    monkeypatch.setattr(module.requests, "get", MagicMock(return_value=download))
    ipfs = MagicMock()
    ipfs.add.return_value = (None, IMAGE_HASH, None)
    monkeypatch.setattr(module, "IPFSTool", MagicMock(return_value=ipfs))
    return client


def _api_keys() -> FakeKeyChain:
    """Build a keychain with the Recraft key."""
    return FakeKeyChain({"recraft_api_key": RECRAFT_KEY})


def _run(api_keys: FakeKeyChain, tool: str = TOOL) -> tuple:
    """Call the tool entry point."""
    return module.run(
        prompt=PROMPT, tool=tool, api_keys=api_keys, counter_callback=None
    )


@pytest.mark.parametrize(
    "make_error", [openai_auth_error, openai_rate_limit_error, openai_server_error]
)
def test_run_raises_on_provider_error(
    boundaries: MagicMock, make_error: Callable[[], Exception]
) -> None:
    """Any provider error propagates; Recraft has one key, so nothing rotates."""
    error = make_error()
    boundaries.images.generate.side_effect = error
    api_keys = _api_keys()
    with pytest.raises(type(error)) as err:
        _run(api_keys)
    assert_not_invalid_request(err.value)
    assert api_keys.rotated == []


def test_provider_bad_request_is_an_invalid_request(boundaries: MagicMock) -> None:
    """A 400 from Recraft is a fault in the request."""
    error = openai_bad_request_error()
    boundaries.images.generate.side_effect = error
    with pytest.raises(module.InvalidRequestError, match="prompt rejected") as err:
        _run(_api_keys())
    assert err.value.__cause__ is error


def test_run_raises_when_response_has_no_url(boundaries: MagicMock) -> None:
    """A response without an image URL is a failed run, not an answer."""
    boundaries.images.generate.return_value = MagicMock(data=[])
    with pytest.raises(ValueError, match="No image URL"):
        _run(_api_keys())


def test_run_raises_when_image_download_fails(boundaries: MagicMock) -> None:
    """An HTTP error on the image download propagates out of run."""
    module.requests.get.return_value.raise_for_status.side_effect = requests.HTTPError(
        "404 Not Found"
    )
    with pytest.raises(requests.HTTPError):
        _run(_api_keys())


def test_successful_run_appends_keychain(boundaries: MagicMock) -> None:
    """A successful run returns the image hash with the keychain appended."""
    api_keys = _api_keys()
    result = _run(api_keys)
    assert json.loads(result[0])["image_hash"] == IMAGE_HASH
    assert result[-1] is api_keys


def test_invalid_request_error_matches_the_mech_contract() -> None:
    """The mech recognises a client-side error by this exact class name."""
    assert_invalid_request_contract(module.InvalidRequestError)


@pytest.mark.parametrize("tool", ["other_tool", "", None])
def test_unsupported_tool_is_an_invalid_request(
    boundaries: MagicMock, tool: Optional[str]
) -> None:
    """An unsupported tool is a fault in the request; nothing is called."""
    with pytest.raises(module.InvalidRequestError, match="not supported"):
        _run(FakeKeyChain({"recraft_api_key": KEY_VALUE}), tool=tool)  # type: ignore[arg-type]
    boundaries.images.generate.assert_not_called()


@pytest.mark.parametrize("keys", [{"recraft_api_key": ""}, {}])
def test_missing_key_is_a_mech_side_error(
    boundaries: MagicMock, keys: Dict[str, str]
) -> None:
    """A missing operator key raises a plain ValueError before any paid call."""
    with pytest.raises(ValueError, match="RECRAFT_API_KEY") as err:
        _run(FakeKeyChain(keys))
    assert_not_invalid_request(err.value)
    boundaries.images.generate.assert_not_called()


def test_keychain_without_rotation_methods_raises(boundaries: MagicMock) -> None:
    """A plain dict in place of the mech KeyChain raises instead of answering."""
    with pytest.raises(TypeError, match="required methods"):
        module.run(prompt=PROMPT, tool=TOOL, api_keys={"recraft_api_key": RECRAFT_KEY})
    boundaries.images.generate.assert_not_called()
