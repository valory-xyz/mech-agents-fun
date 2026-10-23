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
from typing import Dict, Optional
from unittest.mock import MagicMock

import pytest
from PIL import Image
from google.genai import errors as genai_errors

from packages.agents_fun.customs.google_image_gen import google_image_gen as module

from tests.tool_test_helpers import (
    FakeKeyChain,
    HTTP_BAD_REQUEST,
    assert_invalid_request_contract,
    assert_not_invalid_request,
)

TOOL = "google_image_gen"
PROMPT = "a lighthouse at dawn"
GEMINI_KEY = "gemini-test"
IMAGE_HASH = "bafy-image"
GOOGLE_SERVICE = "gemini_api_key"
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500
KEY_VALUE = GEMINI_KEY
HTTP_NOT_FOUND = 404
HTTP_FORBIDDEN = 403


def _genai_rate_limit() -> genai_errors.ClientError:
    """Build the error the Google GenAI SDK raises on HTTP 429."""
    return _genai_client_error(HTTP_TOO_MANY_REQUESTS, "RESOURCE_EXHAUSTED", "quota")


def _genai_client_error(
    code: int, status: str = "INVALID_ARGUMENT", message: str = "prompt rejected"
) -> genai_errors.ClientError:
    """Build the error the Google GenAI SDK raises on a 4xx answer."""
    return genai_errors.ClientError(
        code, {"error": {"code": code, "message": message, "status": status}}
    )


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
        _genai_client_error(HTTP_FORBIDDEN, "PERMISSION_DENIED", "key rejected"),
        genai_errors.ServerError(
            HTTP_SERVER_ERROR,
            {
                "error": {
                    "code": HTTP_SERVER_ERROR,
                    "message": "down",
                    "status": "INTERNAL",
                }
            },
        ),
        ConnectionError("network unreachable"),
    ],
)
def test_run_raises_on_provider_error(
    genai_client: MagicMock, error: Exception
) -> None:
    """A provider error that is not a rate limit propagates and rotates nothing."""
    genai_client.models.generate_images.side_effect = error
    api_keys = FakeKeyChain({"gemini_api_key": GEMINI_KEY}, {GOOGLE_SERVICE: 1})
    with pytest.raises(type(error)):
        _run(api_keys)
    assert api_keys.rotated == []


def test_provider_bad_request_is_an_invalid_request(genai_client: MagicMock) -> None:
    """A 400 from the image provider is a fault in the request."""
    error = _genai_client_error(HTTP_BAD_REQUEST)
    genai_client.models.generate_images.side_effect = error
    with pytest.raises(module.InvalidRequestError, match="prompt rejected") as err:
        _run(FakeKeyChain({"gemini_api_key": GEMINI_KEY}))
    assert err.value.__cause__ is error


@pytest.mark.parametrize(
    ("code", "status", "message"),
    [
        (HTTP_FORBIDDEN, "PERMISSION_DENIED", "forbidden"),
        (HTTP_NOT_FOUND, "NOT_FOUND", "model not found"),
        (
            HTTP_BAD_REQUEST,
            "INVALID_ARGUMENT",
            "API key not valid. Please pass a valid API key.",
        ),
        (HTTP_BAD_REQUEST, "FAILED_PRECONDITION", "User location is not supported"),
    ],
    ids=["forbidden", "not-found", "invalid-key", "failed-precondition"],
)
def test_key_account_and_other_client_errors_are_mech_side(
    genai_client: MagicMock, code: int, status: str, message: str
) -> None:
    """A bad key, a failed precondition or a non-400 error stays mech-side."""
    genai_client.models.generate_images.side_effect = _genai_client_error(
        code, status, message
    )
    with pytest.raises(genai_errors.ClientError) as err:
        _run(FakeKeyChain({"gemini_api_key": GEMINI_KEY}))
    assert_not_invalid_request(err.value)


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
        _genai_rate_limit(),
        _image_response(),
    ]
    api_keys = FakeKeyChain({"gemini_api_key": GEMINI_KEY}, {GOOGLE_SERVICE: 1})

    result = _run(api_keys)

    assert api_keys.rotated == [GOOGLE_SERVICE]
    assert json.loads(result[0])["image_hash"] == IMAGE_HASH
    assert result[-1] is api_keys


def test_rate_limit_reraises_when_keychain_has_no_gemini_service(
    genai_client: MagicMock,
) -> None:
    """A rate limit with no Gemini keys to rotate propagates, not a KeyError."""
    genai_client.models.generate_images.side_effect = _genai_rate_limit()
    api_keys = FakeKeyChain({"gemini_api_key": GEMINI_KEY}, {"openai": 1})
    with pytest.raises(genai_errors.ClientError):
        _run(api_keys)
    assert api_keys.rotated == []


def test_rate_limit_reraises_when_no_retries_left(genai_client: MagicMock) -> None:
    """With the retry budget spent, the rate limit propagates."""
    genai_client.models.generate_images.side_effect = _genai_rate_limit()
    api_keys = FakeKeyChain({"gemini_api_key": GEMINI_KEY}, {GOOGLE_SERVICE: 0})
    with pytest.raises(genai_errors.ClientError):
        _run(api_keys)
    assert api_keys.rotated == []


def test_invalid_request_error_matches_the_mech_contract() -> None:
    """The mech recognises a client-side error by this exact class name."""
    assert_invalid_request_contract(module.InvalidRequestError)


@pytest.mark.parametrize("tool", ["other_tool", "", None])
def test_unsupported_tool_is_an_invalid_request(
    genai_client: MagicMock, tool: Optional[str]
) -> None:
    """An unsupported tool is a fault in the request; nothing is called."""
    with pytest.raises(module.InvalidRequestError, match="not supported"):
        _run(FakeKeyChain({"gemini_api_key": KEY_VALUE}), tool=tool)  # type: ignore[arg-type]
    genai_client.models.generate_images.assert_not_called()


@pytest.mark.parametrize("keys", [{"gemini_api_key": ""}, {}])
def test_missing_key_is_a_mech_side_error(
    genai_client: MagicMock, keys: Dict[str, str]
) -> None:
    """A missing operator key raises a plain ValueError before any paid call."""
    with pytest.raises(ValueError, match="GEMINI_API_KEY") as err:
        _run(FakeKeyChain(keys))
    assert_not_invalid_request(err.value)
    genai_client.models.generate_images.assert_not_called()


def test_keychain_without_rotation_methods_raises(genai_client: MagicMock) -> None:
    """A plain dict in place of the mech KeyChain raises instead of answering."""
    with pytest.raises(TypeError, match="required methods"):
        module.run(prompt=PROMPT, tool=TOOL, api_keys={"gemini_api_key": GEMINI_KEY})
    genai_client.models.generate_images.assert_not_called()
