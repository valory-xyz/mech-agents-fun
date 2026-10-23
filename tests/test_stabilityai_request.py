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

"""Unit tests for stabilityai_request failure handling."""

import json
from unittest.mock import MagicMock

import pytest
import requests

from packages.agents_fun.customs.stability_ai_request import (
    stabilityai_request as module,
)

from tests.tool_test_helpers import (
    FakeKeyChain,
    assert_invalid_request_contract,
    assert_not_invalid_request,
)

TOOL = "stabilityai-stable-diffusion-xl-1024-v1-0"
PROMPT = "a quiet forest"
STABILITY_KEY = "stability-test"
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_FORBIDDEN = 403
HTTP_SERVER_ERROR = 500
SUCCESS_BODY = {"artifacts": [{"base64": "AAAA"}]}


@pytest.fixture
def post(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the HTTP boundary to the Stability API."""
    mock_post = MagicMock()
    monkeypatch.setattr(module.requests, "post", mock_post)
    return mock_post


def _api_keys() -> FakeKeyChain:
    """Build a keychain with the Stability key and one OpenAI retry."""
    return FakeKeyChain(
        {"stabilityai": STABILITY_KEY, "openai": "", "openrouter": ""},
        {"openai": 1, "openrouter": 1},
    )


def _run(api_keys: FakeKeyChain, tool: str = TOOL) -> tuple:
    """Call the tool entry point."""
    return module.run(prompt=PROMPT, tool=tool, api_keys=api_keys)


def test_bad_request_is_an_invalid_request(post: MagicMock) -> None:
    """A 400 from Stability is a fault in the request, with the reason kept."""
    post.return_value = MagicMock(status_code=HTTP_BAD_REQUEST, text="bad height")
    with pytest.raises(module.InvalidRequestError, match="rejected.*bad height"):
        _run(_api_keys())


@pytest.mark.parametrize("status", [HTTP_FORBIDDEN, HTTP_SERVER_ERROR])
def test_other_non_200_responses_are_mech_side(post: MagicMock, status: int) -> None:
    """Any other non-200 answer raises HTTPError with the status in the message."""
    post.return_value = MagicMock(status_code=status, text="no access")
    with pytest.raises(requests.HTTPError, match=f"{status}.*no access") as err:
        _run(_api_keys())
    assert_not_invalid_request(err.value)


def test_invalid_request_error_matches_the_mech_contract() -> None:
    """The mech recognises a client-side error by this exact class name."""
    assert_invalid_request_contract(module.InvalidRequestError)


def test_run_raises_on_network_error(post: MagicMock) -> None:
    """A network failure propagates out of run."""
    post.side_effect = requests.ConnectionError("unreachable")
    with pytest.raises(requests.ConnectionError):
        _run(_api_keys())


def test_successful_run_returns_body_and_keychain(post: MagicMock) -> None:
    """A 200 answer returns the body with the keychain appended."""
    post.return_value = MagicMock(status_code=HTTP_OK, json=lambda: SUCCESS_BODY)
    api_keys = _api_keys()
    result = _run(api_keys)
    assert json.loads(result[0]) == SUCCESS_BODY
    assert result[-1] is api_keys


HTTP_TOO_MANY_REQUESTS = 429


@pytest.mark.parametrize("retries", [{"stabilityai": 1}, {"stabilityai": 2}])
def test_stability_rate_limit_rotates_key_and_retries(
    post: MagicMock, retries: dict
) -> None:
    """A Stability 429 rotates the Stability key and the retry answers."""
    post.side_effect = [
        MagicMock(status_code=HTTP_TOO_MANY_REQUESTS, text="slow down"),
        MagicMock(status_code=HTTP_OK, json=lambda: SUCCESS_BODY),
    ]
    api_keys = FakeKeyChain({"stabilityai": STABILITY_KEY}, retries)

    result = _run(api_keys)

    assert api_keys.rotated == ["stabilityai"]
    assert json.loads(result[0]) == SUCCESS_BODY
    assert post.call_count == 2


@pytest.mark.parametrize("retries", [{"stabilityai": 0}, {}])
def test_stability_rate_limit_propagates_without_keys_left(
    post: MagicMock, retries: dict
) -> None:
    """With no Stability key left to rotate, the 429 propagates as HTTPError."""
    post.return_value = MagicMock(status_code=HTTP_TOO_MANY_REQUESTS, text="slow")
    api_keys = FakeKeyChain({"stabilityai": STABILITY_KEY}, retries)
    with pytest.raises(requests.HTTPError, match="429") as err:
        _run(api_keys)
    assert_not_invalid_request(err.value)
    assert api_keys.rotated == []
    post.assert_called_once()


def test_non_rate_limit_http_error_does_not_rotate(post: MagicMock) -> None:
    """A Stability 500 propagates without touching the keys."""
    post.return_value = MagicMock(status_code=HTTP_SERVER_ERROR, text="down")
    api_keys = FakeKeyChain({"stabilityai": STABILITY_KEY}, {"stabilityai": 1})
    with pytest.raises(requests.HTTPError):
        _run(api_keys)
    assert api_keys.rotated == []


def test_wrapper_propagates_non_rate_limit_error() -> None:
    """The wrapper does not turn other errors into an answer."""
    func = MagicMock(side_effect=[KeyError("stabilityai")])
    api_keys = _api_keys()
    with pytest.raises(KeyError):
        module.with_key_rotation(func)(api_keys=api_keys)
    assert api_keys.rotated == []


@pytest.mark.parametrize("tool", ["other_tool", "stabilityai-", ""])
def test_unsupported_tool_raises_before_any_provider_call(
    post: MagicMock, tool: str
) -> None:
    """An unsupported tool is a fault in the request; nothing is called."""
    with pytest.raises(
        module.InvalidRequestError, match="not in the list of supported tools"
    ):
        _run(_api_keys(), tool=tool)
    post.assert_not_called()
