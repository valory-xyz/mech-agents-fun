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

"""Shared fakes for the custom tool unit tests."""

from typing import Dict, List, Optional

import httpx
import openai

OPENAI_TEST_URL = "https://api.openai.com/v1/test"
HTTP_TOO_MANY_REQUESTS = 429
HTTP_UNAUTHORIZED = 401
HTTP_BAD_REQUEST = 400
HTTP_SERVER_ERROR = 500
INVALID_REQUEST_ERROR_NAME = "InvalidRequestError"


class FakeKeyChain(dict):
    """A KeyChain stand-in that records every rotation."""

    def __init__(
        self, keys: Dict[str, str], retries: Optional[Dict[str, int]] = None
    ) -> None:
        """Store the keys and the retry budget per service."""
        super().__init__(keys)
        self._retries = dict(retries or {service: 0 for service in keys})
        self.rotated: List[str] = []

    def max_retries(self) -> Dict[str, int]:
        """Return a copy of the retry budget."""
        return dict(self._retries)

    def rotate(self, service_name: str) -> None:
        """Record the rotation; an unknown service raises, as in the KeyChain."""
        if service_name not in self:
            raise KeyError(service_name)
        self.rotated.append(service_name)


def _openai_response(status_code: int) -> httpx.Response:
    """Build an httpx response for an OpenAI SDK error."""
    return httpx.Response(status_code, request=httpx.Request("POST", OPENAI_TEST_URL))


def openai_rate_limit_error() -> openai.RateLimitError:
    """Build the error the OpenAI SDK raises on HTTP 429."""
    return openai.RateLimitError(
        "rate limited", response=_openai_response(HTTP_TOO_MANY_REQUESTS), body=None
    )


def openai_auth_error() -> openai.AuthenticationError:
    """Build the error the OpenAI SDK raises on HTTP 401."""
    return openai.AuthenticationError(
        "invalid key", response=_openai_response(HTTP_UNAUTHORIZED), body=None
    )


def openai_bad_request_error() -> openai.BadRequestError:
    """Build the error the OpenAI SDK raises on HTTP 400."""
    return openai.BadRequestError(
        "prompt rejected", response=_openai_response(HTTP_BAD_REQUEST), body=None
    )


def openai_server_error() -> openai.InternalServerError:
    """Build the error the OpenAI SDK raises on HTTP 500."""
    return openai.InternalServerError(
        "backend down", response=_openai_response(HTTP_SERVER_ERROR), body=None
    )


def assert_invalid_request_contract(error_cls: type) -> None:
    """Check a tool's client-side error class against the mech contract.

    :param error_cls: the tool module's InvalidRequestError.
    """
    assert error_cls.__name__ == INVALID_REQUEST_ERROR_NAME
    assert issubclass(error_cls, ValueError)


def assert_not_invalid_request(error: BaseException) -> None:
    """Check an error is reported as a failure on the mech's side.

    :param error: the raised exception.
    """
    assert type(error).__name__ != INVALID_REQUEST_ERROR_NAME
