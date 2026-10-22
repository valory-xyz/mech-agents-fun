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
        """Record the rotation."""
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
