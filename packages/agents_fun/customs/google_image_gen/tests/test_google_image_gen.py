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

"""Tests for the google_image_gen tool: model, cost reporting and result shape."""

import base64
import io
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

from PIL import Image

import packages.agents_fun.customs.google_image_gen.google_image_gen as module
from packages.agents_fun.customs.google_image_gen.google_image_gen import (
    DEFAULT_MODEL,
    IMAGE_COST_USD,
    IMAGE_SIZE,
    run,
)


class _CostRecorder:
    """Stands in for the mech's cost callback and keeps every report."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def charges(self) -> List[Tuple[str, float]]:
        """The (model, call_cost) pairs reported, in order."""
        return [(c["model"], c.get("call_cost", 0.0)) for c in self.calls]


def _api_keys(**services: str) -> MagicMock:
    """A KeyChain-like object with the given services and the rotation methods."""
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key]
    mock.get = lambda key, default=None: services.get(key, default)
    mock.max_retries = lambda: {name: 1 for name in services}
    return mock


PROMPT = "a lighthouse at dusk"
IMAGE_HASH = "bafyimagehash"


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    return buffer.getvalue()


def _client_returning(image_bytes: Any) -> MagicMock:
    client = MagicMock()
    output = (
        None
        if image_bytes is None
        else SimpleNamespace(data=base64.b64encode(image_bytes).decode())
    )
    client.interactions.create.return_value = SimpleNamespace(output_image=output)
    return client


def _run(client: MagicMock, recorder: _CostRecorder) -> Tuple[Any, ...]:
    ipfs = MagicMock()
    ipfs.add.return_value = (None, IMAGE_HASH, None)
    with (
        patch.object(module.genai, "Client", return_value=client),
        patch.object(module, "IPFSTool", return_value=ipfs),
    ):
        return run(
            prompt=PROMPT,
            tool="google_image_gen",
            api_keys=_api_keys(gemini_api_key="gk"),
            counter_callback=recorder,
        )


def test_run_uses_nano_banana_and_reports_the_per_image_price() -> None:
    """One image at the configured size is charged at Google's per-image price."""
    recorder = _CostRecorder()
    client = _client_returning(_png_bytes())

    result, prompt, _, callback, _ = _run(client, recorder)

    body = json.loads(result)
    assert body["image_hash"] == IMAGE_HASH
    assert body["model"] == DEFAULT_MODEL == "gemini-nano-banana-2.1"
    assert prompt == PROMPT and callback is recorder
    assert client.interactions.create.call_args.kwargs["model"] == DEFAULT_MODEL
    assert client.interactions.create.call_args.kwargs["response_format"] == {
        "type": "image",
        "image_size": IMAGE_SIZE,
    }
    assert recorder.charges() == [(DEFAULT_MODEL, IMAGE_COST_USD[IMAGE_SIZE])]


def test_run_charges_nothing_when_no_image_comes_back() -> None:
    """A response without image data is an error and costs the requester nothing."""
    recorder = _CostRecorder()

    result, _, _, callback, _ = _run(_client_returning(None), recorder)

    assert "No image data" in result
    assert callback is recorder
    assert recorder.charges() == []


def test_run_refuses_an_unknown_tool_before_calling_google() -> None:
    """The tool name is validated before any client is built."""
    recorder = _CostRecorder()
    with patch.object(module.genai, "Client") as client_cls:
        result, _, _, _, _ = run(
            prompt=PROMPT,
            tool="not-this-tool",
            api_keys=_api_keys(gemini_api_key="gk"),
            counter_callback=recorder,
        )
    assert "not supported" in result
    client_cls.assert_not_called()
    assert recorder.charges() == []
