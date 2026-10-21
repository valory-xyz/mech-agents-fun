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

"""Tests for the recraft_image_gen tool: pinned model and per-image cost reporting."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import packages.agents_fun.customs.recraft_image_gen.recraft_image_gen as module
from packages.agents_fun.customs.recraft_image_gen.recraft_image_gen import (
    RECRAFT_COST_PER_IMAGE_USD,
    RECRAFT_MODEL,
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


IMAGE_HASH = "bafyrecraft"


def _run(recorder: _CostRecorder, url: Any) -> Tuple[Any, ...]:
    client = MagicMock()
    client.images.generate.return_value = SimpleNamespace(
        data=[SimpleNamespace(url=url)] if url else []
    )
    with (
        patch.object(module, "OpenAI", return_value=client),
        patch.object(module, "_download_image", return_value=b"png"),
        patch.object(
            module, "_process_image_and_upload", return_value=(IMAGE_HASH, None)
        ),
    ):
        result = run(
            prompt="a fox",
            tool="recraft-image-gen",
            api_keys=_api_keys(recraft_api_key="rk"),
            counter_callback=recorder,
        )
    return result, client


def test_the_model_is_pinned_and_one_image_is_charged() -> None:
    """The call names the model, so Recraft's default cannot move the price or the styles."""
    recorder = _CostRecorder()

    (result, _, _, callback, _), client = _run(recorder, "https://img/fox.png")

    assert json.loads(result)["image_hash"] == IMAGE_HASH
    assert callback is recorder
    assert (
        client.images.generate.call_args.kwargs["model"] == RECRAFT_MODEL == "recraftv3"
    )
    assert recorder.charges() == [(RECRAFT_MODEL, RECRAFT_COST_PER_IMAGE_USD)]


def test_no_image_url_means_no_charge() -> None:
    """A response without an image costs the requester nothing."""
    recorder = _CostRecorder()

    (result, _, _, callback, _), _ = _run(recorder, None)

    assert "No image" in result or "image" in result.lower()
    assert callback is recorder
    assert recorder.charges() == []
