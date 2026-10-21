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

"""Tests for the stability_ai_request tool: cost reporting on the v1 SDXL endpoint."""

import json
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

import packages.agents_fun.customs.stability_ai_request.stabilityai_request as module
from packages.agents_fun.customs.stability_ai_request.stabilityai_request import (
    SDXL_CREDITS_PER_IMAGE,
    USD_PER_CREDIT,
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


TOOL = "stabilityai-stable-diffusion-xl-1024-v1-0"
ENGINE = "stable-diffusion-xl-1024-v1-0"


def _run(recorder: _CostRecorder, status: int, **kwargs: Any) -> Tuple[Any, ...]:
    response = MagicMock(status_code=status, text="err")
    response.json.return_value = {"artifacts": [{"base64": "..."}]}
    with patch.object(module.requests, "post", return_value=response) as post:
        result = run(
            prompt="a cat",
            tool=TOOL,
            api_keys=_api_keys(stabilityai="sk"),
            counter_callback=recorder,
            **kwargs,
        )
    return result, post


def test_a_generated_image_is_charged_at_the_default_credit_price() -> None:
    """One sample at the default settings costs 0.9 credits at a cent a credit."""
    recorder = _CostRecorder()

    (result, _, _, callback, _), post = _run(recorder, 200)

    assert json.loads(result)["artifacts"]
    assert callback is recorder
    assert post.call_args.kwargs["json"]["samples"] == 1
    assert recorder.charges() == pytest.approx(
        [(ENGINE, SDXL_CREDITS_PER_IMAGE * USD_PER_CREDIT)]
    )


def test_several_samples_are_charged_each() -> None:
    """The charge scales with the number of images asked for."""
    recorder = _CostRecorder()
    _run(recorder, 200, samples=3)
    assert recorder.charges() == pytest.approx(
        [(ENGINE, 3 * SDXL_CREDITS_PER_IMAGE * USD_PER_CREDIT)]
    )


def test_an_exception_still_hands_the_callback_back() -> None:
    """The rotation wrapper's error path returns the callback, not None."""
    recorder = _CostRecorder()
    with patch.object(module.requests, "post", side_effect=RuntimeError("down")):
        result = run(
            prompt="a cat",
            tool=TOOL,
            api_keys=_api_keys(stabilityai="sk"),
            counter_callback=recorder,
        )
    assert result[3] is recorder
    assert recorder.charges() == []


def test_a_failed_call_returns_the_callback_without_a_charge() -> None:
    """A non-200 answer is an error, costs nothing, and still hands the callback back."""
    recorder = _CostRecorder()

    (result, _, _, callback, _), _ = _run(recorder, 500)

    assert result.startswith("Error")
    assert callback is recorder
    assert recorder.charges() == []
