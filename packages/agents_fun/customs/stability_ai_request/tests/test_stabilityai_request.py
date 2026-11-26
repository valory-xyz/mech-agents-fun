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
    ENGINE_SIZE_CHART,
    MAX_SAMPLES,
    MAX_STEPS,
    MIN_SAMPLES,
    MIN_STEPS,
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
HTTP_OK = 200
HTTP_SERVER_ERROR = 500
SAMPLES = 3
LANDSCAPE = {"height": 768, "width": 1344}
OFF_CHART_SIDE = 2048


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

    (result, _, _, callback, _), post = _run(recorder, HTTP_OK)

    assert json.loads(result)["artifacts"]
    assert callback is recorder
    assert post.call_args.kwargs["json"]["samples"] == 1
    assert recorder.charges() == pytest.approx(
        [(ENGINE, SDXL_CREDITS_PER_IMAGE * USD_PER_CREDIT)]
    )


def test_several_samples_are_charged_each() -> None:
    """The charge scales with the number of images asked for."""
    recorder = _CostRecorder()
    _run(recorder, HTTP_OK, samples=SAMPLES)
    assert recorder.charges() == pytest.approx(
        [(ENGINE, SAMPLES * SDXL_CREDITS_PER_IMAGE * USD_PER_CREDIT)]
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"steps": MIN_STEPS},
        {"steps": MAX_STEPS},
        {"samples": MIN_SAMPLES},
        {"samples": MAX_SAMPLES},
        LANDSCAPE,
        ENGINE_SIZE_CHART[ENGINE][-1],
    ],
)
def test_settings_the_price_covers_are_sent_and_charged(
    settings: Dict[str, int],
) -> None:
    """Settings within the priced range reach Stability and are charged per image."""
    recorder = _CostRecorder()

    _, post = _run(recorder, HTTP_OK, **settings)

    sent = post.call_args.kwargs["json"]
    assert {key: sent[key] for key in settings} == settings
    ((model, cost),) = recorder.charges()
    assert model == ENGINE
    assert cost == pytest.approx(
        sent["samples"] * SDXL_CREDITS_PER_IMAGE * USD_PER_CREDIT
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"steps": MAX_STEPS + 1},
        {"steps": MIN_STEPS - 1},
        {"steps": 0},
        {"steps": -MAX_STEPS},
        {"steps": float(MAX_STEPS)},
        {"steps": str(MAX_STEPS)},
        {"steps": True},
        {"steps": None},
        {"samples": MAX_SAMPLES + 1},
        {"samples": MIN_SAMPLES - 1},
        {"samples": -MIN_SAMPLES},
        {"samples": 1.5},
        {"samples": str(MIN_SAMPLES)},
        {"samples": True},
        {"samples": None},
        {"height": OFF_CHART_SIDE},
        {"width": OFF_CHART_SIDE},
        {"height": OFF_CHART_SIDE, "width": OFF_CHART_SIDE},
        {"height": LANDSCAPE["width"], "width": LANDSCAPE["width"]},
        {"height": str(LANDSCAPE["height"]), "width": LANDSCAPE["width"]},
        {"height": None},
        {"width": None},
    ],
)
def test_settings_outside_the_priced_range_are_refused_before_the_paid_call(
    settings: Dict[str, Any],
) -> None:
    """A size, step count or sample count the price does not cover is a request fault."""
    recorder = _CostRecorder()

    with pytest.raises(module.InvalidRequestError):
        _run(recorder, HTTP_OK, **settings)

    assert recorder.charges() == []


def test_refused_settings_never_reach_stability() -> None:
    """The settings are checked before the HTTP call is made."""
    with patch.object(module.requests, "post") as post:
        with pytest.raises(module.InvalidRequestError, match="steps"):
            run(
                prompt="a cat",
                tool=TOOL,
                api_keys=_api_keys(stabilityai="sk"),
                steps=MAX_STEPS + 1,
            )
    post.assert_not_called()


def test_an_exception_raises_without_a_charge() -> None:
    """An error from the provider call propagates out of the rotation wrapper."""
    recorder = _CostRecorder()
    with patch.object(module.requests, "post", side_effect=RuntimeError("down")):
        with pytest.raises(RuntimeError, match="down"):
            run(
                prompt="a cat",
                tool=TOOL,
                api_keys=_api_keys(stabilityai="sk"),
                counter_callback=recorder,
            )
    assert recorder.charges() == []


def test_a_failed_call_raises_without_a_charge() -> None:
    """A non-200 answer is a provider fault: it raises and costs nothing."""
    recorder = _CostRecorder()

    with pytest.raises(module.requests.HTTPError, match="Non-200 response"):
        _run(recorder, HTTP_SERVER_ERROR)

    assert recorder.charges() == []
