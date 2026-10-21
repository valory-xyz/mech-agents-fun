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

"""Tests for the google_video_gen tool: current models and usage-based cost reporting."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

import packages.agents_fun.customs.google_video_gen.google_video_gen as module
from packages.agents_fun.customs.google_video_gen.google_video_gen import (
    SCRIPT_INPUT_USD_PER_M,
    SCRIPT_MODEL,
    SCRIPT_OUTPUT_USD_PER_M,
    TTS_INPUT_USD_PER_M,
    TTS_MODEL,
    TTS_OUTPUT_USD_PER_M,
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


PROMPT = "why the sky is blue"
AUDIO_HASH = "bafyaudiohash"
SCRIPT_TOKENS = (100, 50)
TTS_TOKENS = (20, 250)


def _usage(prompt_tokens: int, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt_tokens, candidates_token_count=output_tokens
    )


def _script_response() -> SimpleNamespace:
    return SimpleNamespace(
        text="Voiceover: The sky scatters blue light.\nVoice: kore",
        usage_metadata=_usage(*SCRIPT_TOKENS),
    )


def _tts_response() -> SimpleNamespace:
    part = SimpleNamespace(inline_data=SimpleNamespace(data=b"\x00\x01"))
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    return SimpleNamespace(candidates=[candidate], usage_metadata=_usage(*TTS_TOKENS))


def _run(recorder: _CostRecorder, responses: List[Any]) -> Tuple[Any, ...]:
    client = MagicMock()
    client.models.generate_content.side_effect = responses
    ipfs = MagicMock()
    ipfs.add.return_value = (None, AUDIO_HASH, None)
    with (
        patch.object(module.genai, "Client", return_value=client),
        patch.object(module, "IPFSTool", return_value=ipfs),
        patch.object(module, "wave_file"),
        patch.object(module.os.path, "exists", return_value=False),
    ):
        result = run(
            prompt=PROMPT,
            tool="google_video_gen",
            api_keys=_api_keys(gemini_api_key="gk"),
            counter_callback=recorder,
        )
    return result, client


def test_run_calls_the_current_models_and_reports_both_charges() -> None:
    """The script and the voice are each charged from their usage at list prices."""
    recorder = _CostRecorder()

    (result, _, _, callback, _), client = _run(
        recorder, [_script_response(), _tts_response()]
    )

    body = json.loads(result)
    assert body["audio_hash"] == AUDIO_HASH
    assert body["model"] == TTS_MODEL
    assert callback is recorder
    models = [c.kwargs["model"] for c in client.models.generate_content.call_args_list]
    assert models == [SCRIPT_MODEL, TTS_MODEL]
    assert (
        SCRIPT_MODEL == "gemini-3.5-flash-lite"
        and TTS_MODEL == "gemini-3.8-flash-lite-tts"
    )
    script_cost = (
        SCRIPT_TOKENS[0] * SCRIPT_INPUT_USD_PER_M
        + SCRIPT_TOKENS[1] * SCRIPT_OUTPUT_USD_PER_M
    ) / 1e6
    tts_cost = (
        TTS_TOKENS[0] * TTS_INPUT_USD_PER_M + TTS_TOKENS[1] * TTS_OUTPUT_USD_PER_M
    ) / 1e6
    assert recorder.charges() == pytest.approx(
        [(SCRIPT_MODEL, script_cost), (TTS_MODEL, tts_cost)]
    )


def test_a_failed_voice_call_leaves_only_the_script_charge() -> None:
    """A charge is reported only for calls that returned."""
    recorder = _CostRecorder()

    (result, _, _, callback, _), _ = _run(
        recorder, [_script_response(), RuntimeError("tts down")]
    )

    assert "error" in result.lower()
    assert callback is recorder
    assert [model for model, _ in recorder.charges()] == [SCRIPT_MODEL]


def test_a_response_without_usage_is_charged_nothing() -> None:
    """Missing usage metadata is reported as zero rather than guessed."""
    cost = module._usage_cost_usd  # pylint: disable=protected-access
    assert cost(SimpleNamespace(), 1.0, 1.0) == 0.0
    assert cost(
        SimpleNamespace(usage_metadata=_usage(10, 0)), 0.3, 2.5
    ) == pytest.approx(3e-6)
