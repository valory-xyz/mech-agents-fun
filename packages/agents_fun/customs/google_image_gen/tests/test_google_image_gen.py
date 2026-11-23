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

import pytest
from PIL import Image

import packages.agents_fun.customs.google_image_gen.google_image_gen as module
from packages.agents_fun.customs.google_image_gen.google_image_gen import (
    DEFAULT_MODEL,
    IMAGE_COST_USD,
    IMAGE_SIZE,
    TOKEN_COST_USD_PER_MILLION,
    image_cost_usd,
    run,
    token_cost_usd,
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
UNLISTED_MODEL = "some-future-image-model"
OTHER_SIZE = "8K"
OTHER_SIZE_PRICE = 0.5
IMAGE_HASH = "bafyimagehash"


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    return buffer.getvalue()


def _client_returning(image_bytes: Any, usage: Any = None) -> MagicMock:
    client = MagicMock()
    output = (
        None
        if image_bytes is None
        else SimpleNamespace(data=base64.b64encode(image_bytes).decode())
    )
    client.interactions.create.return_value = SimpleNamespace(
        output_image=output, usage=usage
    )
    return client


INPUT_TOKENS = 2_000
IMAGE_TOKENS = 1_120
TEXT_TOKENS = 300
THOUGHT_TOKENS = 500


def _usage(**overrides: Any) -> SimpleNamespace:
    """Usage of one interaction: a prompt, one image, some text and thinking."""
    fields: Dict[str, Any] = {
        "total_input_tokens": INPUT_TOKENS,
        "total_output_tokens": IMAGE_TOKENS + TEXT_TOKENS,
        "total_thought_tokens": THOUGHT_TOKENS,
        "output_tokens_by_modality": [
            SimpleNamespace(modality="image", tokens=IMAGE_TOKENS),
            SimpleNamespace(modality="text", tokens=TEXT_TOKENS),
        ],
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


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

    response = _run(client, recorder)
    result, prompt, callback = response[0], response[1], response[3]

    body = json.loads(result)
    assert body["image_hash"] == IMAGE_HASH
    assert body["model"] == DEFAULT_MODEL == "gemini-nano-banana-2.1"
    assert prompt == PROMPT and callback is recorder
    assert client.interactions.create.call_args.kwargs["model"] == DEFAULT_MODEL
    assert client.interactions.create.call_args.kwargs["response_format"] == {
        "type": "image",
        "image_size": IMAGE_SIZE,
    }
    assert recorder.charges() == [
        (DEFAULT_MODEL, IMAGE_COST_USD[DEFAULT_MODEL][IMAGE_SIZE])
    ]


def test_the_component_default_model_is_the_served_one() -> None:
    """The mech passes component.yaml's ``default_model`` as ``model``; it must be the current id."""
    component = module.__file__.replace("google_image_gen.py", "component.yaml")
    with open(component, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if "default_model:" in line]
    assert lines == [f"default_model: {DEFAULT_MODEL}"]


def test_run_uses_the_model_the_mech_passes() -> None:
    """A ``model`` kwarg, as the mech sends it, is the one the request names."""
    recorder = _CostRecorder()
    client = _client_returning(_png_bytes())
    ipfs = MagicMock()
    ipfs.add.return_value = (None, IMAGE_HASH, None)
    with (
        patch.object(module.genai, "Client", return_value=client),
        patch.object(module, "IPFSTool", return_value=ipfs),
    ):
        run(
            prompt=PROMPT,
            tool="google_image_gen",
            model=DEFAULT_MODEL,
            api_keys=_api_keys(gemini_api_key="gk"),
            counter_callback=recorder,
        )
    assert client.interactions.create.call_args.kwargs["model"] == DEFAULT_MODEL


def test_another_listed_model_is_charged_at_its_own_price() -> None:
    """The charge follows the model the request named, not the default."""
    recorder = _CostRecorder()
    client = _client_returning(_png_bytes())
    ipfs = MagicMock()
    ipfs.add.return_value = (None, IMAGE_HASH, None)
    with (
        patch.object(module.genai, "Client", return_value=client),
        patch.object(module, "IPFSTool", return_value=ipfs),
    ):
        run(
            prompt=PROMPT,
            tool="google_image_gen",
            model="gemini-3-pro-image",
            api_keys=_api_keys(gemini_api_key="gk"),
            counter_callback=recorder,
        )
    assert recorder.charges() == [("gemini-3-pro-image", 0.134)]
    assert image_cost_usd("gemini-3-pro-image", IMAGE_SIZE) == 0.134


@pytest.mark.parametrize(
    "model", [UNLISTED_MODEL, DEFAULT_MODEL.upper(), f"{DEFAULT_MODEL} ", ["gemini"]]
)
def test_a_model_without_a_listed_price_is_refused_before_calling_google(
    model: Any,
) -> None:
    """A model the tool has no price for is a request fault; no client is built."""
    recorder = _CostRecorder()
    with patch.object(module.genai, "Client") as client_cls:
        with pytest.raises(module.InvalidRequestError, match="is not supported"):
            run(
                prompt=PROMPT,
                tool="google_image_gen",
                model=model,
                api_keys=_api_keys(gemini_api_key="gk"),
                counter_callback=recorder,
            )
    client_cls.assert_not_called()
    assert recorder.charges() == []
    assert image_cost_usd(model, IMAGE_SIZE) == 0.0


def test_a_listed_model_without_a_price_at_the_served_size_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model is served only when it has a price at the size the tool generates."""
    monkeypatch.setitem(IMAGE_COST_USD, UNLISTED_MODEL, {OTHER_SIZE: OTHER_SIZE_PRICE})
    with patch.object(module.genai, "Client") as client_cls:
        with pytest.raises(module.InvalidRequestError, match="is not supported"):
            run(
                prompt=PROMPT,
                tool="google_image_gen",
                model=UNLISTED_MODEL,
                api_keys=_api_keys(gemini_api_key="gk"),
                counter_callback=_CostRecorder(),
            )
    client_cls.assert_not_called()


@pytest.mark.parametrize("model", sorted(IMAGE_COST_USD))
def test_every_listed_model_is_served_and_charged_its_listed_price(
    model: str,
) -> None:
    """Each model in the price table generates and reports its own price."""
    recorder = _CostRecorder()
    client = _client_returning(_png_bytes())
    ipfs = MagicMock()
    ipfs.add.return_value = (None, IMAGE_HASH, None)
    with (
        patch.object(module.genai, "Client", return_value=client),
        patch.object(module, "IPFSTool", return_value=ipfs),
    ):
        run(
            prompt=PROMPT,
            tool="google_image_gen",
            model=model,
            api_keys=_api_keys(gemini_api_key="gk"),
            counter_callback=recorder,
        )
    assert client.interactions.create.call_args.kwargs["model"] == model
    assert recorder.charges() == [(model, IMAGE_COST_USD[model][IMAGE_SIZE])]


def test_run_raises_and_charges_nothing_when_no_image_comes_back() -> None:
    """A response without image data is a provider fault and costs the requester nothing."""
    recorder = _CostRecorder()

    with pytest.raises(ValueError, match="No image data"):
        _run(_client_returning(None), recorder)

    assert recorder.charges() == []


def test_run_refuses_an_unknown_tool_before_calling_google() -> None:
    """The tool name is validated before any client is built."""
    recorder = _CostRecorder()
    with patch.object(module.genai, "Client") as client_cls:
        with pytest.raises(module.InvalidRequestError, match="not supported"):
            run(
                prompt=PROMPT,
                tool="not-this-tool",
                api_keys=_api_keys(gemini_api_key="gk"),
                counter_callback=recorder,
            )
    client_cls.assert_not_called()
    assert recorder.charges() == []


def test_run_adds_the_prompt_and_text_tokens_to_the_image_price() -> None:
    """The reported charge is the image plus the prompt, text and thinking tokens."""
    recorder = _CostRecorder()

    _run(_client_returning(_png_bytes(), _usage()), recorder)

    input_rate, text_rate = TOKEN_COST_USD_PER_MILLION[DEFAULT_MODEL]
    tokens = (
        INPUT_TOKENS * input_rate + (TEXT_TOKENS + THOUGHT_TOKENS) * text_rate
    ) / 1_000_000
    ((model, charge),) = recorder.charges()
    assert model == DEFAULT_MODEL
    assert charge == pytest.approx(IMAGE_COST_USD[DEFAULT_MODEL][IMAGE_SIZE] + tokens)
    assert tokens > 0


@pytest.mark.parametrize(
    "usage, input_tokens, text_tokens",
    [
        (None, 0, 0),
        (SimpleNamespace(), 0, 0),
        (_usage(output_tokens_by_modality=None), INPUT_TOKENS, THOUGHT_TOKENS),
        (_usage(total_thought_tokens=None), INPUT_TOKENS, TEXT_TOKENS),
        (_usage(total_input_tokens=None), 0, TEXT_TOKENS + THOUGHT_TOKENS),
        (_usage(total_input_tokens=-5), 0, TEXT_TOKENS + THOUGHT_TOKENS),
        (_usage(total_input_tokens="2000"), 0, TEXT_TOKENS + THOUGHT_TOKENS),
        (_usage(total_input_tokens=True), 0, TEXT_TOKENS + THOUGHT_TOKENS),
        (
            _usage(output_tokens_by_modality=[SimpleNamespace(modality="image")]),
            INPUT_TOKENS,
            THOUGHT_TOKENS,
        ),
    ],
    ids=[
        "no_usage",
        "empty_usage",
        "no_modality_breakdown",
        "no_thought_tokens",
        "no_input_tokens",
        "negative_input",
        "text_input",
        "bool_input",
        "image_entry_without_tokens",
    ],
)
def test_token_cost_counts_only_what_the_usage_states(
    usage: Any, input_tokens: int, text_tokens: int
) -> None:
    """A missing or malformed field counts as zero; image tokens are never charged here."""
    input_rate, text_rate = TOKEN_COST_USD_PER_MILLION[DEFAULT_MODEL]
    expected = (input_tokens * input_rate + text_tokens * text_rate) / 1_000_000
    assert token_cost_usd(DEFAULT_MODEL, usage) == pytest.approx(expected)


def test_token_cost_is_zero_for_a_model_without_rates() -> None:
    """A model with no listed token rates adds nothing to the image price."""
    assert token_cost_usd(UNLISTED_MODEL, _usage()) == 0.0


def test_every_model_with_an_image_price_has_token_rates() -> None:
    """The two price tables cover the same models."""
    assert set(TOKEN_COST_USD_PER_MILLION) == set(IMAGE_COST_USD)
