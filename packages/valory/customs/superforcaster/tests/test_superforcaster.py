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

"""Unit tests for superforcaster: thread-safe client, structured outputs, source_content."""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest
import requests

import packages.valory.customs.superforcaster.superforcaster as module
from packages.valory.customs.superforcaster.superforcaster import (
    OpenAIClientManager,
    PredictionResult,
    _parse_completion,
    run,
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_openai_client(self) -> None:
        """__enter__ returns a fresh openai.OpenAI client, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(
            "packages.valory.customs.superforcaster.superforcaster.OpenAI"
        ) as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance

            with mgr as client:
                assert client is mock_instance
                MockOpenAI.assert_called_once_with(api_key="sk-test")

            mock_instance.close.assert_called_once()

    def test_no_global_client_variable(self) -> None:
        """The module must not define a module-level 'client' variable."""
        source = Path(module.__file__).read_text(encoding="utf-8")
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("client:") or stripped.startswith("client ="):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client' variable found at line {i}: {line}"
                    )

    def test_parse_completion_requires_client_param(self) -> None:
        """_parse_completion requires client as first param."""
        params = list(inspect.signature(_parse_completion).parameters)
        assert params[0] == "client"


SF_MODULE = "packages.valory.customs.superforcaster.superforcaster"

FAKE_SERPER_RESPONSE = {
    "searchParameters": {"q": "test query", "type": "search"},
    "organic": [
        {
            "title": "Test Result",
            "link": "http://example.com/result",
            "snippet": "Test snippet content",
            "position": 1,
        },
    ],
    "peopleAlsoAsk": [
        {"question": "What is test?", "snippet": "A test answer."},
    ],
}

FAKE_PREDICTION = PredictionResult(
    facts="Fact 1. Fact 2.",
    reasons_no="No 1 (strength 6). No 2 (strength 4).",
    reasons_yes="Yes 1 (strength 5). Yes 2 (strength 3).",
    aggregation="Base rate 0.5. Tentative: 0.5.",
    reflection="Passes evidence bar.",
    p_yes=0.5,
    p_no=0.5,
    confidence=0.5,
    info_utility=0.5,
)

PREDICTION_PROMPT = (
    'With the given question "Will X happen?" '
    "and the `yes` option represented by `Yes` and the `no` option represented by `No`, "
    "what are the respective probabilities of `p_yes` and `p_no` occurring?"
)


def _make_mock_api_keys(return_source_content: str = "false") -> MagicMock:
    """Create a mock KeyChain-like api_keys object."""
    services = {
        "openai": ["sk-test"],
        "serperapi": ["serper-test"],
        "return_source_content": [return_source_content],
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key][0]
    mock.get = lambda key, default="": services.get(key, [default])[0]
    return mock


def _mock_parse_response() -> MagicMock:
    """Build a fake response object matching the beta.chat.completions.parse shape."""
    return MagicMock(
        choices=[MagicMock(message=MagicMock(parsed=FAKE_PREDICTION, refusal=None))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )


class TestStructuredOutputContract:
    """Verify the tool uses OpenAI Structured Outputs and returns only the 4 on-chain fields."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_uses_beta_parse_with_prediction_result_schema(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """run() calls client.beta.chat.completions.parse with response_format=PredictionResult."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        mock_client.beta.chat.completions.parse.assert_called_once()
        kwargs = mock_client.beta.chat.completions.parse.call_args.kwargs
        assert kwargs["response_format"] is PredictionResult
        assert kwargs["model"] == "gpt-4.1-2025-04-14"
        mock_client.chat.completions.create.assert_not_called()

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_returns_only_four_mech_fields_on_chain(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """The on-chain result JSON contains exactly p_yes/p_no/confidence/info_utility."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        on_chain_json = result[0]
        parsed = json.loads(on_chain_json)
        assert set(parsed.keys()) == {"p_yes", "p_no", "confidence", "info_utility"}
        assert "facts" not in parsed
        assert "reasons_no" not in parsed
        assert "aggregation" not in parsed
        assert "reflection" not in parsed
        assert parsed["p_yes"] == 0.5
        assert parsed["p_no"] == 0.5


class TestPredictionResultValidator:
    """Pydantic sum validator rejects malformed probabilities."""

    def test_valid_sum_accepted(self) -> None:
        """p_yes + p_no = 1.0 is accepted."""
        obj = PredictionResult(
            facts="f",
            reasons_no="n",
            reasons_yes="y",
            aggregation="a",
            reflection="r",
            p_yes=0.7,
            p_no=0.3,
            confidence=0.6,
            info_utility=0.5,
        )
        assert obj.p_yes == 0.7

    def test_sum_violation_rejected(self) -> None:
        """p_yes + p_no outside 0.01 tolerance raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PredictionResult(
                facts="f",
                reasons_no="n",
                reasons_yes="y",
                aggregation="a",
                reflection="r",
                p_yes=0.7,
                p_no=0.2,
                confidence=0.6,
                info_utility=0.5,
            )


class TestSuperforcasterSourceContent:
    """Verify superforcaster captures and replays source_content correctly."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_live_capture_wraps_serper_json(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Live run wraps Serper response in {'serper_response': ...}."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
        )

        used_params = result[4]
        assert "source_content" in used_params
        assert "mode" in used_params["source_content"]
        assert "serper_response" in used_params["source_content"]
        assert used_params["source_content"]["serper_response"] == FAKE_SERPER_RESPONSE

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_replay_with_serper_response_format(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Replay with {'serper_response': ...} uses organic and peopleAlsoAsk."""
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        source_content = {"serper_response": FAKE_SERPER_RESPONSE}
        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        # Verify the prompt contains the organic result
        prediction_prompt = result[1]
        assert "Test Result" in prediction_prompt
        assert "Test snippet content" in prediction_prompt
        assert "What is test?" in prediction_prompt

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_flag_off_no_source_content(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """When return_source_content is false, source_content is not in used_params."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        used_params = result[4]
        assert "source_content" not in used_params


HTTP_TOO_MANY_REQUESTS = 429
HTTP_UNAUTHORIZED = 401
MODEL = "gpt-4.1-2025-04-14"


class _FakeKeyChain(dict):
    """A KeyChain stand-in that records every rotation."""

    def __init__(self, retries: int) -> None:
        """Store one key per service and the retry budget."""
        super().__init__(
            openai="sk-test", openrouter="or-test", serperapi="serper-test"
        )
        self._retries = {"openai": retries, "openrouter": retries}
        self.rotated: list = []

    def max_retries(self) -> dict:
        """Return a copy of the retry budget."""
        return dict(self._retries)

    def rotate(self, service_name: str) -> None:
        """Record the rotation."""
        self.rotated.append(service_name)


def _openai_error(error_cls: type, status_code: int) -> Exception:
    """Build an OpenAI SDK status error."""
    response = MagicMock(status_code=status_code, headers={})
    return error_cls("provider error", response=response, body=None)


class TestFailedRunRaises:
    """A failed run raises, so the mech can reject it instead of charging."""

    @pytest.fixture
    def parse(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Patch the OpenAI client, the Serper HTTP call and the retry sleep."""
        client = MagicMock()
        monkeypatch.setattr(module, "OpenAI", MagicMock(return_value=client))
        serper = MagicMock()
        serper.json.return_value = FAKE_SERPER_RESPONSE
        monkeypatch.setattr(module.requests, "request", MagicMock(return_value=serper))
        monkeypatch.setattr(module.time, "sleep", MagicMock())
        return client.beta.chat.completions.parse

    def _run(self, api_keys: _FakeKeyChain, **extra: object) -> tuple:
        """Call the tool entry point."""
        return run(
            tool="superforcaster",
            model=MODEL,
            prompt=PREDICTION_PROMPT,
            api_keys=api_keys,
            **extra,
        )

    def test_non_retryable_llm_error_propagates(self, parse: MagicMock) -> None:
        """An OpenAI error outside the retry set propagates out of run."""
        parse.side_effect = _openai_error(openai.AuthenticationError, HTTP_UNAUTHORIZED)
        api_keys = _FakeKeyChain(retries=1)
        with pytest.raises(openai.AuthenticationError):
            self._run(api_keys)
        assert api_keys.rotated == []

    def test_search_network_error_propagates(self, parse: MagicMock) -> None:
        """A Serper network failure propagates out of run."""
        module.requests.request.side_effect = requests.ConnectionError("down")
        with pytest.raises(requests.ConnectionError):
            self._run(_FakeKeyChain(retries=1))
        parse.assert_not_called()

    def test_exhausted_completion_retries_raise_runtime_error(
        self, parse: MagicMock
    ) -> None:
        """Repeated unparseable output ends in RuntimeError, not an answer."""
        parse.side_effect = [ValueError("bad output")] * module.COMPLETION_RETRIES
        with pytest.raises(RuntimeError, match="after retries"):
            self._run(_FakeKeyChain(retries=1))

    def test_rate_limit_rotates_key_and_returns_retry_result(
        self, parse: MagicMock
    ) -> None:
        """After same-key retries hit rate limits, the key rotates and the retry answers."""
        rate_limit = _openai_error(openai.RateLimitError, HTTP_TOO_MANY_REQUESTS)
        parse.side_effect = [rate_limit] * module.COMPLETION_RETRIES + [
            _mock_parse_response()
        ]
        api_keys = _FakeKeyChain(retries=1)

        result = self._run(api_keys)

        assert api_keys.rotated == ["openai", "openrouter"]
        assert json.loads(result[0])["p_yes"] == FAKE_PREDICTION.p_yes
        assert result[-1] is api_keys

    def test_rate_limit_propagates_when_no_retries_left(self, parse: MagicMock) -> None:
        """With the retry budget spent, the rate limit propagates."""
        rate_limit = _openai_error(openai.RateLimitError, HTTP_TOO_MANY_REQUESTS)
        parse.side_effect = [rate_limit] * module.COMPLETION_RETRIES
        api_keys = _FakeKeyChain(retries=0)
        with pytest.raises(openai.RateLimitError):
            self._run(api_keys)
        assert api_keys.rotated == []

    def test_delivery_rate_zero_runs_normally(self, parse: MagicMock) -> None:
        """A delivery_rate of 0 is ignored: no max_cost call, a normal answer."""
        parse.return_value = _mock_parse_response()
        counter_callback = MagicMock()

        result = self._run(
            _FakeKeyChain(retries=1),
            delivery_rate=0,
            counter_callback=counter_callback,
        )

        assert isinstance(result, tuple)
        assert json.loads(result[0])["p_no"] == FAKE_PREDICTION.p_no
        assert all("max_cost" not in c.kwargs for c in counter_callback.call_args_list)
        counter_callback.assert_called_once()


class TestClientFaultClassification:
    """Only faults in the request raise InvalidRequestError."""

    def test_invalid_request_error_matches_the_mech_contract(self) -> None:
        """The mech recognises a client-side error by this exact class name."""
        assert module.InvalidRequestError.__name__ == "InvalidRequestError"
        assert issubclass(module.InvalidRequestError, ValueError)

    @pytest.mark.parametrize("tool", ["other_tool", "", None])
    def test_unsupported_tool_is_an_invalid_request(self, tool: object) -> None:
        """An unsupported tool is a fault in the request."""
        with pytest.raises(module.InvalidRequestError, match="is not supported"):
            run(
                tool=tool,
                model=MODEL,
                prompt=PREDICTION_PROMPT,
                api_keys=_FakeKeyChain(retries=0),
            )

    def test_missing_model_is_a_mech_side_error(self) -> None:
        """No model means no operator default, a failure on the mech's side."""
        with pytest.raises(ValueError, match="Model not supplied") as err:
            run(
                tool="superforcaster",
                model=None,
                prompt=PREDICTION_PROMPT,
                api_keys=_FakeKeyChain(retries=0),
            )
        assert type(err.value).__name__ != "InvalidRequestError"
