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

"""Tests for the short_maker tool's paid calls: models, inputs and cost reporting."""

import base64
import json
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

import packages.agents_fun.customs.short_maker.short_maker as module
from packages.agents_fun.customs.short_maker.short_maker import (
    A100_USD_PER_SECOND,
    IMAGE_INPUT_USD_PER_M,
    IMAGE_MODEL,
    IMAGE_OUTPUT_USD_PER_M,
    MINIMAX_TTS_USD_PER_1K_TOKENS,
    REPLICATE_MINIMAX_TTS,
    REPLICATE_MUSIC_GEN,
    REPLICATE_WAN_I2V,
    TEXT_MODEL,
    WAN_COST_PER_VIDEO_USD,
    generate_images,
    get_audio_prompts,
    get_shot_prompts,
    process_first_shots,
    process_soundtrack,
    process_voiceover,
)


class _CostRecorder:
    """Stands in for the mech's cost callback and keeps every report."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.actual_model: Any = None

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


AUDIO_JSON = '{"voiceover_script": "Hello there.", "soundtrack_prompt": "calm", "voice": "English_CalmWoman"}'
SHOTS_JSON = '{"shot1_prompt": "a rocket", "shot2_prompt": "the moon"}'
CHAT_TOKENS = (120, 40)


def _chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=CHAT_TOKENS[0], completion_tokens=CHAT_TOKENS[1]
        ),
    )


def _with_chat(content: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(content)
    return client


def test_audio_prompts_use_the_current_text_model_and_report_tokens() -> None:
    """The chat model's token usage goes to the mech as tokens, which it prices itself."""
    recorder = _CostRecorder()
    with patch.object(module, "client", _with_chat(AUDIO_JSON)) as client:
        prompts = get_audio_prompts("space", counter_callback=recorder)
    assert prompts["voice"] == "English_CalmWoman"
    assert (
        client.chat.completions.create.call_args.kwargs["model"]
        == TEXT_MODEL
        == "gpt-6-luna"
    )
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert (call["model"], call["input_tokens"], call["output_tokens"]) == (
        TEXT_MODEL,
        CHAT_TOKENS[0],
        CHAT_TOKENS[1],
    )
    assert "call_cost" not in call


def test_shot_prompts_report_tokens_too() -> None:
    """Both chat calls of a run are metered."""
    recorder = _CostRecorder()
    with patch.object(module, "client", _with_chat(SHOTS_JSON)):
        shots = get_shot_prompts(
            "space", voiceover_length=5.0, counter_callback=recorder
        )
    assert shots == {"shot1_prompt": "a rocket", "shot2_prompt": "the moon"}
    assert [c["model"] for c in recorder.calls] == [TEXT_MODEL]


def test_images_are_saved_as_files_and_charged_from_usage(tmp_path: Any) -> None:
    """The image model returns base64 and token usage; the usage is the charge."""
    recorder = _CostRecorder()
    client = MagicMock()
    client.images.generate.return_value = SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(b"png-bytes").decode())],
        usage=SimpleNamespace(input_tokens=10, output_tokens=272),
    )
    with patch.object(module, "client", client):
        paths = generate_images(
            {"shot1_prompt": "a rocket"}, str(tmp_path), counter_callback=recorder
        )
    assert list(paths) == ["shot1_prompt"]
    assert open(paths["shot1_prompt"], "rb").read() == b"png-bytes"
    assert (
        client.images.generate.call_args.kwargs["model"]
        == IMAGE_MODEL
        == "gpt-image-2.5-flare"
    )
    expected = (10 * IMAGE_INPUT_USD_PER_M + 272 * IMAGE_OUTPUT_USD_PER_M) / 1e6
    assert recorder.charges() == pytest.approx([(IMAGE_MODEL, expected)])


def _image_response(data: bytes, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(data).decode())],
        usage=SimpleNamespace(input_tokens=10, output_tokens=output_tokens),
    )


def _openai_error(error_cls: type, status_code: int) -> Exception:
    """An OpenAI API status error for a response with the given status."""
    response = MagicMock(status_code=status_code)
    return error_cls(f"status {status_code}", response=response, body=None)


def test_a_content_rejected_image_is_skipped_without_a_charge(tmp_path: Any) -> None:
    """A shot the image model rejects is left out and costs nothing; the rest are kept."""
    recorder = _CostRecorder()
    client = MagicMock()
    client.images.generate.side_effect = [
        _openai_error(module.openai.BadRequestError, 400),
        _image_response(b"moon", 272),
    ]
    with patch.object(module, "client", client):
        paths = generate_images(
            {"shot1_prompt": "x", "shot2_prompt": "y"},
            str(tmp_path),
            counter_callback=recorder,
        )
    assert list(paths) == ["shot2_prompt"]
    assert [model for model, _ in recorder.charges()] == [IMAGE_MODEL]


def test_no_image_at_all_raises_without_a_charge(tmp_path: Any) -> None:
    """When every shot fails there is nothing to animate, so the run fails."""
    recorder = _CostRecorder()
    client = MagicMock()
    client.images.generate.side_effect = _openai_error(
        module.openai.BadRequestError, 400
    )
    with patch.object(module, "client", client):
        with pytest.raises(ValueError, match="No shot image"):
            generate_images(
                {"shot1_prompt": "x"}, str(tmp_path), counter_callback=recorder
            )
    assert recorder.charges() == []


@pytest.mark.parametrize(
    ("error_cls", "status_code"),
    [
        (module.openai.AuthenticationError, 401),
        (module.openai.RateLimitError, 429),
        (module.openai.InternalServerError, 500),
    ],
)
def test_an_api_error_on_a_shot_stops_the_image_step(
    tmp_path: Any, error_cls: type, status_code: int
) -> None:
    """Errors that are not about the shot itself leave the loop at once, uncharged."""
    recorder = _CostRecorder()
    client = MagicMock()
    client.images.generate.side_effect = [
        _openai_error(error_cls, status_code),
        _image_response(b"moon", 272),
    ]
    with patch.object(module, "client", client):
        with pytest.raises(error_cls):
            generate_images(
                {"shot1_prompt": "x", "shot2_prompt": "y"},
                str(tmp_path),
                counter_callback=recorder,
            )
    assert client.images.generate.call_count == 1
    assert recorder.charges() == []


def test_first_shots_animate_the_file_with_wan_and_charge_per_video(
    tmp_path: Any,
) -> None:
    """Wan 2.2 takes the image file and a prompt and is charged per clip."""
    recorder = _CostRecorder()
    image = tmp_path / "shot1_prompt.png"
    image.write_bytes(b"png")
    replicate = MagicMock()
    replicate.run.return_value = "https://replicate.delivery/clip.mp4"

    url = process_first_shots(replicate, str(image), "a rocket", recorder)

    assert url == "https://replicate.delivery/clip.mp4"
    args, kwargs = replicate.run.call_args
    assert args[0] == REPLICATE_WAN_I2V == "wan-video/wan-2.2-i2v-fast"
    assert kwargs["input"]["prompt"] == "a rocket"
    assert kwargs["input"]["resolution"] == "480p"
    assert recorder.charges() == [(REPLICATE_WAN_I2V, WAN_COST_PER_VIDEO_USD)]


def test_a_missing_shot_image_is_skipped_without_calling_replicate(
    tmp_path: Any,
) -> None:
    """No file, no clip, no charge."""
    recorder = _CostRecorder()
    replicate = MagicMock()
    assert (
        process_first_shots(replicate, str(tmp_path / "none.png"), "x", recorder)
        is None
    )
    replicate.run.assert_not_called()
    assert recorder.charges() == []


def test_voiceover_is_charged_per_token_and_music_by_compute_time() -> None:
    """The voice is billed per input token (one per character), the music by the A100 time it ran."""
    recorder = _CostRecorder()
    replicate = MagicMock()
    replicate.run.return_value = "https://r/voice.mp3"
    prediction = SimpleNamespace(
        status="succeeded",
        error=None,
        output="https://r/music.mp3",
        metrics={"predict_time": 12.5},
        wait=lambda: None,
    )
    replicate.predictions.create.return_value = prediction
    text = "Hello there, this is a voiceover."

    assert (
        process_voiceover(replicate, text, "English_CalmWoman", recorder)
        == "https://r/voice.mp3"
    )
    assert process_soundtrack(replicate, "calm", 5, recorder) == "https://r/music.mp3"

    version = replicate.predictions.create.call_args.kwargs["version"]
    assert version == REPLICATE_MUSIC_GEN.split(":", 1)[1]
    assert recorder.charges() == pytest.approx(
        [
            (REPLICATE_MINIMAX_TTS, len(text) / 1000 * MINIMAX_TTS_USD_PER_1K_TOKENS),
            (REPLICATE_MUSIC_GEN, 12.5 * A100_USD_PER_SECOND),
        ]
    )


def test_a_failed_soundtrack_raises_and_is_not_charged() -> None:
    """A prediction that did not succeed is an error, not a billed run."""
    recorder = _CostRecorder()
    replicate = MagicMock()
    replicate.predictions.create.return_value = SimpleNamespace(
        status="failed", error="oom", output=None, metrics={}, wait=lambda: None
    )
    with pytest.raises(RuntimeError, match="failed"):
        process_soundtrack(replicate, "calm", 5, recorder)
    assert recorder.charges() == []


def test_a_failed_run_raises_after_reporting_earlier_charges() -> None:
    """Charges reported before a later step failed have reached the mech when it raises."""
    recorder = _CostRecorder()
    with (
        patch.object(module, "client", _with_chat(AUDIO_JSON)),
        patch.object(module, "OpenAI", return_value=_with_chat(AUDIO_JSON)),
        patch.object(module, "ReplicateClient") as replicate_cls,
    ):
        replicate_cls.return_value.run.side_effect = RuntimeError("replicate down")
        with pytest.raises(RuntimeError, match="replicate down"):
            module.run(
                prompt="space",
                tool="short_maker",
                api_keys=_api_keys(openai="ok", replicate="rk"),
                counter_callback=recorder,
            )
    assert [c["model"] for c in recorder.calls] == [TEXT_MODEL]


@contextmanager
def _run_pipeline(tmp_path: Any, openai_client: Any) -> Iterator[None]:
    """Stub every external service of ``run`` except the OpenAI image client."""
    ipfs = MagicMock()
    ipfs.add.return_value = (None, "bafyvideo", None)
    ipfs.client.add.return_value = {"Hash": "bafyimage"}
    audio = {"voiceover_script": "Hi", "soundtrack_prompt": "calm", "voice": "v"}
    stubs = {
        "OpenAI": {"return_value": openai_client},
        "ReplicateClient": {},
        "get_audio_prompts": {"return_value": audio},
        "process_voiceover": {"return_value": "https://a/voice.mp3"},
        "download_file": {"side_effect": lambda _url, path: path},
        "get_audio_duration": {"return_value": 3.0},
        "get_shot_prompts": {"return_value": {"shot1": "a rocket"}},
        "process_first_shots": {"return_value": "https://a/shot.mp4"},
        "process_soundtrack": {"return_value": "https://a/music.mp3"},
        "compose_final_video": {"return_value": str(tmp_path / "f.mp4")},
        "IPFSTool": {"return_value": ipfs},
        "cleanup_tempdir": {},
    }
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(module.tempfile, "mkdtemp", return_value=str(tmp_path))
        )
        for name, kwargs in stubs.items():
            stack.enter_context(patch.object(module, name, **kwargs))
        yield


def test_a_rate_limited_shot_rotates_the_openai_key_and_retries(
    tmp_path: Any,
) -> None:
    """A rate limit on the image step reaches the key rotation and the run retries."""
    client = MagicMock()
    client.images.generate.side_effect = [
        _openai_error(module.openai.RateLimitError, 429),
        _image_response(b"rocket", 272),
    ]
    api_keys = _api_keys(openai="ok", replicate="rk")
    with _run_pipeline(tmp_path, client):
        result = module.run(prompt="space", tool="short_maker", api_keys=api_keys)
    api_keys.rotate.assert_called_once_with("openai")
    assert client.images.generate.call_count == 2
    assert json.loads(result[0])["video"] == "bafyvideo"


def test_a_successful_run_reports_the_script_model_as_the_one_used(
    tmp_path: Any,
) -> None:
    """The metadata model is the script model, not the last service the run paid."""
    recorder = _CostRecorder()
    recorder.actual_model = REPLICATE_MUSIC_GEN
    image = tmp_path / "shot1.png"
    image.write_bytes(b"png")
    with (
        _run_pipeline(tmp_path, MagicMock()),
        patch.object(module, "generate_images", return_value={"shot1": str(image)}),
    ):
        result = module.run(
            prompt="space",
            tool="short_maker",
            api_keys=_api_keys(openai="ok", replicate="rk"),
            counter_callback=recorder,
        )
    assert result[3] is recorder
    assert recorder.actual_model == TEXT_MODEL
