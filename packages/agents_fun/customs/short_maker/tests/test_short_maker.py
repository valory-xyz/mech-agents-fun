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
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

import packages.agents_fun.customs.short_maker.short_maker as module
from packages.agents_fun.customs.short_maker.short_maker import (
    IMAGE_INPUT_USD_PER_M,
    IMAGE_MODEL,
    IMAGE_OUTPUT_USD_PER_M,
    MINIMAX_TTS_USD_PER_1K_CHARS,
    MUSICGEN_COST_PER_RUN_USD,
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
    """gpt-3.5 is retired; the replacement's token usage goes to the mech as tokens."""
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
    """DALL-E is retired; the image model returns base64 and token usage."""
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
        == "gpt-image-1-mini"
    )
    expected = (10 * IMAGE_INPUT_USD_PER_M + 272 * IMAGE_OUTPUT_USD_PER_M) / 1e6
    assert recorder.charges() == pytest.approx([(IMAGE_MODEL, expected)])


def test_a_failed_image_is_skipped_without_a_charge(tmp_path: Any) -> None:
    """An image call that raises produces no file and no charge."""
    recorder = _CostRecorder()
    client = MagicMock()
    client.images.generate.side_effect = RuntimeError("quota")
    with patch.object(module, "client", client):
        paths = generate_images(
            {"shot1_prompt": "x"}, str(tmp_path), counter_callback=recorder
        )
    assert not paths
    assert recorder.charges() == []


def test_first_shots_animate_the_file_with_wan_and_charge_per_video(
    tmp_path: Any,
) -> None:
    """Stable Video Diffusion is gone; Wan 2.2 takes the image file and a prompt."""
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


def test_voiceover_is_charged_per_character_and_music_per_run() -> None:
    """The two Replicate audio models report at their list prices."""
    recorder = _CostRecorder()
    replicate = MagicMock()
    replicate.run.side_effect = ["https://r/voice.mp3", "https://r/music.mp3"]
    text = "Hello there, this is a voiceover."

    assert (
        process_voiceover(replicate, text, "English_CalmWoman", recorder)
        == "https://r/voice.mp3"
    )
    assert process_soundtrack(replicate, "calm", 5, recorder) == "https://r/music.mp3"

    assert recorder.charges() == pytest.approx(
        [
            (REPLICATE_MINIMAX_TTS, len(text) / 1000 * MINIMAX_TTS_USD_PER_1K_CHARS),
            (REPLICATE_MUSIC_GEN, MUSICGEN_COST_PER_RUN_USD),
        ]
    )
