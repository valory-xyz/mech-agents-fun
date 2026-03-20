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

"""Tool tests running in isolated venvs matching production component.yaml dependencies.

Each test class specifies a component.yaml path. The test creates (or reuses) a
virtual environment with exactly the dependencies declared in the component.yaml,
then runs the tool inside that environment as a subprocess. This ensures tests
exercise the same dependency versions as production.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.conftest import run_tool_in_isolated_venv
from tests.shared_constants import (
    DEFAULT_CALLABLE,
    DELIVER_MSG_PREVIEW_LENGTH,
    RESULT_KEY_DELIVER_MSG,
    RESULT_KEY_ERRORS,
    RESULT_KEY_MODEL,
    RESULT_KEY_RESULTS,
    RESULT_KEY_SUCCESS,
    RESULT_KEY_TOOL,
)

PACKAGES_DIR = Path(__file__).parent.parent / "packages"
COMPONENT_YAML_FILENAME = "component.yaml"

# Test prompts
IMAGE_GEN_PROMPT = "Generate an image of a futuristic cityscape at sunset."
VIDEO_GEN_PROMPT = "Create a short video of ocean waves crashing on a rocky coast."


def _component_config(relative_path: str) -> str:
    """Build the full path to a component.yaml from a relative package path."""
    return str(PACKAGES_DIR / relative_path / COMPONENT_YAML_FILENAME)


def _module_path_from_config(component_yaml: str) -> str:
    """Derive the Python module path from a component.yaml's entry_point field.

    Reads entry_point from the yaml (e.g. 'stabilityai_request.py') and
    combines it with the package path to produce the full import path.
    """
    import yaml

    component_dir = Path(component_yaml).parent
    with open(component_yaml) as f:
        data = yaml.safe_load(f)
    entry_point = data["entry_point"]
    module_name = entry_point.removesuffix(".py")
    packages_idx = component_dir.parts.index("packages")
    package_parts = component_dir.parts[packages_idx:]
    return ".".join((*package_parts, module_name))


# Component configs
GOOGLE_IMAGE_GEN_CONFIG = _component_config("agents_fun/customs/google_image_gen")
GOOGLE_VIDEO_GEN_CONFIG = _component_config("agents_fun/customs/google_video_gen")
RECRAFT_IMAGE_GEN_CONFIG = _component_config("agents_fun/customs/recraft_image_gen")
SHORT_MAKER_CONFIG = _component_config("agents_fun/customs/short_maker")
STABILITY_AI_REQUEST_CONFIG = _component_config("agents_fun/customs/stability_ai_request")


def _format_failure(failure: Dict[str, Any]) -> str:
    """Format a single test failure into a readable string."""
    deliver_msg = failure.get(RESULT_KEY_DELIVER_MSG, "")[:DELIVER_MSG_PREVIEW_LENGTH]
    errors = "; ".join(failure[RESULT_KEY_ERRORS])
    return (
        f"  model={failure[RESULT_KEY_MODEL]}, tool={failure[RESULT_KEY_TOOL]}:\n"
        f"    errors: {errors}\n"
        f"    deliver_msg: {deliver_msg}"
    )


def _assert_response_fields(results: List[Dict[str, Any]], required_fields: List[str]) -> None:
    """Assert that required fields appear in each result's deliver_msg."""
    if not required_fields:
        return
    for r in results:
        deliver_msg = r.get(RESULT_KEY_DELIVER_MSG, "")
        missing = [f for f in required_fields if f not in deliver_msg]
        if not missing:
            continue
        pytest.fail(
            f"model={r[RESULT_KEY_MODEL]}, tool={r[RESULT_KEY_TOOL]}: "
            f"missing required fields {missing} in deliver_msg: "
            f"{deliver_msg[:DELIVER_MSG_PREVIEW_LENGTH]}"
        )


def _assert_all_passed(results: List[Dict[str, Any]]) -> None:
    """Assert all tool invocation results passed, with detailed failure messages."""
    assert results, "No test results returned from isolated runner."
    failures = [r for r in results if not r[RESULT_KEY_SUCCESS]]
    if not failures:
        return
    details = "\n".join(_format_failure(f) for f in failures)
    pytest.fail(f"{len(failures)}/{len(results)} tool invocations failed:\n{details}")


class BaseIsolatedToolTest:
    """Base class for tool tests that run in isolated component.yaml venvs."""

    component_yaml: str
    prompts: list
    callable_name: str = DEFAULT_CALLABLE
    required_response_fields: List[str] = []
    required_env_vars: List[str] = []

    def test_run(self) -> None:
        """Run the tool in an isolated venv and validate results."""
        output = run_tool_in_isolated_venv(
            component_yaml=self.component_yaml,
            module_path=_module_path_from_config(self.component_yaml),
            prompts=self.prompts,
            callable_name=self.callable_name,
            required_env_vars=self.required_env_vars or None,
        )
        results = output[RESULT_KEY_RESULTS]
        _assert_all_passed(results)
        _assert_response_fields(results, self.required_response_fields)


class TestGoogleImageGen(BaseIsolatedToolTest):
    """Test Google Image Generation."""

    component_yaml = GOOGLE_IMAGE_GEN_CONFIG
    prompts = [IMAGE_GEN_PROMPT]
    required_response_fields = ["image_hash", "prompt", "model"]
    required_env_vars = ["GEMINI_API_KEY"]


@pytest.mark.xfail(reason="Video generation is not implemented yet — tool only generates audio")
class TestGoogleVideoGen(BaseIsolatedToolTest):
    """Test Google Video Generation."""

    component_yaml = GOOGLE_VIDEO_GEN_CONFIG
    prompts = [VIDEO_GEN_PROMPT]
    required_response_fields = ["video_hash"]


@pytest.mark.skip(reason="Tool not used yet and no API keys obtained for testing.")
class TestRecraftImageGen(BaseIsolatedToolTest):
    """Test Recraft Image Generation."""

    component_yaml = RECRAFT_IMAGE_GEN_CONFIG
    prompts = [IMAGE_GEN_PROMPT]


@pytest.mark.xfail(reason="replicate==0.28.0 uses pydantic v1 which is incompatible with Python 3.14")
class TestShortMaker(BaseIsolatedToolTest):
    """Test Short Maker."""

    component_yaml = SHORT_MAKER_CONFIG
    prompts = [VIDEO_GEN_PROMPT]


class TestStabilityAIRequest(BaseIsolatedToolTest):
    """Test Stability AI Request."""

    component_yaml = STABILITY_AI_REQUEST_CONFIG
    prompts = [IMAGE_GEN_PROMPT]
    required_response_fields = ["artifacts"]
    required_env_vars = ["STABILITY_API_KEY"]
