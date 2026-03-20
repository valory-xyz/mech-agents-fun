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

"""Standalone runner for executing tool tests inside isolated virtual environments.

This script is invoked as a subprocess by the test suite. It runs inside a venv
whose dependencies match the tool's component.yaml, ensuring tests exercise
the same dependency versions as production.

Usage:
    <venv>/bin/python tests/isolated_runner.py '<json_config>' <results_file>
"""

import importlib
import json
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List

# Suppress known harmless warnings from pinned dependency versions
warnings.filterwarnings("ignore", message="urllib3.*or chardet.*doesn't match a supported version")
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

ENV_FILE_NAME = ".env"
ENV_COMMENT_PREFIX = "#"

# Expected response tuple length from tool `run()` calls (4 from run + 1 api_keys from decorator)
EXPECTED_RESPONSE_LENGTH = 5

# Known error patterns in deliver_msg that indicate tool failure
ERROR_PATTERNS = (
    "An error occurred:",
    "Google API error:",
    "Unexpected error:",
    "Error: Non-200 response",
    "api_keys object does not have required methods",
    "is not in the list of supported tools",
    "is not supported by this agent",
    "not provided",
    "was not found",
)

# Module attribute names for discovering tools and models
TOOLS_ATTR_NAMES = ("ALLOWED_TOOLS", "AVAILABLE_TOOLS")
ALLOWED_MODELS_ATTR = "ALLOWED_MODELS"

# Config keys
CONFIG_PROJECT_ROOT = "project_root"
CONFIG_MODULE_PATH = "module_path"
CONFIG_CALLABLE = "callable"
CONFIG_PROMPTS = "prompts"
CONFIG_REQUIRED_ENV_VARS = "required_env_vars"
CONFIG_SERVICE_TO_ENV_VAR = "service_to_env_var"

# Truncation limits
PROMPT_TRUNCATE_LENGTH = 100
DELIVER_MSG_TRUNCATE_LENGTH = 1000
ERROR_MSG_TRUNCATE_LENGTH = 200



def _load_env_file(project_root: str) -> None:
    """Load a .env file into os.environ without requiring python-dotenv.

    Only sets variables not already present, so CI env vars take precedence.
    """
    env_path = Path(project_root) / ENV_FILE_NAME
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        _apply_env_line(line.strip())


def _apply_env_line(line: str) -> None:
    """Parse and apply a single .env line to os.environ."""
    if not line or line.startswith(ENV_COMMENT_PREFIX) or "=" not in line:
        return
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip().strip("\"'")
    if not key or key in os.environ:
        return
    os.environ[key] = value


class KeyChain(dict):
    """Dict subclass with max_retries/rotate methods required by with_key_rotation."""

    def max_retries(self) -> Dict[str, int]:
        """Return 1 retry per service (single key per service in tests)."""
        return {service: 1 for service in self}

    def rotate(self, service_name: str) -> None:
        """No-op — tests use a single key per service."""


def build_api_keys(service_to_env_var: Dict[str, str]) -> KeyChain:
    """Build a KeyChain from environment variables."""
    return KeyChain({
        service: os.environ.get(env_var, "")
        for service, env_var in service_to_env_var.items()
    })


def validate_response(response: Any) -> List[str]:
    """Validate a tool response. Returns a list of error strings (empty = pass)."""
    if not isinstance(response, tuple):
        return ["Response of the tool must be a tuple."]
    if len(response) != EXPECTED_RESPONSE_LENGTH:
        return [f"Response must have {EXPECTED_RESPONSE_LENGTH} elements, got {len(response)}."]

    errors: List[str] = []
    deliver_msg = response[0]
    if not isinstance(deliver_msg, str):
        errors.append("Response[0] must be a string.")
    else:
        for pattern in ERROR_PATTERNS:
            if pattern in deliver_msg:
                errors.append(f"Tool error: {deliver_msg[:ERROR_MSG_TRUNCATE_LENGTH]}")
                break

    if not (isinstance(response[1], (str, type(None)))):
        errors.append("Response[1] must be a string or None.")
    if not (isinstance(response[2], dict) or response[2] is None):
        errors.append("Response[2] must be a dictionary or None.")

    return errors


def _check_required_env_vars(required_env_vars: List[str]) -> List[str]:
    """Return list of missing env var names (empty if all present)."""
    return [var for var in required_env_vars if not os.environ.get(var)]


def _make_error_result(errors: List[str]) -> Dict[str, Any]:
    """Build a single failure result dict."""
    return {
        "model": None,
        "tool": None,
        "prompt": None,
        "success": False,
        "errors": errors,
    }


def _get_tools_from_module(module: Any) -> List[str]:
    """Discover available tools from a module, checking multiple attribute names."""
    for attr in TOOLS_ATTR_NAMES:
        value = getattr(module, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            return list(value.keys())
        return list(value)
    return []


def _run_single_invocation(
    func: Any,
    prompt: str,
    tool: str,
    keys: KeyChain,
    model: Any,
) -> Dict[str, Any]:
    """Run a single tool invocation and return the result dict."""
    result: Dict[str, Any] = {
        "model": model,
        "tool": tool,
        "prompt": prompt[:PROMPT_TRUNCATE_LENGTH],
        "success": False,
        "errors": [],
    }
    try:
        response = func(
            prompt=prompt,
            tool=tool,
            api_keys=keys,
            counter_callback=None,
            model=model,
        )
        errs = validate_response(response)
        result["success"] = len(errs) == 0
        result["errors"] = errs
        if isinstance(response, tuple) and len(response) > 0:
            result["deliver_msg"] = str(response[0])[:DELIVER_MSG_TRUNCATE_LENGTH]
    except Exception as e:
        result["errors"] = [f"{type(e).__name__}: {e}"]
    return result


def _missing_env_vars_result(missing_vars: List[str]) -> Dict[str, Any]:
    """Build the output dict for missing env vars."""
    return {
        "results": [
            _make_error_result([
                f"Missing required environment variables: {', '.join(sorted(missing_vars))}. "
                "Either export them in your shell or set them in a .env file "
                "in the project root."
            ])
        ]
    }


def _execute_all_combinations(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run all model x tool x prompt combinations and return results."""
    module = importlib.import_module(config[CONFIG_MODULE_PATH])

    keys = build_api_keys(config[CONFIG_SERVICE_TO_ENV_VAR])
    tools = _get_tools_from_module(module)
    models = getattr(module, ALLOWED_MODELS_ATTR, [None])
    func = getattr(module, config[CONFIG_CALLABLE])
    prompts = config[CONFIG_PROMPTS]

    results = [
        _run_single_invocation(func, prompt, tool, keys, model)
        for model in models
        for tool in tools
        for prompt in prompts
    ]
    return {"results": results}


def run_tests(config: Dict[str, Any]) -> Dict[str, Any]:
    """Import the tool module, run all tool/model/prompt combinations, validate."""
    project_root = config[CONFIG_PROJECT_ROOT]
    _load_env_file(project_root)

    missing_vars = _check_required_env_vars(config.get(CONFIG_REQUIRED_ENV_VARS, []))
    if missing_vars:
        return _missing_env_vars_result(missing_vars)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        return _execute_all_combinations(config)
    except Exception as e:
        return {
            "results": [
                _make_error_result([
                    f"Setup error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                ])
            ]
        }


def main() -> None:
    """Entry point — read config from argv, write JSON results to file."""
    config = json.loads(sys.argv[1])
    results_file = sys.argv[2]
    output = run_tests(config)
    Path(results_file).write_text(json.dumps(output))
    all_passed = all(r["success"] for r in output["results"])
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
