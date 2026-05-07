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

"""Smoke tests covering tool-tests support modules without touching live APIs."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "tests.shared_constants",
        "tests.venv_manager",
        "tests.isolated_runner",
    ],
)
def test_support_modules_import_cleanly(module: str) -> None:
    """The conftest support modules must import without side effects."""
    importlib.import_module(module)
