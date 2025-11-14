# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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
"""Test the short_maker tool."""
import os
from packages.agents_fun.customs.short_maker.short_maker import run


# Define a wrapper class for API keys
class ApiKeysWrapper:
    def __init__(self, keys):
        self._keys = keys
        self._services = list(keys.keys())
        self._current_key_indices = {service: 0 for service in self._services}
        # Define a default max_retries, can be adjusted
        self._max_retries = {service: 3 for service in self._services}

    def get(self, service: str):
        # Return the key if present, otherwise None (consistent with dict.get)
        return self._keys.get(service)

    def __getitem__(self, service: str):
        # Allow dictionary-style access, raises KeyError if service is not found
        return self._keys[service]

    def max_retries(self) -> dict:
        return self._max_retries.copy()  # Return a copy

    def rotate(self, service: str) -> None:
        # This is a placeholder for rotation logic.
        # If actual rotation of keys is needed (e.g., multiple keys per service),
        # this method would need to be implemented more thoroughly.
        # For now, it just logs that rotation is called.
        print(f"Key rotation called for service: {service}")
        # A simple rotation might involve advancing an index if multiple keys are stored per service.
        # For example:
        # if service in self._keys and isinstance(self._keys[service], list) and len(self._keys[service]) > 1:
        #     self._current_key_indices[service] = (self._current_key_indices[service] + 1) % len(self._keys[service])
        #     print(f"Rotated key for {service} to index {self._current_key_indices[service]}")
        # else:
        #     print(f"No keys to rotate for service: {service} or only one key available.")
        pass  # Placeholder for actual rotation logic


# Test the short_maker tool
if __name__ == "__main__":
    # Get API keys from environment variables
    raw_api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "replicate": os.getenv("REPLICATE_API_KEY"),
        # Add other services expected by short_maker if any, e.g., "anthropic", "google_api_key"
        # For the purpose of this fix, we'll assume short_maker primarily uses openai and replicate for now.
        # The with_key_rotation decorator seems to handle 'anthropic', 'openai', 'openrouter', 'google_api_key'.
        # We should ensure these are present if the tool actually tries to use them.
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),  # Assuming this might be needed
        "openrouter": os.getenv("OPENROUTER_API_KEY"),  # Assuming this might be needed
        "google_api_key": os.getenv("GOOGLE_API_KEY"),  # Assuming this might be needed
    }

    # Filter out None keys to prevent issues if some env vars are not set
    # and the tool doesn't strictly require all of them for every path.
    filtered_api_keys = {k: v for k, v in raw_api_keys.items() if v is not None}

    # Wrap the filtered API keys
    api_keys_wrapper = ApiKeysWrapper(filtered_api_keys)

    # Sample input parameters
    kwargs = {
        "prompt": "cat playing with a ball",
        "api_keys": api_keys_wrapper,  # Use the wrapper object
    }

    # Run the short maker
    result, input_prompt, metadata, callback, _ = run(**kwargs)
    print(f"Input prompt: {input_prompt}")
    print(f"Result: {result}")
    print(f"Metadata: {metadata}")
