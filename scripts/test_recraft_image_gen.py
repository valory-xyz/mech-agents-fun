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
"""Test the recraft_image_gen tool."""
import os
import json
from typing import Dict, Optional

from dotenv import load_dotenv

# Import the run function from your tool
from packages.agents_fun.customs.recraft_image_gen.recraft_image_gen import run


# Minimal KeyChain class to mimic the expected structure
class KeyChain:
    def __init__(self, api_keys: Dict[str, Optional[str]]):
        self._keys = api_keys
        self._max_retries = {k: 1 for k in api_keys}  # Simple retry count

    def get(self, key_name: str) -> Optional[str]:
        return self._keys.get(key_name)

    def max_retries(self) -> Dict[str, int]:
        # Return a copy to prevent modification
        return self._max_retries.copy()

    def rotate(self, service: str):
        # Placeholder for rotation logic if needed for testing complex scenarios
        print(f"[KeyChain] Rotating key for {service} (placeholder)")
        # In a real scenario, this would fetch a new key
        pass


if __name__ == "__main__":
    load_dotenv()

    # Get API key from environment variable
    recraft_api_key = os.getenv("RECRAFT_API_KEY")
    print(f"recraft_api_key: {recraft_api_key}")
    if not recraft_api_key:
        print("Error: RECRAFT_API_KEY environment variable not set.")
        exit(1)

    # Create the KeyChain object
    api_keys = KeyChain({"recraft_api_key": recraft_api_key})

    # Sample input parameters
    kwargs = {
        "prompt": "popeye in spain",
        "api_keys": api_keys,
        "tool": "recraft-image-gen",  # Make sure this matches ALLOWED_TOOLS
        "counter_callback": None,  # Optional callback if your tool uses it
    }

    print(f"Running recraft_image_gen with prompt: {kwargs['prompt']}")

    # Run the recraft image gen tool
    # The decorated run function returns 5 elements
    result_str, input_prompt, metadata, callback, _api_keys_obj = run(**kwargs)

    print("\n--- Results ---")
    print(f"Input prompt: {input_prompt}")
    print(f"Metadata: {metadata}")
    print(f"Callback Data: {callback}")
    print(f"Result String: {result_str}")

    # Try parsing the result string if it's JSON
    try:
        result_data = json.loads(result_str)
        print("\nParsed Result Data:")
        print(f"  Image Hash: {result_data.get('image_hash')}")
    except json.JSONDecodeError:
        print("\nResult is not valid JSON.")
    except Exception as e:
        print(f"\nCould not parse result string: {e}")
