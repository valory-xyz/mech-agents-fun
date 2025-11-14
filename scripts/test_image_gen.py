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
"""Test the google_image_gen tool."""
import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv

# Import the run function from your tool
from packages.agents_fun.customs.google_image_gen.google_image_gen import run
# Import the robust KeyChain class
from packages.valory.skills.task_execution.utils.apis import KeyChain


# Test the google_image_gen tool
if __name__ == "__main__":

    load_dotenv()

    # Get API key from environment variable
    gemini_api_key_str = os.getenv("GEMINI_API_KEY")
    print(f"GEMINI_API_KEY from env: {gemini_api_key_str}")
    if not gemini_api_key_str:
        print("Error: GEMINI_API_KEY environment variable not set.")
        exit(1)

    # KeyChain expects a dictionary where values are lists of keys.
    # For a single key, it should be in a list.
    services_config = {
        "gemini_api_key": [gemini_api_key_str]
    }

    # Filter out services with no valid keys to prevent KeyChain initialization error
    # if an env var for a secondary service isn't set.
    valid_services_config = {
        s: k_list
        for s, k_list in services_config.items()
        if k_list and k_list[0] is not None
    }

    if not valid_services_config.get("gemini_api_key"):
        print(
            "Error: GEMINI_API_KEY was not properly configured for KeyChain (e.g., it's None even after os.getenv)."
        )
        exit(1)

    api_keys_instance = KeyChain(valid_services_config)

    # Sample input parameters
    kwargs = {
        "prompt": "Zoro in cyberpunk 2077 universe",
        "api_keys": api_keys_instance, 
        "tool": "google_image_gen",  
        "counter_callback": None,  
    }

    print(f"Running google_image_gen with prompt: {kwargs['prompt']}")

    # Run the google image gen tool
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
        print(f"  Model Used: {result_data.get('model')}")
    except json.JSONDecodeError:
        print("\nResult is not valid JSON.")
    except Exception as e:
        print(f"\nCould not parse result string: {e}")
