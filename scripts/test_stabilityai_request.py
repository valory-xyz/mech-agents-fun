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
"""Test the stabilityai_request tool."""
import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv

# Import the run function from your tool
from packages.agents_fun.customs.stability_ai_request.stabilityai_request import run


# Minimal KeyChain class to mimic the expected structure
class KeyChain:
    def __init__(self, api_keys: Dict[str, Optional[str]]):
        self._keys = api_keys
        self._max_retries = {k: 1 for k in api_keys}  # Simple retry count

    def get(self, key_name: str) -> Optional[str]:
        return self._keys.get(key_name)

    def __getitem__(self, key_name: str) -> Optional[str]:
        """Allow dictionary-style access to the API keys."""
        return self._keys[
            key_name
        ]  # Or self._keys.get(key_name) if None is preferred over KeyError for missing keys

    def max_retries(self) -> Dict[str, int]:
        # Return a copy to prevent modification
        return self._max_retries.copy()

    def rotate(self, service: str):
        # Placeholder for rotation logic if needed for testing complex scenarios
        print(f"[KeyChain] Rotating key for {service} (placeholder)")
        # In a real scenario, this would fetch a new key
        pass


# Test the stabilityai_request tool
if __name__ == "__main__":
    load_dotenv()

    # Get API key from environment variable
    stability_api_key = os.getenv("STABILITY_API_KEY")
    if not stability_api_key:
        print("Error: STABILITY_API_KEY environment variable not set.")
        print(
            "Please set it in your .env file (e.g., STABILITY_API_KEY='your_api_key_here')."
        )
        exit(1)

    print(
        f"Using Stability AI API Key: {'*' * (len(stability_api_key) - 4) + stability_api_key[-4:] if stability_api_key else 'Not Set'}"
    )

    # Create the KeyChain object
    api_keys = KeyChain({"stabilityai": stability_api_key})

    # Sample input parameters
    # Use a tool from ALLOWED_TOOLS in stabilityai_request.py
    # e.g., "stabilityai-stable-diffusion-xl-1024-v1-0" or "stabilityai-stable-diffusion-v1-6"
    kwargs = {
        "prompt": "A beautiful landscape painting in the style of Monet.",
        "api_keys": api_keys,
        "tool": "stabilityai-stable-diffusion-xl-1024-v1-0",
        # Optional parameters from stabilityai_request.py can be added here
        # "cfg_scale": 7,
        # "weight": 0.5,
        # "clip_guidance_preset": "FAST_BLUE",
        # "height": 1024, # For "stable-diffusion-xl-1024-v1-0"
        # "width": 1024,  # For "stable-diffusion-xl-1024-v1-0"
        # "samples": 1,
        # "steps": 30,
        # "style_preset": "enhance", # e.g., "enhance", "photographic", "digital-art"
    }

    print(
        f"Running stabilityai_request with tool: {kwargs['tool']} and prompt: \"{kwargs['prompt']}\""
    )

    # Run the stabilityai_request tool
    # The decorated run function returns 5 elements:
    # result_str, error_details, metadata, extra_data, api_keys_object
    result_str, error_details, metadata, extra_data, _api_keys_obj = run(**kwargs)

    print("\n--- Results ---")
    if error_details:
        print(f"Error Details: {error_details}")
    if metadata:
        print(f"Metadata: {metadata}")
    if extra_data:  # This might be None or have other data depending on execution path
        print(f"Extra Data: {extra_data}")

    print(f"Result String: {result_str}")

    # Try parsing the result string if it's JSON
    if result_str:
        try:
            result_data = json.loads(result_str)
            print("\nParsed Result Data:")
            if isinstance(result_data, dict) and "artifacts" in result_data:
                for i, artifact in enumerate(result_data["artifacts"]):
                    print(f"  Artifact {i+1}:")
                    print(f"    Base64 Length: {len(artifact.get('base64', ''))}")
                    print(f"    Finish Reason: {artifact.get('finishReason')}")
                    print(f"    Seed: {artifact.get('seed')}")
            elif isinstance(
                result_data, dict
            ):  # Handle other JSON structures, e.g. error messages
                print(json.dumps(result_data, indent=2))
            else:
                print(f"  Unexpected JSON structure: {result_data}")

        except json.JSONDecodeError:
            print(
                "\nResult is not valid JSON. This might indicate an error response or a non-JSON output."
            )
        except Exception as e:
            print(f"\nCould not process result string: {e}")
    else:
        print("\nResult string is empty.")

    print("\n--- Test Complete ---")
