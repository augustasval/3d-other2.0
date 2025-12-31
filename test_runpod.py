#!/usr/bin/env python3
"""
Test RunPod endpoint from terminal.

Usage:
    1. Fill in .env file with RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY
    2. Run: python test_runpod.py --image test.jpeg
"""

import argparse
import base64
import requests
import time
import json
import os

def load_env():
    """Load .env file from script directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env")

    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    load_env()

    parser = argparse.ArgumentParser(description="Test RunPod 3D generation endpoint")
    parser.add_argument("--image", default="test.jpeg", help="Path to input image")
    parser.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID", ""), help="RunPod endpoint ID")
    parser.add_argument("--key", default=os.environ.get("RUNPOD_API_KEY", ""), help="RunPod API key")
    parser.add_argument("--output", default="output.glb", help="Output GLB file path")
    args = parser.parse_args()

    if not args.endpoint or not args.key:
        print("Error: Missing endpoint ID or API key.")
        print("Fill in .env file or pass --endpoint and --key arguments.")
        return

    # Encode image
    print(f"Encoding image: {args.image}")
    image_base64 = encode_image(args.image)
    print(f"Image size: {len(image_base64):,} characters")

    # Submit job
    url = f"https://api.runpod.ai/v2/{args.endpoint}/run"
    headers = {
        "Authorization": f"Bearer {args.key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "image": image_base64,
            "remove_background": True,
            "generate_texture": True
        }
    }

    print(f"\nSubmitting job to RunPod...")
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    if "id" not in result:
        print(f"Error: {result}")
        return

    job_id = result["id"]
    print(f"Job ID: {job_id}")

    # Poll for result
    status_url = f"https://api.runpod.ai/v2/{args.endpoint}/status/{job_id}"

    while True:
        print("Checking status...", end=" ")
        response = requests.get(status_url, headers=headers)
        result = response.json()
        status = result.get("status")
        print(status)

        if status == "COMPLETED":
            output = result.get("output", {})
            if output.get("success"):
                # Save GLB
                model_base64 = output.get("model_base64")
                if model_base64:
                    with open(args.output, "wb") as f:
                        f.write(base64.b64decode(model_base64))
                    print(f"\nSuccess! Saved to: {args.output}")
                    print(f"File size: {output.get('file_size', 'N/A')} bytes")
                    print(f"Timings: {json.dumps(output.get('timings', {}), indent=2)}")
                else:
                    print(f"\nNo model in output: {output}")
            else:
                print(f"\nGeneration failed: {output.get('error')}")
            break
        elif status == "FAILED":
            print(f"\nJob failed: {result}")
            break

        time.sleep(2)

if __name__ == "__main__":
    main()
