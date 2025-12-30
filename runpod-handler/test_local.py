#!/usr/bin/env python3
"""
Local Testing Script for 2D-to-3D Handler

This script allows you to test the handler locally without deploying to RunPod.
It simulates the RunPod serverless environment.

Usage:
    # Test with a sample image
    python test_local.py --image path/to/image.png

    # Test with base64 input
    python test_local.py --base64 "..."

    # Run mock test (no GPU required)
    python test_local.py --mock

Requirements:
    - Python 3.10+
    - For full testing: NVIDIA GPU with 48GB+ VRAM
    - For mock testing: No GPU required
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64_to_file(base64_data: str, output_path: str):
    """Decode base64 data to a file."""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(base64_data))


def run_mock_test():
    """
    Run a mock test without requiring GPU or models.
    Useful for testing the handler structure and API.
    """
    print("\n" + "="*60)
    print("MOCK TEST MODE (No GPU required)")
    print("="*60 + "\n")

    # Create a mock job
    mock_job = {
        "input": {
            "image": "mock_base64_image_data",
            "remove_background": True,
            "texture_resolution": 2048,
            "mesh_detail": "high",
            "generate_pbr": True,
        }
    }

    print("Mock Job Input:")
    print(json.dumps(mock_job, indent=2))
    print()

    # Simulate pipeline stages
    stages = [
        ("Background Removal (SAM)", 2.5),
        ("Multi-View Generation (Era3D)", 18.0),
        ("3D Reconstruction (GeoLRM)", 12.0),
        ("PBR Texture Generation (Hunyuan3D-2.1)", 28.0),
        ("GLB Export", 2.0),
    ]

    total_time = 0
    for stage_name, duration in stages:
        print(f"[MOCK] {stage_name}...")
        time.sleep(0.5)  # Brief pause for visibility
        total_time += duration
        print(f"        Completed in {duration:.1f}s (simulated)")

    print()
    print("="*60)
    print("MOCK RESULT")
    print("="*60)

    mock_result = {
        "success": True,
        "model_base64": "<mock_glb_base64_data>",
        "file_size": 15_000_000,  # 15MB mock
        "format": "glb",
        "textured": True,
        "texture_resolution": 2048,
        "vertices": 150_000,
        "faces": 280_000,
        "timings": {
            "background_removal": 2.5,
            "multiview_generation": 18.0,
            "reconstruction": 12.0,
            "texture_generation": 28.0,
            "glb_export": 2.0,
        },
        "total_time": total_time,
        "execution_time": total_time,
    }

    print(json.dumps(mock_result, indent=2))
    print()
    print(f"Total simulated time: {total_time:.1f}s")
    print("\nMock test completed successfully!")
    return mock_result


def run_full_test(image_path: str, output_dir: str = "./test_output"):
    """
    Run a full test with actual models (requires GPU).
    """
    print("\n" + "="*60)
    print("FULL TEST MODE (GPU required)")
    print("="*60 + "\n")

    # Check if GPU is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Use --mock for testing without GPU.")
            sys.exit(1)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("ERROR: PyTorch not installed. Install dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Verify image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Encode image
    print(f"\nInput image: {image_path}")
    image_base64 = encode_image_to_base64(image_path)
    print(f"Base64 size: {len(image_base64):,} characters")

    # Create job
    job = {
        "input": {
            "image": image_base64,
            "remove_background": True,
            "texture_resolution": 2048,
            "mesh_detail": "high",
            "generate_pbr": True,
        }
    }

    # Import and run handler
    print("\nImporting handler...")
    from handler import handler

    print("Running pipeline...")
    start_time = time.time()
    result = handler(job)
    elapsed = time.time() - start_time

    print(f"\nPipeline completed in {elapsed:.1f}s")
    print()

    if result.get("success"):
        # Save output GLB
        output_path = os.path.join(output_dir, "output_model.glb")
        decode_base64_to_file(result["model_base64"], output_path)
        print(f"Output saved: {output_path}")
        print(f"File size: {result.get('file_size', 0):,} bytes")
        print(f"Vertices: {result.get('vertices', 'N/A'):,}")
        print(f"Faces: {result.get('faces', 'N/A'):,}")

        # Print timings
        if "timings" in result:
            print("\nStage timings:")
            for stage, duration in result["timings"].items():
                print(f"  {stage}: {duration:.2f}s")
    else:
        print(f"ERROR: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    return result


def run_http_server(port: int = 8000):
    """
    Run a simple HTTP server that mimics RunPod's API.
    Useful for testing the panel integration locally.
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json

    class RunPodMockHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if "/run" in self.path:
                # Read request body
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                job = json.loads(body)

                print(f"\n[SERVER] Received job request")
                print(f"[SERVER] Image size: {len(job.get('input', {}).get('image', '')):,} chars")

                # Return job ID immediately
                job_id = f"mock-job-{int(time.time())}"
                response = {"id": job_id, "status": "IN_PROGRESS"}

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            elif "/status" in self.path:
                # Simulate completion after a few polls
                response = {
                    "status": "COMPLETED",
                    "output": {
                        "success": True,
                        "model_base64": "bW9ja19nbGJfZGF0YQ==",  # "mock_glb_data"
                        "file_size": 1000,
                        "format": "glb",
                        "textured": True,
                        "vertices": 10000,
                        "faces": 20000,
                        "execution_time": 5.0,
                    }
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if "/health" in self.path:
                response = {"workers": {"ready": 1}}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            print(f"[SERVER] {args[0]}")

    print(f"\n" + "="*60)
    print(f"Starting mock RunPod server on http://localhost:{port}")
    print("="*60)
    print("\nUse this for testing the panel locally:")
    print(f"  Endpoint URL: http://localhost:{port}")
    print("  API Key: any_value")
    print("\nPress Ctrl+C to stop the server.\n")

    server = HTTPServer(("localhost", port), RunPodMockHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Local testing script for 2D-to-3D handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run mock test (no GPU required)
  python test_local.py --mock

  # Test with an image file (requires GPU)
  python test_local.py --image ./test_image.png

  # Start mock HTTP server for panel testing
  python test_local.py --server

  # Specify output directory
  python test_local.py --image ./test.png --output ./my_output
        """
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run mock test without GPU"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image for testing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./test_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start mock HTTP server for panel testing"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for mock HTTP server"
    )

    args = parser.parse_args()

    if args.server:
        run_http_server(args.port)
    elif args.mock:
        run_mock_test()
    elif args.image:
        run_full_test(args.image, args.output)
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Quick start:")
        print("  python test_local.py --mock    # Test without GPU")
        print("  python test_local.py --server  # Start mock server")
        print("="*60)


if __name__ == "__main__":
    main()
