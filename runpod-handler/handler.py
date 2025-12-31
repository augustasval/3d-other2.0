"""
RunPod Serverless Handler for 2D-to-3D Generation

This is a simplified, reliable handler using:
- rembg for background removal (proven, fast)
- Hunyuan3D-2 for 3D generation (from your working setup)

The full multi-model pipeline (Era3D → GeoLRM → Hunyuan Paint)
can be added incrementally once this baseline is working.
"""

import runpod
import base64
import time
import io
import os
import gc
import logging
from PIL import Image
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_models = {}


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))

    if image.mode == "RGBA":
        pass
    elif image.mode != "RGB":
        image = image.convert("RGB")

    return image


def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background using rembg."""
    from rembg import remove

    logger.info("Removing background with rembg...")
    result = remove(image)
    logger.info("Background removed")
    return result


def load_triposr():
    """Load TripoSR model for 3D generation."""
    global _models

    if "triposr" not in _models:
        logger.info("Loading TripoSR model...")

        try:
            import torch
            from tsr.system import TSR

            model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            model.renderer.set_chunk_size(8192)
            model.to("cuda")

            _models["triposr"] = model
            logger.info("TripoSR loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load TripoSR: {e}")
            logger.info("Falling back to placeholder mode")
            _models["triposr"] = None

    return _models.get("triposr")


def generate_3d_model(
    image: Image.Image,
    generate_texture: bool = True,
    remove_bg: bool = True,
) -> dict:
    """
    Generate 3D model from image.

    Returns dict with model_base64, file_size, etc.
    """
    import tempfile
    import torch

    timings = {}

    # Step 1: Background removal
    start = time.time()
    if remove_bg:
        image = remove_background(image)
    timings["background_removal"] = time.time() - start
    clear_gpu_memory()

    # Step 2: Load model
    start = time.time()
    model = load_triposr()
    timings["model_loading"] = time.time() - start

    if model is None:
        logger.warning("Model not available, returning placeholder")
        return {
            "success": False,
            "error": "TripoSR model not loaded. Check logs for details.",
            "timings": timings,
        }

    # Step 3: Generate 3D
    start = time.time()
    logger.info("Generating 3D model with TripoSR...")

    try:
        # Preprocess image for TripoSR
        if image.mode == "RGBA":
            # TripoSR expects RGBA with transparent background
            processed_image = image
        else:
            processed_image = image.convert("RGB")

        with torch.inference_mode():
            # Run TripoSR
            scene_codes = model([processed_image], device="cuda")
            meshes = model.extract_mesh(scene_codes, resolution=256)
            mesh = meshes[0]

        timings["generation"] = time.time() - start
        clear_gpu_memory()

        # Step 4: Export to GLB
        start = time.time()
        logger.info("Exporting to GLB...")

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp_path = tmp.name

        # Export mesh
        mesh.export(tmp_path)

        timings["export"] = time.time() - start

        # Read and encode the GLB
        model_base64 = encode_file_to_base64(tmp_path)
        file_size = os.path.getsize(tmp_path)

        # Cleanup
        os.remove(tmp_path)

        logger.info(f"Generated GLB: {file_size} bytes")

        return {
            "success": True,
            "model_base64": model_base64,
            "file_size": file_size,
            "format": "glb",
            "textured": generate_texture,
            "timings": timings,
            "total_time": sum(timings.values()),
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "timings": timings,
        }


def handler(job: dict) -> dict:
    """
    RunPod serverless handler function.

    Input:
    {
        "input": {
            "image": "<base64-encoded-image>",
            "remove_background": true,
            "generate_texture": true
        }
    }

    Output:
    {
        "model_base64": "<base64-encoded-glb>",
        "file_size": 12345678,
        "format": "glb",
        "textured": true,
        "execution_time": 45.2
    }
    """
    try:
        job_input = job.get("input", {})

        # Validate input
        image_base64 = job_input.get("image")
        if not image_base64:
            return {"error": "No image provided", "success": False}

        # Parse options
        remove_bg = job_input.get("remove_background", True)
        generate_texture = job_input.get("generate_texture", True)

        # Decode image
        logger.info("Decoding input image...")
        image = decode_base64_image(image_base64)
        logger.info(f"Input image: {image.size}, mode: {image.mode}")

        # Generate
        start_time = time.time()
        result = generate_3d_model(
            image=image,
            generate_texture=generate_texture,
            remove_bg=remove_bg,
        )

        result["execution_time"] = time.time() - start_time

        if result.get("success"):
            logger.info(f"Success: {result['file_size']} bytes in {result['execution_time']:.1f}s")
        else:
            logger.error(f"Failed: {result.get('error')}")

        return result

    except Exception as e:
        logger.exception(f"Handler error: {e}")
        return {"error": str(e), "success": False}


# RunPod entrypoint
runpod.serverless.start({"handler": handler})
