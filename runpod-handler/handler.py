"""
RunPod Serverless Handler - Hunyuan3D-2 Shape Generation
Stage 1: Geometry only (no textures)
"""

import runpod
import base64
import time
import io
import os
import gc
import logging
import tempfile
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_pipeline = None


def clear_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def decode_base64_image(image_base64: str) -> Image.Image:
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    image_data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_data))


def encode_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading Hunyuan3D-2 shape generation pipeline...")

    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2'
    )

    logger.info("Pipeline loaded successfully")
    return _pipeline


def generate_3d(image: Image.Image, remove_bg: bool = True) -> dict:
    import torch
    timings = {}

    # Background removal (using Hunyuan3D-2's built-in rembg)
    start = time.time()
    if remove_bg:
        from hy3dgen.rembg import BackgroundRemover
        logger.info("Removing background...")
        rembg = BackgroundRemover()
        image = rembg(image)
    timings["background_removal"] = time.time() - start

    # Load pipeline
    start = time.time()
    pipeline = load_pipeline()
    timings["model_loading"] = time.time() - start

    # Generate mesh
    start = time.time()
    logger.info("Generating 3D mesh...")

    with torch.inference_mode():
        mesh = pipeline(image=image)[0]

    timings["generation"] = time.time() - start
    clear_gpu_memory()

    # Export to GLB
    start = time.time()
    glb_path = tempfile.mktemp(suffix=".glb")
    mesh.export(glb_path)
    timings["export"] = time.time() - start

    # Read and encode
    model_base64 = encode_file_to_base64(glb_path)
    file_size = os.path.getsize(glb_path)
    os.remove(glb_path)

    logger.info(f"Generated GLB: {file_size} bytes")

    return {
        "success": True,
        "model_base64": model_base64,
        "file_size": file_size,
        "format": "glb",
        "textured": False,  # Stage 1: no textures
        "timings": timings,
        "total_time": sum(timings.values()),
    }


def handler(job: dict) -> dict:
    try:
        job_input = job.get("input", {})

        image_base64 = job_input.get("image")
        if not image_base64:
            return {"error": "No image provided", "success": False}

        remove_bg = job_input.get("remove_background", True)

        logger.info("Decoding input image...")
        image = decode_base64_image(image_base64)
        logger.info(f"Input: {image.size}, mode: {image.mode}")

        start_time = time.time()
        result = generate_3d(image=image, remove_bg=remove_bg)
        result["execution_time"] = time.time() - start_time

        return result

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.exception(f"Handler error: {error_msg}")
        return {"error": error_msg, "success": False}


runpod.serverless.start({"handler": handler})
