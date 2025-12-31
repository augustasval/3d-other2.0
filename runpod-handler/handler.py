"""
RunPod Serverless Handler - Hunyuan3D-2
Stage 2: Shape + Texture Generation
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
_shape_pipeline = None
_paint_pipeline = None


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


def load_shape_pipeline():
    global _shape_pipeline
    if _shape_pipeline is not None:
        return _shape_pipeline

    logger.info("Loading Hunyuan3D-2 shape pipeline...")
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2'
    )
    logger.info("Shape pipeline loaded")
    return _shape_pipeline


def load_paint_pipeline():
    global _paint_pipeline
    if _paint_pipeline is not None:
        return _paint_pipeline

    logger.info("Loading Hunyuan3D-2 paint pipeline...")
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    _paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        'tencent/Hunyuan3D-2'
    )
    logger.info("Paint pipeline loaded")
    return _paint_pipeline


def generate_3d(image: Image.Image, remove_bg: bool = True, generate_texture: bool = True) -> dict:
    import torch
    timings = {}

    # Background removal
    start = time.time()
    if remove_bg:
        from hy3dgen.rembg import BackgroundRemover
        logger.info("Removing background...")
        rembg = BackgroundRemover()
        image = rembg(image)
    timings["background_removal"] = time.time() - start

    # Load shape pipeline
    start = time.time()
    shape_pipeline = load_shape_pipeline()
    timings["shape_model_loading"] = time.time() - start

    # Generate mesh
    start = time.time()
    logger.info("Generating 3D mesh...")
    with torch.inference_mode():
        mesh = shape_pipeline(image=image)[0]
    timings["shape_generation"] = time.time() - start
    clear_gpu_memory()

    # Texture generation (if requested)
    if generate_texture:
        start = time.time()
        paint_pipeline = load_paint_pipeline()
        timings["paint_model_loading"] = time.time() - start

        start = time.time()
        logger.info("Generating textures...")
        with torch.inference_mode():
            mesh = paint_pipeline(mesh, image=image)
        timings["texture_generation"] = time.time() - start
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

    logger.info(f"Generated GLB: {file_size} bytes, textured: {generate_texture}")

    return {
        "success": True,
        "model_base64": model_base64,
        "file_size": file_size,
        "format": "glb",
        "textured": generate_texture,
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
        generate_texture = job_input.get("generate_texture", True)

        logger.info("Decoding input image...")
        image = decode_base64_image(image_base64)
        logger.info(f"Input: {image.size}, mode: {image.mode}")

        start_time = time.time()
        result = generate_3d(
            image=image,
            remove_bg=remove_bg,
            generate_texture=generate_texture
        )
        result["execution_time"] = time.time() - start_time

        return result

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.exception(f"Handler error: {error_msg}")
        return {"error": error_msg, "success": False}


runpod.serverless.start({"handler": handler})
