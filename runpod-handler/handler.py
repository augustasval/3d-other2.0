"""
RunPod Serverless Handler for High-Quality 2D-to-3D Generation

Pipeline:
1. SAM Background Removal
2. Era3D Multi-View Generation (6 views + normal maps)
3. GeoLRM 3D Reconstruction
4. Hunyuan3D-2.1 Paint (PBR Textures)
5. GLB Export with Draco Compression

Optimized for A40 (48GB) GPU - Quality over speed.
"""

import runpod
import base64
import time
import io
import os
import gc
import torch
import logging
from PIL import Image
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances (loaded on first request, cached)
_models = {
    "sam": None,
    "era3d": None,
    "geolrm": None,
    "hunyuan_paint": None,
}


def clear_gpu_memory():
    """Clear GPU memory between pipeline stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    # Handle data URL format
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))

    # Convert to RGB if necessary
    if image.mode == "RGBA":
        # Keep alpha for transparency support
        pass
    elif image.mode != "RGB":
        image = image.convert("RGB")

    return image


def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_models():
    """Load all models to GPU (called on first request)."""
    global _models

    # Import model loaders
    from models.background import load_sam_model
    from models.era3d import load_era3d_model
    from models.geolrm import load_geolrm_model
    from models.hunyuan_paint import load_hunyuan_paint_model

    if _models["sam"] is None:
        logger.info("Loading SAM model...")
        _models["sam"] = load_sam_model()
        logger.info("SAM model loaded.")

    if _models["era3d"] is None:
        logger.info("Loading Era3D model...")
        _models["era3d"] = load_era3d_model()
        logger.info("Era3D model loaded.")

    if _models["geolrm"] is None:
        logger.info("Loading GeoLRM model...")
        _models["geolrm"] = load_geolrm_model()
        logger.info("GeoLRM model loaded.")

    if _models["hunyuan_paint"] is None:
        logger.info("Loading Hunyuan3D-2.1 Paint model...")
        _models["hunyuan_paint"] = load_hunyuan_paint_model()
        logger.info("Hunyuan3D-2.1 Paint model loaded.")

    return _models


def run_pipeline(
    image: Image.Image,
    remove_background: bool = True,
    texture_resolution: int = 2048,
    mesh_detail: str = "high",
    generate_pbr: bool = True,
) -> dict:
    """
    Run the full quality 2D-to-3D pipeline.

    Args:
        image: Input PIL Image
        remove_background: Whether to remove background with SAM
        texture_resolution: Output texture resolution (1024, 2048, 4096)
        mesh_detail: Mesh detail level ("low", "medium", "high")
        generate_pbr: Generate full PBR materials

    Returns:
        dict with model_base64, file_size, metadata
    """
    from models.background import remove_background as bg_remove
    from models.era3d import generate_multiview
    from models.geolrm import reconstruct_3d
    from models.hunyuan_paint import generate_pbr_textures
    from utils.glb_export import export_to_glb

    models = load_models()
    timings = {}

    # Stage 1: Background Removal
    start_time = time.time()
    if remove_background:
        logger.info("Stage 1: Removing background with SAM...")
        image = bg_remove(models["sam"], image)
    timings["background_removal"] = time.time() - start_time
    clear_gpu_memory()

    # Stage 2: Multi-View Generation with Era3D
    start_time = time.time()
    logger.info("Stage 2: Generating multi-views with Era3D...")
    multiview_images, normal_maps = generate_multiview(
        models["era3d"],
        image,
        num_views=6,
    )
    timings["multiview_generation"] = time.time() - start_time
    clear_gpu_memory()

    # Stage 3: 3D Reconstruction with GeoLRM
    start_time = time.time()
    logger.info("Stage 3: Reconstructing 3D mesh with GeoLRM...")
    mesh = reconstruct_3d(
        models["geolrm"],
        multiview_images,
        normal_maps,
        detail_level=mesh_detail,
    )
    timings["reconstruction"] = time.time() - start_time
    clear_gpu_memory()

    # Stage 4: PBR Texture Generation with Hunyuan3D-2.1
    start_time = time.time()
    if generate_pbr:
        logger.info("Stage 4: Generating PBR textures with Hunyuan3D-2.1...")
        textured_mesh = generate_pbr_textures(
            models["hunyuan_paint"],
            mesh,
            image,
            multiview_images,
            texture_resolution=texture_resolution,
        )
    else:
        textured_mesh = mesh
    timings["texture_generation"] = time.time() - start_time
    clear_gpu_memory()

    # Stage 5: GLB Export
    start_time = time.time()
    logger.info("Stage 5: Exporting to GLB...")
    glb_path, metadata = export_to_glb(
        textured_mesh,
        use_draco=True,
    )
    timings["glb_export"] = time.time() - start_time

    # Encode GLB to base64
    model_base64 = encode_file_to_base64(glb_path)
    file_size = os.path.getsize(glb_path)

    # Cleanup temporary file
    os.remove(glb_path)

    return {
        "model_base64": model_base64,
        "file_size": file_size,
        "format": "glb",
        "textured": generate_pbr,
        "texture_resolution": texture_resolution if generate_pbr else None,
        "vertices": metadata.get("vertices", 0),
        "faces": metadata.get("faces", 0),
        "timings": timings,
        "total_time": sum(timings.values()),
    }


def handler(job: dict) -> dict:
    """
    RunPod serverless handler function.

    Expected input:
    {
        "input": {
            "image": "<base64-encoded-image>",
            "remove_background": true,
            "texture_resolution": 2048,
            "mesh_detail": "high",
            "generate_pbr": true
        }
    }

    Returns:
    {
        "model_base64": "<base64-encoded-glb>",
        "file_size": 12345678,
        "format": "glb",
        "textured": true,
        "texture_resolution": 2048,
        "vertices": 150000,
        "faces": 300000,
        "timings": {...},
        "total_time": 65.5
    }
    """
    try:
        job_input = job.get("input", {})

        # Validate required input
        image_base64 = job_input.get("image")
        if not image_base64:
            return {"error": "No image provided", "success": False}

        # Parse options with defaults
        remove_background = job_input.get("remove_background", True)
        texture_resolution = job_input.get("texture_resolution", 2048)
        mesh_detail = job_input.get("mesh_detail", "high")
        generate_pbr = job_input.get("generate_pbr", True)

        # Validate texture resolution
        if texture_resolution not in [1024, 2048, 4096]:
            texture_resolution = 2048

        # Validate mesh detail
        if mesh_detail not in ["low", "medium", "high"]:
            mesh_detail = "high"

        # Decode input image
        logger.info("Decoding input image...")
        image = decode_base64_image(image_base64)
        logger.info(f"Input image size: {image.size}, mode: {image.mode}")

        # Run pipeline
        start_time = time.time()
        result = run_pipeline(
            image=image,
            remove_background=remove_background,
            texture_resolution=texture_resolution,
            mesh_detail=mesh_detail,
            generate_pbr=generate_pbr,
        )

        result["success"] = True
        result["execution_time"] = time.time() - start_time

        logger.info(f"Pipeline completed in {result['execution_time']:.2f}s")
        logger.info(f"Output: {result['vertices']} vertices, {result['faces']} faces, {result['file_size']} bytes")

        return result

    except Exception as e:
        logger.exception(f"Pipeline error: {str(e)}")
        return {
            "error": str(e),
            "success": False,
        }


# RunPod serverless entrypoint
runpod.serverless.start({"handler": handler})
