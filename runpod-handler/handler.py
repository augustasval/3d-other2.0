"""
RunPod Serverless Handler for 2D-to-3D Generation
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_models = {}
_load_errors = {}


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


def load_shap_e():
    """Load Shap-E model for 3D generation."""
    global _models, _load_errors

    if "shap_e" in _models:
        return _models.get("shap_e")

    if "shap_e" in _load_errors:
        return None

    logger.info("Loading Shap-E model...")

    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        import diffusers
        logger.info(f"Diffusers version: {diffusers.__version__}")

        from diffusers import ShapEImg2ImgPipeline

        logger.info("Loading ShapEImg2ImgPipeline...")
        pipe = ShapEImg2ImgPipeline.from_pretrained(
            "openai/shap-e-img2img",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = pipe.to("cuda")

        _models["shap_e"] = pipe
        logger.info("Shap-E loaded successfully")
        return pipe

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Failed to load Shap-E: {error_msg}")
        _load_errors["shap_e"] = error_msg
        _models["shap_e"] = None
        return None


def generate_3d_model(
    image: Image.Image,
    generate_texture: bool = True,
    remove_bg: bool = True,
) -> dict:
    """Generate 3D model from image."""
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
    pipe = load_shap_e()
    timings["model_loading"] = time.time() - start

    if pipe is None:
        error_detail = _load_errors.get("shap_e", "Unknown error")
        logger.warning(f"Model not available: {error_detail}")
        return {
            "success": False,
            "error": f"Shap-E model not loaded: {error_detail}",
            "timings": timings,
        }

    # Step 3: Generate 3D
    start = time.time()
    logger.info("Generating 3D model with Shap-E...")

    try:
        # Ensure RGB for Shap-E
        if image.mode == "RGBA":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            processed_image = rgb_image
        else:
            processed_image = image.convert("RGB")

        # Resize (Shap-E works best with 256x256)
        processed_image = processed_image.resize((256, 256), Image.LANCZOS)

        with torch.inference_mode():
            outputs = pipe(
                processed_image,
                guidance_scale=3.0,
                num_inference_steps=64,
                frame_size=256,
                output_type="mesh",
            )

        timings["generation"] = time.time() - start
        clear_gpu_memory()

        # Step 4: Export to GLB
        start = time.time()
        logger.info("Exporting to GLB...")

        mesh = outputs.images[0]

        ply_path = tempfile.mktemp(suffix=".ply")
        glb_path = tempfile.mktemp(suffix=".glb")

        # Export to PLY
        try:
            from diffusers.utils import export_to_ply
        except ImportError:
            from diffusers.utils.export_utils import export_to_ply

        export_to_ply(mesh, ply_path)

        # Convert PLY to GLB
        import trimesh
        mesh_data = trimesh.load(ply_path)
        mesh_data.export(glb_path)

        timings["export"] = time.time() - start

        # Read and encode
        model_base64 = encode_file_to_base64(glb_path)
        file_size = os.path.getsize(glb_path)

        # Cleanup
        os.remove(ply_path)
        os.remove(glb_path)

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
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Generation failed: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "timings": timings,
        }


def handler(job: dict) -> dict:
    """RunPod serverless handler function."""
    try:
        job_input = job.get("input", {})

        image_base64 = job_input.get("image")
        if not image_base64:
            return {"error": "No image provided", "success": False}

        remove_bg = job_input.get("remove_background", True)
        generate_texture = job_input.get("generate_texture", True)

        logger.info("Decoding input image...")
        image = decode_base64_image(image_base64)
        logger.info(f"Input image: {image.size}, mode: {image.mode}")

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
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.exception(f"Handler error: {error_msg}")
        return {"error": error_msg, "success": False}


# RunPod entrypoint
runpod.serverless.start({"handler": handler})
