"""
RunPod Serverless Handler - Hunyuan3D-2.1
Shape + PBR Texture Generation with improved cross-view consistency
"""

import sys
# Add Hunyuan3D-2.1 modules to path
sys.path.insert(0, '/opt/Hunyuan3D-2.1')
sys.path.insert(0, '/opt/Hunyuan3D-2.1/hy3dshape')
sys.path.insert(0, '/opt/Hunyuan3D-2.1/hy3dpaint')

import runpod
import base64
import time
import io
import os
import gc
import logging
import tempfile
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(name)s:%(lineno)d %(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# Global model cache
_shape_pipeline = None
_paint_pipeline = None


def log_gpu_status(label: str = ""):
    """Log current GPU memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"[GPU {label}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
    except Exception as e:
        logger.warning(f"Could not get GPU status: {e}")


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


def check_extensions():
    """Check if all required C++ extensions are available"""
    logger.info("=" * 50)
    logger.info("CHECKING C++ EXTENSIONS (Hunyuan3D-2.1)")
    logger.info("=" * 50)

    # Check custom_rasterizer
    try:
        from hy3dpaint import custom_rasterizer
        logger.info("✓ custom_rasterizer: LOADED")
    except ImportError as e:
        logger.error(f"✗ custom_rasterizer: FAILED - {e}")

    # Check mesh_processor from DifferentiableRenderer
    try:
        from hy3dpaint.DifferentiableRenderer import mesh_processor
        logger.info("✓ mesh_processor: LOADED")
    except ImportError as e:
        logger.warning(f"✗ mesh_processor: {e}")

    logger.info("=" * 50)


def check_diffusers_version():
    """Log diffusers and related package versions"""
    logger.info("=" * 50)
    logger.info("PACKAGE VERSIONS")
    logger.info("=" * 50)

    packages = ['torch', 'diffusers', 'transformers', 'accelerate', 'huggingface_hub']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            logger.info(f"{pkg}: {version}")
        except ImportError:
            logger.warning(f"{pkg}: NOT INSTALLED")

    logger.info("=" * 50)


def load_shape_pipeline():
    global _shape_pipeline
    if _shape_pipeline is not None:
        logger.info("Shape pipeline already loaded (cached)")
        return _shape_pipeline

    logger.info("Loading Hunyuan3D-2.1 shape pipeline...")
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    logger.info(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    log_gpu_status("before shape pipeline load")

    _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2.1'
    )

    log_gpu_status("after shape pipeline load")
    logger.info("Shape pipeline loaded successfully")
    return _shape_pipeline


def load_paint_pipeline():
    global _paint_pipeline
    if _paint_pipeline is not None:
        logger.info("Paint pipeline already loaded (cached)")
        return _paint_pipeline

    logger.info("Loading Hunyuan3D-2.1 paint pipeline...")
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    log_gpu_status("before paint pipeline load")

    # Configure paint pipeline
    config = Hunyuan3DPaintConfig()
    # Use more views for better coverage
    if hasattr(config, 'max_num_view'):
        config.max_num_view = 6

    _paint_pipeline = Hunyuan3DPaintPipeline(config)

    log_gpu_status("after paint pipeline load")
    logger.info("Paint pipeline loaded successfully")
    return _paint_pipeline


def generate_3d(image: Image.Image, remove_bg: bool = True, generate_texture: bool = True) -> dict:
    import torch
    timings = {}

    logger.info("=" * 50)
    logger.info("STARTING 3D GENERATION (Hunyuan3D-2.1)")
    logger.info("=" * 50)
    logger.info(f"Options: remove_bg={remove_bg}, generate_texture={generate_texture}")

    # Create temp directory for file-based API
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, "input.png")
    temp_mesh_path = os.path.join(temp_dir, "mesh.glb")
    temp_output_path = os.path.join(temp_dir, "textured.glb")

    try:
        # Background removal
        start = time.time()
        if remove_bg:
            from hy3dshape.rembg import BackgroundRemover
            logger.info("Removing background...")
            rembg = BackgroundRemover()
            image = rembg(image)
            logger.info(f"Background removed, image size: {image.size}")
        timings["background_removal"] = time.time() - start
        logger.info(f"Background removal took: {timings['background_removal']:.2f}s")

        # Save image for texture pipeline (needs file path)
        image.save(temp_image_path)

        # Load shape pipeline
        start = time.time()
        shape_pipeline = load_shape_pipeline()
        timings["shape_model_loading"] = time.time() - start
        logger.info(f"Shape model loading took: {timings['shape_model_loading']:.2f}s")

        # Generate mesh
        start = time.time()
        logger.info("Generating 3D mesh...")
        log_gpu_status("before shape generation")
        with torch.inference_mode():
            mesh = shape_pipeline(image=image)[0]
        log_gpu_status("after shape generation")
        timings["shape_generation"] = time.time() - start
        logger.info(f"Shape generation took: {timings['shape_generation']:.2f}s")

        logger.info(f"Mesh info: vertices={len(mesh.vertices) if hasattr(mesh, 'vertices') else 'unknown'}")

        # Export mesh for texture pipeline
        mesh.export(temp_mesh_path)
        logger.info(f"Mesh exported to {temp_mesh_path}")

        clear_gpu_memory()
        log_gpu_status("after shape GPU cleanup")

        # Texture generation (if requested)
        final_glb_path = temp_mesh_path  # Default to untextured mesh

        if generate_texture:
            logger.info("=" * 50)
            logger.info("STARTING TEXTURE GENERATION (PBR)")
            logger.info("=" * 50)

            start = time.time()
            paint_pipeline = load_paint_pipeline()
            timings["paint_model_loading"] = time.time() - start
            logger.info(f"Paint model loading took: {timings['paint_model_loading']:.2f}s")

            start = time.time()
            logger.info("Generating PBR textures... (this may take a while)")
            log_gpu_status("before texture generation")

            with torch.inference_mode():
                logger.info("Calling paint_pipeline...")
                # 2.1 API: takes file paths, not objects
                result_path = paint_pipeline(
                    mesh_path=temp_mesh_path,
                    image_path=temp_image_path,
                    output_mesh_path=temp_output_path,
                    use_remesh=True,
                    save_glb=True
                )
                logger.info(f"paint_pipeline returned: {result_path}")

                # Use the output path
                if result_path and os.path.exists(result_path):
                    final_glb_path = result_path
                elif os.path.exists(temp_output_path):
                    final_glb_path = temp_output_path
                else:
                    logger.warning("Texture output not found, using untextured mesh")

            log_gpu_status("after texture generation")
            timings["texture_generation"] = time.time() - start
            logger.info(f"Texture generation took: {timings['texture_generation']:.2f}s")
            clear_gpu_memory()
            log_gpu_status("after texture GPU cleanup")

        # Read and encode final GLB
        logger.info(f"Reading final GLB from: {final_glb_path}")
        model_base64 = encode_file_to_base64(final_glb_path)
        file_size = os.path.getsize(final_glb_path)

        logger.info("=" * 50)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Generated GLB: {file_size} bytes, textured: {generate_texture}")
        logger.info(f"Total time: {sum(timings.values()):.2f}s")
        for key, val in timings.items():
            logger.info(f"  {key}: {val:.2f}s")

        return {
            "success": True,
            "model_base64": model_base64,
            "file_size": file_size,
            "format": "glb",
            "textured": generate_texture,
            "version": "2.1",
            "timings": timings,
            "total_time": sum(timings.values()),
        }

    finally:
        # Cleanup temp files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir: {e}")


def handler(job: dict) -> dict:
    try:
        logger.info("=" * 50)
        logger.info("NEW JOB RECEIVED (Hunyuan3D-2.1)")
        logger.info("=" * 50)

        # Run checks on first job
        check_diffusers_version()
        check_extensions()

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
        result = generate_3d(
            image=image,
            remove_bg=remove_bg,
            generate_texture=generate_texture
        )
        result["execution_time"] = time.time() - start_time

        logger.info(f"Job completed successfully in {result['execution_time']:.2f}s")
        return result

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.exception(f"Handler error: {error_msg}")
        return {"error": error_msg, "success": False}


runpod.serverless.start({"handler": handler})
