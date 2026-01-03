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
    logger.info("CHECKING C++ EXTENSIONS")
    logger.info("=" * 50)

    # Check mesh_processor
    try:
        from hy3dgen.texgen.differentiable_renderer import mesh_processor
        logger.info("✓ mesh_processor: LOADED")
        if hasattr(mesh_processor, 'meshVerticeInpaint'):
            logger.info("  ✓ meshVerticeInpaint function: AVAILABLE")
        else:
            logger.warning("  ✗ meshVerticeInpaint function: NOT FOUND")
    except ImportError as e:
        logger.error(f"✗ mesh_processor: FAILED - {e}")

    # Check custom_rasterizer
    try:
        from hy3dgen.texgen import custom_rasterizer
        logger.info("✓ custom_rasterizer: LOADED")
    except ImportError as e:
        logger.error(f"✗ custom_rasterizer: FAILED - {e}")

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

    logger.info("Loading Hunyuan3D-2 shape pipeline...")
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    logger.info(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    log_gpu_status("before shape pipeline load")

    _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2'
    )

    log_gpu_status("after shape pipeline load")
    logger.info("Shape pipeline loaded successfully")
    return _shape_pipeline


def load_paint_pipeline():
    global _paint_pipeline
    if _paint_pipeline is not None:
        logger.info("Paint pipeline already loaded (cached)")
        return _paint_pipeline

    logger.info("Loading Hunyuan3D-2 paint pipeline...")
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    log_gpu_status("before paint pipeline load")

    _paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-paint-v2-0'  # Use non-turbo (has VAE safetensors)
    )

    # Debug: Log pipeline configuration
    if hasattr(_paint_pipeline, 'config'):
        config = _paint_pipeline.config
        logger.info(f"Paint pipeline config:")
        logger.info(f"  device: {getattr(config, 'device', 'unknown')}")
        logger.info(f"  render_size: {getattr(config, 'render_size', 'unknown')}")
        logger.info(f"  texture_size: {getattr(config, 'texture_size', 'unknown')}")

    # Check models loaded
    if hasattr(_paint_pipeline, 'models'):
        logger.info(f"Paint pipeline models: {list(_paint_pipeline.models.keys())}")

    log_gpu_status("after paint pipeline load")
    logger.info("Paint pipeline loaded successfully")
    return _paint_pipeline


def generate_3d(image: Image.Image, remove_bg: bool = True, generate_texture: bool = True) -> dict:
    import torch
    timings = {}

    logger.info("=" * 50)
    logger.info("STARTING 3D GENERATION")
    logger.info("=" * 50)
    logger.info(f"Options: remove_bg={remove_bg}, generate_texture={generate_texture}")

    # Background removal
    start = time.time()
    if remove_bg:
        from hy3dgen.rembg import BackgroundRemover
        logger.info("Removing background...")
        rembg = BackgroundRemover()
        image = rembg(image)
        logger.info(f"Background removed, image size: {image.size}")
    timings["background_removal"] = time.time() - start
    logger.info(f"Background removal took: {timings['background_removal']:.2f}s")

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
    clear_gpu_memory()
    log_gpu_status("after shape GPU cleanup")

    # Texture generation (if requested)
    if generate_texture:
        logger.info("=" * 50)
        logger.info("STARTING TEXTURE GENERATION")
        logger.info("=" * 50)

        # Simplify mesh if too complex (>100k vertices causes slow UV wrapping)
        vertex_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
        if vertex_count > 100000:
            logger.info(f"Mesh has {vertex_count} vertices - simplifying to speed up texture generation...")
            start_simplify = time.time()
            try:
                import pymeshlab
                # Save mesh temporarily
                temp_mesh_path = tempfile.mktemp(suffix=".ply")
                mesh.export(temp_mesh_path)

                # Use pymeshlab for robust decimation
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(temp_mesh_path)

                # Simplify to ~50k faces using quadric edge collapse
                target_faces = 50000
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)

                # Save simplified mesh
                temp_simplified_path = tempfile.mktemp(suffix=".ply")
                ms.save_current_mesh(temp_simplified_path)

                # Reload as trimesh
                import trimesh
                mesh = trimesh.load(temp_simplified_path)

                # Cleanup
                os.remove(temp_mesh_path)
                os.remove(temp_simplified_path)

                new_vertex_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
                logger.info(f"Mesh simplified: {vertex_count} -> {new_vertex_count} vertices in {time.time() - start_simplify:.2f}s")
            except Exception as e:
                logger.warning(f"Mesh simplification failed: {e}, continuing with original mesh")

        start = time.time()
        paint_pipeline = load_paint_pipeline()
        timings["paint_model_loading"] = time.time() - start
        logger.info(f"Paint model loading took: {timings['paint_model_loading']:.2f}s")

        start = time.time()
        logger.info("Generating textures... (this may take a while)")
        log_gpu_status("before texture generation")

        with torch.inference_mode():
            logger.info("Calling paint_pipeline...")
            mesh = paint_pipeline(mesh, image=image)
            logger.info("paint_pipeline returned")

        log_gpu_status("after texture generation")
        timings["texture_generation"] = time.time() - start
        logger.info(f"Texture generation took: {timings['texture_generation']:.2f}s")
        clear_gpu_memory()
        log_gpu_status("after texture GPU cleanup")

    # Export to GLB
    logger.info("Exporting to GLB...")
    start = time.time()
    glb_path = tempfile.mktemp(suffix=".glb")
    mesh.export(glb_path)
    timings["export"] = time.time() - start
    logger.info(f"Export took: {timings['export']:.2f}s")

    # Read and encode
    model_base64 = encode_file_to_base64(glb_path)
    file_size = os.path.getsize(glb_path)
    os.remove(glb_path)

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
        "timings": timings,
        "total_time": sum(timings.values()),
    }


def handler(job: dict) -> dict:
    try:
        logger.info("=" * 50)
        logger.info("NEW JOB RECEIVED")
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
