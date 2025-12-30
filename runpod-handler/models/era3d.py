"""
Era3D Multi-View Generation Module

Generates 6 consistent views + 6 normal maps from a single image.
Based on: https://github.com/pengHTYX/Era3D

Input: Single image (512x512)
Output: 6 RGB views + 6 normal maps at canonical camera poses
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = "/app/models/era3d"


@dataclass
class Era3DConfig:
    """Configuration for Era3D model."""
    model_path: str = f"{MODEL_DIR}/era3d_pipeline"
    image_size: int = 512
    num_views: int = 6
    guidance_scale: float = 3.0
    num_inference_steps: int = 50
    elevation: float = 0.0  # Camera elevation in degrees
    # Camera azimuths for 6 views (front, front-right, right, back, left, front-left)
    azimuths: Tuple[float, ...] = (0, 60, 120, 180, 240, 300)


class Era3DPipeline:
    """
    Era3D pipeline for multi-view generation with normal maps.
    """

    def __init__(self, config: Era3DConfig = None):
        self.config = config or Era3DConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.loaded = False

    def load(self):
        """Load Era3D pipeline to GPU."""
        try:
            from diffusers import DiffusionPipeline
            import os

            logger.info(f"Loading Era3D from {self.config.model_path}")

            # Check if model exists locally
            if os.path.exists(self.config.model_path):
                model_id = self.config.model_path
            else:
                # Download from HuggingFace
                model_id = "pengHTYX/Era3D"
                logger.info(f"Downloading Era3D from {model_id}")

            # Load the pipeline
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.pipe = self.pipe.to(self.device)

            # Enable memory optimizations
            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                self.pipe.enable_xformers_memory_efficient_attention()

            self.loaded = True
            logger.info("Era3D pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Era3D: {e}")
            raise

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess input image for Era3D.

        - Resize to 512x512
        - Center crop if needed
        - Ensure RGBA with proper alpha
        """
        # Convert to RGBA if needed
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Get dimensions
        w, h = image.size

        # Determine crop box for center square
        if w > h:
            left = (w - h) // 2
            image = image.crop((left, 0, left + h, h))
        elif h > w:
            top = (h - w) // 2
            image = image.crop((0, top, w, top + w))

        # Resize to target size
        image = image.resize(
            (self.config.image_size, self.config.image_size),
            Image.Resampling.LANCZOS
        )

        return image

    def generate(
        self,
        image: Image.Image,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        Generate multi-view images and normal maps from a single image.

        Args:
            image: Input RGBA image
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps

        Returns:
            Tuple of (rgb_images, normal_maps) - each a list of 6 PIL Images
        """
        if not self.loaded:
            self.load()

        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = num_inference_steps or self.config.num_inference_steps

        # Preprocess image
        image = self.preprocess_image(image)

        logger.info(f"Generating {self.config.num_views} views with Era3D...")

        try:
            # Run Era3D pipeline
            with torch.autocast("cuda", dtype=torch.float16):
                output = self.pipe(
                    image,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )

            # Extract RGB images and normal maps
            # Era3D outputs both in a grid or as separate tensors
            if hasattr(output, "images") and hasattr(output, "normals"):
                rgb_images = output.images
                normal_maps = output.normals
            else:
                # Parse from combined output
                rgb_images, normal_maps = self._parse_output(output)

            logger.info(f"Generated {len(rgb_images)} RGB views and {len(normal_maps)} normal maps")
            return rgb_images, normal_maps

        except Exception as e:
            logger.error(f"Era3D generation failed: {e}")
            raise

    def _parse_output(self, output) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        Parse Era3D output into separate RGB and normal images.

        Era3D outputs a 2x6 grid: top row RGB, bottom row normals.
        """
        # Handle different output formats
        if hasattr(output, "images"):
            combined = output.images[0] if isinstance(output.images, list) else output.images
        else:
            combined = output

        # If it's a tensor, convert to PIL
        if isinstance(combined, torch.Tensor):
            combined = self._tensor_to_pil(combined)

        # If it's a grid image, split it
        if isinstance(combined, Image.Image):
            w, h = combined.size

            # Check if it's a 2-row grid (RGB on top, normals on bottom)
            if h == w // 3:  # 2 rows of 6 images (6:2 aspect ratio)
                cell_size = w // 6
                rgb_images = []
                normal_maps = []

                for i in range(6):
                    # Top row: RGB
                    left = i * cell_size
                    rgb = combined.crop((left, 0, left + cell_size, cell_size))
                    rgb_images.append(rgb)

                    # Bottom row: Normals
                    normal = combined.crop((left, cell_size, left + cell_size, cell_size * 2))
                    normal_maps.append(normal)

                return rgb_images, normal_maps

        # If we got a list directly
        if isinstance(combined, list):
            n = len(combined)
            if n >= 12:
                return combined[:6], combined[6:12]
            elif n >= 6:
                return combined[:6], combined[:6]  # Use RGB as fallback for normals

        # Fallback: return the image repeated
        logger.warning("Could not parse Era3D output, using single image")
        return [combined] * 6, [combined] * 6

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]  # Remove batch dimension

        # Move channels to last dimension
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)

        # Convert to numpy
        array = tensor.cpu().numpy()

        # Normalize to 0-255
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

        return Image.fromarray(array)


# Global instance
_era3d_pipeline: Optional[Era3DPipeline] = None


def load_era3d_model(config: Era3DConfig = None) -> Era3DPipeline:
    """
    Load and return the Era3D pipeline.

    Args:
        config: Era3D configuration

    Returns:
        Initialized Era3DPipeline instance
    """
    global _era3d_pipeline

    if _era3d_pipeline is None:
        _era3d_pipeline = Era3DPipeline(config=config)
        _era3d_pipeline.load()

    return _era3d_pipeline


def generate_multiview(
    model: Era3DPipeline,
    image: Image.Image,
    num_views: int = 6,
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Generate multi-view images and normal maps from a single image.

    Args:
        model: Era3DPipeline instance
        image: Input RGBA image
        num_views: Number of views to generate (default 6)

    Returns:
        Tuple of (rgb_images, normal_maps)
    """
    return model.generate(image)


def images_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """
    Convert list of PIL Images to batched tensor for reconstruction.

    Args:
        images: List of PIL Images

    Returns:
        Tensor of shape (N, C, H, W)
    """
    tensors = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")

        array = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append(tensor)

    return torch.stack(tensors)
