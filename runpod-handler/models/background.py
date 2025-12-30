"""
Background Removal Module using SAM (Segment Anything Model)

Uses SAM 2 for high-quality foreground segmentation with automatic prompt generation.
Falls back to rembg/U2-Net if SAM fails.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Model cache directory
MODEL_DIR = "/app/models/sam"


class SAMBackgroundRemover:
    """High-quality background removal using SAM 2."""

    def __init__(self, model_type: str = "sam2_hiera_large"):
        """
        Initialize SAM model.

        Args:
            model_type: SAM model variant
                - "sam2_hiera_large": Best quality, ~2GB VRAM
                - "sam2_hiera_base_plus": Good balance
                - "sam2_hiera_small": Fastest
        """
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.predictor = None

    def load(self):
        """Load SAM model to GPU."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Model checkpoint paths
            checkpoints = {
                "sam2_hiera_large": "sam2_hiera_large.pt",
                "sam2_hiera_base_plus": "sam2_hiera_base_plus.pt",
                "sam2_hiera_small": "sam2_hiera_small.pt",
            }

            checkpoint = f"{MODEL_DIR}/{checkpoints[self.model_type]}"
            config = f"sam2_hiera_{self.model_type.split('_')[-1]}.yaml"

            logger.info(f"Loading SAM2 model: {self.model_type}")
            self.model = build_sam2(config, checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)

            logger.info("SAM2 loaded successfully")

        except Exception as e:
            logger.warning(f"SAM2 failed to load: {e}, falling back to rembg")
            self._setup_rembg_fallback()

    def _setup_rembg_fallback(self):
        """Setup rembg as fallback background remover."""
        try:
            from rembg import new_session
            self.rembg_session = new_session("u2net")
            self.use_rembg = True
            logger.info("Using rembg/U2-Net as fallback")
        except Exception as e:
            logger.error(f"Failed to load rembg fallback: {e}")
            self.use_rembg = False

    def _find_foreground_center(self, image: Image.Image) -> Tuple[int, int]:
        """
        Find the center of the foreground object using edge detection.

        Returns approximate center point for SAM prompt.
        """
        import cv2

        # Convert to grayscale
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)

        # Fallback to image center
        return (image.width // 2, image.height // 2)

    def _segment_with_sam(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Segment foreground using SAM with automatic point prompt.

        Returns binary mask as numpy array.
        """
        if self.predictor is None:
            return None

        try:
            # Convert to numpy
            image_np = np.array(image.convert("RGB"))

            # Set image for predictor
            self.predictor.set_image(image_np)

            # Find foreground center for point prompt
            center_point = self._find_foreground_center(image)
            input_point = np.array([[center_point[0], center_point[1]]])
            input_label = np.array([1])  # 1 = foreground

            # Predict mask
            masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            # Select best mask (highest score)
            best_idx = np.argmax(scores)
            mask = masks[best_idx]

            return mask.astype(np.uint8) * 255

        except Exception as e:
            logger.warning(f"SAM segmentation failed: {e}")
            return None

    def _segment_with_rembg(self, image: Image.Image) -> Image.Image:
        """Fallback segmentation using rembg."""
        from rembg import remove

        if hasattr(self, "rembg_session"):
            return remove(image, session=self.rembg_session)
        else:
            return remove(image)

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove background from image.

        Args:
            image: Input PIL Image

        Returns:
            RGBA image with transparent background
        """
        # Convert to RGB if needed
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        # Try SAM first
        mask = self._segment_with_sam(image)

        if mask is not None:
            # Apply mask to create RGBA image
            image_rgba = image.convert("RGBA")
            image_np = np.array(image_rgba)

            # Create alpha channel from mask
            image_np[:, :, 3] = mask

            return Image.fromarray(image_np, "RGBA")

        # Fallback to rembg
        if hasattr(self, "use_rembg") and self.use_rembg:
            logger.info("Using rembg fallback for background removal")
            return self._segment_with_rembg(image)

        # If all else fails, return original with white background treated as transparent
        logger.warning("Background removal failed, returning original image")
        return image.convert("RGBA")


# Global instance
_background_remover: Optional[SAMBackgroundRemover] = None


def load_sam_model(model_type: str = "sam2_hiera_large") -> SAMBackgroundRemover:
    """
    Load and return the SAM background remover.

    Args:
        model_type: SAM model variant

    Returns:
        Initialized SAMBackgroundRemover instance
    """
    global _background_remover

    if _background_remover is None:
        _background_remover = SAMBackgroundRemover(model_type=model_type)
        _background_remover.load()

    return _background_remover


def remove_background(model: SAMBackgroundRemover, image: Image.Image) -> Image.Image:
    """
    Remove background from image using SAM.

    Args:
        model: SAMBackgroundRemover instance
        image: Input PIL Image

    Returns:
        RGBA image with transparent background
    """
    return model.remove_background(image)


def has_transparency(image: Image.Image, threshold: float = 0.95) -> bool:
    """
    Check if image already has significant transparency.

    Args:
        image: PIL Image to check
        threshold: Ratio of opaque pixels to consider "no transparency"

    Returns:
        True if image has transparency, False otherwise
    """
    if image.mode != "RGBA":
        return False

    alpha = np.array(image.split()[-1])
    opaque_ratio = np.sum(alpha > 250) / alpha.size

    return opaque_ratio < threshold
