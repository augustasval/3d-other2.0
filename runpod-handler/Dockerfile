# Simplified 2D-to-3D Handler for RunPod Serverless
# Using Hunyuan3D-2 as the primary model (proven to work)
#
# This is a simplified version that focuses on reliability.
# Full multi-model pipeline can be added incrementally.

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models
ENV TRANSFORMERS_CACHE=/runpod-volume/models
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install core Python dependencies in one layer
RUN pip install --no-cache-dir \
    runpod==1.6.2 \
    huggingface-hub==0.21.0 \
    transformers==4.38.0 \
    diffusers==0.27.0 \
    accelerate==0.27.0 \
    safetensors==0.4.2 \
    pillow>=10.0.0 \
    numpy>=1.24.0,<2.0.0 \
    scipy>=1.11.0 \
    tqdm>=4.66.0 \
    einops>=0.7.0 \
    omegaconf>=2.3.0

# Install background removal (rembg is reliable and well-tested)
RUN pip install --no-cache-dir rembg[gpu]==2.0.55

# Install 3D processing libraries
RUN pip install --no-cache-dir \
    trimesh>=4.0.0 \
    pygltflib>=1.16.0 \
    opencv-python-headless>=4.8.0

# Install PyMCubes (correct package name)
RUN pip install --no-cache-dir PyMCubes

# Copy application code (simplified handler is self-contained)
COPY handler.py /app/handler.py

# Create model cache directory
RUN mkdir -p /runpod-volume/models

# Health check endpoint support
EXPOSE 8000

# Set the entrypoint
CMD ["python", "-u", "/app/handler.py"]
