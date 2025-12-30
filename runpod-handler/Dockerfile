# High-Quality 2D-to-3D Generation Pipeline
# Optimized for RunPod Serverless with A40 (48GB) GPU
#
# Pipeline: SAM → Era3D → GeoLRM → Hunyuan3D-2.1 Paint → GLB
#
# Build: docker build -t 2d-to-3d-handler .
# Run locally: docker run --gpus all -p 8000:8000 2d-to-3d-handler

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set up environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    HUGGINGFACE_HUB_CACHE=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    git-lfs \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ninja-build \
    build-essential \
    cmake \
    libeigen3-dev \
    libboost-all-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt /app/requirements.txt

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install -r requirements.txt

# Install additional ML libraries
RUN pip install \
    transformers==4.38.0 \
    diffusers==0.27.0 \
    accelerate==0.27.0 \
    safetensors==0.4.2 \
    huggingface-hub==0.21.0 \
    xformers==0.0.24 \
    triton==2.2.0

# Install 3D processing libraries
RUN pip install \
    trimesh==4.1.0 \
    pygltflib==1.16.1 \
    pymeshlab==2023.12.post1 \
    open3d==0.18.0 \
    pyrender==0.1.45

# Install xatlas for UV unwrapping
RUN pip install xatlas==0.0.9

# Install marching cubes
RUN pip install mcubes==0.1.0

# Install SAM 2 dependencies
RUN pip install \
    segment-anything-2==0.1.0 \
    rembg==2.0.55

# Install OpenCV
RUN pip install opencv-python-headless==4.9.0.80

# Install RunPod SDK
RUN pip install runpod==1.6.2

# Create model directories
RUN mkdir -p /app/models/sam \
    /app/models/era3d \
    /app/models/geolrm \
    /app/models/hunyuan3d

# Download SAM 2 model weights (optional - can be done at runtime)
ARG DOWNLOAD_MODELS=false
RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
    echo "Downloading SAM 2 model..." && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large.pt \
        -O /app/models/sam/sam2_hiera_large.pt; \
    fi

# Copy application code
COPY . /app/

# Make handler executable
RUN chmod +x /app/handler.py

# Expose port for local testing
EXPOSE 8000

# Set the entrypoint
CMD ["python", "-u", "/app/handler.py"]
