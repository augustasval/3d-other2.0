# Hunyuan3D-2 Stage 2: Shape + Texture Generation
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models
ENV CUDA_HOME=/usr/local/cuda
# Required for CUDA extension compilation without GPU present during build
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"

# Install Python 3.11 + build tools + OpenGL libs
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.4 (MUST be first)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install pybind11 (needed for CUDA extension compilation)
RUN pip install pybind11

# Clone Hunyuan3D-2
WORKDIR /opt
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git

WORKDIR /opt/Hunyuan3D-2
RUN pip install -r requirements.txt
RUN pip install -e .

# Compile texture generation CUDA extensions
WORKDIR /opt/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python setup.py install

WORKDIR /opt/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python setup.py install

# Install RunPod
RUN pip install runpod

WORKDIR /app
COPY runpod-handler/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
