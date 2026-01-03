# Hunyuan3D-2.1: Shape + PBR Texture Generation
# Improved cross-view consistency with 3D-Aware RoPE
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"

# Install Python 3.11 (required for bpy 4.0) + build tools + OpenGL libs
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip git wget \
    build-essential libgl1-mesa-glx libglib2.0-0 libopengl0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN python -m ensurepip --upgrade && python -m pip install --upgrade pip

# Install PyTorch 2.5.1 with CUDA 12.4
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install pybind11
RUN pip install --no-cache-dir pybind11

# Clone Hunyuan3D-2.1
WORKDIR /opt
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
WORKDIR /opt/Hunyuan3D-2.1

# Install requirements (replace bpy==4.0 with available version)
RUN sed 's/bpy==4.0/bpy>=4.2.0/' requirements.txt > requirements_fixed.txt && \
    pip install --no-cache-dir -r requirements_fixed.txt

# Pin diffusers to 0.30.0 as specified
RUN pip install --no-cache-dir diffusers==0.30.0

# Compile custom_rasterizer (hy3dpaint version)
WORKDIR /opt/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
RUN python setup.py install

# Compile DifferentiableRenderer
WORKDIR /opt/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh

# Download RealESRGAN upsampler model
WORKDIR /opt/Hunyuan3D-2.1
RUN wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -P hy3dpaint/ckpt

# Install RunPod
RUN pip install --no-cache-dir runpod

# Cleanup
RUN rm -rf /root/.cache/pip

WORKDIR /app
COPY runpod-handler/handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
