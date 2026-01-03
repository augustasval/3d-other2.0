# Hunyuan3D-2 Stage 2: Shape + Texture Generation
# Using NVIDIA CUDA 12.4 base + PyTorch 2.6.0 for compatibility
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"

# Install Python 3.11 + build tools + OpenGL libs
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip git \
    build-essential libgl1-mesa-glx libglib2.0-0 libopengl0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip

# Install PyTorch 2.6.0 with CUDA 12.4 (required for transformers security + RunPod compatibility)
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install pybind11
RUN pip install --no-cache-dir pybind11

# Clone Hunyuan3D-2
WORKDIR /opt
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
WORKDIR /opt/Hunyuan3D-2

# Install requirements and package
RUN pip install --no-cache-dir -r requirements.txt

# CRITICAL FIX: Disable super_model upscaler that hangs on CPU
# The stabilityai/stable-diffusion-x4-upscaler falls back to single-threaded CPU and hangs
RUN sed -i "s/multiviews\[i\] = self.models\['super_model'\](multiviews\[i\])/# multiviews[i] = self.models['super_model'](multiviews[i])  # DISABLED - hangs on CPU/" hy3dgen/texgen/pipelines.py

RUN pip install --no-cache-dir -e .

# Use diffusers 0.31.0 - newer versions have UNet type issues, older versions have module loading issues
RUN pip install --no-cache-dir diffusers==0.31.0

# Compile texture CUDA extensions
WORKDIR /opt/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python setup.py install

# Compile mesh_processor (CRITICAL for GPU texture inpainting - without this, falls back to slow CPU)
WORKDIR /opt/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python setup.py install
# Also build in-place for local imports
RUN python setup.py build_ext --inplace

# Install RunPod
RUN pip install --no-cache-dir runpod

# Cleanup
RUN rm -rf /root/.cache/pip

WORKDIR /app
COPY runpod-handler/handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
