# Hunyuan3D-2 Stage 2: Shape + Texture Generation
# Using RunPod's pre-cached base image to reduce build size
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models
ENV CUDA_HOME=/usr/local/cuda
# Required for CUDA extension compilation without GPU present during build
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"

# Install build tools + OpenGL libs (PyTorch already installed in base image)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install pybind11 (needed for CUDA extension compilation)
RUN pip install --no-cache-dir pybind11

# Clone Hunyuan3D-2
WORKDIR /opt
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git

WORKDIR /opt/Hunyuan3D-2

# CRITICAL FIX: Disable super_model upscaling which hangs on CPU
# See: https://huggingface.co/tencent/Hunyuan3D-2/discussions/6
RUN sed -i "s/multiviews\[i\] = self.models\['super_model'\](multiviews\[i\])/# multiviews[i] = self.models['super_model'](multiviews[i])  # DISABLED - hangs on CPU/" hy3dgen/texgen/pipelines.py

# Install requirements (skip torch since already installed)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Compile texture generation CUDA extensions
WORKDIR /opt/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python setup.py install

WORKDIR /opt/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python setup.py install

# Install RunPod
RUN pip install --no-cache-dir runpod

# Cleanup to reduce image size
RUN rm -rf /root/.cache/pip && \
    find /opt/Hunyuan3D-2 -name "*.pyc" -delete && \
    find /opt/Hunyuan3D-2 -name "__pycache__" -type d -delete

WORKDIR /app
COPY runpod-handler/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
