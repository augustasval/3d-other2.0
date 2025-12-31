# Minimal 2D-to-3D Handler for RunPod Serverless v2
# Uses runpod/pytorch base which has most dependencies pre-installed

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models

# Install dependencies
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    runpod \
    onnxruntime-gpu \
    rembg \
    trimesh \
    pygltflib \
    PyMCubes \
    transformers \
    huggingface_hub

# Install TripoSR from GitHub
RUN pip install git+https://github.com/VAST-AI-Research/TripoSR.git

WORKDIR /app

# Copy handler
COPY runpod-handler/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
