# Minimal 2D-to-3D Handler for RunPod Serverless v3
# Uses runpod/pytorch base which has most dependencies pre-installed

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models

# CRITICAL: Pin versions compatible with PyTorch 2.2.0
# Latest diffusers/accelerate require torch.xpu which doesn't exist in PyTorch 2.2
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    runpod \
    onnxruntime \
    rembg \
    trimesh \
    pygltflib \
    PyMCubes \
    "diffusers==0.25.0" \
    "accelerate==0.25.0" \
    "transformers==4.36.0" \
    huggingface_hub \
    safetensors

WORKDIR /app

# Copy handler
COPY runpod-handler/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
