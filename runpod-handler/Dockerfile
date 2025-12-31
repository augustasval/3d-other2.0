# USE PYTORCH 2.8 - required by transformers for CVE-2025-32434 fix
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models

# Install ALL dependencies - no version pinning needed with PyTorch 2.4+
RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    huggingface_hub \
    safetensors \
    onnxruntime-gpu \
    rembg \
    trimesh \
    scipy \
    Pillow

WORKDIR /app

COPY runpod-handler/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
