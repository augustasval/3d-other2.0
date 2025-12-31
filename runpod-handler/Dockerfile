# Hunyuan3D-2 Stage 1: Shape Generation Only
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.4
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Clone Hunyuan3D-2 and install (shape-only, skip texture compilation)
WORKDIR /opt
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git

WORKDIR /opt/Hunyuan3D-2
RUN pip install -r requirements.txt
RUN pip install -e .

# Install RunPod
RUN pip install runpod

WORKDIR /app
COPY runpod-handler/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
