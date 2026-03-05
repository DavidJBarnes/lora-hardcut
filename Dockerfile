FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git wget ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Clone musubi-tuner
WORKDIR /workspace
RUN git clone https://github.com/kohya-ss/musubi-tuner.git

# Install musubi-tuner deps
WORKDIR /workspace/musubi-tuner
RUN pip install --no-cache-dir -e ".[cu124]" && \
    pip install --no-cache-dir tensorboard

# Create directory structure
RUN mkdir -p /workspace/dataset/videos /workspace/dataset/cache \
    /workspace/configs /workspace/output/high_noise /workspace/output/low_noise \
    /workspace/logs /workspace/models

# Copy training scripts
COPY train_runpod.sh /workspace/train_runpod.sh
COPY download_models.sh /workspace/download_models.sh
RUN chmod +x /workspace/train_runpod.sh /workspace/download_models.sh

WORKDIR /workspace
