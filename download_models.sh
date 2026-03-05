#!/bin/bash
# Download models from HuggingFace to /workspace/models/
set -e

MODELS=/workspace/models
cd "$MODELS"

echo "=== Downloading models from HuggingFace ==="

# T5 text encoder (11GB)
if [ ! -f "$MODELS/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "Downloading T5 text encoder..."
    wget -q --show-progress -O models_t5_umt5-xxl-enc-bf16.pth \
        "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth"
fi

# VAE (485MB)
if [ ! -f "$MODELS/Wan2.1_VAE.pth" ]; then
    echo "Downloading VAE..."
    wget -q --show-progress -O Wan2.1_VAE.pth \
        "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/Wan2.1_VAE.pth"
fi

# DiT high noise (27GB)
if [ ! -f "$MODELS/wan2.2_i2v_high_noise_14B_fp16.safetensors" ]; then
    echo "Downloading DiT high noise model..."
    wget -q --show-progress -O wan2.2_i2v_high_noise_14B_fp16.safetensors \
        "https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-720P/resolve/main/wan2.2_i2v_high_noise_14B_fp16.safetensors"
fi

# DiT low noise (27GB)
if [ ! -f "$MODELS/wan2.2_i2v_low_noise_14B_fp16.safetensors" ]; then
    echo "Downloading DiT low noise model..."
    wget -q --show-progress -O wan2.2_i2v_low_noise_14B_fp16.safetensors \
        "https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-720P/resolve/main/wan2.2_i2v_low_noise_14B_fp16.safetensors"
fi

echo ""
echo "=== All models downloaded ==="
ls -lh "$MODELS"
