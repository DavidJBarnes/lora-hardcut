# HARDCUT LoRA Training Setup Guide
## WAN 2.2 I2V on RTX 3090 with Musubi-Tuner

---

## Understanding WAN 2.2 Architecture

<cite index="34-4">WAN 2.2 introduces Mixture-of-Experts (MoE) architecture into the video generation diffusion model. The A14B model series adopts a two-expert design: a **high-noise expert** for early stages focusing on overall layout, and a **low-noise expert** for later stages refining video details.</cite>

**This means you need to train TWO separate LoRAs** (or train them jointly):
- **High-noise LoRA**: Timesteps 900-1000 (handles transitions/layout)
- **Low-noise LoRA**: Timesteps 0-900 (handles details)

For HARDCUT transitions, the **high-noise model is critical** since hard cuts happen in the early denoising stages.

---

## Directory Structure

```
~/projects/lora-hardcut/
├── musubi-tuner/                    # Cloned repository
├── dataset/
│   ├── videos/                      # Your 61-frame videos
│   │   ├── video001.mp4
│   │   ├── video001.txt             # Caption file
│   │   ├── video002.mp4
│   │   ├── video002.txt
│   │   └── ...
│   └── cache/                       # Latent cache (auto-generated)
├── configs/
│   └── dataset.toml                 # Dataset configuration
├── output/                          # Training output
│   ├── high_noise/
│   └── low_noise/
└── logs/                            # TensorBoard logs

# Your existing ComfyUI models (NO DUPLICATES NEEDED):
~/StabilityMatrix-linux-x64/Data/Packages/ComfyUI/models/
├── diffusion_models/
│   ├── wan2.2_i2v_high_noise_14B_fp16.safetensors  ✓
│   ├── wan2.2_i2v_low_noise_14B_fp16.safetensors   ✓
│   └── Wan2.1_VAE.pth                               ✓
└── text_encoders/
    └── umt5_xxl_fp8_e4m3fn_scaled.safetensors       ✓
```

---

## Step 1: Installation

```bash
cd ~/projects/lora-hardcut

# Clone musubi-tuner
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner

# Create virtual environment (Python 3.10 recommended)
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch 2.5.1+ with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install musubi-tuner
pip install -e .

# Install additional dependencies
pip install ascii-magic matplotlib tensorboard prompt-toolkit

# Configure accelerate
accelerate config
# Answer the prompts:
# - This machine
# - No distributed training
# - No DeepSpeed
# - No FSDP
# - bf16 (or fp16)
```

---

## Step 2: Model Paths (Using Your Existing ComfyUI Models)

You already have all required models! No downloads needed.

```bash
# Your existing model locations:
COMFYUI_MODELS=~/StabilityMatrix-linux-x64/Data/Packages/ComfyUI/models

# DiT models (you have these ✓)
DIT_HIGH=$COMFYUI_MODELS/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
DIT_LOW=$COMFYUI_MODELS/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors

# VAE (you have this ✓)
VAE=$COMFYUI_MODELS/diffusion_models/Wan2.1_VAE.pth

# T5 text encoder (you have this ✓)
T5=$COMFYUI_MODELS/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
```

### If T5 Caching Fails

Your FP8 T5 should work, but if you get errors during text encoder caching, download the bf16 version:

```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
  models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir ~/StabilityMatrix-linux-x64/Data/Packages/ComfyUI/models/text_encoders

# Then use this path instead:
T5=$COMFYUI_MODELS/text_encoders/models_t5_umt5-xxl-enc-bf16.pth
```

**Note:** <cite index="36-1">For WAN 2.2, CLIP model is not required (unlike WAN 2.1).</cite>

---

## Step 3: Prepare Your Dataset

### Video Requirements
Your videos are **61 frames at 480x832** - this is perfect because:
- <cite index="10-2">For WAN2.1/2.2, the number of target_frames must be "N*4+1" (N=0,1,2,...)</cite>
- 61 = 15×4+1 ✓
- <cite index="10-1">It is recommended to use videos with a frame rate of 16fps for Wan2.1</cite> (same for 2.2)

### Caption Files
Each video needs a matching `.txt` caption file:
```
dataset/videos/
├── hardcut_001.mp4
├── hardcut_001.txt    # "A hard cut transition from a forest scene to a city street..."
├── hardcut_002.mp4
├── hardcut_002.txt
...
```

**Caption example for HARDCUT:**
```
A hard cut transition. The scene abruptly changes from [describe scene A] to [describe scene B]. 
The transition is instantaneous with no fade or blend between the two distinct scenes.
HARDCUT style.
```

### Dataset TOML Configuration

Create `configs/dataset.toml`:

```toml
# Dataset configuration for HARDCUT LoRA training
# WAN 2.2 I2V - 61 frames @ 480x832

[general]
resolution = [480, 832]  # [width, height] - your 480w x 832h videos (portrait)
caption_extension = ".txt"
batch_size = 1
enable_bucket = false     # Disable since all videos are same resolution
bucket_no_upscale = true

[[datasets]]
video_directory = "/home/YOUR_USER/projects/lora-hardcut/dataset/videos"
cache_directory = "/home/YOUR_USER/projects/lora-hardcut/dataset/cache"
target_frames = [61]      # Your exact frame count (N*4+1 = 61)
frame_extraction = "full" # Use all frames
num_repeats = 1           # Increase if you have few videos
```

---

## Step 4: Pre-Cache Latents and Text Encoder Outputs

**CRITICAL:** You must run these caching steps before training!

```bash
cd ~/projects/lora-hardcut/musubi-tuner

# Define your model paths
COMFYUI_MODELS=~/StabilityMatrix-linux-x64/Data/Packages/ComfyUI/models
VAE=$COMFYUI_MODELS/diffusion_models/Wan2.1_VAE.pth
T5=$COMFYUI_MODELS/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

# 1. Cache latents (add --i2v for Image-to-Video training!)
python src/musubi_tuner/wan_cache_latents.py \
  --dataset_config ../configs/dataset.toml \
  --vae $VAE \
  --i2v \
  --vae_cache_cpu \
  --batch_size 1

# 2. Cache text encoder outputs
python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config ../configs/dataset.toml \
  --t5 $T5 \
  --batch_size 4
```

**Debug tip:** Add `--debug_mode video` to verify your dataset is being processed correctly.

---

## Step 5: Training Commands (RTX 3090 Optimized)

```bash
# Define model paths (use these in all commands below)
COMFYUI_MODELS=~/StabilityMatrix-linux-x64/Data/Packages/ComfyUI/models
DIT_HIGH=$COMFYUI_MODELS/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
DIT_LOW=$COMFYUI_MODELS/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
VAE=$COMFYUI_MODELS/diffusion_models/Wan2.1_VAE.pth
T5=$COMFYUI_MODELS/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
```

### Train HIGH-NOISE Model (Most important for HARDCUT)

```bash
cd ~/projects/lora-hardcut/musubi-tuner

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  src/musubi_tuner/wan_train_network.py \
  --task i2v-A14B \
  --dit $DIT_HIGH \
  --vae $VAE \
  --t5 $T5 \
  --dataset_config ../configs/dataset.toml \
  \
  --network_module networks.lora_wan \
  --network_dim 16 \
  --network_alpha 16 \
  --network_args "loraplus_lr_ratio=4" \
  \
  --timestep_sampling shift \
  --discrete_flow_shift 5.0 \
  --min_timestep 900 \
  --max_timestep 1000 \
  --preserve_distribution_shape \
  \
  --optimizer_type adamw8bit \
  --learning_rate 2e-4 \
  --lr_scheduler cosine \
  --max_train_epochs 20 \
  --save_every_n_epochs 2 \
  \
  --sdpa \
  --mixed_precision fp16 \
  --fp8_base \
  --fp8_scaled \
  --gradient_checkpointing \
  --blocks_to_swap 20 \
  \
  --max_data_loader_n_workers 2 \
  --persistent_data_loader_workers \
  --seed 42 \
  \
  --output_dir ../output/high_noise \
  --output_name hardcut_high \
  --log_with tensorboard \
  --logging_dir ../logs
```

### Train LOW-NOISE Model

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  src/musubi_tuner/wan_train_network.py \
  --task i2v-A14B \
  --dit $DIT_LOW \
  --vae $VAE \
  --t5 $T5 \
  --dataset_config ../configs/dataset.toml \
  \
  --network_module networks.lora_wan \
  --network_dim 16 \
  --network_alpha 16 \
  --network_args "loraplus_lr_ratio=4" \
  \
  --timestep_sampling shift \
  --discrete_flow_shift 5.0 \
  --min_timestep 0 \
  --max_timestep 900 \
  --preserve_distribution_shape \
  \
  --optimizer_type adamw8bit \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --max_train_epochs 20 \
  --save_every_n_epochs 2 \
  \
  --sdpa \
  --mixed_precision fp16 \
  --fp8_base \
  --fp8_scaled \
  --gradient_checkpointing \
  --blocks_to_swap 20 \
  \
  --max_data_loader_n_workers 2 \
  --persistent_data_loader_workers \
  --seed 42 \
  \
  --output_dir ../output/low_noise \
  --output_name hardcut_low \
  --log_with tensorboard \
  --logging_dir ../logs
```

---

## Key Parameters Explained

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--task i2v-A14B` | WAN 2.2 I2V task identifier |
| `--network_dim 16` | LoRA rank - 16 is good balance for 3090 |
| `--fp8_base --fp8_scaled` | Run DiT in FP8 - critical for 24GB VRAM |
| `--blocks_to_swap 20` | Offload blocks to CPU - adjust up if OOM |
| `--gradient_checkpointing` | Trades compute for VRAM |
| `--min_timestep 900 --max_timestep 1000` | HIGH noise range |
| `--min_timestep 0 --max_timestep 900` | LOW noise range |
| `--preserve_distribution_shape` | Maintain proper noise distribution |
| `--discrete_flow_shift 5.0` | Recommended for WAN 2.2 |
| `--loraplus_lr_ratio=4` | LoRA+ for faster convergence |

---

## About CLAUDE.md

The repository mentions creating a `CLAUDE.md` file - <cite index="27-1">this repository provides recommended instructions to help AI agents like Claude understand the project context and coding standards. Create a CLAUDE.md file in the project root and add the following line to import the repository's recommended prompt.</cite>

This is **not** a training config - it's for AI assistants helping with the codebase. You don't need it for training.

---

## Troubleshooting

### OOM (Out of Memory)
- Increase `--blocks_to_swap` (max ~38)
- Lower `--network_dim` to 8
- Reduce batch_size to 1
- Add `--vae_cache_cpu` to cache commands

### Training shows no effect
- Ensure you're using **both** high and low noise LoRAs during inference
- Check that `--i2v` flag was used during latent caching
- Verify captions describe the HARDCUT transition concept

### KeyError: 'latents_image'
- You forgot `--i2v` flag when caching latents
- Re-run latent caching with `--i2v`

---

## Inference

After training, use both LoRAs in ComfyUI with the WAN 2.2 I2V workflow, applying:
- `hardcut_high.safetensors` to the high-noise model
- `hardcut_low.safetensors` to the low-noise model

---

## Quick Start Script

Save as `train_hardcut.sh` (or use the provided script):

```bash
#!/bin/bash
set -e

PROJECT_ROOT=~/projects/lora-hardcut
MUSUBI=$PROJECT_ROOT/musubi-tuner
COMFYUI_MODELS=~/StabilityMatrix-linux-x64/Data/Packages/ComfyUI/models
CONFIG=$PROJECT_ROOT/configs/dataset.toml

VAE=$COMFYUI_MODELS/diffusion_models/Wan2.1_VAE.pth
T5=$COMFYUI_MODELS/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

cd $MUSUBI

echo "=== Caching Latents ==="
python src/musubi_tuner/wan_cache_latents.py \
  --dataset_config $CONFIG \
  --vae $VAE \
  --i2v --vae_cache_cpu --batch_size 1

echo "=== Caching Text Encoder ==="
python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config $CONFIG \
  --t5 $T5 \
  --batch_size 4

echo "=== Training HIGH Noise ==="
# (insert high noise training command)

echo "=== Training LOW Noise ==="
# (insert low noise training command)

echo "=== Done! ==="
```

---

## Summary

1. **Clone** musubi-tuner and install dependencies
2. **Download** WAN 2.2 I2V models (high + low noise), VAE, and T5
3. **Prepare** dataset with videos + caption files
4. **Create** dataset.toml config
5. **Cache** latents with `--i2v` flag
6. **Cache** text encoder outputs
7. **Train** HIGH noise model (timesteps 900-1000)
8. **Train** LOW noise model (timesteps 0-900)
9. **Use** both LoRAs during inference
