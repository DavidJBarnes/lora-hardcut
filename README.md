# HARDCUT LoRA

WAN 2.2 I2V LoRA training using [musubi-tuner](https://github.com/kohya-ss/musubi-tuner). Trains two LoRA models (high noise + low noise) on portrait video clips for image-to-video generation.

## Prerequisites

- NVIDIA GPU with 24GB+ VRAM (RTX 3090 tested)
- [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) cloned and installed
- WAN 2.2 models:
  - `wan2.2_i2v_high_noise_14B_fp16.safetensors` — from [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
  - `wan2.2_i2v_low_noise_14B_fp16.safetensors` — same repo
  - `Wan2.1_VAE.pth` — from [Wan-AI/Wan2.1-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)
  - `models_t5_umt5-xxl-enc-bf16.pth` — same repo (must be bf16 `.pth`, not fp8 safetensors)

## Dataset

Place video clips and matching caption files in `dataset/videos/`:

```
dataset/videos/
├── clip_001.mp4
├── clip_001.txt
├── clip_002.mp4
├── clip_002.txt
└── ...
```

- Videos: 480x832 portrait, 61 frames @ 16fps
- Captions: one `.txt` per video with the same filename

Edit `configs/dataset.toml` to match your paths and video specs.

## Setup

```bash
# Clone musubi-tuner into project root
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
pip install -e ".[cu124]"
cd ..

# Edit model paths in train_hardcut.sh
vim train_hardcut.sh
```

## Training

```bash
# 1. Cache latents and text encoder outputs (required once per dataset)
bash train_hardcut.sh cache

# 2. Train both models sequentially
bash train_hardcut.sh train

# Or train individually
bash train_hardcut.sh train-high   # High noise (timesteps 900-1000)
bash train_hardcut.sh train-low    # Low noise (timesteps 0-900)
```

Output LoRAs are saved to `output/high_noise/` and `output/low_noise/`.

## Benchmarks

Tested with 238 videos at 480x832, 61 frames, batch_size=1, network_dim=16, fp8_base + fp8_scaled.

### RTX 3090 (24GB)

| Setting | blocks_to_swap | VRAM Used | Speed | Status |
|---------|---------------|-----------|-------|--------|
| Default | 20 | OOM | — | OOM during forward pass |
| Moderate | 28 | OOM | — | OOM during forward pass |
| Conservative | 32 | OOM | — | OOM during forward pass |
| **Working** | **36** | **23.4 GB** | **~70-78s/step** | Stable |
| Combined (2 DiTs) | 36 | OOM | — | Two 14B models don't fit |

- **Latent caching**: ~6.8s/video, ~27 min for 238 videos
- **Text encoder caching**: ~2 min for 238 captions
- **Training (per model)**: 2380 steps @ 10 epochs ≈ 50 hours, 4760 steps @ 20 epochs ≈ 100 hours
- **Checkpoint saves**: Every 2 epochs (~5 hours at 10 epoch config)
- Key memory flags: `--fp8_base --fp8_scaled --gradient_checkpointing --force_v2_1_time_embedding`
- Combined single-run mode (`--dit` + `--dit_high_noise`) does **not** fit on 24GB — use separate runs

### L40S (48GB, RunPod)

| Setting | blocks_to_swap | VRAM Used | Speed |
|---------|---------------|-----------|-------|
| No swapping | 0 | 45 GB | ~35s/step |

- ~2.2x faster than 3090, but cloud rental cost ($0.89/hr) makes it ~$80+ for full training of both models

### RunPod / Cloud GPU

Use `train_runpod.sh` for 48GB+ GPUs (A6000, L40S). With more VRAM you can set `blocks_to_swap=0` and skip memory optimizations. See `download_models.sh` for fetching models on a fresh instance.

## Files

| File | Description |
|------|-------------|
| `train_hardcut.sh` | Main training pipeline (3090 optimized) |
| `train_runpod.sh` | Cloud GPU training script |
| `configs/dataset.toml` | Dataset configuration |
| `download_models.sh` | Model download helper for cloud setups |
| `Dockerfile` | RunPod container image |
| `hardcut-lora-setup-guide.md` | Detailed parameter reference |
