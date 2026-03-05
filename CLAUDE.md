# HARDCUT LoRA Training

WAN 2.2 I2V LoRA training project using [musubi-tuner](https://github.com/kohya-ss/musubi-tuner).

## Project Structure

```
configs/dataset.toml     - Dataset config (video paths, resolution, frame count)
train_hardcut.sh         - Main training pipeline script (3090 optimized)
train_runpod.sh          - RunPod cloud GPU training script
download_models.sh       - Model download helper for cloud setups
Dockerfile               - RunPod container image
dataset-prep/            - Dataset preparation scripts (video cutting, captioning)
hardcut-lora-setup-guide.md - Full parameter reference guide
```

## Training Setup

- **GPU**: RTX 3090 (24GB) on `3090.zero` via SSH
- **Framework**: musubi-tuner (cloned to `musubi-tuner/`, not committed)
- **Models**: Stored in ComfyUI at `~/StabilityMatrix-linux-x64/Data/Packages/ComfyUI/models/`
  - DiT high noise: `diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors`
  - DiT low noise: `diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors`
  - VAE: `diffusion_models/Wan2.1_VAE.pth`
  - T5: `text_encoders/models_t5_umt5-xxl-enc-bf16.pth` (must be bf16 .pth, NOT fp8 safetensors)

## Dataset

- 238 video clips, 480x832 portrait, 61 frames @ 16fps
- Each video has a matching `.txt` caption file
- Videos live in `dataset/videos/` (gitignored)
- Latent/text caches in `dataset/cache/` (gitignored)

## Training Parameters (3090)

- **blocks_to_swap=36** — minimum viable for 24GB with 480x832x61 frame videos. Values below 36 OOM.
- **Two separate models**: high noise (timesteps 900-1000, lr 2e-4) and low noise (0-900, lr 1e-4)
- Combined single-run approach (`--dit` + `--dit_high_noise` + `--timestep_boundary`) does NOT fit on 3090 (two 14B models exceed 24GB)
- Key flags: `--fp8_base --fp8_scaled --gradient_checkpointing --force_v2_1_time_embedding`
- Speed: ~70-78s/step, ~2380 steps per model at 10 epochs

## Commands

```bash
# Cache latents and text encoder outputs (required before training)
bash train_hardcut.sh cache

# Train high noise model only
bash train_hardcut.sh train-high

# Train both models sequentially
bash train_hardcut.sh train
```

## Notes

- RunPod SSH requires PTY (`ssh -tt`), no SCP support. Use `dataset-prep/rpod.py` for remote commands.
- The `convert_t5_to_musubi.py` script converts HuggingFace T5 checkpoints to musubi-tuner format (only needed if downloading from HF directly).
- Output LoRAs go to `output/high_noise/` and `output/low_noise/` (gitignored).
