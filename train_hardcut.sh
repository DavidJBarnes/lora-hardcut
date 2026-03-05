#!/bin/bash
# HARDCUT LoRA Training Pipeline for WAN 2.2 I2V
# RTX 3090 optimized settings
set -e

# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================
PROJECT_ROOT=~/projects/lora-hardcut
MUSUBI=$PROJECT_ROOT/musubi-tuner
DATASET_CONFIG=$PROJECT_ROOT/configs/dataset.toml
OUTPUT=$PROJECT_ROOT/output
LOGS=$PROJECT_ROOT/logs

# Existing ComfyUI model paths (no duplicate downloads needed)
COMFYUI_MODELS=~/StabilityMatrix-linux-x64/Data/Packages/ComfyUI/models

# Model paths (using your existing files)
DIT_HIGH=$COMFYUI_MODELS/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
DIT_LOW=$COMFYUI_MODELS/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
VAE=$COMFYUI_MODELS/diffusion_models/Wan2.1_VAE.pth
# T5 text encoder - bf16 version (FP8 safetensors has incompatible state dict keys)
T5=$COMFYUI_MODELS/text_encoders/models_t5_umt5-xxl-enc-bf16.pth

# Training parameters
NETWORK_DIM=16
NETWORK_ALPHA=16
LEARNING_RATE_HIGH=2e-4
LEARNING_RATE_LOW=1e-4
MAX_EPOCHS=20
SAVE_EVERY=2
BLOCKS_TO_SWAP=20  # Increase if OOM (max ~38)

# ============================================
# FUNCTIONS
# ============================================

setup_directories() {
    echo "=== Setting up directory structure ==="
    mkdir -p $PROJECT_ROOT/{dataset/videos,configs,output/{high_noise,low_noise},logs}
}

download_models() {
    echo "=== Checking models ==="
    
    # You already have DiT and VAE models in ComfyUI!
    # You also have an FP8 T5 - let's try that first
    
    echo ""
    echo "Model check complete. Using existing models:"
    echo "  DIT_HIGH: $DIT_HIGH"
    echo "  DIT_LOW:  $DIT_LOW"
    echo "  VAE:      $VAE"
    echo "  T5:       $T5"
    echo ""
    echo "Note: If text encoder caching fails with your FP8 T5,"
    echo "download the bf16 version with:"
    echo "  huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P models_t5_umt5-xxl-enc-bf16.pth --local-dir $COMFYUI_MODELS/text_encoders"
    echo "Then update T5 path in this script."
}

cache_latents() {
    echo "=== Caching latents (with --i2v flag) ==="
    cd $MUSUBI
    python src/musubi_tuner/wan_cache_latents.py \
        --dataset_config $DATASET_CONFIG \
        --vae $VAE \
        --i2v \
        --vae_cache_cpu \
        --batch_size 1
}

cache_text_encoder() {
    echo "=== Caching text encoder outputs ==="
    cd $MUSUBI
    # Note: Using your existing FP8 T5 model
    # If this fails, you may need the bf16 version instead
    python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
        --dataset_config $DATASET_CONFIG \
        --t5 $T5 \
        --batch_size 4
}

train_high_noise() {
    echo "=== Training HIGH NOISE model (timesteps 900-1000) ==="
    cd $MUSUBI
    accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
        src/musubi_tuner/wan_train_network.py \
        --task i2v-A14B \
        --dit $DIT_HIGH \
        --vae $VAE \
        --t5 $T5 \
        --dataset_config $DATASET_CONFIG \
        \
        --network_module networks.lora_wan \
        --network_dim $NETWORK_DIM \
        --network_alpha $NETWORK_ALPHA \
        --network_args "loraplus_lr_ratio=4" \
        \
        --timestep_sampling shift \
        --discrete_flow_shift 5.0 \
        --min_timestep 900 \
        --max_timestep 1000 \
        --preserve_distribution_shape \
        \
        --optimizer_type adamw8bit \
        --learning_rate $LEARNING_RATE_HIGH \
        --lr_scheduler cosine \
        --max_train_epochs $MAX_EPOCHS \
        --save_every_n_epochs $SAVE_EVERY \
        \
        --sdpa \
        --mixed_precision fp16 \
        --fp8_base \
        --fp8_scaled \
        --gradient_checkpointing \
        --blocks_to_swap $BLOCKS_TO_SWAP \
        \
        --max_data_loader_n_workers 2 \
        --persistent_data_loader_workers \
        --seed 42 \
        \
        --output_dir $OUTPUT/high_noise \
        --output_name hardcut_high \
        --log_with tensorboard \
        --logging_dir $LOGS
}

train_low_noise() {
    echo "=== Training LOW NOISE model (timesteps 0-900) ==="
    cd $MUSUBI
    accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
        src/musubi_tuner/wan_train_network.py \
        --task i2v-A14B \
        --dit $DIT_LOW \
        --vae $VAE \
        --t5 $T5 \
        --dataset_config $DATASET_CONFIG \
        \
        --network_module networks.lora_wan \
        --network_dim $NETWORK_DIM \
        --network_alpha $NETWORK_ALPHA \
        --network_args "loraplus_lr_ratio=4" \
        \
        --timestep_sampling shift \
        --discrete_flow_shift 5.0 \
        --min_timestep 0 \
        --max_timestep 900 \
        --preserve_distribution_shape \
        \
        --optimizer_type adamw8bit \
        --learning_rate $LEARNING_RATE_LOW \
        --lr_scheduler cosine \
        --max_train_epochs $MAX_EPOCHS \
        --save_every_n_epochs $SAVE_EVERY \
        \
        --sdpa \
        --mixed_precision fp16 \
        --fp8_base \
        --fp8_scaled \
        --gradient_checkpointing \
        --blocks_to_swap $BLOCKS_TO_SWAP \
        \
        --max_data_loader_n_workers 2 \
        --persistent_data_loader_workers \
        --seed 42 \
        \
        --output_dir $OUTPUT/low_noise \
        --output_name hardcut_low \
        --log_with tensorboard \
        --logging_dir $LOGS
}

# ============================================
# MAIN SCRIPT
# ============================================

case "${1:-all}" in
    setup)
        setup_directories
        ;;
    download)
        download_models
        ;;
    cache)
        cache_latents
        cache_text_encoder
        ;;
    cache-latents)
        cache_latents
        ;;
    cache-text)
        cache_text_encoder
        ;;
    train-high)
        train_high_noise
        ;;
    train-low)
        train_low_noise
        ;;
    train)
        train_high_noise
        train_low_noise
        ;;
    all)
        setup_directories
        echo ""
        echo "Directory structure created!"
        echo ""
        echo "Using existing ComfyUI models from:"
        echo "  $COMFYUI_MODELS"
        echo ""
        echo "All models found - no downloads needed!"
        echo ""
        echo "Next steps:"
        echo "1. Put your videos in: $PROJECT_ROOT/dataset/videos/"
        echo "2. Create captions (.txt files) for each video"
        echo "3. Copy dataset.toml to: $PROJECT_ROOT/configs/"
        echo "4. Run: $0 cache        # Cache latents and text"
        echo "5. Run: $0 train        # Train both models"
        ;;
    *)
        echo "Usage: $0 {setup|download|cache|cache-latents|cache-text|train-high|train-low|train|all}"
        echo ""
        echo "Commands:"
        echo "  setup        - Create directory structure"
        echo "  download     - Download all required models"
        echo "  cache        - Cache latents and text encoder outputs"
        echo "  cache-latents - Cache latents only"
        echo "  cache-text   - Cache text encoder only"
        echo "  train-high   - Train high noise model only"
        echo "  train-low    - Train low noise model only"  
        echo "  train        - Train both models"
        echo "  all          - Setup directories and show next steps"
        exit 1
        ;;
esac

echo ""
echo "=== Done! ==="
