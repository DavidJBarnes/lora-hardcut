#!/bin/bash
# Full training pipeline for RunPod (A6000 48GB)
set -e

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MUSUBI=/workspace/musubi-tuner
MODELS=/workspace/models
DATASET_CONFIG=/workspace/configs/dataset.toml
OUTPUT=/workspace/output
LOGS=/workspace/logs

DIT_HIGH=$MODELS/wan2.2_i2v_high_noise_14B_fp16.safetensors
DIT_LOW=$MODELS/wan2.2_i2v_low_noise_14B_fp16.safetensors
VAE=$MODELS/Wan2.1_VAE.pth
T5=$MODELS/models_t5_umt5-xxl-enc-bf16.pth

# A6000 48GB: blocks_to_swap=0, no memory pressure
BLOCKS_TO_SWAP=0

# ============================================
# FUNCTIONS
# ============================================

cache_latents() {
    echo "=== Caching Latents - $(date) ==="
    cd $MUSUBI
    python src/musubi_tuner/wan_cache_latents.py \
        --dataset_config $DATASET_CONFIG \
        --vae $VAE \
        --i2v \
        --batch_size 1
}

cache_text() {
    echo "=== Caching Text Encoder - $(date) ==="
    cd $MUSUBI
    python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
        --dataset_config $DATASET_CONFIG \
        --t5 $T5 \
        --batch_size 4
}

train_high() {
    echo ""
    echo "=== Train High Noise (timesteps 900-1000) - $(date) ==="
    cd $MUSUBI
    accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
        src/musubi_tuner/wan_train_network.py \
        --task i2v-A14B \
        --dit $DIT_HIGH \
        --vae $VAE \
        --t5 $T5 \
        --dataset_config $DATASET_CONFIG \
        --network_module networks.lora_wan \
        --network_dim 16 --network_alpha 16 \
        --network_args "loraplus_lr_ratio=4" \
        --timestep_sampling shift --discrete_flow_shift 5.0 \
        --min_timestep 900 --max_timestep 1000 \
        --preserve_distribution_shape \
        --optimizer_type adamw8bit --learning_rate 2e-4 \
        --lr_scheduler cosine --max_train_epochs 20 --save_every_n_epochs 2 \
        --sdpa --mixed_precision fp16 --fp8_base --fp8_scaled \
        --gradient_checkpointing \
        ${BLOCKS_TO_SWAP:+--blocks_to_swap $BLOCKS_TO_SWAP} \
        --max_data_loader_n_workers 2 --persistent_data_loader_workers \
        --seed 42 \
        --output_dir $OUTPUT/high_noise \
        --output_name hardcut_high \
        --log_with tensorboard --logging_dir $LOGS
    echo "High noise training complete: $(date)"
}

train_low() {
    echo ""
    echo "=== Train Low Noise (timesteps 0-900) - $(date) ==="
    cd $MUSUBI
    accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
        src/musubi_tuner/wan_train_network.py \
        --task i2v-A14B \
        --dit $DIT_LOW \
        --vae $VAE \
        --t5 $T5 \
        --dataset_config $DATASET_CONFIG \
        --network_module networks.lora_wan \
        --network_dim 16 --network_alpha 16 \
        --network_args "loraplus_lr_ratio=4" \
        --timestep_sampling shift --discrete_flow_shift 5.0 \
        --min_timestep 0 --max_timestep 900 \
        --preserve_distribution_shape \
        --optimizer_type adamw8bit --learning_rate 1e-4 \
        --lr_scheduler cosine --max_train_epochs 20 --save_every_n_epochs 2 \
        --sdpa --mixed_precision fp16 --fp8_base --fp8_scaled \
        --gradient_checkpointing \
        ${BLOCKS_TO_SWAP:+--blocks_to_swap $BLOCKS_TO_SWAP} \
        --max_data_loader_n_workers 2 --persistent_data_loader_workers \
        --seed 42 \
        --output_dir $OUTPUT/low_noise \
        --output_name hardcut_low \
        --log_with tensorboard --logging_dir $LOGS
    echo "Low noise training complete: $(date)"
}

# ============================================
# MAIN
# ============================================

case "${1:-all}" in
    cache)
        cache_latents
        cache_text
        ;;
    train-high)
        train_high
        ;;
    train-low)
        train_low
        ;;
    train)
        train_high
        train_low
        ;;
    all)
        cache_latents
        cache_text
        train_high
        train_low
        ;;
    *)
        echo "Usage: $0 {cache|train-high|train-low|train|all}"
        exit 1
        ;;
esac

echo ""
echo "=== ALL DONE - $(date) ==="
