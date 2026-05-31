#!/bin/bash
# =============================================================
# SGA Training — Sharpness-aware Geometric Alignment (Step 2)
#
# Trains three LoRA adapters (sport, toy, sport_toy) using the
# SAM optimizer, guiding each adapter toward flat minima so they
# can be merged without geometric parameter interference.
#
# Run from the project root:  bash scripts/run_sga_train.sh
# Output: models/ST/{sport,toy,sport_toy}/
# =============================================================

# Prevent CUDA allocator fragmentation on long training runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# =====================  Configuration  =======================

BASE_MODEL_PATH="models/meta-llama/Llama-2-7b-chat-hf"
TRAIN_DATA_DIR="data/processed/traindata"
OUTPUT_DIR="models/ST"

# SAM perturbation radius (ρ)
RHO=0.01

# LoRA fine-tuning hyperparameters
EPOCHS=2
BATCH_SIZE=4
MAX_LENGTH=1200
LR=2e-4

# =============================================================

# Main.py dynamically computes gradient accumulation steps so that
# the effective batch size stays fixed at 8 regardless of GPU count:
#   accum_steps = target_global_bs // (world_size * batch_size_per_gpu)
# On a single GPU this yields accum_steps = 2.

# --- Sport adapter ---
echo "=================================================="
echo "[1/3] Training Sport adapter..."
echo "=================================================="
accelerate launch --num_processes 1 --main_process_port 29500 src/main.py \
    --model_name  "${BASE_MODEL_PATH}" \
    --data_path   "${TRAIN_DATA_DIR}/sport.jsonl" \
    --output_dir  "${OUTPUT_DIR}/sport" \
    --use_sam \
    --rho         ${RHO} \
    --epochs      ${EPOCHS} \
    --batch_size  ${BATCH_SIZE} \
    --max_length  ${MAX_LENGTH} \
    --lr          ${LR}

# --- Toy adapter ---
echo ""
echo "=================================================="
echo "[2/3] Training Toy adapter..."
echo "=================================================="
accelerate launch --num_processes 1 --main_process_port 29501 src/main.py \
    --model_name  "${BASE_MODEL_PATH}" \
    --data_path   "${TRAIN_DATA_DIR}/toy.jsonl" \
    --output_dir  "${OUTPUT_DIR}/toy" \
    --use_sam \
    --rho         ${RHO} \
    --epochs      ${EPOCHS} \
    --batch_size  ${BATCH_SIZE} \
    --max_length  ${MAX_LENGTH} \
    --lr          ${LR}

# --- Sport+Toy hybrid adapter ---
echo ""
echo "=================================================="
echo "[3/3] Training Sport+Toy hybrid adapter..."
echo "=================================================="
accelerate launch --num_processes 1 --main_process_port 29502 src/main.py \
    --model_name  "${BASE_MODEL_PATH}" \
    --data_path   "${TRAIN_DATA_DIR}/sport_toy.jsonl" \
    --output_dir  "${OUTPUT_DIR}/sport_toy" \
    --use_sam \
    --rho         ${RHO} \
    --epochs      ${EPOCHS} \
    --batch_size  ${BATCH_SIZE} \
    --max_length  ${MAX_LENGTH} \
    --lr          ${LR}

echo ""
echo "SGA training complete. Adapters saved to ${OUTPUT_DIR}/"
