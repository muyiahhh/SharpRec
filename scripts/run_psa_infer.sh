#!/bin/bash
# =============================================================
# PSA Inference Pipeline — Preference Salience Activation (Step 3)
#
# Four-stage pipeline per target domain:
#   1. init_weights_robust.py — compute per-parameter fusion weights
#   2. merge_weights.py       — fuse adapters into a full merged model
#   3. psa.py                 — post-fusion non-linear reparameterization
#   4. vllmtest.py            — vLLM inference on the PSA model
#
# Run from the project root:  bash scripts/run_psa_infer.sh
# Output: result/sport-toy/{sport,toy}/output/
# =============================================================

# =====================  Configuration  =======================

# Adapter paths (outputs of run_sga_train.sh)
ADAPTER_SPORT="models/ST/sport"
ADAPTER_TOY="models/ST/toy"
ADAPTER_HYBRID="models/ST/sport_toy"

# Base LLM (needed by PSA to compute ΔW = W_merged − W_base)
BASE_MODEL_PATH="models/meta-llama/Llama-2-7b-chat-hf"

# Merging function — softmax-weighted average (unchanged from WeaveRec)
MERGE_FUNC="softmax"
MERGE_SOFTMAX_T=1

# PSA hyperparameters (paper defaults, Section 5.1.4)
PSA_SIGMA=0.0001   # σ_g: Gaussian noise std for disentanglement
PSA_GAMMA=0.98     # γ:   tail-heaviness exponent
PSA_ALPHA=0.1      # α:   smoothness coefficient
PSA_BETA=10        # β:   decay coefficient

# Result root
RESULT_DIR="result/sport-toy"
mkdir -p "${RESULT_DIR}"

# =============================================================

# Helper: run the full 4-stage pipeline for one target domain.
#
# Arguments:
#   $1  Human-readable scenario name
#   $2  Adapter list string (Python list literal)
#   $3  Test data file
#   $4  Sub-directory name under RESULT_DIR
#   $5  vLLM inference batch size
run_scenario() {
    local scenario_name="$1"
    local adapter_list="$2"
    local test_file="$3"
    local work_dir="${RESULT_DIR}/$4"
    local vllm_batch="$5"

    echo ""
    echo "=================================================="
    echo "Scenario: ${scenario_name}"
    echo "=================================================="

    mkdir -p "${work_dir}"

    local merging_weights_pth="${work_dir}/merging_weights.pth"
    local temp_merged_model="${work_dir}/merged_model"
    local temp_psa_model="${work_dir}/psa_model"
    local output_dir="${work_dir}/output"
    mkdir -p "${output_dir}"

    # --- Stage 1: compute robust per-parameter merging weights ---
    echo "[Stage 1] Computing merging weights..."
    python src/init_weights_robust.py \
        --adapter_list "${adapter_list}" \
        --output_path  "${merging_weights_pth}"

    # --- Stage 2: fuse adapters into a full merged model ---
    echo "[Stage 2] Fusing adapters..."
    python src/merge_weights.py \
        --adapter_list    "${adapter_list}" \
        --weight_path     "${merging_weights_pth}" \
        --func            "${MERGE_FUNC}" \
        --softmax_t       "${MERGE_SOFTMAX_T}" \
        --base_model_path "${BASE_MODEL_PATH}"

    # merge_weights.py names the output directory after the .pth file (strip suffix)
    local raw_merged_dir="${work_dir}/merging_weights"
    if [ ! -d "${raw_merged_dir}" ]; then
        echo "Error: expected merged model directory not found: ${raw_merged_dir}"
        exit 1
    fi
    mv "${raw_merged_dir}" "${temp_merged_model}"

    # --- Stage 3: Preference Salience Activation ---
    echo "[Stage 3] Applying PSA..."
    python src/psa.py \
        --merged_model_path "${temp_merged_model}" \
        --base_model_path   "${BASE_MODEL_PATH}" \
        --output_model_path "${temp_psa_model}" \
        --sigma "${PSA_SIGMA}" \
        --gamma "${PSA_GAMMA}" \
        --alpha "${PSA_ALPHA}" \
        --beta  "${PSA_BETA}"

    # --- Stage 4: vLLM inference on the PSA-transformed model ---
    echo "[Stage 4] Running inference..."
    CUDA_VISIBLE_DEVICES=0 python src/vllmtest.py \
        --outpath "${output_dir}" \
        --outname output \
        --input   "${test_file}" \
        --memory  0.9 \
        --batch   "${vllm_batch}" \
        --max_new_tokens 512 \
        --model "${temp_psa_model}"

    # Clean up large intermediate files to free disk space
    rm -rf "${temp_merged_model}"
    rm -rf "${temp_psa_model}"
    rm -f  "${merging_weights_pth}"

    echo "Inference complete. Results saved to: ${output_dir}"
}

# ==================== Run Scenarios ====================

# Scenario A: predict on Sport domain, assisted by Toy knowledge
run_scenario \
    "Sport (enhanced by Toy)" \
    "['${ADAPTER_SPORT}', '${ADAPTER_HYBRID}']" \
    "data/processed/testdata/sport.jsonl" \
    "sport" \
    50

# Scenario B: predict on Toy domain, assisted by Sport knowledge
run_scenario \
    "Toy (enhanced by Sport)" \
    "['${ADAPTER_TOY}', '${ADAPTER_HYBRID}']" \
    "data/processed/testdata/toy.jsonl" \
    "toy" \
    32

echo ""
echo "All inference tasks finished!"
echo "Run 'bash scripts/run_metric.sh' to compute evaluation metrics."
