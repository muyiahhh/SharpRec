#!/bin/bash
# =============================================================
# Evaluation — Compute HR / NDCG / MRR (Step 4)
#
# Evaluates the inference outputs produced by run_psa_infer.sh
# against the ground-truth test files.
#
# Run from the project root:  bash scripts/run_metric.sh
# =============================================================

# =====================  Configuration  =======================

RESULT_DIR="result/sport-toy"
TEST_DATA_DIR="data/processed/testdata"

# =============================================================

echo "=================================================="
echo "Evaluating: Toy → Sport"
echo "=================================================="
python src/metrics.py \
    --res_path   "${RESULT_DIR}/sport/output" \
    --truth_file "${TEST_DATA_DIR}/sport.jsonl"

echo ""
echo "=================================================="
echo "Evaluating: Sport → Toy"
echo "=================================================="
python src/metrics.py \
    --res_path   "${RESULT_DIR}/toy/output" \
    --truth_file "${TEST_DATA_DIR}/toy.jsonl"

echo ""
echo "Evaluation complete."
