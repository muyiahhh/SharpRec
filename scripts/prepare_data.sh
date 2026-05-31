#!/bin/bash
# =============================================================
# Data Preprocessing Pipeline for SharpRec
#
# Run from the project root:  bash scripts/prepare_data.sh
#
# Input:  data/raw/review_categories/ and data/raw/meta_categories/
# Output: data/processed/traindata/ and data/processed/testdata/
# =============================================================

set -e

# All preprocessing scripts live here and use paths relative to this directory
PREPROCESS_DIR="data/preprocess"
cd "${PREPROCESS_DIR}"

echo "========= [Stage 0] Convert JSONL to CSV ========="
python3 prepareData.py

echo -e "\n========= [Stage 1] Interaction / Item / User Filtering ========="
# _0dataMain.py orchestrates _1dataFliter, _2itemFilter, _3userFliter internally
python3 _0dataMain.py

echo -e "\n========= [Stage 2] Build Cross-Domain Sequences ========="
python3 _4.py

echo -e "\n========= [Stage 3] Subsequence Sampling (Time Window) ========="
python3 _5.py

echo -e "\n========= [Stage 4] Iterative K-Core Cleaning ========="
python3 _6.py

cd ../..

echo -e "\n========= Preprocessing Complete ========="
echo "Processed data written to data/processed/"
