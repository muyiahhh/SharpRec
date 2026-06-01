# SharpRec

Official implementation of the KDD '26 paper:

> **Sharpness-aware Model Merging with Salience Recovery for LLM-based Cross-Domain Sequential Recommendation**  
> Huwei Ji, JiaJie Su, Yuyuan Li, Xiaohua Feng, Chaochao Chen  
> *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)*

## Overview

SharpRec tackles two fundamental bottlenecks that arise when merging domain-specific LLM adapters for cross-domain sequential recommendation:

| Bottleneck | Cause | SharpRec Module |
|------------|-------|-----------------|
| **B1** Cross-domain knowledge conflict | Geometric incompatibility of domain-specific adapters | **SGA** — Sharpness-aware Geometric Alignment |
| **B2** Performance saturation in multi-domain fusion | Statistical homogenization of merged parameters | **PSA** — Preference Salience Activation |

This repository uses the **Sport ↔ Toy** (Amazon Reviews 2023) pair as the running example.

## Repository Structure

```
SharpRec/
├── data/
│   ├── preprocess/          # Preprocessing pipeline scripts (_0–_6, prepareData)
│   ├── raw/                 # Place raw Amazon Review 2023 files here (if preprocessing from scratch)
│   └── processed/           # ✅ Included: traindata/ and testdata/ for Sport & Toy
├── result/
│   └── sport-toy/           # ✅ Included: pre-computed vLLM inference outputs
│       ├── sport/output/output.jsonl
│       └── toy/output/output.jsonl
├── models/
│   ├── meta-llama/          # Place Llama-2-7b-chat-hf here
│   └── ST/                  # Place downloaded adapters here (sport/, toy/, sport_toy/)
├── scripts/
│   ├── prepare_data.sh      # Step 1: Data preprocessing
│   ├── run_sga_train.sh     # Step 2: SGA adapter training
│   ├── run_psa_infer.sh     # Step 3: PSA inference pipeline
│   └── run_metric.sh        # Step 4: Evaluation
└── src/
    ├── main.py              # SGA training entry point
    ├── sam.py               # SAM optimizer
    ├── finetune.py          # Custom trainer with SAM integration
    ├── data_moudle.py       # Dataset and collator utilities
    ├── init_weights_robust.py
    ├── merge_weights.py
    ├── psa.py               # PSA post-fusion reparameterization
    ├── vllmtest.py          # vLLM inference
    └── metrics.py           # HR / NDCG / MRR evaluation
```

## Installation

```bash
git clone https://github.com/muyiahhh/SharpRec.git
cd SharpRec
pip install -r requirements.txt
```

The backbone used in all experiments is **Llama-2-7b-chat-hf**. Download it and place it under `models/meta-llama/Llama-2-7b-chat-hf/`, then set `BASE_MODEL_PATH` at the top of each script accordingly.

---

## Workflow — Sport ↔ Toy


### Step 1 — Data Preparation

Run the full preprocessing pipeline on the raw Amazon Reviews 2023 data:

**1. Download the raw data** from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/). You need four files:

| File | Type |
|------|------|
| `Sports_and_Outdoors.jsonl` | Review data |
| `Toys_and_Games.jsonl` | Review data |
| `meta_Sports_and_Outdoors.jsonl` | Item metadata |
| `meta_Toys_and_Games.jsonl` | Item metadata |

**2. Place all four files** into `data/preprocess/`:

```
data/preprocess/
├── Sports_and_Outdoors.jsonl
├── Toys_and_Games.jsonl
├── meta_Sports_and_Outdoors.jsonl
└── meta_Toys_and_Games.jsonl
```

**3. Run the preprocessing pipeline:**

```bash
bash scripts/prepare_data.sh
```

The pipeline runs six stages: JSONL→CSV conversion → interaction filtering → item and user filtering → cross-domain sequence construction → subsequence sampling → iterative k-core cleaning. Outputs are written to `data/processed/`.

> **Pre-processed data already included.** The processed Sport and Toy files are committed to this repository under `data/processed/traindata/` and `data/processed/testdata/`. You can skip this step entirely.

---

### Step 2 — SGA Training

Train three LoRA adapters (sport, toy, sport\_toy) with SAM optimization. Fine-tuning toward flat minima ensures geometric compatibility when the adapters are later merged (SGA).

```bash
bash scripts/run_sga_train.sh
```

Adapters are saved to `saft_output/sport/`, `saft_output/toy/`, and `saft_output/sport_toy/`.

> **Pre-trained adapters available.** To skip training, download our adapters from Google Drive and place them under `models/ST/`:[Google Drive](https://drive.google.com/drive/folders/1uco6LQYNbyG4FCUD-nYgKbb2rCtGMDfv?usp=drive_link). 
> Expected layout: `models/ST/sport/`, `models/ST/toy/`, `models/ST/sport_toy/`

---

### Step 3 — PSA Inference

Merge the trained adapters and apply Preference Salience Activation to recover heavy-tailed parameter distributions before inference (PSA).

```bash
bash scripts/run_psa_infer.sh
```

The pipeline runs four stages for each target domain:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `init_weights_robust.py` | Compute per-parameter fusion weights |
| 2 | `merge_weights.py` | Fuse adapters into a full merged model |
| 3 | `psa.py` | Apply PSA non-linear reparameterization to ΔW |
| 4 | `vllmtest.py` | vLLM inference on the PSA model |


Results are written to `result/sport-toy/{sport,toy}/output/`.

> **Pre-computed inference outputs already included.** The vLLM outputs for both domains are committed to this repository under `result/sport-toy/`. You can skip this step and proceed directly to evaluation.

---

### Step 4 — Evaluation

Compute HR@k, NDCG@k, and MRR@k on the inference outputs:

```bash
bash scripts/run_metric.sh
```

---

## Citation

```bibtex
@inproceedings{ji2026sharprec,
  title     = {Sharpness-aware Model Merging with Salience Recovery for LLM-based CDSR},
  author    = {Ji, Huwei and Su, JiaJie and Li, Yuyuan and Feng, Xiaohua and Chen, Chaochao},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  doi       = {10.1145/3770855.3817945}
}
```

---

## Acknowledgement

This codebase builds upon [WeaveRec](https://github.com/mertell/WeaveRec) (LLM-Based Cross-Domain Sequential Recommendation with Negative Transfer Mitigation). We thank the authors for releasing their code.
