# SharpRec

Official implementation of the KDD '26 paper:

> **Sharpness-aware Model Merging with Salience Recovery for LLM-based Cross-Domain Sequential Recommendation**  
> Huwei Ji, JiaJie Su, Yuyuan Li, Xiaohua Feng, Chaochao Chen  
> *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)*

## Overview

SharpRec tackles two fundamental bottlenecks that limit existing LLM-based CDSR model-merging approaches:

| Bottleneck | Cause | SharpRec Module |
|------------|-------|-----------------|
| **B1** Cross-domain knowledge conflict | Geometric incompatibility of domain-specific adapters | **SGA** — Sharpness-aware Geometric Alignment |
| **B2** Performance saturation in multi-domain fusion | Statistical homogenization of merged parameters | **PSA** — Preference Salience Activation |

## Repository Structure

```
SharpRec/
├── data/
│   ├── preprocess/          # Preprocessing pipeline scripts (_0–_6, prepareData)
│   ├── raw/                 # Place raw Amazon Review 2023 files here
│   └── processed/           # Output: traindata/ and testdata/
├── scripts/
│   ├── prepare_data.sh      # Step 1: Data preprocessing
│   ├── run_sga_train.sh     # Step 2: SGA training
│   ├── run_psa_infer.sh     # Step 3: PSA inference
│   └── run_metric.sh        # Step 4: Evaluation
└── src/
    ├── main.py              # SGA training entry point
    ├── sam.py               # SAM optimizer
    ├── finetune.py          # Custom trainer with SAM integration
    ├── data_moudle.py       # Dataset and collator utilities
    ├── init_weights_robust.py
    ├── merge_weights.py
    ├── psa.py               # PSA post-processing
    ├── vllmtest.py          # vLLM inference
    └── metrics.py           # HR / NDCG / MRR evaluation
```

## Installation

```bash
git clone https://github.com/muyiahhh/SharpRec.git
cd SharpRec
pip install -r requirements.txt
```

The backbone model used in all experiments is **Llama-2-7b-chat-hf**. Set its local path via the `BASE_MODEL_PATH` variable at the top of each script.

---

## Workflow — Sport ↔ Toy Example

All scripts are designed to be run from the **project root**.

### Step 1 — Data Preparation

**Option A: Use our pre-processed Sport-Toy data (recommended)**

Download and place the files as follows:

```
data/processed/traindata/sport.jsonl
data/processed/traindata/toy.jsonl
data/processed/traindata/sport_toy.jsonl
data/processed/testdata/sport.jsonl
data/processed/testdata/toy.jsonl
```

> **Download link**: [TODO — Google Drive](#)

Then skip to Step 2.

**Option B: Preprocess from scratch**

1. Download the `review_Sports_and_Outdoors.jsonl` and `review_Toys_and_Games.jsonl` (with their corresponding `meta_*.jsonl` files) from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).

2. Place them under `data/raw/review_categories/` and `data/raw/meta_categories/`.

3. Run the full preprocessing pipeline:

```bash
bash scripts/prepare_data.sh
```

The pipeline runs six stages: JSONL→CSV conversion → interaction filtering → item and user filtering → cross-domain sequence construction → subsequence sampling → iterative k-core cleaning. Final outputs are written to `data/processed/`.

---

### Step 2 — SGA Training

Train three LoRA adapters (sport, toy, sport\_toy) with SAM optimization. Fine-tuning toward flat minima establishes a stable geometric foundation that prevents parameter interference during merging (SGA, Section 4.2).

**Option A: Use our pre-trained adapters (recommended)**

> **Download link**: [TODO — Google Drive](#)

Place the downloaded adapters under `saft_output/sport/`, `saft_output/toy/`, and `saft_output/sport_toy/`, then skip to Step 3.

**Option B: Train from scratch**

```bash
bash scripts/run_sga_train.sh
```

Key hyperparameters (paper Section 5.1.4):

| Parameter | Value |
|-----------|-------|
| Backbone | Llama-2-7b-chat-hf |
| Epochs | 2 |
| Learning rate | 2×10⁻⁴ |
| LoRA rank / alpha | 16 / 32 |
| SAM perturbation ρ | 0.01 |
| Effective batch size | 8 |

Training produces three adapters saved to `saft_output/`.

---

### Step 3 — PSA Inference

Merge the domain-specific adapters and apply Preference Salience Activation to recover heavy-tailed parameter distributions before inference (PSA, Section 4.3).

**Option A: Use our pre-computed inference outputs**

> **Download link**: [TODO — Google Drive](#)

Place the downloaded outputs under `result/sport-toy/sport/output/` and `result/sport-toy/toy/output/`, then skip to Step 4.

**Option B: Run inference yourself**

Configure the adapter paths at the top of `scripts/run_psa_infer.sh`, then:

```bash
bash scripts/run_psa_infer.sh
```

The pipeline runs the following four stages for each target domain:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `init_weights_robust.py` | Compute per-parameter fusion weights |
| 2 | `merge_weights.py` | Fuse adapters into a full merged model |
| 3 | `psa.py` | Apply PSA non-linear reparameterization |
| 4 | `vllmtest.py` | vLLM inference on the PSA model |

PSA hyperparameters (paper defaults):

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Gaussian noise std | σ_g | 0.0001 |
| Tail-heaviness exponent | γ | 0.98 |
| Smoothness coefficient | α | 0.1 |
| Decay coefficient | β | 10 |

---

### Step 4 — Evaluation

Compute HR@k, NDCG@k, and MRR@k on the inference outputs from Step 3:

```bash
bash scripts/run_metric.sh
```

Expected results on Sport ↔ Toy (Table 2 in the paper, expressed as %):

| Task | HR@3 | NDCG@3 | HR@5 | NDCG@5 | MRR |
|------|------|--------|------|--------|-----|
| Toy → Sport | 76.14 | 74.26 | 78.78 | 75.34 | 74.21 |
| Sport → Toy | 66.34 | 63.88 | 71.66 | 66.05 | 64.23 |

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
