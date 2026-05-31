"""
PSA: Preference Salience Activation for SharpRec (KDD '26).
"""

import os
import json
import shutil
import torch
import argparse
from safetensors.torch import load_file, save_file


def psa_transform(delta: torch.Tensor, sigma: float, gamma: float,
                  alpha: float, beta: float) -> torch.Tensor:
    original_dtype = delta.dtype
    w = delta.to(torch.float32)

    # Step 1: Gaussian disentanglement — θ̃ = θ_merge - G
    if sigma > 0:
        w_tilde = w - torch.randn_like(w) * sigma
    else:
        w_tilde = w

    # Step 2: non-linear heavy-tail activation
    w_abs = w_tilde.abs() + 1e-8
    w_psa = w_tilde.sign() * (w_abs ** gamma) * (1.0 + alpha * torch.exp(-beta * w_abs))

    return w_psa.to(original_dtype)


def load_sharded_state_dict(model_dir: str):
    """
    Load a safetensors model directory (single-file or sharded) into a flat state dict.
    Returns (state_dict, weight_map) where weight_map maps param_name -> shard_filename.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        state_dict = {}
        for shard_file in shard_files:
            print(f"  Loading shard: {shard_file}")
            state_dict.update(load_file(os.path.join(model_dir, shard_file)))
        return state_dict, weight_map

    # Single-file fallback
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        state_dict = load_file(single)
        weight_map = {k: "model.safetensors" for k in state_dict}
        return state_dict, weight_map

    raise FileNotFoundError(f"No safetensors model found in {model_dir}")


def save_sharded_state_dict(state_dict: dict, weight_map: dict, output_dir: str):
    """Save state dict preserving the original shard structure from weight_map."""
    shards: dict[str, dict] = {}
    for key, shard_file in weight_map.items():
        if key in state_dict:
            shards.setdefault(shard_file, {})[key] = state_dict[key]

    for shard_file, tensors in shards.items():
        out_path = os.path.join(output_dir, shard_file)
        save_file(tensors, out_path)
        print(f"  Saved shard: {shard_file}")


def main():
    parser = argparse.ArgumentParser(
        description="PSA: Preference Salience Activation (SharpRec). "
                    "Post-fusion non-linear reparameterization of merged adapter weights."
    )
    parser.add_argument("--merged_model_path", required=True,
                        help="Full merged model directory (output of merge_weights.py)")
    parser.add_argument("--base_model_path", required=True,
                        help="Original base LLM directory (e.g., Llama-2-7b-chat-hf)")
    parser.add_argument("--output_model_path", required=True,
                        help="Directory to save the PSA-transformed model")
    parser.add_argument("--sigma", type=float, default=0.0001,
                        help="Gaussian noise std σ_g for disentanglement (paper default: 0.0001)")
    parser.add_argument("--gamma", type=float, default=0.98,
                        help="Tail-heaviness exponent γ ∈ (0, 1) (paper default: 0.98)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Smoothness coefficient α (paper default: 0.1)")
    parser.add_argument("--beta", type=float, default=10.0,
                        help="Decay coefficient β (paper default: 10)")
    parser.add_argument("--delta_threshold", type=float, default=1e-6,
                        help="Min peak |ΔW| to apply PSA; layers below threshold are unchanged")
    args = parser.parse_args()

    print("[PSA] Preference Salience Activation — SharpRec")
    print(f"  merged_model : {args.merged_model_path}")
    print(f"  base_model   : {args.base_model_path}")
    print(f"  output       : {args.output_model_path}")
    print(f"  sigma={args.sigma}, gamma={args.gamma}, alpha={args.alpha}, beta={args.beta}")

    # Copy merged model directory so config, tokenizer, index.json, etc. are preserved
    if os.path.exists(args.output_model_path):
        shutil.rmtree(args.output_model_path)
    shutil.copytree(args.merged_model_path, args.output_model_path)

    # Load both models into CPU memory
    print("[PSA] Loading merged model...")
    merged_sd, weight_map = load_sharded_state_dict(args.merged_model_path)

    print("[PSA] Loading base model...")
    base_sd, _ = load_sharded_state_dict(args.base_model_path)

    # Apply PSA to LoRA-modified parameters (those with non-trivial ΔW)
    psa_count, skip_count = 0, 0
    for key in merged_sd:
        if key not in base_sd:
            skip_count += 1
            continue

        delta = merged_sd[key].to(torch.float32) - base_sd[key].to(torch.float32)

        # Skip layers untouched by LoRA (embeddings, layernorms, etc.)
        if delta.abs().max().item() < args.delta_threshold:
            skip_count += 1
            continue

        psa_delta = psa_transform(delta, args.sigma, args.gamma, args.alpha, args.beta)
        merged_sd[key] = (base_sd[key].to(torch.float32) + psa_delta).to(merged_sd[key].dtype)
        psa_count += 1

    print(f"[PSA] Applied to {psa_count} tensors, skipped {skip_count} (unchanged / not in base).")

    # Overwrite the safetensors shards in the output directory
    print("[PSA] Saving output model...")
    save_sharded_state_dict(merged_sd, weight_map, args.output_model_path)

    print("[PSA] Done.")


if __name__ == "__main__":
    main()
