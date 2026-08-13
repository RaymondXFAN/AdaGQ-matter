"""
Ablation experiment runner — tests each component's contribution.

Ablation configurations from configs/ablation.yaml:
- Full: All components active (baseline)
- No EC: No error compensation
- No Quant: κ=0.2, FP32 values (no quantization)
- No DP: No differential privacy (ε=∞)
- No Adaptive κ: Fixed κ=0.2
- No Adaptive W_agg: Fixed 500ms window
- Naive Combination: Top-k + QSGD serial (no co-optimization)
- No Shuffling: No random permutation privacy amplification
- No Feature Grouping: No Matter-aware priority boost

Usage:
    python experiments/run_ablation.py --dataset iotid20 --seed 1
    python experiments/run_ablation.py --config configs/ablation.yaml
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# --- Fix Python import path for package-level imports ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.run_main import load_config, load_processed_data, run_single_experiment


def load_ablation_config(config_path: str) -> dict:
    """Load ablation YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_ablation_experiment(
    base_config: dict,
    ablation_name: str,
    ablation_overrides: dict,
    dataset: str,
    seed: int,
) -> dict:
    """
    Run one ablation experiment by modifying base config.

    Args:
        base_config: Base configuration dict
        ablation_name: Name of the ablation variant
        ablation_overrides: Config overrides for this ablation
        dataset: Dataset name
        seed: Random seed

    Returns:
        Results dict
    """
    # Create modified config for this ablation
    modified_config = base_config.copy()

    # Apply overrides
    for key, value in ablation_overrides.items():
        if key == "epsilon" and value == "infinity":
            modified_config[key] = float("inf")
            modified_config["dp_noise_multiplier"] = 0.0
        elif key == "b_default" and value == 32:
            modified_config[key] = 32
            modified_config["b_min"] = 32
            modified_config["b_max"] = 32
        else:
            modified_config[key] = value

    # Run experiment
    alpha = base_config.get("alpha", 0.5)
    result = run_single_experiment(modified_config, dataset, "adagq", seed, alpha)

    # Add ablation metadata
    result["ablation_name"] = ablation_name
    result["ablation_overrides"] = ablation_overrides

    return result


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--dataset", default="iotid20",
                        choices=["iotid20", "ciciot2023"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--base_config", default="configs/base_cpu.yaml")
    parser.add_argument("--ablation_config", default="configs/ablation.yaml")
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    ablation_config = load_ablation_config(args.ablation_config)

    # Run all ablation variants
    ablation_configs = ablation_config.get("ablation_configs", {})
    all_results = {}

    print(f"\n{'='*60}")
    print(f"ABLATION EXPERIMENTS — Dataset: {args.dataset}, Seed: {args.seed}")
    print(f"{'='*60}")

    for name, overrides in ablation_configs.items():
        description = overrides.get("description", name)
        print(f"\n--- Running: {name} ({description}) ---")

        result = run_ablation_experiment(
            base_config, name, overrides, args.dataset, args.seed
        )
        all_results[name] = result

    # Summary table
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'F1':>8} {'MIA%':>8} {'KB/Round':>10}")
    print("-" * 58)

    for name, result in all_results.items():
        fm = result["final_metrics"]
        f1 = fm.get("f1", 0.0)
        comm = fm.get("avg_comm_kb", 0.0)
        # MIA would be computed separately; placeholder here
        mia = "—"
        print(f"{name:<30} {f1:>8.4f} {mia:>8} {comm:>10.1f}")

    # Save results
    results_dir = base_config.get("results_dir", "results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    output_path = os.path.join(results_dir, f"ablation_{args.dataset}_seed{args.seed}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
