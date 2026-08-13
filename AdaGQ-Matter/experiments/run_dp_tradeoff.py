"""
DP trade-off experiment runner — tests ε vs F1 curve.

Runs experiments at ε = 1, 3, 5, 8, 10, ∞ (no DP)
to generate the privacy-utility trade-off curve.

Usage:
    python experiments/run_dp_tradeoff.py --dataset iotid20 --seed 1
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np
import torch
from pathlib import Path

# --- Fix Python import path for package-level imports ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.run_main import load_config, run_single_experiment


def main():
    parser = argparse.ArgumentParser(description="DP trade-off experiments")
    parser.add_argument("--dataset", default="iotid20",
                        choices=["iotid20", "ciciot2023"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--config", default="configs/base_cpu.yaml")
    parser.add_argument("--epsilon_list", nargs="+", type=float,
                        default=[1, 3, 5, 8, 10])
    args = parser.parse_args()

    config = load_config(args.config)

    epsilon_values = args.epsilon_list + [float("inf")]  # Add no-DP baseline

    print(f"\n{'='*60}")
    print(f"DP TRADE-OFF — Dataset: {args.dataset}, Seed: {args.seed}")
    print(f"{'='*60}")

    results = {}

    for epsilon in epsilon_values:
        eps_str = "inf" if epsilon == float("inf") else str(epsilon)
        print(f"\n--- ε = {eps_str} ---")

        modified_config = config.copy()
        modified_config["epsilon"] = epsilon

        if epsilon == float("inf"):
            modified_config["dp_noise_multiplier"] = 0.0

        # Run AdaGQ-Matter
        result = run_single_experiment(
            modified_config, args.dataset, "adagq", args.seed, args.alpha
        )
        results[eps_str] = result

        # Also run FedAvg+DP for comparison
        if epsilon != float("inf"):
            result_fedavg_dp = run_single_experiment(
                modified_config, args.dataset, "dp_fedavg", args.seed, args.alpha
            )
            results[f"fedavg_dp_eps{eps_str}"] = result_fedavg_dp

    # Print trade-off table
    print(f"\n{'='*60}")
    print("DP TRADE-OFF SUMMARY")
    print(f"{'='*60}")
    print(f"{'ε':>6} {'AdaGQ F1':>10} {'FedAvg+DP F1':>12} {'ΔF1':>8}")
    print("-" * 40)

    for epsilon in args.epsilon_list:
        eps_str = str(epsilon)
        adagq_f1 = results.get(eps_str, {}).get("final_metrics", {}).get("f1", 0.0)
        fedavg_dp_f1 = results.get(f"fedavg_dp_eps{eps_str}", {}).get("final_metrics", {}).get("f1", 0.0)
        delta_f1 = adagq_f1 - fedavg_dp_f1
        print(f"{eps_str:>6} {adagq_f1:>10.4f} {fedavg_dp_f1:>12.4f} {delta_f1:>8.2f}")

    # Save
    results_dir = config.get("results_dir", "results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(results_dir, f"dp_tradeoff_{args.dataset}_seed{args.seed}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
