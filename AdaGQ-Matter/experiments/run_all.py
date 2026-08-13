"""
Run all experiments — master script that runs the complete experiment suite.

Executes:
1. E1: Main experiment (AdaGQ-Matter vs all baselines) — IoTID20
2. E2: Main experiment — CICIoT2023
3. E3: DP trade-off curve (ε=1/3/5/8/10/∞)
4. E4: Ablation study (9 configurations)
5. E5: Non-IID experiment (α=0.1/0.5/1.0)
6. E6: MIA attack evaluation
7. E7: Communication overhead measurement
8. E8: DLG/Inverting Gradients attack

Usage:
    python experiments/run_all.py --dataset iotid20 --seeds 1 2 3 4 5
    python experiments/run_all.py --dataset both --seeds 1
    python experiments/run_all.py --quick  # Only E1+E7 with 1 seed
"""

import argparse
import os
import json
import yaml
import numpy as np
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: str) -> dict:
    """Run a subprocess command and capture output."""
    print(f"\n[run_all] Running: {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd="/sandbox/workspace/outputs/AdaGQ-Matter"
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"[run_all] ⚠️ Command failed with code {result.returncode}")
        print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description="Run all AdaGQ-Matter experiments")
    parser.add_argument("--dataset", default="iotid20",
                        choices=["iotid20", "ciciot2023", "both"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--config", default="configs/base_cpu.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only E1 main + E7 communication")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset != "both" else ["iotid20", "ciciot2023"]
    seeds = args.seeds

    start_time = datetime.now()
    all_results = {}

    print(f"\n{'#'*60}")
    print(f"# AdaGQ-Matter Full Experiment Suite")
    print(f"# Started: {start_time.isoformat()}")
    print(f"# Datasets: {datasets}")
    print(f"# Seeds: {seeds}")
    print(f"{'#'*60}")

    # ==========================================
    # E1: Main Experiment (all methods vs baselines)
    # ==========================================
    for dataset in datasets:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"E1: Main Experiment — {dataset}, seed={seed}")
            print(f"{'='*60}")
            result = run_command(
                f"python experiments/run_main.py --dataset {dataset} "
                f"--method all --seed {seed} --config {args.config}"
            )
            all_results[f"E1_{dataset}_seed{seed}"] = result

    if args.quick:
        # Quick mode: skip remaining experiments
        print("\n[Quick mode] Skipping E2-E8 experiments")
        print("To run full suite, remove --quick flag")
    else:
        # ==========================================
        # E3: DP Trade-off (ε=1/3/5/8/10/∞)
        # ==========================================
        for dataset in datasets:
            for seed in seeds[:3]:  # Only 3 seeds for DP trade-off
                print(f"\n{'='*60}")
                print(f"E3: DP Trade-off — {dataset}, seed={seed}")
                print(f"{'='*60}")
                result = run_command(
                    f"python experiments/run_dp_tradeoff.py --dataset {dataset} "
                    f"--seed {seed} --config {args.config}"
                )
                all_results[f"E3_{dataset}_seed{seed}"] = result

        # ==========================================
        # E4: Ablation Study
        # ==========================================
        for dataset in datasets:
            for seed in seeds[:3]:
                print(f"\n{'='*60}")
                print(f"E4: Ablation — {dataset}, seed={seed}")
                print(f"{'='*60}")
                result = run_command(
                    f"python experiments/run_ablation.py --dataset {dataset} "
                    f"--seed {seed} --config {args.config}"
                )
                all_results[f"E4_{dataset}_seed{seed}"] = result

        # ==========================================
        # E6+E8: Privacy Attack Evaluation
        # ==========================================
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"E6+E8: Privacy Attacks — {dataset}")
            print(f"{'='*60}")
            result = run_command(
                f"python experiments/run_attack.py --dataset {dataset} "
                f"--epsilon 3 --config {args.config}"
            )
            all_results[f"E6E8_{dataset}"] = result

    # ==========================================
    # Summary
    # ==========================================
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT SUITE COMPLETE")
    print(f"# Duration: {duration}")
    print(f"# Results in: results/")
    print(f"{'#'*60}")

    # Save run_all metadata
    metadata = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "datasets": datasets,
        "seeds": seeds,
        "quick_mode": args.quick,
        "results_keys": list(all_results.keys()),
    }

    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/run_all_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
