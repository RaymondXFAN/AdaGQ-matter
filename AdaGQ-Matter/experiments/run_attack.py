"""
Privacy attack evaluation runner — MIA + DLG + Inverting Gradients.

Runs all three attack types on trained models:
1. Membership Inference Attack (MIA)
2. Deep Leakage from Gradients (DLG)
3. Inverting Gradients Attack

Usage:
    python experiments/run_attack.py --dataset iotid20 --seed 1 --epsilon 3
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

from experiments.run_main import load_config, load_processed_data
from models.factory import create_model
from attacks.mia import MembershipInferenceAttack, compute_mia_auc
from attacks.dlg import DeepLeakageAttack, compute_reconstruction_mse
from attacks.invgrad import InvertingGradientsAttack


def main():
    parser = argparse.ArgumentParser(description="Privacy attack evaluation")
    parser.add_argument("--dataset", default="iotid20",
                        choices=["iotid20", "ciciot2023"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=3.0)
    parser.add_argument("--config", default="configs/base_cpu.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"PRIVACY ATTACK EVALUATION — Dataset: {args.dataset}, ε={args.epsilon}")
    print(f"{'='*60}")

    # Load data
    data_dir = config.get("data_dir", "data/processed")
    X_train, y_train, X_test, y_test, partitions = load_processed_data(data_dir, args.dataset)

    # Create model
    model = create_model(config)

    # --- MIA Attack ---
    print("\n--- Membership Inference Attack ---")
    mia_attack = MembershipInferenceAttack()

    # Split train data into member/non-member
    n_member = int(len(X_train) * 0.5)
    member_data = X_train[:n_member]
    non_member_data = X_train[n_member:]

    # Train attack model
    mia_attack.train_attack_model(
        member_data=member_data,
        non_member_data=non_member_data,
        target_model=model,
    )

    # Evaluate
    mia_results = mia_attack.evaluate(
        target_model=model,
        data=X_test,
        labels=y_test,
    )

    print(f"  MIA Accuracy: {mia_results['accuracy']:.4f}")
    print(f"  MIA AUC-ROC: {mia_results['auc']:.4f}")

    # --- DLG Attack ---
    print("\n--- Deep Leakage from Gradients ---")
    dlg_attack = DeepLeakageAttack()

    # Create dummy sample for reconstruction
    sample_idx = 0
    original_data = X_train[sample_idx:sample_idx + 1]
    original_label = y_train[sample_idx]

    # Compute gradient from original data
    model.eval()
    X_tensor = torch.tensor(original_data, dtype=torch.float32)
    y_tensor = torch.tensor([original_label], dtype=torch.int64)

    model.zero_grad()
    output = model(X_tensor)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, y_tensor)
    loss.backward()

    # Full gradient (FedAvg scenario)
    full_gradient = model.get_gradients_flat().detach().cpu().numpy()
    reconstructed_full, _ = dlg_attack.reconstruct(
        dummy_data=torch.randn(1, config.get("input_dim_iotid20", 79)),
        observed_gradient=torch.tensor(full_gradient),
        model=model,
        n_iter=300,
    )
    mse_full = compute_reconstruction_mse(X_tensor.detach().cpu().numpy(), reconstructed_full)

    # Sparse gradient (AdaGQ-Matter scenario)
    from core.compression import top_k_sparsify
    sparse_values, sparse_indices, mask = top_k_sparsify(full_gradient, kappa=0.2)
    reconstructed_sparse, _ = dlg_attack.reconstruct_sparse(
        dummy_data=torch.randn(1, config.get("input_dim_iotid20", 79)),
        sparse_values=sparse_values,
        sparse_indices=sparse_indices,
        model=model,
        n_iter=300,
    )
    mse_sparse = compute_reconstruction_mse(X_tensor.detach().cpu().numpy(), reconstructed_sparse)

    print(f"  DLG MSE (FedAvg, full gradient): {mse_full:.4f}")
    print(f"  DLG MSE (AdaGQ, sparse gradient): {mse_sparse:.4f}")

    # --- Inverting Gradients Attack ---
    print("\n--- Inverting Gradients Attack ---")
    inv_attack = InvertingGradientsAttack()

    reconstructed_inv_full, _ = inv_attack.reconstruct(
        dummy_data=torch.randn(1, config.get("input_dim_iotid20", 79)),
        observed_gradient=torch.tensor(full_gradient),
        model=model,
        n_iter=300,
    )
    mse_inv_full = compute_reconstruction_mse(X_tensor.detach().cpu().numpy(), reconstructed_inv_full)

    reconstructed_inv_sparse, _ = inv_attack.reconstruct_sparse(
        dummy_data=torch.randn(1, config.get("input_dim_iotid20", 79)),
        sparse_values=sparse_values,
        sparse_indices=sparse_indices,
        model=model,
        n_iter=300,
    )
    mse_inv_sparse = compute_reconstruction_mse(X_tensor.detach().cpu().numpy(), reconstructed_inv_sparse)

    print(f"  InvGrad MSE (FedAvg, full gradient): {mse_inv_full:.4f}")
    print(f"  InvGrad MSE (AdaGQ, sparse gradient): {mse_inv_sparse:.4f}")

    # Save results
    attack_results = {
        "dataset": args.dataset,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "mia": mia_results,
        "dlg": {
            "mse_full": mse_full,
            "mse_sparse": mse_sparse,
        },
        "inverting_gradients": {
            "mse_full": mse_inv_full,
            "mse_sparse": mse_inv_sparse,
        },
    }

    results_dir = config.get("results_dir", "results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    output_path = os.path.join(results_dir,
                               f"attacks_{args.dataset}_eps{args.epsilon}_seed{args.seed}.json")
    with open(output_path, "w") as f:
        json.dump(attack_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
