"""
Classification Evaluation Metrics Module.

Computes standard classification metrics for IoT anomaly detection:
- Cross-Entropy loss
- Accuracy, Precision, Recall, F1 (macro)
- AUC (ROC-AUC for binary, multi-class AUC for multi-class)

Supports both binary and multi-class scenarios, consistent with
AnomalyDNN (output_dim=2 for binary anomaly detection).

Reference: AdaGQ-Matter, Section 5 (Evaluation Protocol)
"""

import numpy as np
from typing import Dict, Optional, Union

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


# ============================================================
# Core Metrics Computation
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    n_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (1D array of ints).
        y_pred: Predicted labels (1D array of ints).
        y_prob: Predicted probabilities (2D array, shape [n_samples, n_classes]).
                Required for computing loss and AUC.
        n_classes: Number of classes (2 for binary, >2 for multi-class).

    Returns:
        Dict with keys: loss, accuracy, precision, recall, f1, auc.
        - loss: Cross-Entropy loss (requires y_prob)
        - accuracy: Classification accuracy
        - precision: Macro-averaged precision
        - recall: Macro-averaged recall
        - f1: Macro-averaged F1 score
        - auc: ROC-AUC (binary: single score; multi-class: averaged over classes)
    """
    metrics: Dict[str, float] = {}

    # --- Accuracy ---
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # --- Precision, Recall, F1 (macro) ---
    average = "macro" if n_classes > 2 else "binary"
    metrics["precision"] = precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["f1"] = f1_score(
        y_true, y_pred, average=average, zero_division=0
    )

    # --- Cross-Entropy Loss ---
    if y_prob is not None:
        metrics["loss"] = log_loss(y_true, y_prob, labels=list(range(n_classes)))

        # --- AUC ---
        if n_classes == 2:
            # Binary: use probability of positive class
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Multi-class: use OvR macro-average AUC
            try:
                metrics["auc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
            except ValueError:
                # If some classes are missing in y_true, AUC is undefined
                metrics["auc"] = 0.0
    else:
        # Without probabilities, loss and AUC cannot be computed
        metrics["loss"] = float("nan")
        metrics["auc"] = float("nan")

    return metrics


# ============================================================
# Metric Formatting
# ============================================================

def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics dict into a readable string for logging.

    Args:
        metrics: Dict from compute_metrics()
        prefix: Optional prefix for each line (e.g., "[Round 5]")

    Returns:
        Multi-line formatted string
    """
    lines = []
    for key in ["loss", "accuracy", "precision", "recall", "f1", "auc"]:
        val = metrics.get(key, float("nan"))
        if prefix:
            lines.append(f"{prefix} {key}: {val:.4f}")
        else:
            lines.append(f"{key}: {val:.4f}")
    return "\n".join(lines)


# ============================================================
# Metric Aggregation Across Clients
# ============================================================

def aggregate_client_metrics(
    client_metrics_list: list[Dict[str, float]],
    weights: Optional[list[int]] = None,
) -> Dict[str, float]:
    """
    Weighted average of metrics from multiple FL clients.

    Args:
        client_metrics_list: List of dicts from compute_metrics()
        weights: Sample counts per client (for weighted averaging).
                 If None, uniform weighting.

    Returns:
        Weighted-average metrics dict
    """
    if weights is None:
        weights = [1] * len(client_metrics_list)

    total_weight = sum(weights)
    aggregated: Dict[str, float] = {}

    for key in ["loss", "accuracy", "precision", "recall", "f1", "auc"]:
        weighted_sum = 0.0
        for m, w in zip(client_metrics_list, weights):
            val = m.get(key, 0.0)
            if not np.isnan(val):
                weighted_sum += val * w
        aggregated[key] = weighted_sum / total_weight

    return aggregated


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":
    print("=== Binary Classification Test ===")
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.dirichlet([1, 1], size=n)  # Random probabilities for 2 classes
    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_metrics(y_true, y_pred, y_prob, n_classes=2)
    print(format_metrics(metrics))

    print("\n=== Multi-Class Classification Test ===")
    n_classes = 5
    y_true_mc = rng.integers(0, n_classes, size=n)
    y_prob_mc = rng.dirichlet(np.ones(n_classes), size=n)
    y_pred_mc = np.argmax(y_prob_mc, axis=1)

    metrics_mc = compute_metrics(y_true_mc, y_pred_mc, y_prob_mc, n_classes=n_classes)
    print(format_metrics(metrics_mc))

    print("\n=== Client Metrics Aggregation Test ===")
    client_metrics = [
        compute_metrics(y_true, y_pred, y_prob, n_classes=2),
        compute_metrics(y_true, y_pred, y_prob, n_classes=2),
    ]
    weights = [80, 20]
    agg = aggregate_client_metrics(client_metrics, weights)
    print(format_metrics(agg, prefix="[Aggregated]"))

    print("\n=== Without Probabilities Test ===")
    metrics_no_prob = compute_metrics(y_true, y_pred, y_prob=None, n_classes=2)
    print(format_metrics(metrics_no_prob))
