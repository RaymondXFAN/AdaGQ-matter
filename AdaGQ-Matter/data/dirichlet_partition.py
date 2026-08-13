"""
Dirichlet partition — create Non-IID data partitions for FL clients.

Implements Dirichlet-based data partitioning where each client gets
a heterogeneous data distribution controlled by concentration parameter α:
- α → ∞: IID distribution
- α = 1.0: moderate heterogeneity
- α = 0.5: moderate Non-IID
- α = 0.1: strong Non-IID (highly heterogeneous)
- α → 0: each client has only one class

Reference: Hsu et al., "Measuring the Effects of Non-Identical Data Distribution
for Federated Visual Object Classification", 2019.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple


def dirichlet_partition(
    labels: np.ndarray,
    N: int = 10,
    alpha: float = 0.5,
    seed: int = 1,
) -> Dict[int, List[int]]:
    """
    Partition data indices among N clients using Dirichlet distribution.

    Args:
        labels: Array of class labels (0/1 for binary, or multi-class)
        N: Number of clients
        alpha: Dirichlet concentration parameter
        seed: Random seed for reproducibility

    Returns:
        Dict mapping client_id → list of data indices assigned to that client

    Raises:
        ValueError: If any client gets zero data points
    """
    rng = np.random.default_rng(seed)
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # For each class, draw Dirichlet proportions for N clients
    client_indices: Dict[int, List[int]] = {i: [] for i in range(N)}

    for c in unique_labels:
        # Get all indices belonging to this class
        class_indices = np.where(labels == c)[0]
        n_class = len(class_indices)

        # Draw Dirichlet distribution for this class
        proportions = rng.dirichlet([alpha] * N)

        # Proportionally assign indices
        # Split class_indices according to proportions
        counts = (proportions * n_class).astype(int)

        # Handle rounding: give remaining samples to the largest-proportion client
        remainder = n_class - counts.sum()
        if remainder > 0:
            # Assign remainder to clients with largest proportions
            largest_clients = np.argsort(proportions)[-remainder:]
            for lc in largest_clients:
                counts[lc] += 1

        # Distribute indices
        perm = rng.permutation(class_indices)  # Shuffle within class
        offset = 0
        for i in range(N):
            client_indices[i].extend(perm[offset:offset + counts[i]].tolist())
            offset += counts[i]

    # Validate: no empty client
    for i, indices in client_indices.items():
        if len(indices) == 0:
            # Re-distribute: take 1 sample from the richest client
            richest = max(client_indices.keys(), key=lambda k: len(client_indices[k]))
            if len(client_indices[richest]) > 1:
                client_indices[i] = [client_indices[richest].pop()]
            else:
                raise ValueError(
                    f"Client {i} has 0 data points and no client has surplus. "
                    f"Try increasing N or decreasing α."
                )

    # Shuffle each client's indices for training order
    for i in client_indices:
        client_indices[i] = rng.permutation(client_indices[i]).tolist()

    # Print statistics
    total = sum(len(v) for v in client_indices.values())
    print(f"[Dirichlet] α={alpha}, N={N}, n_samples={total}")
    for i, indices in client_indices.items():
        client_labels = labels[indices]
        label_counts = {
            int(c): int((client_labels == c).sum()) for c in unique_labels
        }
        print(f"  Client {i}: {len(indices)} samples, label_dist={label_counts}")

    return client_indices


def save_partitions(
    partitions: Dict[int, List[int]],
    output_path: str,
) -> None:
    """Save partition mapping to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(partitions, f, indent=2)
    print(f"[Dirichlet] Partitions saved to {output_path}")


def load_partitions(path: str) -> Dict[int, List[int]]:
    """Load partition mapping from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_client_dataset(
    X: np.ndarray,
    y: np.ndarray,
    partitions: Dict[int, List[int]],
    client_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a specific client's data from the partition."""
    indices = partitions[client_id]
    return X[indices], y[indices]


if __name__ == "__main__":
    # Quick test
    labels = np.array([0] * 300 + [1] * 700)
    partitions = dirichlet_partition(labels, N=10, alpha=0.5, seed=1)
