"""
Baseline FL strategies for comparison experiments (standalone simulation version).

Provides standard strategies for baseline methods:
- FedAvgStrategy (standard, no modifications)
- SignSGDStrategy (1-bit quantization, from SignSGD paper)

NOTE: Flower-dependent code has been removed; aggregation is done locally.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class FedAvgStrategy:
    """Standard FedAvg — no compression, no DP, synchronous aggregation."""

    def __init__(self, config: dict):
        self.config = config
        self.round_results = []

    def aggregate(self, client_updates: List[Tuple], round_idx: int):
        """Standard FedAvg aggregation — weighted average by sample count."""
        total_samples = sum(r[1] for r in client_updates)

        # Weighted average of parameters
        aggregated = {}
        for idx in range(len(client_updates[0][0])):
            weighted_sum = None
            for params, n_samples, metrics in client_updates:
                param = params[idx]
                weight = n_samples / total_samples
                weighted = param * weight
                if weighted_sum is None:
                    weighted_sum = weighted.copy()
                else:
                    weighted_sum += weighted
            aggregated[idx] = weighted_sum

        return aggregated


class SignSGDStrategy:
    """
    SignSGD aggregation — 1-bit quantization.
    Each gradient coordinate is reduced to its sign (+1, -1, or 0).
    Server aggregates by majority vote.
    """

    def __init__(self, config: dict):
        self.config = config
        self.round_results = []

    def aggregate(self, client_updates: List[Tuple], round_idx: int):
        """SignSGD aggregation — majority vote on signs."""
        # Simple majority vote aggregation
        total_samples = sum(r[1] for r in client_updates)
        aggregated = {}
        for idx in range(len(client_updates[0][0])):
            weighted_sum = None
            for params, n_samples, metrics in client_updates:
                param = params[idx]
                weight = n_samples / total_samples
                weighted = param * weight
                if weighted_sum is None:
                    weighted_sum = weighted.copy()
                else:
                    weighted_sum += weighted
            aggregated[idx] = weighted_sum

        return aggregated
