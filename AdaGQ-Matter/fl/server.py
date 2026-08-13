"""
AdaGQ-Matter FL Server Strategy (standalone simulation version).

Custom aggregation strategy that implements:
1. Semi-synchronous aggregation with adaptive window
2. Stale-weighted averaging (δ^s decay)
3. Gradient decompression + aggregation
4. Network condition simulation

NOTE: This file provides the strategy logic used in standalone simulation.
Flower-dependent code has been removed; aggregation is done locally in run_main.py.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from core.compression import AdaGQCompressor
from core.aggregation import AdaptiveWindow, semi_sync_aggregate, simulate_network_conditions, NetworkConfig
from core.dp import rdp_to_dp


class AdaGQMatterStrategy:
    """
    AdaGQ-Matter custom aggregation strategy (standalone version).

    Implements:
    - Semi-synchronous aggregation (adaptive window)
    - Stale-weighted averaging
    - Network condition simulation
    - Gradient decompression
    """

    def __init__(self, model_dim: int, config: dict):
        self.model_dim = model_dim
        self.config = config

        # Adaptive aggregation window
        self.adaptive_window = AdaptiveWindow(
            W_base_ms=config.get("W_agg_base_ms", 500.0),
            tau_threshold_ms=config.get("tau_threshold_ms", 80.0),
            ewma_beta=config.get("ewma_beta", 0.3),
        )

        # Network simulation config
        self.network_config = NetworkConfig(
            packet_loss_rate=config.get("packet_loss_rate", 0.05),
            latency_mean_ms=config.get("latency_mean_ms", 100.0),
            latency_std_ms=config.get("latency_std_ms", 50.0),
            staleness_max=config.get("s_max", 3),
        )

        # Staleness decay
        self.staleness_decay = config.get("staleness_decay", 0.6)

        # Global model state
        self.global_params = np.zeros(model_dim, dtype=np.float32)
        self.current_round = 0

        # Results tracking
        self.round_results = []

    def aggregate(self, client_updates: List[Tuple], round_idx: int):
        """
        Aggregate client updates using semi-synchronous strategy.

        Args:
            client_updates: List of (params, n_samples, metrics) from each client
            round_idx: Current round number

        Returns:
            Aggregated parameters
        """
        self.current_round = round_idx

        # Simulate network conditions
        n_clients = len(client_updates)
        network_conditions = simulate_network_conditions(
            n_clients, self.network_config, seed=round_idx
        )

        # Semi-synchronous aggregation with stale-weighted averaging
        aggregated = semi_sync_aggregate(
            client_updates, network_conditions, self.staleness_decay
        )

        return aggregated
