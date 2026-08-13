"""
Semi-Synchronous Aggregation Module.

Implements link-aware adaptive aggregation window (Section 4.5):
1. EWMA-based adaptive window estimation
2. Stale-weighted averaging (δ^s decay)
3. Network condition simulation (packet loss + latency)

Reference: AdaGQ-Matter, Section 4.5 (Link-Aware Semi-Synchronous Aggregation)
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================
# Adaptive Aggregation Window
# ============================================================

class AdaptiveWindow:
    """
    EWMA-based adaptive aggregation window W_agg(t).

    - During congestion (τ_est > 80ms): window expands to include more clients
    - During stability (τ_est < 80ms): window contracts to W_base = 500ms

    EWMA latency estimation:
    τ_est(t) = β · τ(t) + (1-β) · τ_est(t-1)

    Window adjustment:
    W_agg(t) = W_base + max(0, τ_est(t) - τ_threshold) · expansion_factor
    """

    def __init__(
        self,
        W_base_ms: float = 500.0,
        tau_threshold_ms: float = 80.0,
        ewma_beta: float = 0.3,
        expansion_factor: float = 5.0,
        W_max_ms: float = 2000.0,
    ):
        self.W_base_ms = W_base_ms
        self.tau_threshold_ms = tau_threshold_ms
        self.ewma_beta = ewma_beta
        self.expansion_factor = expansion_factor
        self.W_max_ms = W_max_ms

        # EWMA estimated latency
        self.tau_est_ms = 0.0

        # History
        self.window_history = []
        self.latency_history = []

    def update(
        self,
        observed_latencies: List[float],
    ) -> float:
        """
        Update EWMA estimate and compute new aggregation window.

        Args:
            observed_latencies: List of client response latencies in ms

        Returns:
            New aggregation window W_agg(t) in ms
        """
        # Compute average observed latency this round
        if len(observed_latencies) > 0:
            tau_current = np.mean(observed_latencies)
        else:
            tau_current = self.tau_est_ms

        # EWMA update
        self.tau_est_ms = self.ewma_beta * tau_current + (1 - self.ewma_beta) * self.tau_est_ms

        # Compute adaptive window
        if self.tau_est_ms > self.tau_threshold_ms:
            # Congestion: expand window
            W_agg = self.W_base_ms + (self.tau_est_ms - self.tau_threshold_ms) * self.expansion_factor
        else:
            # Stable: contract to base
            W_agg = self.W_base_ms

        # Clamp
        W_agg = max(self.W_base_ms, min(self.W_max_ms, W_agg))

        self.window_history.append(W_agg)
        self.latency_history.append(self.tau_est_ms)

        return W_agg

    def get_current_window(self) -> float:
        """Get current aggregation window."""
        if self.window_history:
            return self.window_history[-1]
        return self.W_base_ms

    def reset(self) -> None:
        """Reset for new experiment."""
        self.tau_est_ms = 0.0
        self.window_history = []
        self.latency_history = []


# ============================================================
# Network Condition Simulation
# ============================================================

@dataclass
class NetworkConfig:
    """Network simulation configuration."""
    packet_loss_rate: float = 0.05   # 5% packet loss
    latency_mean_ms: float = 100.0   # 100ms mean latency
    latency_std_ms: float = 50.0     # 50ms latency variation
    staleness_max: int = 3           # Maximum staleness (rounds)


def simulate_network_conditions(
    n_clients: int,
    config: NetworkConfig,
    seed: int = 1,
) -> Dict[int, Dict]:
    """
    Simulate network conditions for each client in a round.

    For each client, generate:
    - Whether the client responds within the window (packet loss → no response)
    - Response latency (Gaussian with config mean/std)
    - Staleness (how many rounds delayed, 0 = on-time)

    Args:
        n_clients: Number of clients
        config: Network configuration
        seed: Random seed

    Returns:
        Dict mapping client_id → {responded, latency_ms, staleness}
    """
    rng = np.random.default_rng(seed)

    results = {}
    for i in range(n_clients):
        # Packet loss: client doesn't respond
        responded = rng.random() > config.packet_loss_rate

        if responded:
            # Latency: Gaussian distribution
            latency = rng.normal(config.latency_mean_ms, config.latency_std_ms)
            latency = max(10.0, latency)  # Minimum 10ms

            # Staleness: probability of being delayed
            staleness = 0
            # High latency → higher chance of being stale
            if latency > config.latency_mean_ms + 2 * config.latency_std_ms:
                staleness = rng.integers(1, config.staleness_max + 1)
        else:
            latency = float("inf")
            staleness = config.staleness_max

        results[i] = {
            "responded": responded,
            "latency_ms": latency,
            "staleness": staleness,
        }

    return results


# ============================================================
# Stale-Weighted Aggregation
# ============================================================

def stale_weighted_average(
    client_updates: Dict[int, np.ndarray],
    client_staleness: Dict[int, int],
    staleness_decay: float = 0.6,
    n_clients: int = 10,
) -> np.ndarray:
    """
    Stale-weighted averaging of client updates.

    Weight for client i delayed by s rounds:
    w_i = δ^s_i / |S_t|

    where δ = 0.6 (staleness decay factor) and |S_t| is the number
    of participating clients this round.

    Reference: Section 4.5, Eq. (stale-weighted aggregation)

    Args:
        client_updates: Dict mapping client_id → flat gradient update
        client_staleness: Dict mapping client_id → staleness (rounds delayed)
        staleness_decay: Decay factor δ (default 0.6)
        n_clients: Total expected clients (for normalization)

    Returns:
        Aggregated gradient update (d-dim)
    """
    participating_ids = list(client_updates.keys())
    n_participating = len(participating_ids)

    if n_participating == 0:
        raise ValueError("No participating clients for aggregation!")

    # Compute weights
    weights = {}
    for client_id in participating_ids:
        s = client_staleness.get(client_id, 0)
        weights[client_id] = staleness_decay ** s

    # Normalize weights
    total_weight = sum(weights.values())
    for client_id in weights:
        weights[client_id] /= total_weight

    # Weighted average
    d = len(client_updates[participating_ids[0]])
    aggregated = np.zeros(d, dtype=np.float32)

    for client_id in participating_ids:
        aggregated += weights[client_id] * client_updates[client_id]

    return aggregated


def semi_sync_aggregate(
    client_updates: Dict[int, Dict],
    adaptive_window: AdaptiveWindow,
    staleness_decay: float = 0.6,
) -> Tuple[np.ndarray, Dict]:
    """
    Semi-synchronous aggregation with adaptive window.

    1. Compute adaptive window based on observed latencies
    2. Filter clients who responded within window
    3. Apply stale-weighted averaging

    Args:
        client_updates: Dict mapping client_id → {
            "compressed": compressed gradient dict,
            "decompressed": decompressed full gradient,
            "latency_ms": response latency,
            "staleness": rounds delayed,
        }
        adaptive_window: AdaptiveWindow instance
        staleness_decay: δ for stale weighting

    Returns:
        aggregated_gradient: Aggregated gradient update (d-dim)
        aggregation_info: Dict with participation statistics
    """
    # Compute adaptive window from observed latencies
    latencies = [info["latency_ms"] for info in client_updates.values()
                 if info["latency_ms"] != float("inf")]
    W_agg = adaptive_window.update(latencies)

    # Filter clients within window
    participating = {}
    dropped = {}
    for client_id, info in client_updates.items():
        if info["latency_ms"] <= W_agg:
            participating[client_id] = info
        else:
            dropped[client_id] = info

    # Extract decompressed updates for aggregation
    updates_for_agg = {
        cid: info["decompressed"] for cid, info in participating.items()
        if "decompressed" in info
    }
    staleness_for_agg = {
        cid: info["staleness"] for cid, info in participating.items()
    }

    # Stale-weighted averaging
    if len(updates_for_agg) > 0:
        aggregated = stale_weighted_average(
            updates_for_agg, staleness_for_agg, staleness_decay
        )
    else:
        # Fallback: use all available updates with maximum staleness decay
        updates_for_agg = {
            cid: info["decompressed"] for cid, info in client_updates.items()
            if "decompressed" in info
        }
        staleness_for_agg = {
            cid: info["staleness"] for cid, info in client_updates.items()
        }
        aggregated = stale_weighted_average(
            updates_for_agg, staleness_for_agg, staleness_decay
        )

    aggregation_info = {
        "W_agg_ms": W_agg,
        "n_participating": len(participating),
        "n_dropped": len(dropped),
        "participation_rate": len(participating) / max(1, len(client_updates)),
        "avg_staleness": np.mean(list(staleness_for_agg.values())) if staleness_for_agg else 0,
    }

    return aggregated, aggregation_info
