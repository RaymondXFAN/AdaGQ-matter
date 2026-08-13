"""
Standalone Rényi Differential Privacy (RDP) Accountant.

Independent RDP accountant extracted from core/dp.py, without Opacus dependency.
Implements:
1. Gaussian RDP computation: RDP(α) = α / (2σ²)
2. RDP → (ε,δ)-DP conversion
3. PrivacyAccountant class for multi-round composition tracking
4. Noise multiplier computation for target privacy budget

This module supports the sparsification-aware DP innovation of AdaGQ-Matter,
where noise is only applied to κ·d dimensions, yielding effective σ_eff = σ/√κ.

Reference: AdaGQ-Matter, Section 4.4
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================
# Core RDP Functions
# ============================================================

def gaussian_rdp(
    sigma: float,
    alpha: float,
) -> float:
    """
    Compute Rényi Differential Privacy guarantee for Gaussian mechanism.

    RDP(α) = α / (2σ²) for Gaussian mechanism with noise multiplier σ.

    Args:
        sigma: Noise multiplier (σ = noise_scale / sensitivity)
        alpha: Rényi divergence order (α ≥ 2)

    Returns:
        RDP value at order α
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if alpha < 1:
        raise ValueError(f"alpha must be >= 1, got {alpha}")
    return alpha / (2.0 * sigma ** 2)


def rdp_to_dp(
    rdp_values: Dict[float, float],
    delta: float = 1e-5,
) -> float:
    """
    Convert RDP guarantees to (ε, δ)-DP bound.

    ε(δ) = min_α { RDP(α) + log(1/δ) / (α - 1) }

    Args:
        rdp_values: Dict mapping α → RDP(α) value
        delta: Target δ parameter

    Returns:
        Best ε bound for given δ
    """
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    best_epsilon = float("inf")
    for alpha, rdp_val in rdp_values.items():
        if alpha <= 1:
            continue
        epsilon = rdp_val + math.log(1.0 / delta) / (alpha - 1)
        best_epsilon = min(best_epsilon, epsilon)
    return best_epsilon


# ============================================================
# Sparsification-Aware RDP
# ============================================================

def sparsification_aware_rdp(
    sigma: float,
    kappa: float,
    alpha: float,
) -> float:
    """
    Compute RDP with sparsification-aware noise amplification.

    When noise is only applied to κ·d dimensions (out of d total),
    the effective noise multiplier is enhanced: σ_eff = σ / √κ.

    RDP_sparsity(α) = α / (2σ_eff²) = α · κ / (2σ²)

    Args:
        sigma: Base noise multiplier
        kappa: Sparsification ratio (0 < κ ≤ 1)
        alpha: Rényi divergence order

    Returns:
        Sparsification-aware RDP value
    """
    if kappa <= 0 or kappa > 1:
        raise ValueError(f"kappa must be in (0, 1], got {kappa}")
    sigma_eff = sigma / math.sqrt(kappa)
    return gaussian_rdp(sigma_eff, alpha)


# ============================================================
# PrivacyAccountant Class
# ============================================================

class PrivacyAccountant:
    """
    RDP accountant that tracks cumulative privacy budget across FL rounds.

    Supports:
    - Per-round RDP computation with sparsification-aware noise
    - Composition across rounds (additive RDP)
    - Automatic ε computation at any point
    - Shuffling amplification (optional)

    Usage:
        accountant = PrivacyAccountant(epsilon_target=3.0, delta=1e-5)
        for round in range(n_rounds):
            accountant.step(sigma=1.5, kappa=0.2, n_clients=10)
            eps = accountant.get_epsilon()
            if eps > accountant.epsilon_target:
                print("Privacy budget exhausted!")
                break
    """

    # Default Rényi divergence orders to track
    DEFAULT_ALPHAS: List[float] = [
        1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        32, 40, 48, 56, 64, 72, 80, 88, 96, 128,
        256, 512
    ]

    def __init__(
        self,
        epsilon_target: float = 3.0,
        delta: float = 1e-5,
        alphas: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            epsilon_target: Target privacy budget (ε)
            delta: Target δ for (ε,δ)-DP
            alphas: List of Rényi orders to track. Uses DEFAULT_ALPHAS if None.
        """
        self.epsilon_target = epsilon_target
        self.delta = delta
        self.alphas = alphas or self.DEFAULT_ALPHAS

        # Accumulated RDP per order
        self.rdp_accumulated: Dict[float, float] = {a: 0.0 for a in self.alphas}

        # History tracking
        self.round_epsilon_history: List[float] = []
        self.round_rdp_history: List[Dict[float, float]] = []
        self.total_rounds: int = 0

    def step(
        self,
        sigma: float,
        kappa: float = 1.0,
        n_clients: int = 10,
        shuffling: bool = False,
    ) -> float:
        """
        Accumulate RDP for one FL round.

        Args:
            sigma: Noise multiplier for this round
            kappa: Sparsification ratio (default 1.0 = no sparsification)
            n_clients: Number of participating clients
            shuffling: Whether shuffling amplification applies

        Returns:
            Current cumulative ε after this round
        """
        # Compute per-round RDP at each order
        rdp_round: Dict[float, float] = {}
        for alpha in self.alphas:
            if shuffling and n_clients > 1:
                # Shuffling amplification: RDP is divided by n_clients
                # (simplified model based on Erlingsson et al., 2019)
                rdp_val = sparsification_aware_rdp(sigma, kappa, alpha) / n_clients
            else:
                rdp_val = sparsification_aware_rdp(sigma, kappa, alpha)
            rdp_round[alpha] = rdp_val

            # Composition: RDP is additive across rounds
            self.rdp_accumulated[alpha] += rdp_val

        # Record history
        self.round_rdp_history.append(rdp_round)
        self.total_rounds += 1

        # Compute current ε
        eps = self.get_epsilon()
        self.round_epsilon_history.append(eps)

        return eps

    def get_epsilon(self) -> float:
        """
        Compute current cumulative (ε, δ)-DP guarantee.

        Returns:
            Current ε bound for the accumulated RDP
        """
        return rdp_to_dp(self.rdp_accumulated, self.delta)

    def get_spent_budget(self) -> float:
        """
        Return how much of the target budget has been spent.

        Returns:
            Fraction of epsilon_target consumed (0.0 to >1.0)
        """
        return self.get_epsilon() / self.epsilon_target

    def is_budget_exhausted(self) -> bool:
        """
        Check if the privacy budget has been exhausted.

        Returns:
            True if current ε > epsilon_target
        """
        return self.get_epsilon() > self.epsilon_target

    def remaining_rounds(
        self,
        sigma: float,
        kappa: float = 1.0,
        n_clients: int = 10,
        shuffling: bool = False,
    ) -> int:
        """
        Estimate how many more rounds can be run before budget exhaustion.

        Args:
            sigma: Noise multiplier (assumed constant for future rounds)
            kappa: Sparsification ratio
            n_clients: Number of clients
            shuffling: Whether shuffling applies

        Returns:
            Estimated remaining rounds
        """
        remaining_eps = self.epsilon_target - self.get_epsilon()
        if remaining_eps <= 0:
            return 0

        # Estimate per-round ε cost
        rdp_per_round: Dict[float, float] = {}
        for alpha in self.alphas:
            if shuffling and n_clients > 1:
                rdp_val = sparsification_aware_rdp(sigma, kappa, alpha) / n_clients
            else:
                rdp_val = sparsification_aware_rdp(sigma, kappa, alpha)
            rdp_per_round[alpha] = rdp_val

        eps_per_round = rdp_to_dp(rdp_per_round, self.delta)
        if eps_per_round <= 0:
            return 9999  # Effectively unlimited

        return max(0, int(remaining_eps / eps_per_round))

    def summary(self) -> str:
        """
        Generate privacy budget summary.

        Returns:
            Multi-line summary string
        """
        lines = ["=== Privacy Budget Summary ==="]
        lines.append(f"Target ε:          {self.epsilon_target}")
        lines.append(f"Target δ:          {self.delta}")
        lines.append(f"Rounds completed:  {self.total_rounds}")
        lines.append(f"Current ε:         {self.get_epsilon():.6f}")
        lines.append(f"Budget spent:      {self.get_spent_budget():.2%}")
        lines.append(f"Budget exhausted:  {self.is_budget_exhausted()}")

        if self.round_epsilon_history:
            lines.append(f"ε per round (last 5): "
                         f"{[f'{e:.4f}' for e in self.round_epsilon_history[-5:]]}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset accountant for a new experiment."""
        self.rdp_accumulated = {a: 0.0 for a in self.alphas}
        self.round_epsilon_history = []
        self.round_rdp_history = []
        self.total_rounds = 0


# ============================================================
# Noise Multiplier Computation
# ============================================================

def compute_noise_multiplier(
    epsilon: float,
    delta: float = 1e-5,
    n_rounds: int = 50,
    n_clients: int = 10,
    kappa: float = 1.0,
    shuffling: bool = False,
) -> float:
    """
    Compute the required noise multiplier σ for a target privacy budget.

    Given (ε, δ) target, number of rounds, and sparsification ratio,
    find σ such that cumulative RDP after n_rounds ≤ ε.

    Uses bisection search over σ values.

    Args:
        epsilon: Target ε budget
        delta: Target δ
        n_rounds: Number of FL rounds
        n_clients: Number of clients (for shuffling amplification)
        kappa: Sparsification ratio (σ_eff = σ / √κ)
        shuffling: Whether shuffling amplification applies

    Returns:
        Required noise multiplier σ
    """
    # Bisection search for σ
    sigma_low = 0.01
    sigma_high = 100.0
    tolerance = 1e-4

    # Use the same alphas as PrivacyAccountant
    alphas = PrivacyAccountant.DEFAULT_ALPHAS

    for _ in range(200):  # Max bisection iterations
        sigma_mid = (sigma_low + sigma_high) / 2.0

        # Compute cumulative RDP after n_rounds
        rdp_accumulated: Dict[float, float] = {}
        for alpha in alphas:
            if shuffling and n_clients > 1:
                rdp_per_round = sparsification_aware_rdp(sigma_mid, kappa, alpha) / n_clients
            else:
                rdp_per_round = sparsification_aware_rdp(sigma_mid, kappa, alpha)
            rdp_accumulated[alpha] = rdp_per_round * n_rounds

        eps = rdp_to_dp(rdp_accumulated, delta)

        if eps > epsilon:
            sigma_low = sigma_mid  # Need more noise
        else:
            sigma_high = sigma_mid  # Can reduce noise

        if sigma_high - sigma_low < tolerance:
            break

    return sigma_high  # Return the conservative (higher) bound


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":
    print("=== Gaussian RDP Test ===")
    sigma = 1.5
    for alpha in [2, 4, 8, 16, 32]:
        rdp = gaussian_rdp(sigma, alpha)
        print(f"  σ={sigma}, α={alpha}: RDP = {rdp:.6f}")

    print("\n=== RDP → DP Conversion Test ===")
    rdp_dict = {alpha: gaussian_rdp(sigma, alpha) for alpha in [2, 4, 8, 16, 32, 64, 128]}
    eps = rdp_to_dp(rdp_dict, delta=1e-5)
    print(f"  σ={sigma}, δ=1e-5: ε = {eps:.6f}")

    print("\n=== Sparsification-Aware RDP Test ===")
    kappa = 0.2
    sigma = 1.5
    for alpha in [2, 8, 32]:
        rdp_base = gaussian_rdp(sigma, alpha)
        rdp_sparse = sparsification_aware_rdp(sigma, kappa, alpha)
        print(f"  α={alpha}: Base RDP={rdp_base:.6f}, Sparsity-aware RDP={rdp_sparse:.6f}")

    print("\n=== PrivacyAccountant Composition Test ===")
    accountant = PrivacyAccountant(epsilon_target=3.0, delta=1e-5)
    sigma = 1.5
    kappa = 0.2
    for r in range(10):
        eps = accountant.step(sigma=sigma, kappa=kappa, n_clients=10)
        print(f"  Round {r}: ε = {eps:.6f}")

    print(accountant.summary())
    print(f"  Remaining rounds (σ={sigma}, κ={kappa}): {accountant.remaining_rounds(sigma, kappa)}")

    print("\n=== Noise Multiplier Computation Test ===")
    sigma_required = compute_noise_multiplier(
        epsilon=3.0, delta=1e-5, n_rounds=50, n_clients=10, kappa=0.2
    )
    print(f"  Required σ for ε=3.0, δ=1e-5, 50 rounds, κ=0.2: {sigma_required:.4f}")

    # Verify: run accountant with computed σ
    accountant2 = PrivacyAccountant(epsilon_target=3.0, delta=1e-5)
    for r in range(50):
        accountant2.step(sigma=sigma_required, kappa=0.2, n_clients=10)
    print(f"  After 50 rounds with σ={sigma_required:.4f}: ε = {accountant2.get_epsilon():.6f}")

    print("\n=== Shuffling Amplification Test ===")
    accountant_shuffle = PrivacyAccountant(epsilon_target=3.0, delta=1e-5)
    for r in range(10):
        eps = accountant_shuffle.step(sigma=sigma, kappa=kappa, n_clients=10, shuffling=True)
        print(f"  Round {r} (shuffle): ε = {eps:.6f}")
    print(f"  Without shuffle ε after 10 rounds: {accountant.round_epsilon_history[-1]:.6f}")
    print(f"  With shuffle ε after 10 rounds:    {accountant_shuffle.round_epsilon_history[-1]:.6f}")
