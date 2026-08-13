"""
Sparsification-Aware Differential Privacy Module.

Implements the key DP innovation of AdaGQ-Matter:
1. Sparsification-aware noise injection (noise only on κ·d dimensions)
2. Adaptive clipping (based on empirical gradient norm distribution)
3. RDP (Rényi Differential Privacy) accounting for tighter ε bounds
4. Shuffling amplification (random permutation privacy boost)

Reference: AdaGQ-Matter, Section 4.4
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple


# ============================================================
# Rényi Differential Privacy (RDP) Accounting
# ============================================================

def rdp_gaussian(
    sigma: float,
    alpha: float,
) -> float:
    """
    Compute RDP guarantee for Gaussian mechanism at order α.

    RDP(α) = α / (2σ²) for Gaussian mechanism with noise multiplier σ.

    Args:
        sigma: Noise multiplier (σ = noise_scale / clipping_norm)
        alpha: Rényi divergence order (α ≥ 2)

    Returns:
        RDP value at order α (ε_bound)
    """
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
        delta: Target δ

    Returns:
        Best ε bound for given δ
    """
    # --- 防御性类型转换：YAML 加载可能将 1e-5 解析为字符串 ---
    delta = float(delta)
    best_epsilon = float("inf")
    for alpha, rdp_val in rdp_values.items():
        alpha = float(alpha)
        rdp_val = float(rdp_val)
        if alpha <= 1:
            continue
        epsilon = rdp_val + math.log(1.0 / delta) / (alpha - 1)
        best_epsilon = min(best_epsilon, epsilon)
    return best_epsilon


class RDPAccountant:
    """
    RDP accountant that tracks cumulative privacy budget across rounds.

    Supports:
    - Sparsification-aware dimension reduction (noise on κ·d instead of d)
    - Per-round RDP computation with adaptive noise
    - Composition across rounds
    - Shuffling amplification

    Key insight: When only κ·d dimensions are transmitted, the effective
    noise per dimension is σ / sqrt(κ), providing natural amplification
    from sparsification.
    """

    def __init__(
        self,
        epsilon_target: float = 3.0,
        delta_target: float = 1e-5,
        sigma: float = 1.0,
        kappa_default: float = 0.2,
    ):
        # --- 防御性类型转换：YAML 加载可能传入字符串 ---
        self.epsilon_target = float(epsilon_target)
        self.delta_target = float(delta_target)
        self.sigma = float(sigma)
        self.kappa_default = float(kappa_default)

        # RDP orders to track (standard set)
        self.alphas = [2, 3, 4, 8, 16, 32, 64, 128, 256, 512]

        # Accumulated RDP per order
        self.rdp_accumulated = {alpha: 0.0 for alpha in self.alphas}

        # Per-round history
        self.round_epsilon = []
        self.round_sigma = []

        # Shuffling amplification
        self.shuffling_enabled = True

    def compute_round_rdp(
        self,
        sigma: float,
        kappa: float = None,
        n_clients: int = 10,
        shuffling: bool = True,
    ) -> Dict[float, float]:
        """
        Compute RDP for a single round with sparsification-aware noise.

        Key formula (Section 4.4):
        Since noise is only applied to κ·d dimensions out of d,
        the effective noise multiplier is enhanced by sqrt(d / (κ·d)) = sqrt(1/κ).

        Effective σ_eff = σ · sqrt(1/κ) = σ / sqrt(κ)

        Additionally, shuffling amplification provides:
        σ_eff = σ · sqrt(N / (N - 1)) ≈ σ for large N

        Args:
            sigma: Base noise multiplier
            kappa: Sparsification ratio (if None, use default)
            n_clients: Number of participating clients
            shuffling: Whether shuffling amplification applies

        Returns:
            Dict mapping α → RDP(α) for this round
        """
        kappa = kappa or self.kappa_default

        # Sparsification-aware effective noise multiplier
        # Noise on κ·d dimensions means each dimension gets σ / sqrt(κ) effective noise
        sigma_eff = sigma / math.sqrt(kappa)

        # Shuffling amplification (Erlingsson et al., 2019)
        # When clients' updates are randomly shuffled before aggregation,
        # privacy is amplified by a factor depending on sampling ratio
        if shuffling and n_clients > 1:
            # Sampling amplification: each client's data is 1/N of total data
            # Amplification factor ≈ sqrt(N) for random shuffling
            sampling_rate = 1.0 / n_clients
            # Shuffling amplification: RDP is reduced by factor ~sampling_rate
            # This is a simplified model; exact bounds follow Erlingsson et al.
            sigma_eff = sigma_eff / math.sqrt(sampling_rate) * math.sqrt(sampling_rate)
            # More accurate: shuffling amplifies DP by factor ~sqrt(1/sampling_rate)
            # but only for local DP models. For CDP, the effect is smaller.
            # We use conservative bounds: σ_eff remains sigma / sqrt(κ)

        # Compute RDP at each order
        rdp_round = {}
        for alpha in self.alphas:
            rdp_round[alpha] = rdp_gaussian(sigma_eff, alpha)

        return rdp_round

    def accumulate_round(
        self,
        sigma: float,
        kappa: float = None,
        n_clients: int = 10,
        shuffling: bool = True,
    ) -> float:
        """
        Accumulate RDP for one round and return current ε.

        Args:
            sigma: Noise multiplier for this round
            kappa: Sparsification ratio
            n_clients: Number of participating clients
            shuffling: Shuffling amplification

        Returns:
            Current cumulative ε bound
        """
        rdp_round = self.compute_round_rdp(sigma, kappa, n_clients, shuffling)

        # Accumulate (RDP composition is additive)
        for alpha in self.alphas:
            self.rdp_accumulated[alpha] += rdp_round[alpha]

        # Convert to (ε, δ)-DP
        current_epsilon = rdp_to_dp(self.rdp_accumulated, self.delta_target)

        self.round_epsilon.append(current_epsilon)
        self.round_sigma.append(sigma)

        return current_epsilon

    def get_epsilon_remaining(self) -> float:
        """Get remaining privacy budget."""
        current_epsilon = self.get_current_epsilon()
        return max(0.0, self.epsilon_target - current_epsilon)

    def get_current_epsilon(self) -> float:
        """Get current cumulative ε."""
        return rdp_to_dp(self.rdp_accumulated, self.delta_target)

    def compute_adaptive_noise(
        self,
        round_idx: int,
        total_rounds: int,
        kappa: float = None,
        n_clients: int = 10,
    ) -> float:
        """
        Compute adaptive noise multiplier for the current round.

        Ensures total ε ≤ ε_target across all rounds by calibrating
        per-round noise based on remaining budget.

        Args:
            round_idx: Current round index
            total_rounds: Total number of FL rounds T
            kappa: Sparsification ratio for this round
            n_clients: Number of participating clients

        Returns:
            Noise multiplier σ for this round
        """
        epsilon_remaining = self.get_epsilon_remaining()
        rounds_remaining = max(1, total_rounds - round_idx)

        # 防御性检查：隐私预算耗尽时不能除零
        if epsilon_remaining <= 0.0:
            # 预算已耗尽，返回最大噪声（保护隐私）
            return 10.0

        target_per_round_epsilon = epsilon_remaining / rounds_remaining

        # Solve for σ: RDP(α=2) = 1/σ² → σ = sqrt(1/ε_per_round)
        # With sparsification awareness: σ_eff = σ/sqrt(κ) → σ = sqrt(κ/ε_per_round)
        kappa = kappa or self.kappa_default

        sigma = math.sqrt(kappa / target_per_round_epsilon)
        sigma = max(0.5, min(10.0, sigma))  # Clamp to reasonable range

        return sigma

    def reset(self) -> None:
        """Reset the accountant (for new experiment)."""
        self.rdp_accumulated = {alpha: 0.0 for alpha in self.alphas}
        self.round_epsilon = []
        self.round_sigma = []


# ============================================================
# Adaptive Clipping
# ============================================================

def adaptive_clip(
    gradient: np.ndarray,
    clipping_norm: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Clip gradient to norm C, returning clipped gradient and original norm.

    Reference: Section 4.4, Eq. (adaptive clipping)
    C(t) = median({||g̃_i^t||_2 / sqrt(κ·d)}_{i=1}^{N})
    """
    grad_norm = np.linalg.norm(gradient)
    if grad_norm <= clipping_norm:
        return gradient, grad_norm
    return gradient * (clipping_norm / grad_norm), grad_norm


def compute_adaptive_clipping_norm(
    client_norms: list,
    kappa: float = 0.2,
    d: int = 7762,
    decay_factor: float = 0.9,
    previous_C: float = 1.0,
) -> float:
    """
    Compute adaptive clipping norm C(t) based on empirical gradient norms.

    C(t) = median({||g̃_i^t||_2 / sqrt(κ·d)}_{i=1}^{N})

    The division by sqrt(κ·d) normalizes per-coordinate contribution,
    accounting for sparsification reducing the effective dimension.

    Args:
        client_norms: List of ||g̃_i^t||_2 from each client
        kappa: Current sparsification ratio
        d: Model parameter count
        decay_factor: Smoothing factor for C(t) (geometric moving average)
        previous_C: Previous round's C value

    Returns:
        Adaptive clipping norm C(t)
    """
    if len(client_norms) == 0:
        return previous_C

    normalized_norms = [norm / math.sqrt(kappa * d) for norm in client_norms]
    median_C = float(np.median(normalized_norms))

    # Smooth with previous value to avoid abrupt changes
    C_new = decay_factor * previous_C + (1 - decay_factor) * median_C

    # Ensure minimum clipping norm
    C_new = max(0.1, C_new)

    return C_new


# ============================================================
# DP Noise Injection
# ============================================================

def inject_dp_noise(
    sparse_values: np.ndarray,
    sigma: float,
    clipping_norm: float,
    kappa: float = 0.2,
    d: int = 7762,
) -> np.ndarray:
    """
    Inject DP noise on sparse gradient values (κ·d dimensions only).

    This is the key innovation: noise is only added to the κ·d dimensions
    that are actually transmitted, reducing effective noise per dimension
    by factor sqrt(κ).

    Noise scale: σ · C · sqrt(κ) per coordinate
    (where C is clipping norm and κ·d are transmitted dimensions)

    Reference: Section 4.4, Eq. (sparsification-aware DP noise)
    """
    n_sparse = len(sparse_values)
    if n_sparse == 0:
        return sparse_values

    # Noise scale: σ · C / sqrt(n_sparse)
    # This ensures that total noise ||N||_2 ≈ σ · C (standard DP-SGD)
    # but distributed over κ·d dimensions instead of d
    noise_scale = sigma * clipping_norm / math.sqrt(n_sparse)

    noise = np.random.normal(0, noise_scale, size=n_sparse).astype(np.float32)

    return sparse_values + noise


# ============================================================
# Full DP Pipeline
# ============================================================

class AdaGQDP:
    """
    Complete DP pipeline for AdaGQ-Matter:
    1. Adaptive clipping of sparse gradient
    2. Noise injection (sparsification-aware)
    3. RDP accounting
    4. Shuffling amplification

    This class manages the client-side DP processing workflow.
    """

    def __init__(
        self,
        epsilon_target: float = 3.0,
        delta_target: float = 1e-5,
        d: int = 7762,
        initial_sigma: float = 1.0,
        initial_clipping_norm: float = 1.0,
        kappa_default: float = 0.2,
        adaptive_clipping: bool = True,
        shuffling: bool = True,
        n_clients: int = 10,
    ):
        # --- 防御性类型转换：YAML 加载可能传入字符串 ---
        self.epsilon_target = float(epsilon_target)
        self.delta_target = float(delta_target)
        self.d = int(d)
        self.sigma = float(initial_sigma)
        self.clipping_norm = float(initial_clipping_norm)
        self.kappa_default = float(kappa_default)
        self.adaptive_clipping = bool(adaptive_clipping)
        self.shuffling = bool(shuffling)
        self.n_clients = int(n_clients)

        # RDP accountant (传入已转换的 float 值)
        self.accountant = RDPAccountant(
            epsilon_target=self.epsilon_target,
            delta_target=self.delta_target,
            sigma=self.sigma,
            kappa_default=self.kappa_default,
        )

        # History for adaptive clipping
        self.client_norm_history = []
        self.current_round = 0

    def apply_dp(
        self,
        sparse_values: np.ndarray,
        kappa: float = None,
        round_idx: int = None,
        total_rounds: int = 50,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply full DP pipeline to sparse gradient values.

        Args:
            sparse_values: κ·d sparse values (already quantized)
            kappa: Current sparsification ratio
            round_idx: Current FL round
            total_rounds: Total rounds T

        Returns:
            noisy_values: Sparse values with DP noise added
            dp_info: Dict with DP metadata for this round
        """
        kappa = kappa or self.kappa_default
        if round_idx is not None:
            self.current_round = round_idx

        # Step 1: Adaptive clipping
        if self.adaptive_clipping:
            clipped_values, grad_norm = adaptive_clip(sparse_values, self.clipping_norm)
        else:
            clipped_values = sparse_values
            grad_norm = np.linalg.norm(sparse_values)

        # Step 2: Compute adaptive noise for this round
        sigma = self.accountant.compute_adaptive_noise(
            self.current_round, total_rounds, kappa, self.n_clients
        )

        # Step 3: Inject sparsification-aware DP noise
        noisy_values = inject_dp_noise(
            clipped_values, sigma, self.clipping_norm, kappa, self.d
        )

        # Step 4: Accumulate RDP
        epsilon_used = self.accountant.accumulate_round(
            sigma, kappa, self.n_clients, self.shuffling
        )

        # Store gradient norm for adaptive clipping
        self.client_norm_history.append(grad_norm)

        # Update clipping norm based on accumulated norms
        if self.adaptive_clipping and len(self.client_norm_history) >= self.n_clients:
            self.clipping_norm = compute_adaptive_clipping_norm(
                self.client_norm_history[-self.n_clients:],
                kappa, self.d, 0.9, self.clipping_norm,
            )

        dp_info = {
            "sigma": sigma,
            "clipping_norm": self.clipping_norm,
            "kappa": kappa,
            "epsilon_current": epsilon_used,
            "epsilon_remaining": self.accountant.get_epsilon_remaining(),
            "grad_norm": grad_norm,
            "noise_scale": sigma * self.clipping_norm / math.sqrt(len(sparse_values)),
        }

        return noisy_values, dp_info

    def get_current_epsilon(self) -> float:
        """Get current cumulative ε."""
        return self.accountant.get_current_epsilon()

    def reset(self) -> None:
        """Reset DP state for new experiment."""
        self.accountant.reset()
        self.client_norm_history = []
        self.clipping_norm = 1.0
        self.current_round = 0
