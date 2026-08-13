"""
AdaGQ-Matter Gradient Compression Module.

Implements the three-stage compression pipeline from the paper:
1. Top-k Sparsification with adaptive κ (based on Gini coefficient)
2. Adaptive Quantization with bit-width b (coupled with κ and privacy budget)
3. Error Compensation (error feedback buffer for residual gradients)

Reference: AdaGQ-Matter, Section 4.2–4.4
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict


# ============================================================
# Gini Coefficient Computation
# ============================================================

def compute_gini(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1D array.

    Gini measures gradient concentration:
    - Gini ≈ 1.0: highly concentrated (few large gradients)
    - Gini ≈ 0.0: diffuse (uniform distribution)

    Args:
        values: 1D numpy array (absolute gradient magnitudes)

    Returns:
        Gini coefficient in [0, 1]
    """
    if len(values) == 0:
        return 0.0
    # Sort absolute values
    sorted_vals = np.sort(np.abs(values))
    n = len(sorted_vals)
    # Gini = 2 * sum(i * sorted_vals[i]) / (n * sum(sorted_vals)) - (n+1)/n
    total = sorted_vals.sum()
    if total == 0:
        return 0.0
    index_sum = np.arange(1, n + 1) * sorted_vals
    gini = (2.0 * index_sum.sum()) / (n * total) - (n + 1.0) / n
    return max(0.0, min(1.0, gini))  # Clamp to [0, 1]


# ============================================================
# Top-k Sparsification
# ============================================================

def top_k_sparsify(
    gradient: np.ndarray,
    kappa: float = 0.2,
    priority_boost: Optional[Dict[int, float]] = None,
    layer_ranges: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Top-k sparsification: select the top κ·d elements by absolute magnitude.

    With Matter-aware feature grouping, coordinates in FG1/FG2 receive
    a priority boost (1.2×) on their absolute magnitude before ranking.

    Args:
        gradient: Full gradient vector (d-dim)
        kappa: Sparsification ratio (0.1 ≤ κ ≤ 0.3)
        priority_boost: Dict mapping group_id → boost factor (e.g., {0: 1.2, 1: 1.2})
        layer_ranges: Dict mapping layer_id → (start_idx, end_idx) in flat vector

    Returns:
        sparse_values: κ·d selected values
        sparse_indices: κ·d indices in the original gradient
        mask: Boolean mask (d-dim) indicating which coordinates were selected
    """
    d = len(gradient)
    k = max(1, int(kappa * d))  # Number of elements to keep

    # Compute ranking scores (with optional priority boost)
    abs_grad = np.abs(gradient)

    if priority_boost and layer_ranges:
        # Apply priority boost to specific feature groups
        ranking_scores = abs_grad.copy()
        # Map layer ranges to feature groups
        # Convention: layers 0-1 → FG1 (Sensors), 2-3 → FG2 (Actuators), etc.
        for layer_id, (start, end) in layer_ranges.items():
            if layer_id in priority_boost:
                boost = priority_boost[layer_id]
                ranking_scores[start:end] *= boost
    else:
        ranking_scores = abs_grad

    # Select top-k indices based on ranking scores
    top_k_indices = np.argsort(ranking_scores)[-k:]

    # Extract sparse gradient
    sparse_values = gradient[top_k_indices]
    sparse_indices = top_k_indices.astype(np.int32)

    # Create mask for error compensation
    mask = np.zeros(d, dtype=bool)
    mask[top_k_indices] = True

    return sparse_values, sparse_indices, mask


def adaptive_kappa(
    gradient: np.ndarray,
    kappa_min: float = 0.1,
    kappa_max: float = 0.3,
    kappa_default: float = 0.2,
    gini_threshold_high: float = 0.7,
    gini_threshold_low: float = 0.4,
) -> float:
    """
    Determine sparsification ratio κ based on gradient Gini coefficient.

    - Gini > 0.7 → κ = kappa_min (gradient concentrated → fewer elements needed)
    - 0.4 ≤ Gini ≤ 0.7 → κ = kappa_default (moderate)
    - Gini < 0.4 → κ = kappa_max (diffuse → need more elements)

    Reference: Section 4.2, Eq. (κ selection rule)
    """
    gini = compute_gini(gradient)

    if gini > gini_threshold_high:
        kappa = kappa_min
    elif gini < gini_threshold_low:
        kappa = kappa_max
    else:
        kappa = kappa_default

    return kappa


# ============================================================
# Adaptive Quantization
# ============================================================

def stochastic_quantize(
    values: np.ndarray,
    b: int = 4,
    norm_max: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Stochastic quantization to b bits.

    Quantization formula (Section 4.2):
    Q_b(g_j) = clip(g_j / ||g||_∞, -1, 1) · ||g||_∞
               + uniform(-||g||_∞/2^b, ||g||_∞/2^b)

    Args:
        values: Values to quantize (κ·d sparse values)
        b: Bit-width (4 ≤ b ≤ 8)
        norm_max: ||g||_∞ (max absolute value). If None, computed from values.

    Returns:
        quantized_values: Quantized values
        norm_max: The infinity norm used for quantization (needed for decoder)
    """
    if norm_max is None:
        norm_max = np.max(np.abs(values)) if len(values) > 0 else 1.0

    if norm_max == 0:
        return values.copy(), 0.0

    # Normalize to [-1, 1]
    normalized = np.clip(values / norm_max, -1.0, 1.0)

    # Quantization levels: 2^b levels from -1 to 1
    num_levels = 2 ** b
    level_spacing = 2.0 / (num_levels - 1)

    # Stochastic rounding: floor with probability proportional to distance to next level
    # This ensures unbiased quantization (A4: E[Q_b(g)] = g)
    quantized_normalized = np.zeros_like(normalized)

    for i in range(len(normalized)):
        # Find nearest quantization levels
        lower_level = np.floor((normalized[i] + 1.0) / level_spacing) * level_spacing - 1.0
        upper_level = lower_level + level_spacing

        # Stochastic choice
        if upper_level > 1.0:
            upper_level = 1.0

        if normalized[i] - lower_level < 1e-10:
            quantized_normalized[i] = lower_level
        else:
            prob_upper = (normalized[i] - lower_level) / level_spacing
            prob_upper = np.clip(prob_upper, 0.0, 1.0)
            if np.random.random() < prob_upper:
                quantized_normalized[i] = upper_level
            else:
                quantized_normalized[i] = lower_level

    # Scale back
    quantized_values = quantized_normalized * norm_max

    return quantized_values, norm_max


def adaptive_bit_width(
    round_idx: int,
    total_rounds: int,
    epsilon_remaining: float,
    b_min: int = 4,
    b_max: int = 8,
) -> int:
    """
    Determine bit-width b for quantization at round t.

    Formula (Section 4.2):
    b(t) = min(b_max, max(b_min, ceil(8 · ε_remaining / (T - t))))

    As privacy budget depletes (ε_remaining → 0), b decreases,
    allowing more noise relative to quantization resolution.
    """
    if total_rounds - round_idx <= 0:
        return b_min

    b = int(np.ceil(8.0 * epsilon_remaining / (total_rounds - round_idx)))
    b = max(b_min, min(b_max, b))
    return b


# ============================================================
# Error Compensation (Error Feedback Buffer)
# ============================================================

class ErrorFeedbackBuffer:
    """
    Error feedback buffer for gradient compression.

    Stores residual gradients from previous rounds and adds them
    back before the next round's sparsification, ensuring that
    information lost in earlier rounds is re-attempted.

    Reference: Section 4.2, Eq. (error feedback):
    g_i^t ← g_i^t + e_i^{t-1}
    e_i^t = g_i^t + e_i^{t-1} - g̃_i^t
    """

    def __init__(self, dim: int, staleness_decay: float = 0.6):
        """
        Args:
            dim: Dimension of the gradient vector (d)
            staleness_decay: Decay factor δ for stale error buffers
        """
        self.dim = dim
        self.staleness_decay = staleness_decay
        self.buffer = np.zeros(dim, dtype=np.float32)
        self.staleness = 0  # Number of rounds since last update

    def accumulate(self, gradient: np.ndarray, sparse_gradient: np.ndarray,
                   sparse_indices: np.ndarray) -> np.ndarray:
        """
        Accumulate error after sparsification.

        e_i^t = (g_i^t + e_i^{t-1}) - g̃_i^t
        where g̃_i^t is the sparse representation reconstructed to full dimension.

        Args:
            gradient: Original full gradient
            sparse_gradient: Selected κ·d values
            sparse_indices: Indices of selected elements

        Returns:
            Updated error buffer
        """
        # Reconstruct sparse gradient to full dimension
        reconstructed = np.zeros(self.dim, dtype=np.float32)
        reconstructed[sparse_indices] = sparse_gradient

        # Compute residual
        # Note: gradient already includes previous error buffer
        residual = gradient - reconstructed

        # Apply staleness decay if this client was delayed
        if self.staleness > 0:
            decay_factor = self.staleness_decay ** self.staleness
            residual *= decay_factor
            self.staleness = 0  # Reset after applying decay

        self.buffer = residual
        return self.buffer

    def apply(self, gradient: np.ndarray) -> np.ndarray:
        """
        Add error buffer to current gradient before sparsification.

        g_i^t ← g_i^t + e_i^{t-1}

        Args:
            gradient: Current round's gradient

        Returns:
            Gradient with error feedback added
        """
        return gradient + self.buffer

    def mark_stale(self, n_rounds: int = 1) -> None:
        """Mark this buffer as stale (client was delayed by n_rounds)."""
        self.staleness += n_rounds

    def reset(self) -> None:
        """Reset the error buffer."""
        self.buffer = np.zeros(self.dim, dtype=np.float32)
        self.staleness = 0


# ============================================================
# Full Compression Pipeline
# ============================================================

class AdaGQCompressor:
    """
    Complete AdaGQ-Matter compression pipeline:
    1. Add error feedback buffer
    2. Compute adaptive κ (Gini-based)
    3. Top-k sparsification (with Matter-aware priority)
    4. Compute adaptive b (privacy-budget-based)
    5. Stochastic quantization
    6. Update error feedback buffer

    This class manages the full client-side gradient processing workflow.
    """

    def __init__(
        self,
        dim: int,
        kappa_min: float = 0.1,
        kappa_max: float = 0.3,
        kappa_default: float = 0.2,
        b_min: int = 4,
        b_max: int = 8,
        b_default: int = 4,
        gini_threshold_high: float = 0.7,
        gini_threshold_low: float = 0.4,
        error_feedback: bool = True,
        staleness_decay: float = 0.6,
        priority_boost: Optional[Dict[int, float]] = None,
        layer_ranges: Optional[Dict[int, Tuple[int, int]]] = None,
        naive_combination: bool = False,
    ):
        # --- 防御性类型转换：YAML加载可能传入字符串 ---
        self.dim = int(dim)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.kappa_default = float(kappa_default)
        self.b_min = int(b_min)
        self.b_max = int(b_max)
        self.b_default = int(b_default)
        self.gini_threshold_high = float(gini_threshold_high)
        self.gini_threshold_low = float(gini_threshold_low)
        self.naive_combination = bool(naive_combination)
        self.priority_boost = priority_boost or {}
        self.layer_ranges = layer_ranges or {}

        # Error feedback buffer
        self.error_buffer = ErrorFeedbackBuffer(dim, staleness_decay) if error_feedback else None

        # Round tracking for adaptive bit-width
        self.current_round = 0
        self.epsilon_remaining = 3.0  # Will be updated by DP module

        # Communication tracking
        self.comm_bytes_history = []

    def compress(
        self,
        gradient: np.ndarray,
        round_idx: int = None,
        epsilon_remaining: float = None,
    ) -> Dict:
        """
        Full compression pipeline for one round.

        Args:
            gradient: Full gradient vector (d-dim)
            round_idx: Current FL round index
            epsilon_remaining: Remaining privacy budget ε

        Returns:
            Dict with:
                - sparse_values: Quantized sparse values (κ·d elements)
                - sparse_indices: Indices of sparse elements
                - kappa: Sparsification ratio used
                - b: Bit-width used
                - norm_max: ||g||_∞ for dequantization
                - mask: Boolean mask for error feedback
                - comm_bytes: Actual transmitted bytes (for communication measurement)
        """
        if round_idx is not None:
            self.current_round = round_idx
        if epsilon_remaining is not None:
            self.epsilon_remaining = epsilon_remaining

        # Step 1: Add error feedback buffer
        if self.error_buffer is not None:
            gradient = self.error_buffer.apply(gradient)

        # Step 2: Compute adaptive κ
        kappa = adaptive_kappa(
            gradient,
            self.kappa_min, self.kappa_max, self.kappa_default,
            self.gini_threshold_high, self.gini_threshold_low,
        )

        # Step 3: Top-k sparsification (with Matter-aware priority)
        sparse_values, sparse_indices, mask = top_k_sparsify(
            gradient, kappa,
            self.priority_boost, self.layer_ranges,
        )

        # Step 4: Compute adaptive bit-width
        b = adaptive_bit_width(
            self.current_round, 50,  # T=50 default
            self.epsilon_remaining,
            self.b_min, self.b_max,
        )

        # Step 5: Stochastic quantization
        quantized_values, norm_max = stochastic_quantize(sparse_values, b)

        # Step 6: Update error feedback buffer
        if self.error_buffer is not None:
            self.error_buffer.accumulate(gradient, quantized_values, sparse_indices)

        # Step 7: Compute actual communication bytes
        # Upload = κ·d·(b/8 + 2) bytes
        #   quantized values: κ·d · b/8 bytes (each value uses b bits)
        #   index encoding: κ·d · 2 bytes (int16 per index)
        #   norm_max: 4 bytes (float32 scalar)
        #   metadata: ~10 bytes
        n_sparse = len(sparse_indices)
        comm_bytes = n_sparse * (b / 8) + n_sparse * 2 + 4 + 10
        self.comm_bytes_history.append(comm_bytes)

        return {
            "sparse_values": quantized_values,
            "sparse_indices": sparse_indices,
            "kappa": kappa,
            "b": b,
            "norm_max": norm_max,
            "mask": mask,
            "comm_bytes": comm_bytes,
        }

    def decompress(self, compressed: Dict, dim: int = None) -> np.ndarray:
        """
        Decompress a sparse+quantized gradient back to full dimension.

        Args:
            compressed: Dict from compress()
            dim: Target dimension (default: self.dim)

        Returns:
            Reconstructed full-dimension gradient
        """
        d = dim or self.dim
        reconstructed = np.zeros(d, dtype=np.float32)
        reconstructed[compressed["sparse_indices"]] = compressed["sparse_values"]
        return reconstructed

    def get_avg_comm_bytes(self) -> float:
        """Average communication bytes per round."""
        if not self.comm_bytes_history:
            return 0.0
        return np.mean(self.comm_bytes_history)

    def get_total_comm_bytes(self) -> float:
        """Total communication bytes across all rounds."""
        return sum(self.comm_bytes_history)


# ============================================================
# Naive Combination (Top-k + QSGD serial, no co-optimization)
# ============================================================

class NaiveCombiner:
    """
    Naive serial combination: Top-k first, then QSGD quantization.
    No coupling between κ and b. Uses fixed κ=0.2, b=8.

    This is the key baseline to show that co-optimization outperforms
    naive serial application.
    """

    def __init__(self, dim: int, kappa: float = 0.2, b: int = 8):
        self.dim = dim
        self.kappa = kappa
        self.b = b
        self.error_buffer = ErrorFeedbackBuffer(dim)

    def compress(self, gradient: np.ndarray) -> Dict:
        """Naive combination: Top-k then QSGD."""
        # Add error feedback
        gradient = self.error_buffer.apply(gradient)

        # Top-k (fixed κ)
        sparse_values, sparse_indices, mask = top_k_sparsify(gradient, self.kappa)

        # QSGD quantization (fixed b=8)
        quantized_values, norm_max = stochastic_quantize(sparse_values, self.b)

        # Update error buffer
        self.error_buffer.accumulate(gradient, quantized_values, sparse_indices)

        # Communication: κ·d·(8/8 + 2) = κ·d·3 bytes + norm_max + metadata
        n_sparse = len(sparse_indices)
        comm_bytes = n_sparse * (self.b / 8) + n_sparse * 2 + 4 + 10

        return {
            "sparse_values": quantized_values,
            "sparse_indices": sparse_indices,
            "kappa": self.kappa,
            "b": self.b,
            "norm_max": norm_max,
            "mask": mask,
            "comm_bytes": comm_bytes,
        }

    def decompress(self, compressed: Dict) -> np.ndarray:
        """Same decompression as AdaGQ."""
        d = self.dim
        reconstructed = np.zeros(d, dtype=np.float32)
        reconstructed[compressed["sparse_indices"]] = compressed["sparse_values"]
        return reconstructed
