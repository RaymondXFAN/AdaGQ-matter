"""
AnomalyDNN — 4-layer lightweight DNN for IoT anomaly detection.

Architecture: Input(dim) → FC(64, ReLU) → FC(32, ReLU) → FC(16, ReLU) → Output(2)
- IoTID20: dim=79 → d ≈ 7,762 parameters (含bias), weights ≈ 30.6 KB
- CICIoT2023: dim=46 → d ≈ 5,650 parameters

The model uses CrossEntropyLoss (which includes log-softmax), so no softmax layer needed.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class AnomalyDNN(nn.Module):
    """4-layer DNN anomaly detection model (lightweight, CPU-friendly)."""

    def __init__(
        self,
        input_dim: int = 79,
        hidden_dims: List[int] = [64, 32, 16],
        output_dim: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        # CrossEntropyLoss includes log-softmax, no need for softmax here
        self.net = nn.Sequential(*layers)

        # Per-layer parameter grouping for Matter-aware feature grouping
        self._layer_param_ranges = self._compute_param_ranges()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_parameters(self) -> int:
        """Total number of trainable parameters (including bias)."""
        return sum(p.numel() for p in self.parameters())

    def get_gradients_flat(self) -> torch.Tensor:
        """Get all gradients as a flat 1D tensor (for compression/DP)."""
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.view(-1))
            else:
                grads.append(torch.zeros_like(p.data).view(-1))
        return torch.cat(grads)

    def set_gradients_from_flat(self, flat_grad: torch.Tensor) -> None:
        """Set gradients from a flat 1D tensor back to per-parameter tensors."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.grad = flat_grad[offset:offset + numel].view_as(p.data).clone()
            offset += numel

    def get_parameters_flat(self) -> torch.Tensor:
        """Get all parameters as a flat 1D tensor."""
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params)

    def set_parameters_from_flat(self, flat_params: torch.Tensor) -> None:
        """Set all parameters from a flat 1D tensor."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = flat_params[offset:offset + numel].view_as(p.data).clone()
            offset += numel

    def get_layer_param_indices(self) -> dict:
        """
        Return mapping from layer index → (start_idx, end_idx) in the flat parameter vector.
        Used for Matter-aware feature grouping to apply priority boosts to specific layers.
        """
        return self._layer_param_ranges

    def _compute_param_ranges(self) -> dict:
        """Compute per-layer parameter ranges in the flat vector."""
        ranges = {}
        offset = 0
        for i, p in enumerate(self.parameters()):
            numel = p.numel()
            ranges[i] = (offset, offset + numel)
            offset += numel
        return ranges

    def model_size_bytes(self) -> int:
        """Total model size in bytes (FP32 = 4 bytes per parameter)."""
        return self.num_parameters() * 4

    def model_size_kb(self) -> float:
        """Total model size in KB."""
        return self.model_size_bytes() / 1024.0

    def __repr__(self) -> str:
        d = self.num_parameters()
        kb = self.model_size_kb()
        return (
            f"AnomalyDNN(input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, output_dim={self.output_dim})\n"
            f"  d = {d} parameters (含bias)\n"
            f"  weights ≈ {kb:.1f} KB (FP32)"
        )


def build_model(
    input_dim: int = 79,
    hidden_dims: List[int] = [64, 32, 16],
    output_dim: int = 2,
) -> AnomalyDNN:
    """Factory function to build and validate a DNN model."""
    model = AnomalyDNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    print(f"[Model] {model}")
    return model


# --- Quick self-test ---
if __name__ == "__main__":
    print("=== IoTID20 model ===")
    m1 = build_model(input_dim=79)
    x1 = torch.randn(4, 79)
    y1 = m1(x1)
    print(f"  Output shape: {y1.shape}")
    print(f"  Flat param dim: {m1.get_parameters_flat().shape}")

    print("\n=== CICIoT2023 model ===")
    m2 = build_model(input_dim=46)
    x2 = torch.randn(4, 46)
    y2 = m2(x2)
    print(f"  Output shape: {y2.shape}")
    print(f"  Flat param dim: {m2.get_parameters_flat().shape}")
