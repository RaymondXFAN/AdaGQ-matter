"""
LSTM-AE — LSTM Autoencoder for anomaly detection (GPU-only, d ≈ 45k+).

This model is designed for larger-model validation experiments (Section 6.8).
It requires GPU for training; CPU is too slow for practical FL rounds.

Architecture:
  Encoder: LSTM(input_dim, hidden_dim, num_layers)
  Decoder: LSTM(hidden_dim, hidden_dim, num_layers) → Linear(hidden_dim, input_dim)
  Anomaly head: Linear(hidden_dim, 2)
"""

import torch
import torch.nn as nn
from typing import List


class LSTMAnomalyDetector(nn.Module):
    """LSTM-based anomaly detection model (encoder + classification head)."""

    def __init__(
        self,
        input_dim: int = 79,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 2,
        seq_len: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_len = seq_len

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # Classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, features) or (batch, seq_len, features)
        Returns:
            (batch, output_dim) classification logits
        """
        if x.dim() == 2:
            # Repeat single timestep to create sequence
            x = x.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, seq_len, F)

        # Encode
        _, (h_n, _) = self.encoder(x)  # h_n: (num_layers, B, hidden_dim)
        last_hidden = h_n[-1]  # (B, hidden_dim) — last layer's hidden state

        # Classify
        logits = self.classifier(last_hidden)  # (B, output_dim)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_parameters_flat(self) -> torch.Tensor:
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params)

    def set_parameters_from_flat(self, flat_params: torch.Tensor) -> None:
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = flat_params[offset:offset + numel].view_as(p.data).clone()
            offset += numel

    def get_gradients_flat(self) -> torch.Tensor:
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.view(-1))
            else:
                grads.append(torch.zeros_like(p.data).view(-1))
        return torch.cat(grads)

    def set_gradients_from_flat(self, flat_grad: torch.Tensor) -> None:
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.grad = flat_grad[offset:offset + numel].view_as(p.data).clone()
            offset += numel

    def model_size_kb(self) -> float:
        return self.num_parameters() * 4 / 1024.0

    def __repr__(self) -> str:
        d = self.num_parameters()
        kb = self.model_size_kb()
        return (
            f"LSTMAnomalyDetector(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers})\n"
            f"  d = {d} parameters\n"
            f"  weights ≈ {kb:.1f} KB (FP32)"
        )


if __name__ == "__main__":
    m = LSTMAnomalyDetector(input_dim=79, hidden_dim=128, num_layers=2)
    print(m)
    x = torch.randn(4, 79)
    y = m(x)
    print(f"Output shape: {y.shape}")
