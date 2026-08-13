"""
AdaGQ-Matter FL Client.

Implements the full AdaGQ-Matter pipeline:
1. Local training (E_local epochs on partitioned data)
2. Gradient processing: Top-k sparsification + adaptive quantization
3. Privacy protection: Sparsification-aware DP noise injection
4. Error compensation: Error feedback buffer management

This client integrates compression.py, dp.py, feature_grouping.py modules.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple, List

from models.dnn import AnomalyDNN
from core.compression import AdaGQCompressor, compute_gini
from core.dp import AdaGQDP
from core.feature_grouping import get_priority_boost_dict
from utils.metrics import compute_metrics


class FLClient:
    """Base FL client class (replaces Flower NumPyClient for standalone simulation)."""
    def get_parameters(self, config=None):
        raise NotImplementedError
    def set_parameters(self, parameters):
        raise NotImplementedError
    def fit(self, parameters, config):
        raise NotImplementedError
    def evaluate(self, parameters, config):
        raise NotImplementedError


class AdaGQMatterClient(FLClient):
    """
    AdaGQ-Matter Flower Client — implements full compression+DP pipeline.
    """

    def __init__(
        self,
        client_id: int,
        model: AnomalyDNN,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: dict,
        compressor: AdaGQCompressor,
        dp_module: AdaGQDP,
    ):
        self.client_id = client_id
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config

        # Training setup (防御性类型转换)
        self.device = config.get("device", "cpu")
        self.eta = float(config.get("eta", 0.01))
        self.E_local = int(config.get("E_local", 5))
        self.batch_size = int(config.get("batch_size", 32))

        # Compression & DP modules
        self.compressor = compressor
        self.dp_module = dp_module

        # Metrics tracking
        self.round_metrics = []

        # Move model to device
        self.model = self.model.to(self.device)

    def get_parameters(self, config=None):
        """Return model parameters as numpy arrays (for Flower)."""
        return [val.detach().cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Set model parameters from numpy arrays (from Flower server)."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Local training + AdaGQ-Matter compression + DP pipeline.

        Steps:
        1. Receive global model parameters from server
        2. Set local model
        3. Train E_local epochs on local data
        4. Compute gradient (difference from received parameters)
        5. Add error feedback buffer
        6. Top-k sparsification (with Matter-aware priority)
        7. Adaptive quantization
        8. DP noise injection (sparsification-aware)
        9. Return compressed+noisy gradient to server

        Args:
            parameters: Global model parameters from server
            config: Server-provided config (round number, etc.)

        Returns:
            (compressed_parameters, n_samples, metrics_dict)
        """
        round_idx = config.get("current_round", 0) if config else 0
        total_rounds = config.get("total_rounds", 50) if config else 50

        # Step 1-2: Set global model parameters
        self.set_parameters(parameters)
        global_params_flat = self.model.get_parameters_flat().detach().cpu().numpy()

        # Step 3: Local training
        self._local_train()

        # Step 4: Compute gradient (difference from global)
        local_params_flat = self.model.get_parameters_flat().detach().cpu().numpy()
        gradient = local_params_flat - global_params_flat

        # Step 5-7: Compression pipeline
        compressed = self.compressor.compress(
            gradient,
            round_idx=round_idx,
            epsilon_remaining=self.dp_module.accountant.get_epsilon_remaining(),
        )

        # Step 8: DP noise injection
        noisy_values, dp_info = self.dp_module.apply_dp(
            compressed["sparse_values"],
            kappa=compressed["kappa"],
            round_idx=round_idx,
            total_rounds=total_rounds,
        )

        # Step 9: Reconstruct full gradient from compressed format for aggregation
        # (Decompress sparse representation back to full dimension, then split per-layer)
        reconstructed = np.zeros(len(gradient), dtype=np.float32)
        reconstructed[compressed["sparse_indices"]] = noisy_values
        # NOTE: noisy_values 已经是原始尺度（stochastic_quantize 内部已做 norm_max 还原）
        #       不再需要额外乘 norm_max，否则会 double-scaling！

        # Split reconstructed gradient into per-layer arrays (reshape to original shape!)
        # Bug fix: 之前返回flat 1D数组, 导致load_state_dict size mismatch
        updated_flat = global_params_flat + reconstructed

        return_params = []
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            param_flat = updated_flat[offset:offset + numel].astype(np.float32)
            # ✅ reshape回原始参数形状 (如 [64,25] 而非 [1600])
            return_params.append(param_flat.reshape(p.shape))
            offset += numel

        # Metrics to send back to server
        metrics = {
            "client_id": self.client_id,
            "n_samples": len(self.X_train),
            "kappa": compressed["kappa"],
            "b": compressed["b"],
            "comm_bytes": compressed["comm_bytes"],
            "epsilon_current": dp_info.get("epsilon_current", 0.0),
            "grad_norm": dp_info.get("grad_norm", 0.0),
        }

        return return_params, len(self.X_train), metrics

    def evaluate(self, parameters, config):
        """Evaluate the global model on local test data."""
        self.set_parameters(parameters)

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            y = torch.tensor(self.y_test, dtype=torch.int64).to(self.device)
            outputs = self.model(X)
            preds = torch.argmax(outputs, dim=1)

            metrics = compute_metrics(y.cpu().numpy(), preds.cpu().numpy())

        return metrics["loss"], len(self.X_test), metrics

    def _local_train(self):
        """Train model locally for E_local epochs."""
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.eta)
        criterion = nn.CrossEntropyLoss()

        X = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y_train, dtype=torch.int64).to(self.device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.E_local):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()


def create_client(
    client_id: int,
    model: AnomalyDNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> AdaGQMatterClient:
    """
    Factory function to create an AdaGQ-Matter client with all modules.
    """
    d = model.num_parameters()
    priority_boost = get_priority_boost_dict(config.get("dataset", "iotid20"))
    layer_ranges = model.get_layer_param_indices()

    # Compression module (防御性类型转换：YAML可能传入字符串)
    compressor = AdaGQCompressor(
        dim=d,
        kappa_min=float(config.get("kappa_min", 0.1)),
        kappa_max=float(config.get("kappa_max", 0.3)),
        kappa_default=float(config.get("kappa_default", 0.2)),
        b_min=int(config.get("b_min", 4)),
        b_max=int(config.get("b_max", 8)),
        b_default=int(config.get("b_default", 4)),
        gini_threshold_high=float(config.get("gini_threshold_high", 0.7)),
        gini_threshold_low=float(config.get("gini_threshold_low", 0.4)),
        error_feedback=bool(config.get("error_feedback_enabled", True)),
        staleness_decay=float(config.get("staleness_decay", 0.6)),
        priority_boost=priority_boost,
        layer_ranges=layer_ranges,
        naive_combination=bool(config.get("naive_combination", False)),
    )

    # DP module
    dp_module = AdaGQDP(
        epsilon_target=float(config.get("epsilon", 3.0)),
        delta_target=float(config.get("delta", 1e-5)),
        d=d,
        initial_sigma=float(config.get("dp_noise_multiplier", 1.0)),
        initial_clipping_norm=float(config.get("dp_clipping_norm", 1.0)),
        kappa_default=float(config.get("kappa_default", 0.2)),
        adaptive_clipping=bool(config.get("dp_adaptive_clipping", True)),
        shuffling=bool(config.get("shuffling", True)),
        n_clients=int(config.get("N", 10)),
    )

    client = AdaGQMatterClient(
        client_id=client_id,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config=config,
        compressor=compressor,
        dp_module=dp_module,
    )

    return client
