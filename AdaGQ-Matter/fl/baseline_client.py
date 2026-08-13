"""
Baseline FL clients for comparison experiments.

Implements standard baseline methods:
1. FedAvg — Standard FL, no compression, no DP
2. FedProx — FL with proximal regularization (μ term)
3. DP-FedAvg — FL with fixed DP noise (no sparsification-aware)
4. Top-k Only — Sparsification without quantization/DP
5. Quantization Only — Quantization without sparsification
6. Naive Combination — Top-k + QSGD serial (no co-optimization)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional

from models.dnn import AnomalyDNN
from core.compression import top_k_sparsify, stochastic_quantize, NaiveCombiner
from core.dp import inject_dp_noise, RDPAccountant, adaptive_clip
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


class BaselineFedAvgClient(FLClient):
    """Standard FedAvg client — no compression, no DP."""

    def __init__(self, client_id, model, X_train, y_train, X_test, y_test, config):
        self.client_id = client_id
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.device = config.get("device", "cpu")
        self.eta = float(config.get("eta", 0.01))
        self.E_local = int(config.get("E_local", 5))
        self.batch_size = int(config.get("batch_size", 32))
        self.model = self.model.to(self.device)

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self._local_train()
        params = self.get_parameters()
        n_samples = len(self.X_train)
        # FedAvg: full gradient, no compression, no DP
        metrics = {
            "method": "fedavg",
            "comm_bytes": self.model.num_parameters() * 4,  # Full FP32 upload
        }
        return params, n_samples, metrics

    def evaluate(self, parameters, config):
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


class BaselineFedProxClient(BaselineFedAvgClient):
    """FedProx client — adds proximal term μ·||w - w_global||²/2."""

    def __init__(self, client_id, model, X_train, y_train, X_test, y_test, config, mu=0.01):
        super().__init__(client_id, model, X_train, y_train, X_test, y_test, config)
        self.mu = mu

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Store global parameters for proximal term
        global_params = [p.clone() for p in self.model.parameters()]
        self._local_train_prox(global_params)
        params = self.get_parameters()
        n_samples = len(self.X_train)
        metrics = {
            "method": "fedprox",
            "comm_bytes": self.model.num_parameters() * 4,
            "mu": self.mu,
        }
        return params, n_samples, metrics

    def _local_train_prox(self, global_params):
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
                # Add proximal term
                prox_term = 0.0
                for local_p, global_p in zip(self.model.parameters(), global_params):
                    prox_term += ((local_p - global_p) ** 2).sum()
                loss = loss + (self.mu / 2) * prox_term
                loss.backward()
                optimizer.step()


class BaselineDPFedAvgClient(BaselineFedAvgClient):
    """DP-FedAvg client — standard DP-SGD: gradient clipping + Gaussian noise + RDP accounting.

    Bug #19 fix: 之前实现 (1) 直接在训练后参数上加噪 (非梯度, 违反DP敏感度分析);
    (2) 噪声尺度 σ≈8 过大导致 F1 崩溃 (seed4=0.62, seed5=0.79);
    (3) metrics 返回键名 "epsilon" 与服务器读取的 "epsilon_current" 不匹配 → ε 永远=0.
    现在改为标准 DP-SGD: 梯度裁剪 → 高斯加噪 → 更新参数, 并用 RDPAccountant 追踪真实 ε.
    """

    def __init__(self, client_id, model, X_train, y_train, X_test, y_test, config,
                 epsilon=None, delta=None, clipping_norm=None):
        super().__init__(client_id, model, X_train, y_train, X_test, y_test, config)
        # 防御性类型转换：YAML可能传入字符串
        self.epsilon = float(epsilon if epsilon is not None else config.get("epsilon", 3.0))
        self.delta = float(delta if delta is not None else config.get("delta", 1e-5))
        self.clipping_norm = float(clipping_norm if clipping_norm is not None
                                   else config.get("dp_clipping_norm", 1.0))
        self.n_clients = int(config.get("N", 10))

        # RDP accountant (kappa=1.0: 标准DP-SGD, 无稀疏化增强)
        self.accountant = RDPAccountant(
            epsilon_target=self.epsilon,
            delta_target=self.delta,
            sigma=1.0,
            kappa_default=1.0,
        )

    def fit(self, parameters, config):
        round_idx = config.get("current_round", 0) if config else 0
        total_rounds = config.get("total_rounds", 50) if config else 50

        self.set_parameters(parameters)
        global_flat = self.model.get_parameters_flat().detach().cpu().numpy()
        self._local_train_dp()
        local_flat = self.model.get_parameters_flat().detach().cpu().numpy()

        # Compute gradient (parameter difference)
        gradient = local_flat - global_flat
        d = len(gradient)

        # Step 1: Clip gradient to norm C (standard DP-SGD)
        clipped_gradient, grad_norm = adaptive_clip(gradient, self.clipping_norm)

        # Step 2: Adaptive noise multiplier σ for this round (κ=1.0, no sparsification)
        sigma = self.accountant.compute_adaptive_noise(
            round_idx, total_rounds, kappa=1.0, n_clients=self.n_clients
        )

        # Step 3: Add Gaussian noise to full gradient (noise_scale = σ·C/√d)
        noisy_gradient = inject_dp_noise(
            clipped_gradient, sigma, self.clipping_norm, kappa=1.0, d=d
        )

        # Step 4: Accumulate RDP → current cumulative ε
        epsilon_used = self.accountant.accumulate_round(
            sigma, kappa=1.0, n_clients=self.n_clients, shuffling=False
        )

        # Updated parameters = global + noisy gradient (标准DP-SGD参数服务器形式)
        updated_flat = global_flat + noisy_gradient

        # Split back to per-layer arrays (reshape to original shape!)
        return_params = []
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            param_flat = updated_flat[offset:offset + numel].astype(np.float32)
            return_params.append(param_flat.reshape(p.shape))  # ✅ reshape回原始形状
            offset += numel

        metrics = {
            "method": "dp_fedavg",
            "comm_bytes": d * 4,  # Full FP32 upload (no compression)
            "epsilon_current": epsilon_used,  # ✅ 键名修复: 服务器读取的是 epsilon_current
            "epsilon_remaining": self.accountant.get_epsilon_remaining(),
            "sigma": sigma,
            "grad_norm": grad_norm,
        }
        return return_params, len(self.X_train), metrics

    def _local_train_dp(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.eta)
        criterion = nn.CrossEntropyLoss()
        X = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y_train, dtype=torch.long).to(self.device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.E_local):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                # Clip gradients per-batch (标准 DP-SGD)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_norm)
                optimizer.step()


class BaselineTopKOnlyClient(FLClient):
    """Top-k Only — sparsification without quantization or DP."""

    def __init__(self, client_id, model, X_train, y_train, X_test, y_test, config, kappa=0.2):
        self.client_id = client_id
        self.model = model.to(config.get("device", "cpu"))
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.kappa = float(kappa)
        self.device = config.get("device", "cpu")
        self.eta = float(config.get("eta", 0.01))
        self.E_local = int(config.get("E_local", 5))
        self.batch_size = int(config.get("batch_size", 32))

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_flat = self.model.get_parameters_flat().detach().cpu().numpy()
        self._local_train()
        local_flat = self.model.get_parameters_flat().detach().cpu().numpy()
        gradient = local_flat - global_flat

        # Top-k sparsification (no quantization, no DP)
        sparse_values, sparse_indices, mask = top_k_sparsify(gradient, self.kappa)

        # Communication: κ·d·4 (FP32 values) + κ·d·2 (int16 indices)
        d = len(gradient)
        k = len(sparse_indices)
        comm_bytes = k * 4 + k * 2 + 4 + 10

        # Return: reconstruct full gradient from sparse, then add back to global params
        reconstructed = np.zeros(d, dtype=np.float32)
        reconstructed[sparse_indices] = sparse_values

        # Fix: 必须返回 global_flat + reconstructed (= 更新后的模型参数)
        #      而非仅返回 reconstructed (= 梯度)，否则服务器误把梯度当参数聚合
        updated_flat = global_flat + reconstructed

        # Split back to per-layer arrays (reshape to original shape!)
        # Bug fix: 之前返回flat 1D数组, 导致load_state_dict size mismatch
        return_params = []
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            param_flat = updated_flat[offset:offset + numel].astype(np.float32)
            return_params.append(param_flat.reshape(p.shape))  # ✅ reshape回原始形状
            offset += numel

        metrics = {"method": "top_k_only", "kappa": self.kappa, "comm_bytes": comm_bytes}
        return return_params, len(self.X_train), metrics

    def evaluate(self, parameters, config):
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
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.eta)
        criterion = nn.CrossEntropyLoss()
        X = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y_train, dtype=torch.long).to(self.device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.E_local):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()


class BaselineQuantOnlyClient(BaselineFedAvgClient):
    """Quantization Only — quantization without sparsification or DP."""

    def __init__(self, client_id, model, X_train, y_train, X_test, y_test, config, b=8):
        super().__init__(client_id, model, X_train, y_train, X_test, y_test, config)
        self.b = int(b)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self._local_train()

        # Quantize full gradient (QSGD style)
        params_flat = self.model.get_parameters_flat().detach().cpu().numpy()
        # QSGD: quantize each parameter to b bits
        quantized, norm_max = stochastic_quantize(params_flat, self.b)

        # Communication: d · b/8 bytes (no index overhead since full gradient)
        d = len(params_flat)
        comm_bytes = d * (self.b / 8) + 4 + 10

        # Dequantize for server
        # Split back to per-layer (reshape to original shape!)
        # Bug fix: 之前返回flat 1D数组, 导致load_state_dict size mismatch
        return_params = []
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            param_flat = quantized[offset:offset + numel].astype(np.float32)
            return_params.append(param_flat.reshape(p.shape))  # ✅ reshape回原始形状
            offset += numel

        metrics = {"method": "quant_only", "b": self.b, "comm_bytes": comm_bytes}
        return return_params, len(self.X_train), metrics


class BaselineNaiveCombClient(FLClient):
    """Naive Combination — Top-k + QSGD serial, no co-optimization."""

    def __init__(self, client_id, model, X_train, y_train, X_test, y_test, config):
        self.client_id = client_id
        self.model = model.to(config.get("device", "cpu"))
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.device = config.get("device", "cpu")
        self.eta = float(config.get("eta", 0.01))
        self.E_local = int(config.get("E_local", 5))
        self.batch_size = int(config.get("batch_size", 32))

        d = model.num_parameters()
        self.combiner = NaiveCombiner(d, kappa=0.2, b=8)

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_flat = self.model.get_parameters_flat().detach().cpu().numpy()
        self._local_train()
        local_flat = self.model.get_parameters_flat().detach().cpu().numpy()
        gradient = local_flat - global_flat

        compressed = self.combiner.compress(gradient)

        # Return reconstructed gradient + global params (= 更新后的模型参数)
        # Split back to per-layer (reshape to original shape!)
        # Bug fix: 之前返回flat 1D数组, 导致load_state_dict size mismatch
        reconstructed = self.combiner.decompress(compressed)
        updated_flat = global_flat + reconstructed  # Fix: 返回参数而非梯度
        return_params = []
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            param_flat = updated_flat[offset:offset + numel].astype(np.float32)
            return_params.append(param_flat.reshape(p.shape))  # ✅ reshape回原始形状
            offset += numel

        metrics = {
            "method": "naive_combination",
            "comm_bytes": compressed["comm_bytes"],
            "kappa": compressed["kappa"],
            "b": compressed["b"],
        }
        return return_params, len(self.X_train), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            y = torch.tensor(self.y_test, dtype=torch.long).to(self.device)
            outputs = self.model(X)
            preds = torch.argmax(outputs, dim=1)
            metrics = compute_metrics(y.cpu().numpy(), preds.cpu().numpy())
        return metrics["loss"], len(self.X_test), metrics

    def _local_train(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.eta)
        criterion = nn.CrossEntropyLoss()
        X = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y_train, dtype=torch.long).to(self.device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.E_local):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
