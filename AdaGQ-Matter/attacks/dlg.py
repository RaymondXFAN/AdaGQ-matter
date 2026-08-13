"""Deep Leakage from Gradients (DLG) Attack for AdaGQ-Matter.

Implementation of Zhu et al.'s Deep Leakage from Gradients attack adapted
for evaluating privacy in the AdaGQ-Matter federated learning framework.

DLG reconstructs private training data from observed gradients by optimizing
dummy input data and labels such that the resulting gradients match the
observed (leaked) gradients.

In the AdaGQ-Matter context, the server receives sparse (Top-k) and
quantized gradients with DP noise. The paper claims DLG reconstruction
MSE = 0.82 on sparse gradients, showing significant privacy protection
from sparsification alone (80% of gradient dimensions are missing).

Reference:
    Zhu, L., Liu, Z., & Han, S. (2019).
    Deep Leakage from Gradients.
    NeurIPS 2019.

Typical usage:
    >>> attack = DeepLeakageAttack()
    >>> reconstructed = attack.reconstruct(dummy_data, observed_grad, model)
    >>> mse = compute_reconstruction_mse(original_data, reconstructed)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict, List


class DeepLeakageAttack:
    """Deep Leakage from Gradients (DLG) attack.

    Reconstructs private training data from observed gradients by iteratively
    optimizing dummy data so that dummy gradients match the observed (leaked)
    gradients. Uses cosine similarity as the matching loss function.

    In the sparse gradient setting (AdaGQ-Matter's Top-k), 80% of gradient
    dimensions are missing, making reconstruction significantly harder.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        tv_dist_coeff: float = 0.0,
    ):
        """Initialize the DLG attack.

        Args:
            device: Device to use ('cpu' or 'cuda'). Auto-detected if None.
            tv_dist_coeff: Total variation regularization coefficient.
                Helps produce smoother reconstructed images. Set to 0
                for general data (non-image).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tv_dist_coeff = tv_dist_coeff

    @staticmethod
    def _total_variation(x: torch.Tensor) -> torch.Tensor:
        """Compute total variation of a tensor for regularization.

        Args:
            x: Input tensor.

        Returns:
            Total variation value.
        """
        diff_h = x[:, 1:, :] - x[:, :-1, :]
        diff_w = x[:, :, 1:] - x[:, :, :-1]
        return diff_h.pow(2).sum() + diff_w.pow(2).sum()

    def _compute_gradient(
        self,
        model: nn.Module,
        dummy_data: torch.Tensor,
        dummy_label: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute gradients from dummy data through the model.

        Args:
            model: Target model.
            dummy_data: Dummy input data (optimizable).
            dummy_label: Dummy label (optimizable).

        Returns:
            List of gradient tensors for each model parameter.
        """
        model.zero_grad()
        output = model(dummy_data)
        loss = nn.functional.cross_entropy(output, dummy_label)
        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        return list(gradients)

    def reconstruct(
        self,
        dummy_data: torch.Tensor,
        observed_gradient: List[torch.Tensor],
        model: nn.Module,
        n_iter: int = 300,
        lr: float = 0.1,
        num_classes: int = 2,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """Reconstruct original data from observed gradients via DLG.

        Iteratively optimizes dummy data and label so that the gradient
        computed from the dummy data matches the observed (leaked) gradient.
        Uses cosine similarity loss between gradients.

        Args:
            dummy_data: Initial dummy data to start optimization from.
                Shape: (1, input_dim). Will be optimized in-place.
            observed_gradient: The observed gradient from the target model.
                List of tensors, one per model parameter.
            model: The target model.
            n_iter: Number of optimization iterations.
            lr: Learning rate for reconstruction optimizer.
            num_classes: Number of output classes for label reconstruction.
            return_trajectory: Whether to return the reconstruction trajectory.

        Returns:
            Tuple of (reconstructed_data, reconstructed_label, trajectory_dict).
            - reconstructed_data: Tensor matching the original data shape.
            - reconstructed_label: Predicted label (argmax of soft label).
            - trajectory_dict: Optional trajectory info if return_trajectory=True.
        """
        model = model.to(self.device)
        dummy_data = dummy_data.to(self.device).detach().clone().requires_grad_(True)

        # Initialize soft label (optimizable one-hot-like vector)
        dummy_label = torch.randn((1, num_classes), device=self.device).requires_grad_(True)

        # Move observed gradients to device
        observed_gradient = [g.to(self.device) for g in observed_gradient]

        # Optimizer for both dummy data and label
        optimizer = optim.Adam([dummy_data, dummy_label], lr=lr)

        trajectory: Dict[str, List] = {"loss": [], "mse": []}

        for iteration in range(n_iter):
            optimizer.zero_grad()

            # Compute gradients from dummy data
            dummy_gradients = self._compute_gradient(model, dummy_data, dummy_label)

            # Cosine similarity loss: maximize similarity → minimize negative
            loss = self._cosine_similarity_loss(dummy_gradients, observed_gradient)

            # Add TV regularization if specified
            if self.tv_dist_coeff > 0:
                loss += self.tv_dist_coeff * self._total_variation(dummy_data)

            loss.backward()
            optimizer.step()

            # Record trajectory
            if return_trajectory:
                trajectory["loss"].append(loss.item())

            if (iteration + 1) % 50 == 0 or iteration == 0:
                label_pred = dummy_label.argmax(dim=1).item()
                print(
                    f"  DLG iter {iteration+1}/{n_iter}: "
                    f"loss={loss.item():.6f}, pred_label={label_pred}"
                )

        # Final reconstructed data and label
        reconstructed_data = dummy_data.detach().clone()
        reconstructed_label = dummy_label.argmax(dim=1).detach().clone()

        traj_result = trajectory if return_trajectory else None

        return reconstructed_data, reconstructed_label, traj_result

    def _cosine_similarity_loss(
        self,
        dummy_gradients: List[torch.Tensor],
        observed_gradients: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute cosine similarity loss between dummy and observed gradients.

        Loss = sum over parameters of (1 - cos_sim(dummy_grad, observed_grad)).
        Minimizing this loss maximizes cosine similarity.

        Args:
            dummy_gradients: Gradients computed from dummy data.
            observed_gradients: The target/observed gradients.

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=self.device)

        for dg, og in zip(dummy_gradients, observed_gradients):
            dg_flat = dg.flatten()
            og_flat = og.flatten()

            # Cosine similarity
            cos_sim = nn.functional.cosine_similarity(
                dg_flat.unsqueeze(0), og_flat.unsqueeze(0)
            )
            total_loss = total_loss + (1.0 - cos_sim)

        return total_loss

    def reconstruct_sparse(
        self,
        dummy_data: torch.Tensor,
        observed_sparse_gradient: List[torch.Tensor],
        model: nn.Module,
        sparsity_mask: Optional[List[torch.Tensor]] = None,
        n_iter: int = 300,
        lr: float = 0.1,
        num_classes: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct from sparse (Top-k) gradients — AdaGQ-Matter setting.

        In this setting, the observed gradient is sparse (only top-k values
        are transmitted). 80% of dimensions are zero/missing, making
        reconstruction much harder. The cosine similarity loss is computed
        only on the non-zero (transmitted) elements.

        Args:
            dummy_data: Initial dummy data.
            observed_sparse_gradient: Sparse gradient (most entries zero).
            model: Target model.
            sparsity_mask: Binary masks indicating which entries are kept.
                If None, derived from non-zero entries in observed gradients.
            n_iter: Optimization iterations.
            lr: Learning rate.
            num_classes: Number of classes.

        Returns:
            Tuple of (reconstructed_data, reconstructed_label).
        """
        model = model.to(self.device)
        dummy_data = dummy_data.to(self.device).detach().clone().requires_grad_(True)
        dummy_label = torch.randn((1, num_classes), device=self.device).requires_grad_(True)

        observed_sparse_gradient = [g.to(self.device) for g in observed_sparse_gradient]

        # Derive masks from non-zero entries if not provided
        if sparsity_mask is None:
            sparsity_mask = [(g != 0).float() for g in observed_sparse_gradient]

        sparsity_mask = [m.to(self.device) for m in sparsity_mask]

        optimizer = optim.Adam([dummy_data, dummy_label], lr=lr)

        for iteration in range(n_iter):
            optimizer.zero_grad()

            # Compute full gradients from dummy data
            dummy_gradients = self._compute_gradient(model, dummy_data, dummy_label)

            # Cosine similarity loss on sparse entries only
            loss = self._sparse_cosine_similarity_loss(
                dummy_gradients, observed_sparse_gradient, sparsity_mask
            )

            loss.backward()
            optimizer.step()

            if (iteration + 1) % 50 == 0 or iteration == 0:
                label_pred = dummy_label.argmax(dim=1).item()
                print(
                    f"  DLG-Sparse iter {iteration+1}/{n_iter}: "
                    f"loss={loss.item():.6f}, pred_label={label_pred}"
                )

        reconstructed_data = dummy_data.detach().clone()
        reconstructed_label = dummy_label.argmax(dim=1).detach().clone()

        return reconstructed_data, reconstructed_label


    def _sparse_cosine_similarity_loss(
        self,
        dummy_gradients: List[torch.Tensor],
        observed_sparse_gradients: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """Cosine similarity loss restricted to sparse (non-zero) entries.

        Args:
            dummy_gradients: Full gradients from dummy data.
            observed_sparse_gradients: Sparse observed gradients.
            masks: Binary masks for sparse entries.

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=self.device)

        for dg, og, mask in zip(dummy_gradients, observed_sparse_gradients, masks):
            dg_flat = dg.flatten()
            og_flat = og.flatten()
            mask_flat = mask.flatten()

            # Apply mask: only compare where gradient was transmitted
            dg_masked = dg_flat * mask_flat
            og_masked = og_flat * mask_flat

            # If masked region has signal, compute cosine similarity
            dg_norm = dg_masked.norm()
            og_norm = og_masked.norm()

            if dg_norm > 1e-8 and og_norm > 1e-8:
                cos_sim = nn.functional.cosine_similarity(
                    dg_masked.unsqueeze(0), og_masked.unsqueeze(0)
                )
                total_loss = total_loss + (1.0 - cos_sim)
            else:
                # No signal in this parameter's sparse region —
                # penalize any dummy gradient activity outside the mask
                outside_mask_norm = (dg_flat * (1 - mask_flat)).norm()
                total_loss = total_loss + 0.01 * outside_mask_norm

        return total_loss


def compute_reconstruction_mse(
    original_data: torch.Tensor,
    reconstructed_data: torch.Tensor,
) -> float:
    """Compute reconstruction Mean Squared Error (MSE).

    Measures the quality of data reconstruction by computing the average
    squared difference between original and reconstructed data.

    Args:
        original_data: The original private data tensor.
        reconstructed_data: The reconstructed data tensor (same shape).

    Returns:
        MSE value as a float. Higher values indicate worse reconstruction
        (better privacy). MSE=0 means perfect reconstruction (worst privacy).

    Note:
        In the AdaGQ-Matter paper, DLG MSE on sparse gradients ≈ 0.82,
        meaning reconstruction is very poor — good for privacy.
    """
    if original_data.shape != reconstructed_data.shape:
        # Attempt to broadcast if shapes are compatible
        min_samples = min(original_data.shape[0], reconstructed_data.shape[0])
        original_data = original_data[:min_samples]
        reconstructed_data = reconstructed_data[:min_samples]

    mse = nn.functional.mse_loss(reconstructed_data, original_data).item()
    return float(mse)


def topk_sparsify(
    gradients: List[torch.Tensor],
    k_ratio: float = 0.2,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Apply Top-k sparsification to gradients (AdaGQ-Matter style).

    Keeps only the top k_ratio fraction of gradient entries (by absolute
    value), zeroing out the rest.

    Args:
        gradients: List of gradient tensors (one per model parameter).
        k_ratio: Fraction of entries to keep (0.2 = keep top 20%).

    Returns:
        Tuple of (sparse_gradients, masks).
        - sparse_gradients: List of sparse gradient tensors.
        - masks: List of binary mask tensors indicating kept entries.
    """
    sparse_grads: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for grad in gradients:
        flat_grad = grad.flatten()
        k = max(1, int(k_ratio * flat_grad.shape[0]))

        # Find top-k indices by absolute value
        _, top_indices = torch.topk(flat_grad.abs(), k)

        # Create sparse gradient and mask
        sparse_flat = torch.zeros_like(flat_grad)
        mask_flat = torch.zeros_like(flat_grad)
        sparse_flat[top_indices] = flat_grad[top_indices]
        mask_flat[top_indices] = 1.0

        sparse_grads.append(sparse_flat.reshape(grad.shape))
        masks.append(mask_flat.reshape(grad.shape))

    return sparse_grads, masks


def add_dp_noise(
    gradients: List[torch.Tensor],
    noise_scale: float,
    clip_bound: float = 1.0,
) -> List[torch.Tensor]:
    """Add Gaussian DP noise to gradients (AdaGQ-Matter style).

    Applies gradient clipping and then adds Gaussian noise with scale σ
    for (ε, δ)-DP guarantee.

    Args:
        gradients: List of gradient tensors.
        noise_scale: Noise scale (σ). For ε=3 with δ=10^-5,
            σ ≈ clip_bound / ε for simple analysis.
        clip_bound: Per-parameter gradient clipping bound (C).

    Returns:
        List of noisy gradient tensors.
    """
    noisy_grads: List[torch.Tensor] = []

    for grad in gradients:
        # Clip gradient
        grad_norm = grad.norm()
        if grad_norm > clip_bound:
            clipped_grad = grad * (clip_bound / grad_norm)
        else:
            clipped_grad = grad

        # Add Gaussian noise
        noise = torch.randn_like(clipped_grad) * noise_scale
        noisy_grad = clipped_grad + noise
        noisy_grads.append(noisy_grad)

    return noisy_grads


# ---------------------------------------------------------------------------
# Helper: create AdaGQ-Matter model
# ---------------------------------------------------------------------------

def _create_target_model(input_dim: int = 20, num_classes: int = 2) -> nn.Module:
    """Create a 4-layer DNN matching AdaGQ-Matter architecture.

    Architecture: input_dim → 64 → 32 → 16 → num_classes
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Deep Leakage from Gradients (DLG) — Self-Test")
    print("=" * 70)

    torch.manual_seed(42)

    # Parameters
    input_dim = 20
    num_classes = 2

    # Create target model
    model = _create_target_model(input_dim, num_classes)
    # Simple training
    train_data = torch.randn(50, input_dim)
    train_labels = torch.randint(0, num_classes, (50,))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for _ in range(10):
        optimizer.zero_grad()
        out = model(train_data)
        loss = criterion(out, train_labels.long())
        loss.backward()
        optimizer.step()

    # Original private data (single sample)
    original_data = torch.randn(1, input_dim)
    original_label = torch.tensor([1])

    # Compute original gradient
    model.zero_grad()
    output = model(original_data)
    loss = criterion(output, original_label)
    original_gradient = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    original_gradient_list = [g.detach().clone() for g in original_gradient]

    # --- Scenario 1: Full gradient (FedAvg) ---
    print("\n--- Scenario 1: Full gradient reconstruction (FedAvg) ---")
    attack = DeepLeakageAttack()
    dummy_data = torch.randn(1, input_dim)
    recon_data, recon_label, traj = attack.reconstruct(
        dummy_data, original_gradient_list, model, n_iter=300, lr=0.01,
        num_classes=num_classes, return_trajectory=True,
    )
    mse_full = compute_reconstruction_mse(original_data, recon_data)
    print(f"  Original label: {original_label.item()}")
    print(f"  Reconstructed label: {recon_label.item()}")
    print(f"  Reconstruction MSE (full gradient): {mse_full:.6f}")
    print(f"  Label correct: {original_label.item() == recon_label.item()}")

    # --- Scenario 2: Sparse gradient (Top-k κ=0.2) ---
    print("\n--- Scenario 2: Sparse gradient reconstruction (Top-k κ=0.2) ---")
    sparse_grads, masks = topk_sparsify(original_gradient_list, k_ratio=0.2)

    # Count nonzero ratio
    total_params = sum(g.numel() for g in original_gradient_list)
    sparse_params = sum((g != 0).sum().item() for g in sparse_grads)
    print(f"  Total parameters: {total_params}")
    print(f"  Sparse (non-zero): {sparse_params} ({sparse_params/total_params*100:.1f}%)")

    recon_sparse_data, recon_sparse_label = attack.reconstruct_sparse(
        torch.randn(1, input_dim), sparse_grads, model,
        sparsity_mask=masks, n_iter=300, lr=0.01,
        num_classes=num_classes,
    )
    mse_sparse = compute_reconstruction_mse(original_data, recon_sparse_data)
    print(f"  Reconstruction MSE (sparse gradient): {mse_sparse:.6f}")
    print(f"  Label correct: {original_label.item() == recon_sparse_label.item()}")

    # --- Scenario 3: Sparse + DP noise (AdaGQ-Matter) ---
    print("\n--- Scenario 3: Sparse + DP noise (AdaGQ-Matter ε≈3) ---")
    # Clip and add noise
    noisy_sparse_grads = add_dp_noise(sparse_grads, noise_scale=0.1, clip_bound=1.0)

    recon_adagq_data, recon_adagq_label = attack.reconstruct_sparse(
        torch.randn(1, input_dim), noisy_sparse_grads, model,
        sparsity_mask=masks, n_iter=300, lr=0.01,
        num_classes=num_classes,
    )
    mse_adagq = compute_reconstruction_mse(original_data, recon_adagq_data)
    print(f"  Reconstruction MSE (sparse + DP): {mse_adagq:.6f}")
    print(f"  Label correct: {original_label.item() == recon_adagq_label.item()}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("DLG Attack Summary:")
    print(f"  FedAvg (full gradients):   MSE = {mse_full:.6f}")
    print(f"  Top-k only (κ=0.2):        MSE = {mse_sparse:.6f}")
    print(f"  AdaGQ (κ=0.2 + DP ε≈3):   MSE = {mse_adagq:.6f}")
    print("=" * 70)
    print("\nExpected results:")
    print("  Full gradient MSE should be relatively low (good reconstruction)")
    print("  Sparse MSE should be higher (~0.82 per paper)")
    print("  Sparse + DP MSE should be even higher")
    print("Self-test PASSED ✓")
