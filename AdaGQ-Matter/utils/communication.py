"""
Communication Cost Measurement Module.

Measures upload/download communication overhead for federated learning:
- AdaGQ compressed upload: Top-k sparsified + quantized gradients
- FedAvg baseline upload: Full FP32 model parameters
- Compression ratio computation
- Cumulative upload tracking across rounds

Formula (Section 4.3, Communication Cost Analysis):
  Compressed upload = n_sparse * (b/8) + n_sparse * 2 + 4 (norm_max) + 10 (metadata)
  FedAvg upload     = d * 4 + metadata

Reference: AdaGQ-Matter, Section 4.3
"""

import math
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ============================================================
# Upload Cost Computation
# ============================================================

def compute_upload_bytes(
    n_sparse: int,
    b: int,
    d: int,
) -> int:
    """
    Compute compressed upload size in bytes for AdaGQ-Matter.

    Each sparse value is quantized to b bits:
    - Quantized values:  n_sparse * (b / 8) bytes
    - Indices (int32):   n_sparse * 4 bytes  → but we use int16 for indices, n_sparse * 2 bytes
    - Norm max (float32): 4 bytes
    - Metadata:         10 bytes (kappa, bit-width, round_id, etc.)

    Total = n_sparse*(b/8) + n_sparse*2 + 4 + 10

    Args:
        n_sparse: Number of sparse elements (κ * d)
        b: Quantization bit-width (4 ≤ b ≤ 8)
        d: Full model dimension (total parameters)

    Returns:
        Upload size in bytes
    """
    quantized_bytes = math.ceil(n_sparse * b / 8)  # Fix: ceil而非int截断，避免丢失字节
    index_bytes = n_sparse * 2       # int16 indices
    norm_max_bytes = 4                # float32
    metadata_bytes = 10               # kappa, b, round_id, etc.

    total = int(quantized_bytes) + index_bytes + norm_max_bytes + metadata_bytes
    return total


def compute_fedavg_upload_bytes(
    d: int,
    metadata_bytes: int = 10,
) -> int:
    """
    Compute FedAvg upload size in bytes (uncompressed FP32 model).

    Each parameter is float32 (4 bytes):
    - Parameters: d * 4 bytes
    - Metadata:   metadata_bytes bytes (round_id, client_id, etc.)

    Args:
        d: Full model dimension (total parameters)
        metadata_bytes: Metadata overhead

    Returns:
        Upload size in bytes
    """
    return d * 4 + metadata_bytes


def compute_compression_ratio(
    fedavg_bytes: int,
    compressed_bytes: int,
) -> float:
    """
    Compute compression ratio vs FedAvg baseline.

    ratio = fedavg_bytes / compressed_bytes

    Args:
        fedavg_bytes: FedAvg upload size in bytes
        compressed_bytes: AdaGQ compressed upload size in bytes

    Returns:
        Compression ratio (higher = better compression)
    """
    if compressed_bytes == 0:
        return float("inf")
    return fedavg_bytes / compressed_bytes


def compute_cumulative_upload(
    bytes_per_round: int,
    n_rounds: int,
    n_clients: int,
) -> int:
    """
    Compute total cumulative upload across all rounds and clients.

    Total = bytes_per_round * n_rounds * n_clients

    Args:
        bytes_per_round: Upload size per round per client
        n_rounds: Number of FL rounds
        n_clients: Number of participating clients per round

    Returns:
        Total cumulative upload in bytes
    """
    return bytes_per_round * n_rounds * n_clients


# ============================================================
# Communication Tracker
# ============================================================

@dataclass
class RoundRecord:
    """Record of communication cost for a single round."""
    round_id: int
    client_id: int
    method: str           # "adagq" or "fedavg"
    upload_bytes: int
    n_sparse: int = 0
    b: int = 0
    d: int = 0


class CommunicationTracker:
    """
    Track per-round communication costs and output statistical summary.

    Usage:
        tracker = CommunicationTracker()
        tracker.record(round_id=0, client_id=0, method="adagq",
                       upload_bytes=500, n_sparse=100, b=4, d=1000)
        tracker.record(round_id=0, client_id=1, method="fedavg",
                       upload_bytes=4010, d=1000)
        summary = tracker.summary()
    """

    def __init__(self) -> None:
        self.records: List[RoundRecord] = []
        self._round_totals: Dict[int, Dict[str, int]] = {}

    def record(
        self,
        round_id: int,
        client_id: int,
        method: str,
        upload_bytes: int,
        n_sparse: int = 0,
        b: int = 0,
        d: int = 0,
    ) -> None:
        """
        Record a communication event.

        Args:
            round_id: FL round index
            client_id: Client identifier
            method: "adagq" or "fedavg"
            upload_bytes: Upload size in bytes
            n_sparse: Number of sparse elements (for adagq)
            b: Quantization bit-width (for adagq)
            d: Full model dimension
        """
        rec = RoundRecord(
            round_id=round_id,
            client_id=client_id,
            method=method,
            upload_bytes=upload_bytes,
            n_sparse=n_sparse,
            b=b,
            d=d,
        )
        self.records.append(rec)

        # Update round totals
        if round_id not in self._round_totals:
            self._round_totals[round_id] = {}
        self._round_totals[round_id][method] = \
            self._round_totals[round_id].get(method, 0) + upload_bytes

    def total_upload(self, method: Optional[str] = None) -> int:
        """
        Total upload bytes across all recorded events.

        Args:
            method: Filter by method ("adagq" or "fedavg"). None = all.

        Returns:
            Total bytes
        """
        if method:
            return sum(r.upload_bytes for r in self.records if r.method == method)
        return sum(r.upload_bytes for r in self.records)

    def avg_upload_per_round(self, method: Optional[str] = None) -> float:
        """
        Average upload bytes per round (summed across clients).

        Args:
            method: Filter by method. None = all.

        Returns:
            Average bytes per round
        """
        if not self._round_totals:
            return 0.0

        totals = []
        for rid, method_dict in self._round_totals.items():
            if method:
                totals.append(method_dict.get(method, 0))
            else:
                totals.append(sum(method_dict.values()))

        return np.mean(totals) if totals else 0.0

    def compression_ratio_vs_fedavg(self) -> float:
        """
        Overall compression ratio: FedAvg total / AdaGQ total.

        Returns:
            Compression ratio (inf if no adagq records)
        """
        fedavg_total = self.total_upload(method="fedavg")
        adagq_total = self.total_upload(method="adagq")
        if adagq_total == 0:
            return float("inf")
        return fedavg_total / adagq_total

    def summary(self) -> str:
        """
        Generate a statistical summary of tracked communication costs.

        Returns:
            Multi-line summary string
        """
        lines = ["=== Communication Cost Summary ==="]

        n_records = len(self.records)
        n_rounds = len(self._round_totals)
        lines.append(f"Total records: {n_records}")
        lines.append(f"Total rounds:  {n_rounds}")

        # Per-method stats
        for method in ["adagq", "fedavg"]:
            method_records = [r for r in self.records if r.method == method]
            if not method_records:
                continue
            total = sum(r.upload_bytes for r in method_records)
            avg = np.mean([r.upload_bytes for r in method_records])
            lines.append(f"\n[{method.upper()}]")
            lines.append(f"  Total upload:    {total:>10,} bytes ({total/1024:.1f} KB)")
            lines.append(f"  Avg per client:  {avg:>10.1f} bytes ({avg/1024:.2f} KB)")
            lines.append(f"  Avg per round:   {self.avg_upload_per_round(method):>10.1f} bytes")

        # Compression ratio
        ratio = self.compression_ratio_vs_fedavg()
        if ratio != float("inf"):
            lines.append(f"\nCompression ratio (FedAvg/AdaGQ): {ratio:.2f}x")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset tracker for a new experiment."""
        self.records = []
        self._round_totals = {}


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":
    print("=== Upload Cost Computation Test ===")

    # IoTID20: d ≈ 7,762, κ=0.2, b=4
    d_iotid20 = 7762
    kappa = 0.2
    b = 4
    n_sparse = int(kappa * d_iotid20)

    compressed = compute_upload_bytes(n_sparse=n_sparse, b=b, d=d_iotid20)
    fedavg = compute_fedavg_upload_bytes(d=d_iotid20)
    ratio = compute_compression_ratio(fedavg, compressed)
    cumulative_compressed = compute_cumulative_upload(compressed, n_rounds=50, n_clients=10)
    cumulative_fedavg = compute_cumulative_upload(fedavg, n_rounds=50, n_clients=10)

    print(f"  IoTID20 (d={d_iotid20}, κ={kappa}, b={b})")
    print(f"  n_sparse = {n_sparse}")
    print(f"  AdaGQ upload:   {compressed:>6} bytes ({compressed/1024:.2f} KB)")
    print(f"  FedAvg upload:  {fedavg:>6} bytes ({fedavg/1024:.2f} KB)")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Cumulative AdaGQ (50 rounds, 10 clients): "
          f"{cumulative_compressed/1024:.1f} KB ({cumulative_compressed/1024/1024:.3f} MB)")
    print(f"  Cumulative FedAvg (50 rounds, 10 clients): "
          f"{cumulative_fedavg/1024:.1f} KB ({cumulative_fedavg/1024/1024:.3f} MB)")

    print("\n=== CICIoT2023 Test ===")
    d_cic = 5650
    n_sparse_cic = int(kappa * d_cic)
    compressed_cic = compute_upload_bytes(n_sparse=n_sparse_cic, b=b, d=d_cic)
    fedavg_cic = compute_fedavg_upload_bytes(d=d_cic)
    ratio_cic = compute_compression_ratio(fedavg_cic, compressed_cic)
    print(f"  CICIoT2023 (d={d_cic}, κ={kappa}, b={b})")
    print(f"  AdaGQ upload:  {compressed_cic:>6} bytes ({compressed_cic/1024:.2f} KB)")
    print(f"  FedAvg upload: {fedavg_cic:>6} bytes ({fedavg_cic/1024:.2f} KB)")
    print(f"  Compression ratio: {ratio_cic:.2f}x")

    print("\n=== CommunicationTracker Test ===")
    tracker = CommunicationTracker()
    for r in range(5):
        for c in range(10):
            tracker.record(round_id=r, client_id=c, method="adagq",
                           upload_bytes=compressed, n_sparse=n_sparse, b=b, d=d_iotid20)
            tracker.record(round_id=r, client_id=c, method="fedavg",
                           upload_bytes=fedavg, d=d_iotid20)

    print(tracker.summary())

    tracker.reset()
    print("\nTracker reset. Records:", len(tracker.records))
