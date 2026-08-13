# AdaGQ-Matter: Deployment & Reproduction Guide

> **Paper**: *AdaGQ: Compute–Communication Co-Design for Privacy-Preserving Federated Learning on 6G Edge Gateways*  
> **License**: MIT

---

## 1. Overview

AdaGQ-Matter is a standalone PyTorch simulation of federated learning with adaptive gradient sparsification, quantisation, and differential privacy. It compares **7 methods** (FedAvg, FedProx, DP-FedAvg, Top-K Only, Quant Only, Naive Combination, AdaGQ) on IoT intrusion-detection benchmarks under non-IID (Dirichlet) data partitions.

**Key features:**

- Gini-driven adaptive Top-k sparsification (κ ∈ [0.1, 0.3])
- Privacy-budget-driven adaptive stochastic quantisation (b ∈ [4, 8] bit)
- Error-feedback compensation
- Matter feature grouping with priority boost (1.2× for early/intermediate layers)
- Sparsity-coupled Gaussian Rényi DP accounting (reports both ε_naive and ε_strict)
- Privacy attack evaluation: MIA, DLG, Inverting Gradients

---

## 2. Software & Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| PyTorch | 2.0+ | 2.0+ with CUDA 11.8+ |
| OS | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU mode available) | NVIDIA GPU ≥ 6 GB VRAM |
| Disk | 2 GB (code + data) | 10 GB (with CICIoT2023) |

**Python dependencies** (see `requirements.txt`):

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
opacus>=1.4.0
pyyaml>=6.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## 3. Dataset Acquisition

### 3.1 IoTID20 (Primary Benchmark)

| Property | Value |
|----------|-------|
| Records | 625,783 |
| Features | 79 (after dropping non-numeric columns) |
| Classes | 2 (binary: normal vs anomaly) |
| Source | Ullah & Mahmoud, *IEEE Access*, 2021 (DOI: [10.1109/ACCESS.2021.3124688](https://doi.org/10.1109/ACCESS.2021.3124688)) |
| Access | https://sites.google.com/view/iot-network-intrusion-dataset |

**Option A — Automatic download:**

```bash
python -m data.download_datasets --dataset iotid20 --output_dir /root/datasets/raw
```

**Option B — Manual download:**

1. Visit the dataset website above and download the CSV file.
2. Place it at `/root/datasets/raw/IoTID20.csv`.

### 3.2 CICIoT2023 (Optional Secondary Benchmark)

| Property | Value |
|----------|-------|
| Records | ~1.3M (subsampled from 46M) |
| Features | 46 |
| Classes | 2 (binary) |
| Source | HuggingFace: `lacg030175/CIC-IoT-2023-raw` |

**Automatic download:**

```bash
python -m data.download_datasets --dataset ciciot2023 --output_dir /root/datasets/raw
```

> **Note**: The CICIoT2023 dataset has known label-encoding inconsistencies in the public release. The preprocessing script handles binary remapping, but we recommend verifying the label distribution before running experiments.

---

## 4. Data Preprocessing

After downloading, preprocess the data into the FL-ready format (train/test splits + Dirichlet partitions):

```bash
# IoTID20 (required)
python -m data.preprocess \
    --dataset iotid20 \
    --raw_dir /root/datasets/raw \
    --output_dir /root/datasets/processed \
    --N 10 --alpha 0.5 --seed 1

# CICIoT2023 (optional)
python -m data.preprocess \
    --dataset ciciot2023 \
    --raw_dir /root/datasets/raw \
    --output_dir /root/datasets/processed \
    --N 10 --alpha 0.5 --seed 1 --subsample_ratio 1.0
```

**Expected output files** in `/root/datasets/processed/`:

```
iotid20_train.npz          # Training features + labels
iotid20_test.npz           # Test features + labels
iotid20_partitions.json    # Dirichlet partition indices for N=10 clients
ciciot2023_train.npz       # (if CICIoT2023 preprocessed)
ciciot2023_test.npz
ciciot2023_partitions.json
```

> **Data–code separation**: The code base is deliberately separated from the data. All configs reference `/root/datasets/processed/` as the default data directory. If that path is missing, `run_main.py` falls back to `data/processed/` within the project directory.

---

## 5. Project Structure

```
AdaGQ-Matter/
├── configs/                          # YAML configuration files
│   ├── base_cpu.yaml                 # Default CPU config (T=50)
│   ├── base_gpu.yaml                 # GPU full config (T=200)
│   ├── base_gpu_T50.yaml             # GPU fast config (T=50)
│   └── ablation.yaml                 # 9 ablation variants
├── core/                             # Core algorithm components
│   ├── compression.py                # Top-k sparsification + QSGD quantisation
│   ├── dp.py                         # Sparsity-coupled Gaussian RDP accountant
│   ├── feature_grouping.py           # Matter FG1–FG4 priority boost
│   └── aggregation.py                # Semi-synchronous federated aggregation
├── fl/                               # Federated learning simulation
│   ├── client.py                     # AdaGQ client (adaptive κ, b, σ, C)
│   ├── baseline_client.py            # Baseline clients (FedAvg, FedProx, DP-FedAvg, etc.)
│   ├── server.py                     # FL server (FedAvg aggregation)
│   └── baseline_strategy.py          # Strategy dispatch for baseline methods
├── models/                           # Neural network architectures
│   ├── dnn.py                        # AnomalyDNN (4-layer: 79→64→32→16→2, d≈4,306)
│   ├── lstm_ae.py                    # LSTM Autoencoder (optional larger model)
│   └── factory.py                    # Model factory (dispatch by name)
├── data/                             # Data utilities
│   ├── download_datasets.py          # Auto-download IoTID20 / CICIoT2023
│   ├── preprocess.py                 # Preprocess → npz + Dirichlet partitions
│   └── dirichlet_partition.py        # Dirichlet non-IID partitioner
├── experiments/                      # Experiment runners
│   ├── run_main.py                   # ★ Main runner (7 methods × 5 seeds)
│   ├── run_all.py                    # Full experiment suite
│   ├── run_ablation.py               # 9 ablation configurations
│   ├── run_dp_tradeoff.py            # ε sweep: {1, 3, 5, 8, 10, ∞}
│   └── run_attack.py                 # MIA + DLG + Inverting Gradients
├── attacks/                          # Privacy attack implementations
│   ├── mia.py                        # Membership Inference Attack
│   ├── dlg.py                        # Deep Leakage from Gradients
│   └── invgrad.py                    # Inverting Gradients Attack
├── utils/                            # Utility modules
│   ├── privacy_accountant.py         # RDP accountant (ε_naive + ε_strict)
│   ├── communication.py              # Byte-level communication tracker
│   ├── metrics.py                    # F1, accuracy, AUC, FPR computation
│   └── visualization.py              # Plotting helpers
├── scripts/                          # Operational scripts
│   ├── diagnose.py                   # Environment + data sanity check
│   ├── gen_summary.py                # Aggregate results → summary table
│   └── plot_paper_figures.py         # Generate paper figures (Morandi palette)
├── setup_autodl.sh                   # One-click setup (install + data download)
├── run_T50.sh                        # ★ Primary experiment runner (GPU, T=50)
├── rerun_dp_fedavg.sh                # Re-run DP-FedAvg at matched σ=1.83
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 6. Key Files Explained

### 6.1 Core Algorithm Components

| File | Purpose | Key Correspondence |
|------|---------|--------------------|
| `core/compression.py` | Top-k sparsification (Gini-driven adaptive κ) + QSGD stochastic quantisation (adaptive b) + error feedback | Algorithm 1 in the paper |
| `core/dp.py` | Gaussian noise injection on the κ·d-sparse payload; RDP accountant with strict (ε, δ)-DP conversion; adaptive clipping via EWMA | Algorithm 2 / Theorem 3 in the paper |
| `core/feature_grouping.py` | Divides 79 IoT features into 4 Matter groups (FG1–FG4); applies 1.2× priority boost to early/intermediate layers | Section 3.4 in the paper |
| `core/aggregation.py` | Semi-synchronous federated aggregation with staleness decay, packet loss, and latency simulation | Algorithm 3 in the paper |

### 6.2 Federated Learning Simulation

| File | Purpose |
|------|---------|
| `fl/client.py` | **AdaGQ client** — per-round: (1) compute gradient, (2) apply error feedback, (3) Gini-driven Top-k, (4) adaptive QSGD, (5) DP noise injection, (6) return sparse payload. Key class: `AdaGQClient` |
| `fl/baseline_client.py` | Baseline clients — `FedAvgClient`, `FedProxClient`, `DPFedAvgClient`, `TopKOnlyClient`, `QuantOnlyClient`, `NaiveCombinationClient` |
| `fl/server.py` | FL server — aggregates client payloads, updates global model, tracks cumulative ε and communication |
| `fl/baseline_strategy.py` | Strategy factory — dispatches client creation based on method name |

### 6.3 Models

| File | Architecture | Parameters |
|------|-------------|-----------|
| `models/dnn.py` | AnomalyDNN: Linear(79→64) → ReLU → Linear(64→32) → ReLU → Linear(32→16) → ReLU → Linear(16→2) | d = 4,306 |
| `models/lstm_ae.py` | LSTM Autoencoder (optional, for larger model experiments) | — |
| `models/factory.py` | `create_model(model_type, input_dim, output_dim, **kwargs)` — dispatches by name | — |

### 6.4 Experiment Runners

| File | Command | Description |
|------|---------|-------------|
| `experiments/run_main.py` | `python -m experiments.run_main --dataset iotid20 --method all --seed 1 --config configs/base_gpu_T50.yaml` | ★ Main experiment runner. Supports `--method {adagq,fedavg,fedprox,dp_fedavg,top_k_only,quant_only,naive_combination,all}`, `--seed 1-5`, `--alpha 0.5`, `--T 50`, `--N 10` |
| `experiments/run_ablation.py` | `python -m experiments.run_ablation --dataset iotid20 --config configs/ablation.yaml` | 9 ablation variants (no EF, no quant, no DP, fixed κ, etc.) |
| `experiments/run_dp_tradeoff.py` | `python -m experiments.run_dp_tradeoff --dataset iotid20 --config configs/base_gpu_T50.yaml` | ε sweep: {1, 3, 5, 8, 10, ∞} |
| `experiments/run_attack.py` | `python -m experiments.run_attack --dataset iotid20 --config configs/base_gpu_T50.yaml` | MIA + DLG + Inverting Gradients |
| `experiments/run_all.py` | `python -m experiments.run_all --config configs/base_gpu_T50.yaml` | Full experiment suite (all datasets × all methods × all seeds) |

### 6.5 Utility Scripts

| File | Command | Description |
|------|---------|-------------|
| `scripts/diagnose.py` | `python scripts/diagnose.py configs/base_gpu_T50.yaml /root/datasets/processed` | Checks environment, data integrity, model creation, and class distribution |
| `scripts/gen_summary.py` | `python scripts/gen_summary.py` | Reads all JSON results and produces `results/results_summary.txt` |
| `scripts/plot_paper_figures.py` | `python scripts/plot_paper_figures.py` | Generates paper figures (Morandi colour palette) from result JSON files |

---

## 7. Quick Start

### 7.1 One-Click Setup (AutoDL or similar GPU cloud)

```bash
# Step 1: Install dependencies + download data
bash setup_autodl.sh

# Step 2: Run the full experiment suite
bash run_T50.sh
```

### 7.2 Manual Setup

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download data
python -m data.download_datasets --dataset iotid20 --output_dir /root/datasets/raw

# Step 3: Preprocess
python -m data.preprocess \
    --dataset iotid20 \
    --raw_dir /root/datasets/raw \
    --output_dir /root/datasets/processed \
    --N 10 --alpha 0.5 --seed 1

# Step 4: Diagnose
python scripts/diagnose.py configs/base_gpu_T50.yaml /root/datasets/processed

# Step 5: Run single experiment
python -m experiments.run_main \
    --dataset iotid20 --method adagq --seed 1 \
    --alpha 0.5 --config configs/base_gpu_T50.yaml

# Step 6: Run all methods × 5 seeds
python -m experiments.run_main \
    --dataset iotid20 --method all \
    --config configs/base_gpu_T50.yaml

# Step 7: Generate summary
python scripts/gen_summary.py

# Step 8: Generate paper figures
python scripts/plot_paper_figures.py
```

---

## 8. Running the Matched-σ Comparison

The paper's key result (Table 4, footnote †) requires re-running DP-FedAvg at σ = 1.83 to match AdaGQ's per-coordinate noise level:

```bash
bash rerun_dp_fedavg.sh
```

This runs DP-FedAvg at σ = 1.83 across 5 seeds and produces the matched-σ F1 = 0.9578 ± 0.0419 result reported in Section 5.4 of the paper.

---

## 9. Configuration Reference

The primary configuration file for paper experiments is `configs/base_gpu_T50.yaml`. Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dataset` | `iotid20` | Primary benchmark |
| `input_dim_iotid20` | 79 | Number of input features |
| `model_type` | `dnn` | AnomalyDNN (4-layer) |
| `hidden_dims` | `[64, 32, 16]` | Hidden layer sizes |
| `T` | 50 | Number of FL rounds |
| `N` | 10 | Number of clients |
| `alpha` | 0.5 | Dirichlet concentration |
| `epsilon` | 3.0 | DP target budget |
| `delta` | 1e-5 | DP delta |
| `kappa_min/max/default` | 0.1/0.3/0.2 | Sparsity ratio range |
| `b_min/max/default` | 4/8/4 | Bit-width range |
| `gini_threshold_high/low` | 0.7/0.4 | Gini thresholds for κ adaptation |
| `feature_grouping_enabled` | true | Matter FG priority boost |
| `fg1/2_priority_boost` | 1.2 | Priority boost for early layers |
| `error_feedback_enabled` | true | Error feedback compensation |
| `dp_adaptive_clipping` | true | EWMA-based adaptive clipping |
| `device` | `cuda` | GPU acceleration |

---

## 10. Results

### 10.1 Output Format

Each experiment produces a JSON file and a human-readable TXT file:

```
results/
├── adagq_iotid20_alpha0.5_seed1.json
├── adagq_iotid20_alpha0.5_seed1.txt
├── fedavg_iotid20_alpha0.5_seed1.json
├── ...
├── results_summary.txt          # Aggregated by gen_summary.py
└── figures/                     # Generated by plot_paper_figures.py
```

**JSON structure:**

```json
{
  "method": "adagq",
  "dataset": "iotid20",
  "seed": 1,
  "alpha": 0.5,
  "accuracy": 0.9926,
  "f1_score": 0.9926,
  "auc": 0.9988,
  "fpr": 0.0067,
  "total_communication_mb": 1.055,
  "compression_ratio": 7.97,
  "dp_epsilon_naive": 3.147,
  "dp_epsilon_strict": 26.3,
  "dp_sigma": 1.83,
  "kappa_final": 0.2,
  "b_final": 4,
  "per_round_metrics": [...]
}
```

### 10.2 Reproducing Paper Tables

| Table | Command |
|-------|---------|
| Table 4 (Main Results) | `python -m experiments.run_main --dataset iotid20 --method all --config configs/base_gpu_T50.yaml` |
| Table 5 (Communication Breakdown) | Computed from `utils/communication.py` output in JSON |
| Table 6 (Adaptive Schedule) | Extracted from `per_round_metrics` in AdaGQ JSON results |
| Table 4 footnote † (Matched-σ) | `bash rerun_dp_fedavg.sh` |

### 10.3 Reproducing Paper Figures

```bash
python scripts/plot_paper_figures.py
```

This generates all 7 figures with the Morandi colour palette used in the paper. Output goes to `results/figures/`.

---

## 11. Ablation Study

9 ablation variants are defined in `configs/ablation.yaml`:

| Config Name | What is Disabled |
|-------------|-----------------|
| `full` | Nothing (full AdaGQ) |
| `no_ec` | Error feedback compensation |
| `no_quant` | Quantisation (b set to 32, i.e. full precision) |
| `no_dp` | Differential privacy (ε = ∞, σ = 0) |
| `no_adaptive_kappa` | Adaptive κ (fixed at 0.2) |
| `no_adaptive_window` | Adaptive window (fixed at 500 ms) |
| `naive_combination` | Naive sequential composition (Top-K then QSGD, no coupling) |
| `no_shuffling` | Shuffling amplification |
| `no_feature_grouping` | Matter feature grouping (priority boost = 1.0 for all) |

```bash
python -m experiments.run_ablation --dataset iotid20 --config configs/ablation.yaml
```

---

## 12. Privacy Attack Evaluation

Three attack types are implemented:

| Attack | File | Metric |
|--------|------|--------|
| Membership Inference (MIA) | `attacks/mia.py` | AUC (lower = more private) |
| Deep Leakage from Gradients (DLG) | `attacks/dlg.py` | MSE (higher = more private) |
| Inverting Gradients | `attacks/invgrad.py` | MSE (higher = more private) |

```bash
python -m experiments.run_attack --dataset iotid20 --config configs/base_gpu_T50.yaml
```

---

## 13. Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `TypeError: unsupported operand type(s)` in `dp.py` | PyYAML parses `1e-5` as string | `load_config()` in `run_main.py` auto-coerces numerics; ensure you are using the latest version |
| `ValueError: Training data has < 2 classes` | Corrupted or misaligned `.npz` file | Re-run `data/preprocess.py` to regenerate |
| `RuntimeError: Expected all tensors on the same device` | Global model on CPU, data on GPU | Fixed in latest `run_main.py` (global model moved to device) |
| `input_dim mismatch` | Config `input_dim` differs from actual data | `run_main.py` dynamically reads `input_dim` from `X_train.shape[1]` |
| `KeyError: 'final_metrics'` | Accessing results of a failed experiment | Check experiment log; `gen_summary.py` handles missing keys |
| CICIoT2023 label errors | Public release has encoding inconsistencies | Verify `y_train` class distribution after preprocessing; use `diagnose.py` |
| `run_all.py` fails with `FileNotFoundError` | Hardcoded working directory | Edit `run_all.py` line setting `cwd` to match your deployment path |

---

## 14. Citation

If you use this code, please cite:

```bibtex
@article{fan2026adagq,
  title={AdaGQ: Compute--Communication Co-Design for Privacy-Preserving Federated Learning on 6G Edge Gateways},
  author={Fan, X.},
  journal={Network},
  year={2026},
  note={Submitted to MDPI Network, Special Issue ``AI-Oriented 6G Networks''}
}
```

---

## 15. Contact

For questions or issues, please open an issue on the repository or contact the corresponding author.
