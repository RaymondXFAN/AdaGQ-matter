# AdaGQ-Matter

**Adaptive Gradient Sparsification, Quantization, and Differential Privacy for Federated Learning-based IoT Anomaly Detection with Matter Feature Grouping**

<p align="center">
  <img src="https://img.shields.io/badge/status-Paper%20Under%20Review-yellow" alt="Status"/>
  <img src="https://img.shields.io/badge/framework-PyTorch%202.0+-orange" alt="Framework"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.8+-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/FL-Standalone%20Simulation-red" alt="FL"/>
</p>

---

## 📖 Overview

**AdaGQ-Matter** is a communication-efficient and privacy-preserving framework for Federated Learning (FL)-based IoT anomaly detection. It co-optimizes **adaptive gradient sparsification (Top-k)**, **adaptive quantization**, **error compensation**, **Matter-aware feature grouping**, and **sparsification-aware differential privacy** to achieve 85%+ communication reduction while maintaining model utility under formal (ε, δ)-DP guarantees.

The core insight: in sparsified FL, only κ·d out of d gradient dimensions are transmitted per round. By injecting DP noise only on those κ·d dimensions, the effective noise per dimension is enhanced by factor √(1/κ), providing a natural "privacy dividend" from compression. AdaGQ-Matter further couples κ(t), b(t), and σ(t) via a unified adaptive controller that responds to gradient Gini coefficient, remaining privacy budget, and round progress.

---

## 🧠 Algorithm Architecture

AdaGQ-Matter consists of **5 co-optimized components**:

| Component | Module | Key Mechanism |
|-----------|--------|---------------|
| **Top-k Sparsification** | `core/compression.py` | Adaptive κ(t) based on gradient Gini coefficient |
| **Adaptive Quantization** | `core/compression.py` | Bit-width b(t) coupled with κ(t) and ε_remaining |
| **Error Compensation** | `core/compression.py` | Error feedback buffer with stale decay |
| **Matter Feature Grouping** | `core/feature_grouping.py` | Priority boost (1.2×) for Sensor/Actuator layers |
| **Sparsification-Aware DP** | `core/dp.py` | RDP accounting + noise on κ·d dims + adaptive clipping |

**Coupling logic** (per round t):
1. Compute Gini(g_t) → determine κ(t) ∈ [κ_min, κ_max]
2. b(t) = clamp(⌈b_max · ε_remaining / (ε_target · (T−t)/T⌋, b_min, b_max)
3. σ(t) = √(2·κ(t)·ln(1/δ)) / ε_remaining  (sparsification-aware)
4. Top-k → Quantize → DP noise → Error feedback accumulation

**Semi-synchronous aggregation** (`core/aggregation.py`): adaptive window W_agg(t) with stale-weighted averaging δ^s.

---

## 📊 Datasets

| Dataset | Features | Samples | Source |
|---------|----------|---------|--------|
| **IoTID20** | 79 (numerical) | ~625K | [Mendeley Data](https://data.mendeley.com/datasets/nzc7grj6jm) / [HuggingFace](https://huggingface.co/datasets/maruuf/iotid20_dataset) |
| **CICIoT2023** | 46 (numerical) | ~1.3M (subsampled) | [HuggingFace](https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-raw) / [CIC UNB](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |

Both datasets are preprocessed into:
- **Binary classification**: Normal (0) vs Attack (1)
- **Z-score standardization** (StandardScaler)
- **Dirichlet Non-IID partition**: α=0.5, N=10 clients
- **Train/test split**: 80%/20% stratified

Output format: `{dataset}_train.npz`, `{dataset}_test.npz`, `{dataset}_partitions.json`

---

## 🖥️ Environment

### Software Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Runtime |
| PyTorch | ≥ 2.0 | Deep learning framework |
| NumPy | ≥ 1.24 | Numerical computation |
| scikit-learn | ≥ 1.3 | Metrics & preprocessing |
| Pandas | ≥ 2.0 | Data loading |
| PyYAML | ≥ 6.0 | Configuration |
| SciPy | ≥ 1.11 | Statistical utilities |
| Matplotlib | ≥ 3.7 | Visualization |
| Seaborn | ≥ 0.12 | Visualization |
| Opacus | ≥ 1.4 | Differential privacy (optional, for Opacus-based baselines) |
| tqdm | ≥ 4.65 | Progress bar |
| datasets | ≥ 2.14 | HuggingFace dataset download (optional) |

### Hardware Requirements

| Configuration | Min | Recommended |
|---------------|-----|-------------|
| CPU | 4 cores | 8+ cores (for parallel experiments) |
| RAM | 8 GB | 16+ GB (CICIoT2023 preprocessing needs ~16GB) |
| GPU | Not required | Any CUDA GPU (RTX 3060+, 6GB+ VRAM) |
| Disk | 2 GB (code) + 600 MB (data) | SSD recommended |

> **Note**: GPU significantly speeds up training but is not strictly required. CPU mode (T=50 rounds) completes in ~15 min; GPU mode (T=200 rounds) in ~30 min per seed.

---

## 📁 Code Architecture

```
AdaGQ-Matter/
├── configs/                          # Experiment configurations (YAML)
│   ├── base_cpu.yaml                 # CPU baseline (ε=3, T=50, batch=32)
│   ├── base_gpu.yaml                 # GPU full (ε=3, T=200, batch=64)
│   ├── base_gpu_T50.yaml             # GPU quick (ε=3, T=50, batch=64)
│   ├── nonsat_eps50_T50.yaml         # Non-saturated (ε=50, T=50) — controllers in dynamic range
│   ├── nonsat_eps3_T200.yaml         # Extended rounds (ε=3, T=200)
│   ├── matched_sigma_dpfedavg.yaml   # Matched-σ DP-FedAvg (σ=1.83, ε=3)
│   ├── ablation.yaml                 # Ablation experiment configs
│   └── seed10.yaml                   # 10-seed statistical validation
│
├── core/                             # Algorithm core modules
│   ├── compression.py                # Top-k sparsification + adaptive quantization + error feedback
│   ├── dp.py                         # Sparsification-aware DP (RDP accounting + adaptive clipping)
│   ├── feature_grouping.py           # Matter-aware feature grouping (FG1-FG4 priority boost)
│   └── aggregation.py                # Semi-synchronous aggregation (adaptive window + stale-weighted)
│
├── fl/                               # Federated learning engine
│   ├── client.py                     # AdaGQ-Matter client (full compression+DP pipeline)
│   ├── baseline_client.py            # Baseline clients (FedAvg, FedProx, DP-FedAvg, Top-k, Quant, Naive)
│   ├── cpsgd_client.py               # cpSGD baseline (discrete Gaussian mechanism)
│   ├── server.py                     # AdaGQ-Matter server strategy
│   └── baseline_strategy.py          # FedAvg / SignSGD strategies
│
├── models/                           # Model definitions
│   ├── dnn.py                        # AnomalyDNN — 4-layer DNN (Input→64→32→16→2)
│   ├── factory.py                    # Model factory (DNN / LSTM-AE)
│   └── lstm_ae.py                    # LSTM autoencoder (GPU-only, for large-model validation)
│
├── data/                             # Data download & preprocessing
│   ├── download_datasets.py          # Auto-download from HuggingFace / Mendeley
│   ├── preprocess.py                 # Unified preprocessing (IoTID20 + CICIoT2023)
│   └── dirichlet_partition.py        # Dirichlet Non-IID data partitioning
│
├── utils/                            # Utilities
│   ├── metrics.py                    # Classification metrics (F1, Acc, AUC, Precision, Recall)
│   ├── communication.py              # Communication cost tracker
│   ├── privacy_accountant.py         # RDP privacy accountant wrapper
│   └── visualization.py              # Paper figure generation (F1 curves, DP tradeoff, etc.)
│
├── attacks/                          # Privacy attack evaluation
│   ├── dlg.py                        # Deep Leakage from Gradients (DLG)
│   ├── mia.py                        # Membership Inference Attack
│   └── invgrad.py                    # Inverting Gradients Attack
│
├── experiments/                      # Experiment runners
│   ├── run_main.py                   # ★ Main experiment entry point
│   ├── run_ablation.py               # Ablation study
│   ├── run_dp_tradeoff.py            # ε-utility tradeoff sweep
│   ├── run_attack.py                 # Attack evaluation
│   └── run_all.py                    # Full experiment suite
│
├── scripts/                          # Analysis & diagnostics
│   ├── gen_summary.py                # Aggregate results into summary tables
│   ├── plot_paper_figures.py         # Generate publication-quality figures
│   ├── diagnose.py                   # Environment & data diagnostics
│   └── analysis_supplement.py        # Supplement experiment analysis
│
├── setup_autodl.sh                   # One-click setup script (AutoDL GPU platform)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Deployment & Running

### Step 1: Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/AdaGQ-Matter.git
cd AdaGQ-Matter
pip install -r requirements.txt
```

### Step 2: Prepare Data

**Option A: One-click setup (AutoDL)**

```bash
bash setup_autodl.sh
```

This script will:
1. Enable academic mirror (AutoDL)
2. Install pip dependencies
3. Verify GPU environment
4. Download datasets from HuggingFace
5. Preprocess into .npz files + Dirichlet partitions
6. Validate data dimensions

**Option B: Manual step-by-step**

```bash
# Download datasets
python -m data.download_datasets --dataset both --output_dir ./data/raw

# Preprocess IoTID20
python -m data.preprocess --dataset iotid20 --raw_dir ./data/raw --output_dir ./data/processed

# Preprocess CICIoT2023
python -m data.preprocess --dataset ciciot2023 --raw_dir ./data/raw --output_dir ./data/processed
```

> **Note**: If using AutoDL, data is stored in `/root/datasets/` (separate from code). The code automatically falls back to `data/processed/` if the system path is not found.

### Step 3: Run Experiments

**Basic usage:**

```bash
# Single experiment: AdaGQ-Matter on IoTID20, seed=1
python -u experiments/run_main.py --dataset iotid20 --method adagq --seed 1 --config configs/base_cpu.yaml

# DP-FedAvg baseline
python -u experiments/run_main.py --dataset iotid20 --method dp_fedavg --seed 1 --config configs/base_cpu.yaml

# FedAvg (no DP, no compression)
python -u experiments/run_main.py --dataset iotid20 --method fedavg --seed 1 --config configs/base_cpu.yaml
```

**Multi-seed batch:**

```bash
# 5 seeds for statistical significance
python -u experiments/run_main.py --dataset iotid20 --method adagq --seeds 1,2,3,4,5 --config configs/base_cpu.yaml

# All methods
python -u experiments/run_main.py --dataset iotid20 --method all --seeds 1,2,3,4,5 --config configs/base_gpu.yaml
```

**Non-saturated configuration (ε=50, controllers in dynamic range):**

```bash
python -u experiments/run_main.py --dataset iotid20 --method adagq --seeds 1,2,3,4,5 \
  --config configs/nonsat_eps50_T50.yaml
```

**Matched-σ experiment (fair comparison):**

```bash
python -u experiments/run_main.py --dataset iotid20 --method dp_fedavg --seeds 1,2,3,4,5 \
  --config configs/matched_sigma_dpfedavg.yaml
```

**Ablation study:**

```bash
python -u experiments/run_ablation.py --dataset iotid20 --config configs/ablation.yaml
```

**DP tradeoff sweep:**

```bash
python -u experiments/run_dp_tradeoff.py --dataset iotid20
```

### Step 4: Parallel Acceleration (Multi-Core Servers)

For servers with many CPU cores, run multiple experiments in parallel:

```bash
# Example: 4-way parallel on a 192-core server
python -u experiments/run_main.py --dataset iotid20 --method adagq --seed 1 --config configs/nonsat_eps50_T50.yaml &
python -u experiments/run_main.py --dataset iotid20 --method adagq --seed 2 --config configs/nonsat_eps50_T50.yaml &
python -u experiments/run_main.py --dataset iotid20 --method adagq --seed 3 --config configs/nonsat_eps50_T50.yaml &
python -u experiments/run_main.py --dataset iotid20 --method adagq --seed 4 --config configs/nonsat_eps50_T50.yaml &
wait
```

> **Tip**: Use `nohup` or `tmux`/`screen` for long-running experiments to survive SSH disconnections.

### Step 5: View Results

Results are saved as JSON files in `results/`:

```bash
# List all result files
ls -lt results/*.json

# View a specific result
python -c "import json; d=json.load(open('results/adagq_iotid20_alpha0.5_seed1.json')); print(f'F1={d[\"final_metrics\"][\"f1\"]:.4f}, Comm={d[\"final_metrics\"][\"avg_comm_kb\"]:.1f}KB')"

# Generate summary
python scripts/gen_summary.py
```

---

## ⚙️ Configuration Reference

### Key Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `T` | 50 | 10–200 | Total FL communication rounds |
| `N` | 10 | 5–100 | Number of FL clients |
| `alpha` | 0.5 | 0.1–∞ | Dirichlet concentration (Non-IID degree) |
| `eta` | 0.01 | — | Learning rate (SGD) |
| `E_local` | 5 | 1–10 | Local training epochs per round |
| `epsilon` | 3.0 | 0.1–100 | DP privacy budget (ε) |
| `delta` | 1e-5 | — | DP delta |
| `kappa_min` | 0.1 | 0.05–0.3 | Minimum sparsification ratio |
| `kappa_max` | 0.3 | 0.1–0.5 | Maximum sparsification ratio |
| `b_min` | 4 | 2–4 | Minimum quantization bit-width |
| `b_max` | 8 | 4–16 | Maximum quantization bit-width |
| `gini_threshold_high` | 0.7 | 0.5–0.9 | Gini > this → κ = κ_min |
| `gini_threshold_low` | 0.4 | 0.2–0.5 | Gini < this → κ = κ_max |
| `staleness_decay` | 0.6 | 0–1 | Stale-weighted averaging decay factor |
| `device` | cpu | cpu/cuda | Compute device |

---

## 📈 Baseline Methods

| Method | Description | Compression | DP | Feature Grouping |
|--------|-------------|:-----------:|:--:|:----------------:|
| **FedAvg** | Standard federated averaging | ✗ | ✗ | ✗ |
| **FedProx** | FL with proximal regularization (μ=0.01) | ✗ | ✗ | ✗ |
| **DP-FedAvg** | DP-SGD: gradient clip + Gaussian noise + RDP | ✗ | ✓ | ✗ |
| **Top-k Only** | Top-k sparsification only (κ=0.2) | ✓ | ✗ | ✗ |
| **Quant Only** | QSGD quantization only (b=8) | ✓ | ✗ | ✗ |
| **Naive Combination** | Top-k + QSGD serial (fixed κ=0.2, b=8) | ✓ | ✗ | ✗ |
| **cpSGD** | Discrete Gaussian mechanism for DP | ✓ | ✓ | ✗ |
| **AdaGQ-Matter (Ours)** | Co-optimized adaptive κ+b+σ + Matter grouping | ✓ | ✓ | ✓ |

---

## 📋 Experiment Results (IoTID20, α=0.5, 5 seeds)

### Saturated Configuration (ε=3, T=50)

| Method | F1 (mean) | Comm/round | ε |
|--------|-----------|------------|---|
| AdaGQ-Matter | 0.9926 | ~21.1 KB | 3.0 |
| DP-FedAvg | 0.9411 | 168.2 KB | ~4.2 |
| FedAvg | 0.9968 | 50.5 KB | 0 |
| Naive Combination | 0.9972 | 25.4 KB | — |
| Top-k Only | 0.9971 | 50.6 KB | — |
| Quant Only | 0.9972 | 42.2 KB | — |

### Non-Saturated Configuration (ε=50, T=50) — Controllers in Dynamic Range

| Method | F1 (mean±std) | Comm/round | ε |
|--------|---------------|------------|---|
| **AdaGQ-Matter** | **0.9951±0.0034** | **24.6 KB** | **50.0** |
| DP-FedAvg | 0.9949±0.0023 | 168.2 KB | 50.0 |
| FedAvg | 0.9973±0.0007 | 168.2 KB | 0 |

> **Key finding**: AdaGQ-Matter achieves **85.4% communication reduction** (24.6 vs 168.2 KB/round) while maintaining F1 comparable to DP-FedAvg (0.9951 vs 0.9949) under ε=50.

---

## 🔬 Ablation Study

The ablation study progressively removes each component to measure its contribution:

| Configuration | Description |
|---------------|-------------|
| Full | All components active |
| w/o Error Compensation | Disable error feedback buffer |
| w/o Quantization | Set b=32 (effectively FP32) |
| w/o DP | Set ε=∞, σ=0 |
| w/o Adaptive κ | Fix κ=0.2 |
| w/o Adaptive Window | Fix W_agg=500ms |
| Naive Combination | Top-k + QSGD serial, no co-optimization |
| w/o Shuffling | Disable shuffling amplification |
| w/o Feature Grouping | Disable Matter priority boost |

---

## 🛡️ Privacy Attack Evaluation

| Attack | Metric | FedAvg | DP-FedAvg | AdaGQ-Matter |
|--------|--------|--------|-----------|---------------|
| DLG | Reconstruction MSE | Low | Medium | **High** |
| MIA | Attack AUC | ~0.80 | ~0.55 | **~0.52** |
| Inverting Gradients | Reconstruction MSE | Low | Medium | **High** |

Higher MSE → better privacy protection. AdaGQ-Matter's sparsity + DP noise combination provides the strongest defense.

---

## 🎯 Key Innovations

1. **Sparsification-Aware DP**: Noise on κ·d dimensions provides √(1/κ) effective noise amplification — the "privacy dividend" of compression
2. **Coupled Adaptive Controllers**: κ(t), b(t), σ(t) are jointly adapted based on Gini coefficient, remaining privacy budget, and round progress
3. **Matter-Aware Feature Grouping**: IoT device semantic groups (Sensors, Actuators, Controllers, Network) receive differentiated priority in Top-k selection
4. **Semi-Synchronous Aggregation**: Link-aware adaptive window with stale-weighted averaging for realistic network conditions
5. **Error Compensation with Stale Decay**: Error feedback buffers account for delayed/stale updates

---

## 📚 Citation

If you use this code or method in your research, please cite:

```bibtex
@article{fan2025adagq,
  title={Adaptive Gradient Sparsification, Quantization, and Differential Privacy for Federated Learning-based IoT Anomaly Detection with Matter Feature Grouping},
  author={Fan, Xiaohu and others},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- **IoTID20** and **CICIoT2023** dataset providers for advancing IoT security research
- **Matter Protocol** (CSA Alliance) for the IoT device interoperability standard
- **PyTorch** and **Opacus** open-source communities
- All reviewers for constructive feedback that improved this work

---

<p align="center">
  <b>AdaGQ-Matter</b> — Co-Optimized Communication Efficiency & Privacy for Federated IoT Anomaly Detection<br>
  ⭐ If this project helps your research, please consider giving it a star!
</p>
