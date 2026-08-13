# AdaGQ-Matter

**Adaptive Gradient Sparsification, Quantization, and Differential Privacy for Federated Learning-based IoT Anomaly Detection with Matter Feature Grouping**

<p align="center">
  <img src="https://img.shields.io/badge/status-Experimental-brightgreen" alt="Status"/>
  <img src="https://img.shields.io/badge/framework-PyTorch-orange" alt="Framework"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License"/>
  <img src="https://img.shields.io/badge/FL-IoT%20Anomaly%20Detection-red" alt="FL"/>
</p>

---

## 📖 项目简介

**AdaGQ-Matter** 是一个面向联邦学习（Federated Learning, FL）场景下物联网异常检测的高效通信与隐私保护框架。本项目实现了完整的算法核心模块、多组基线对比实验、消融实验以及抗攻击评估，旨在验证 **自适应梯度稀疏化 + 量化 + 差分隐私 + Matter 特征分组** 在 IoT 异常检测任务上的综合性能。

核心思想：在通信受限且隐私敏感的 IoT 场景中，通过 **Top-k 稀疏化** 减少通信量，**自适应量化** 进一步压缩梯度，**Error Feedback** 补偿压缩误差，**Matter 特征分组** 对齐 IoT 设备异构特征空间，**稀疏化感知差分隐私（Sparsity-Aware DP）** 注入可控噪声，实现通信效率与隐私保护的最优权衡。

---

## 📄 论文信息

- **论文标题**：Adaptive Gradient Sparsification, Quantization, and Differential Privacy for Federated Learning-based IoT Anomaly Detection with Matter Feature Grouping
- **作者**：范小虎（Xiaohu Fan）等
- **关键词**：Federated Learning, Gradient Sparsification, Quantization, Differential Privacy, IoT Anomaly Detection, Matter Protocol, Feature Grouping, Communication Efficiency

---

## 🧠 算法核心组件

AdaGQ-Matter 由以下 **5 个核心组件** 构成：

### 1. Top-k 梯度稀疏化（Top-k Gradient Sparsification）
在每次本地训练完成后，仅保留梯度张量中绝对值最大的 `k` 个元素（其余置零），可大幅降低上行通信量。稀疏率 `k/d` 为可调超参数，默认保留 10% 的梯度。

### 2. 自适应量化（Adaptive Quantization）
基于每轮梯度的动态范围（min/max），自适应选择量化位数（2-bit ~ 8-bit）。梯度波动大时自动提高量化精度以保留信息，波动小时降低位数以节省带宽。

### 3. Error Feedback 误差补偿机制
累积每次压缩带来的量化/稀疏化误差，在下一轮训练前将误差加回梯度，保证收敛性不受压缩损失影响。理论保证：在有界误差假设下收敛速率与全精度 SGD 一致。

### 4. Matter 特征分组（Matter Feature Grouping）
借鉴 Matter 协议中设备类型的语义分组思想，将 IoT 设备按功能类型（传感器、执行器、网关等）划分为语义组，在同组设备间进行特征级对齐（Feature-wise Alignment），缓解 heterogeneous 场景下的数据分布漂移问题。

### 5. 稀疏化感知差分隐私（Sparsity-Aware DP）
在稀疏化后的梯度上注入高斯噪声，利用稀疏结构的先验信息自适应当轮隐私预算分配。相比传统 DP-FedAvg，在相同隐私预算（ε）下可获得更高的模型精度。

---

## 📊 数据集

### IoTID20
| 属性 | 值 |
|------|------|
| 特征数 | 79 |
| 样本量 | ~625,000 条 |
| 任务类型 | 二分类 / 多分类异常检测 |
| 攻击类型 | DoS, DDoS, Scan, MITM, Mirai 等 |

### CICIoT2023
| 属性 | 值 |
|------|------|
| 特征数 | 46 |
| 样本量 | 1,300,000 条（子采样） |
| 任务类型 | 多分类异常检测 |
| 攻击类型 | DDoS, DoS, Recon, Web-based, Brute Force, Spoofing, Mirai 等 |

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- 推荐：CPU 8 核 / GPU 6GB+ 显存

### 第 1 步：安装依赖

```bash
git clone https://github.com/your-org/AdaGQ-Matter.git
cd AdaGQ-Matter
pip install -r requirements.txt
```

主要依赖（`requirements.txt`）：
```
torch>=1.12.0
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0
tqdm>=4.62.0
matplotlib>=3.4.0
seaborn>=0.11.0
opacus>=1.3.0
```

### 第 2 步：数据预处理

```bash
# 预处理 IoTID20 数据集
python data/preprocess_iotid20.py

# 预处理 CICIoT2023 数据集
python data/preprocess_ciot2023.py

# 或一键预处理两个数据集
python data/preprocess_all.py
```

预处理脚本将执行：
- 缺失值填充（均值/中位数）
- 类别特征编码（Label Encoding / One-Hot）
- 数值特征标准化（Z-Score Normalization）
- 训练/测试集划分（80%/20%）
- 按设备 ID 进行 non-IID 分区（Dirichlet 分布 α=0.5）

### 第 3 步：运行实验

```bash
# 运行主实验（对比所有方法）
python experiments/run_main.py --dataset iotid20 --method all --seed 1

# 运行消融实验
python experiments/run_ablation.py

# 运行 DP 隐私预算权衡实验
python experiments/run_dp_tradeoff.py

# 运行攻击评估实验
python experiments/run_attack.py
```

---

## 📁 项目结构

```
AdaGQ-Matter/
├── configs/                      # 配置文件
│   ├── default.yaml              # 默认配置（CPU版）
│   ├── gpu_config.yaml           # GPU版配置
│   ├── iotid20.yaml              # IoTID20 数据集配置
│   └── cioc2023.yaml             # CICIoT2023 数据集配置
│
├── data/                         # 数据处理
│   ├── preprocess_iotid20.py     # IoTID20 预处理
│   ├── preprocess_ciot2023.py    # CICIoT2023 预处理
│   ├── preprocess_all.py         # 一键预处理
│   ├── dataset.py                # 数据集加载与划分
│   └── partition.py              # Non-IID 分区策略
│
├── models/                       # 模型定义
│   ├── mlp.py                    # MLP 分类器（79/46 features）
│   ├── cnn.py                    # CNN 1D 分类器
│   └── base.py                   # 模型基类
│
├── core/                         # 算法核心
│   ├── sparsification.py         # Top-k 稀疏化
│   ├── quantization.py           # 自适应量化
│   ├── error_feedback.py         # Error Feedback 机制
│   ├── matter_grouping.py        # Matter 特征分组
│   ├── dp_mechanism.py           # 稀疏化感知 DP
│   └── adagq_matter.py           # AdaGQ-Matter 主算法
│
├── fl/                           # 联邦学习引擎
│   ├── client.py                 # 客户端实现
│   ├── server.py                 # 服务端聚合
│   ├── aggregator.py             # 聚合策略（FedAvg / FedProx）
│   └── evaluator.py              # 评估指标计算
│
├── utils/                        # 工具函数
│   ├── metrics.py                # 评估指标（Accuracy, F1, AUC, etc.）
│   ├── logger.py                 # 日志记录
│   ├── visualizer.py             # 可视化工具
│   └── helpers.py                # 通用辅助函数
│
├── attacks/                      # 攻击评估
│   ├── gradient_leakage.py       # 梯度泄露攻击（DLG）
│   ├── membership_inference.py   # 成员推断攻击
│   └── defense_eval.py           # 防御效果评估
│
├── experiments/                  # 实验脚本
│   ├── run_main.py               # 主实验入口
│   ├── run_ablation.py           # 消融实验
│   ├── run_dp_tradeoff.py        # DP 权衡实验
│   ├── run_attack.py             # 攻击评估实验
│   └── run_all.py                # 全套实验（多种子）
│
├── results/                      # 结果输出
│   ├── main/                     # 主实验结果
│   ├── ablation/                 # 消融实验结果
│   ├── dp_tradeoff/              # DP 权衡结果
│   └── attack/                   # 攻击评估结果
│
├── requirements.txt              # Python 依赖
├── README.md                     # 本文件
└── LICENSE                       # MIT License
```

---

## ⚙️ 实验命令详解

### 主实验：方法对比

在所有方法上运行完整 FL 训练流程，输出 Accuracy、F1-Score、AUC、通信量、收敛轮数等指标。

```bash
# 单数据集 + 单种子
python experiments/run_main.py --dataset iotid20 --method all --seed 1

# 指定方法（FedAvg, FedProx, DP-FedAvg, Top-k Only, Quant Only, Naive, AdaGQ-Matter）
python experiments/run_main.py --dataset cioc2023 --method adagq_matter --seed 42

# 多方法对比
python experiments/run_main.py --dataset both --method fedavg,fedprox,adagq_matter --seed 0
```

### 消融实验

逐一移除 AdaGQ-Matter 的每个组件，验证各组件贡献度。

```bash
python experiments/run_ablation.py
```

消融模式：
- `w/o Top-k`：禁用稀疏化，仅保留量化+DP+分组
- `w/o Quant`：禁用自适应量化
- `w/o ErrorFB`：禁用 Error Feedback
- `w/o Matter`：禁用 Matter 特征分组
- `w/o DP`：禁用差分隐私
- `Full`：完整 AdaGQ-Matter

### DP 隐私预算权衡实验

在不同 ε 值（0.1, 0.5, 1.0, 2.0, 5.0, 10.0）下评估模型性能，绘制隐私-效用 Pareto 前沿。

```bash
python experiments/run_dp_tradeoff.py
```

### 攻击评估实验

评估模型在梯度泄露攻击（DLG）和成员推断攻击（MIA）下的防御效果。

```bash
python experiments/run_attack.py
```

### 全套实验

一键运行所有实验，支持多数据集和多随机种子，确保结果统计显著性。

```bash
# 双数据集 + 5 个种子
python experiments/run_all.py --dataset both --seeds 1 2 3 4 5

# 仅 IoTID20 + 3 个种子
python experiments/run_all.py --dataset iotid20 --seeds 10 20 30
```

---

## 🔧 配置说明

### CPU 版（默认配置：`configs/default.yaml`）

适用于本地开发调试或算力受限环境。

| 参数 | 值 | 说明 |
|------|------|------|
| `num_clients` | 10 | 参与客户端数量 |
| `num_rounds` | 50 | 联邦训练轮数 |
| `local_epochs` | 5 | 每轮本地训练 epoch |
| `batch_size` | 64 | 本地 batch size |
| `learning_rate` | 0.01 | 学习率 |
| `top_k_ratio` | 0.1 | Top-k 稀疏率 |
| `quant_bits` | [2, 4, 8] | 自适应量化候选位数 |
| `dp_epsilon` | 1.0 | 隐私预算 ε |
| `dp_delta` | 1e-5 | 隐私预算 δ |
| `matter_groups` | 5 | Matter 特征分组数 |

### GPU 版（`configs/gpu_config.yaml`）

适用于 AutoDL、Colab Pro 等 GPU 环境，更多轮数以获得更稳定收敛。

| 参数 | 值 | 说明 |
|------|------|------|
| `num_clients` | 10 | 参与客户端数量 |
| `num_rounds` | 200 | 联邦训练轮数（GPU 加速） |
| `local_epochs` | 10 | 每轮本地训练 epoch |
| `batch_size` | 128 | 本地 batch size |
| `learning_rate` | 0.001 | 学习率（Adam 优化器） |
| `top_k_ratio` | 0.1 | Top-k 稀疏率 |
| `quant_bits` | [2, 4, 8] | 自适应量化候选位数 |
| `dp_epsilon` | 1.0 | 隐私预算 ε |
| `dp_delta` | 1e-5 | 隐私预算 δ |
| `matter_groups` | 5 | Matter 特征分组数 |

切换方式：

```bash
# 使用 GPU 配置运行
python experiments/run_main.py --config configs/gpu_config.yaml --dataset iotid20
```

---

## 📈 基线方法

| 方法 | 描述 | 通信压缩 | 隐私保护 | 特征对齐 |
|------|------|:--------:|:--------:|:--------:|
| **FedAvg** | 标准联邦平均基线 | ✗ | ✗ | ✗ |
| **FedProx** | 带近端项的正则化 FL | ✗ | ✗ | ✗ |
| **DP-FedAvg** | 差分隐私 FedAvg | ✗ | ✓ | ✗ |
| **Top-k Only** | 仅 Top-k 稀疏化，无量化/DP | ✓ | ✗ | ✗ |
| **Quant Only** | 仅自适应量化，无稀疏化/DP | ✓ | ✗ | ✗ |
| **Naive Combination** | 简单组合（固定稀疏+量化+DP） | ✓ | ✓ | ✗ |
| **AdaGQ-Matter（Ours）** | 自适应稀疏+量化+DP+特征分组 | ✓ | ✓ | ✓ |

---

## 📋 结果格式

所有实验结果以 JSON 格式保存在 `results/` 目录下，按实验类型组织。

### 主实验结果示例（`results/main/main_iotid20_seed1.json`）

```json
{
  "dataset": "iotid20",
  "seed": 1,
  "config": {
    "num_clients": 10,
    "num_rounds": 50,
    "top_k_ratio": 0.1
  },
  "methods": {
    "fedavg": {
      "accuracy": 0.9523,
      "f1_score": 0.9487,
      "auc": 0.9812,
      "total_communication_mb": 1250.5,
      "convergence_round": 35,
      "final_loss": 0.1854
    },
    "adagq_matter": {
      "accuracy": 0.9618,
      "f1_score": 0.9589,
      "auc": 0.9876,
      "total_communication_mb": 86.3,
      "compression_ratio": 14.49,
      "convergence_round": 28,
      "final_loss": 0.1521,
      "dp_epsilon_used": 1.02
    }
  }
}
```

### 消融实验结果示例（`results/ablation/ablation_iotid20.json`）

```json
{
  "dataset": "iotid20",
  "seed": 1,
  "ablations": {
    "full": {"accuracy": 0.9618, "f1": 0.9589},
    "w/o_topk": {"accuracy": 0.9541, "f1": 0.9512},
    "w/o_quant": {"accuracy": 0.9592, "f1": 0.9560},
    "w/o_errorfb": {"accuracy": 0.9475, "f1": 0.9443},
    "w/o_matter": {"accuracy": 0.9510, "f1": 0.9481},
    "w/o_dp": {"accuracy": 0.9635, "f1": 0.9607}
  }
}
```

### DP 权衡结果示例（`results/dp_tradeoff/dp_tradeoff.json`）

```json
{
  "dataset": "iotid20",
  "tradeoffs": [
    {"epsilon": 0.1, "accuracy": 0.9132, "f1": 0.9087},
    {"epsilon": 0.5, "accuracy": 0.9385, "f1": 0.9350},
    {"epsilon": 1.0, "accuracy": 0.9618, "f1": 0.9589},
    {"epsilon": 2.0, "accuracy": 0.9681, "f1": 0.9654},
    {"epsilon": 5.0, "accuracy": 0.9723, "f1": 0.9698},
    {"epsilon": 10.0, "accuracy": 0.9745, "f1": 0.9721}
  ]
}
```

### 攻击评估结果示例（`results/attack/attack_eval.json`）

```json
{
  "dataset": "iotid20",
  "attack_results": {
    "gradient_leakage": {
      "fedavg": {"attack_success_rate": 0.872, "mse": 0.014},
      "dp_fedavg": {"attack_success_rate": 0.215, "mse": 0.423},
      "adagq_matter": {"attack_success_rate": 0.098, "mse": 0.687}
    },
    "membership_inference": {
      "fedavg": {"attack_accuracy": 0.815, "advantage": 0.327},
      "dp_fedavg": {"attack_accuracy": 0.542, "advantage": 0.058},
      "adagq_matter": {"attack_accuracy": 0.513, "advantage": 0.027}
    }
  }
}
```

---

## 🖥️ GPU 部署指南

### 推荐环境

| 平台 | 推荐配置 | 预估费用 |
|------|---------|---------|
| **AutoDL** | RTX 3090 / 4090, 24GB | ¥2~4/小时 |
| **Colab Pro** | T4 / V100, 16GB | $9.99/月 |
| **Colab Pro+** | A100, 40GB | $49.99/月 |

### AutoDL 快速部署

```bash
# 1. 创建实例（PyTorch 镜像）
# 2. 克隆项目
git clone https://github.com/your-org/AdaGQ-Matter.git
cd AdaGQ-Matter

# 3. 安装依赖
pip install -r requirements.txt

# 4. 数据预处理
python data/preprocess_all.py

# 5. 运行全套实验（GPU 配置）
python experiments/run_all.py --config configs/gpu_config.yaml --dataset both --seeds 1 2 3 4 5
```

### Colab 使用说明

1. 在 Google Drive 中挂载项目
2. 选择运行时类型为 **GPU（T4 / V100）**
3. 运行 `notebooks/colab_setup.ipynb` 安装依赖
4. 执行实验脚本

---

## 📚 引用

如果您使用了本项目的代码或算法，请引用：

```bibtex
@article{fan2025adagq,
  title={Adaptive Gradient Sparsification, Quantization, and Differential Privacy for Federated Learning-based IoT Anomaly Detection with Matter Feature Grouping},
  author={Fan, Xiaohu and others},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```

---

## 📄 License

本项目采用 **MIT License** 开源。

```
MIT License

Copyright (c) 2025 Xiaohu Fan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 致谢

- 感谢 **IoTID20** 和 **CICIoT2023** 数据集提供方为 IoT 安全研究社区做出的贡献
- 感谢 **Matter 协议**（CSA 联盟）为 IoT 设备互操作性提供的标准化框架
- 感谢 **PyTorch** 和 **Opacus** 开源社区提供的深度学习与差分隐私工具支持
- 感谢所有在实验过程中提供宝贵建议的同行和评审专家

---

<p align="center">
  <b>AdaGQ-Matter</b> — Making Federated Learning Smarter, Faster, and Safer for IoT.<br>
  🌟 如果该项目对您的研究有帮助，欢迎 Star 支持！
</p>