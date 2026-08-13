"""
Main experiment runner — AdaGQ-Matter vs baselines on IoTID20 / CICIoT2023.

This script runs the full FL experiment pipeline:
1. Load processed data (from data/processed/*.npz)
2. Create FL clients (AdaGQ-Matter + all baselines)
3. Run Flower simulation
4. Collect results (F1, comm bytes, ε, etc.)
5. Save results to results/

Usage:
    python experiments/run_main.py --dataset iotid20 --method adagq --seed 1 --alpha 0.5
    python experiments/run_main.py --dataset ciciot2023 --method all --seed 1
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# --- Fix Python import path for package-level imports ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Project imports
from models.factory import create_model, get_input_dim
from data.dirichlet_partition import dirichlet_partition, save_partitions
from fl.client import create_client
from fl.baseline_client import (
    BaselineFedAvgClient,
    BaselineFedProxClient,
    BaselineDPFedAvgClient,
    BaselineTopKOnlyClient,
    BaselineQuantOnlyClient,
    BaselineNaiveCombClient,
)
from utils.metrics import compute_metrics
from utils.communication import CommunicationTracker


def load_config(config_path: str) -> dict:
    """Load YAML configuration file and sanitize numeric types."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- 防御性类型转换：确保所有数值参数是正确的 Python 类型 ---
    # PyYAML 在某些环境/版本下可能将 1e-5 解析为字符串而非浮点数
    _numeric_keys_float = [
        "delta", "epsilon", "eta", "alpha",
        "kappa_min", "kappa_max", "kappa_default",
        "b_min", "b_max", "b_default",
        "gini_threshold_high", "gini_threshold_low",
        "dp_noise_multiplier", "dp_clipping_norm",
        "staleness_decay", "ewma_beta",
        "fg1_priority_boost", "fg2_priority_boost", "fg3_priority_boost", "fg4_priority_boost",
        "error_decay_stale",
        "W_agg_base_ms", "tau_threshold_ms",
        "packet_loss_rate", "latency_mean_ms", "latency_std_ms",
    ]
    _numeric_keys_int = [
        "T", "N", "E_local", "batch_size", "num_seeds", "num_workers", "s_max",
        "input_dim_iotid20", "input_dim_ciciot2023", "output_dim",
    ]

    for key in _numeric_keys_float:
        if key in config and config[key] is not None:
            config[key] = float(config[key])

    for key in _numeric_keys_int:
        if key in config and config[key] is not None:
            config[key] = int(config[key])

    # Convert hidden_dims list to int elements
    if "hidden_dims" in config and isinstance(config["hidden_dims"], list):
        config["hidden_dims"] = [int(x) for x in config["hidden_dims"]]

    # Convert seed_range to int elements
    if "seed_range" in config and isinstance(config["seed_range"], list):
        config["seed_range"] = [int(x) for x in config["seed_range"]]

    return config


def load_processed_data(data_dir: str, dataset: str):
    """
    Load processed .npz data and partition JSON.

    Returns:
        X_train, y_train, X_test, y_test, partitions
    """
    train_path = os.path.join(data_dir, f"{dataset}_train.npz")
    test_path = os.path.join(data_dir, f"{dataset}_test.npz")
    partition_path = os.path.join(data_dir, f"{dataset}_partitions.json")

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]

    with open(partition_path, "r") as f:
        partitions = json.load(f)

    return X_train, y_train, X_test, y_test, partitions


def create_client_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    partitions: Dict,
    client_id: int,
) -> tuple:
    """Get client-specific training data from partitions."""
    # Support both int keys (from dirichlet_partition) and str keys (from JSON)
    indices = partitions.get(client_id) or partitions.get(str(client_id))
    if indices is None:
        raise KeyError(f"Client {client_id} not found in partitions. Available keys: {list(partitions.keys())[:5]}")
    return X_train[indices], y_train[indices]


def run_single_experiment(
    config: dict,
    dataset: str,
    method: str,
    seed: int,
    alpha: float,
) -> Dict:
    """
    Run a single FL experiment (one method, one seed, one alpha).

    Args:
        config: Configuration dict
        dataset: Dataset name ("iotid20" or "ciciot2023")
        method: Method name
        seed: Random seed
        alpha: Dirichlet concentration

    Returns:
        Results dict with F1, comm bytes, ε, etc.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data (优先从 /root/datasets/processed/，fallback到 data/processed/)
    data_dir = config.get("data_dir", "/root/datasets/processed")
    # 如果新位置没有数据，尝试老位置 (data/processed/)
    if not os.path.exists(os.path.join(data_dir, f"{dataset}_train.npz")):
        fallback_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
        if os.path.exists(os.path.join(fallback_dir, f"{dataset}_train.npz")):
            print(f"⚠️ 新位置 {data_dir} 无数据, 自动使用老位置 {fallback_dir}")
            data_dir = fallback_dir
        else:
            print(f"❌ 数据文件不存在: {data_dir}/{dataset}_train.npz")
            print(f"   也未在老位置找到: {fallback_dir}/{dataset}_train.npz")
            raise FileNotFoundError(f"数据缺失! 请先运行: bash setup_autodl.sh")
    X_train, y_train, X_test, y_test, partitions = load_processed_data(data_dir, dataset)

    # === 动态 input_dim：从实际数据维度读取，而非硬编码 ===
    actual_input_dim = X_train.shape[1]
    expected_input_dim = None
    if dataset == "iotid20":
        expected_input_dim = config.get("input_dim_iotid20", 79)
    elif dataset == "ciciot2023":
        expected_input_dim = config.get("input_dim_ciciot2023", 46)

    if expected_input_dim and actual_input_dim != expected_input_dim:
        print(f"⚠️ 数据维度与配置不一致!")
        print(f"   配置期望: {expected_input_dim} 维, 实际数据: {actual_input_dim} 维")
        print(f"   → 自动使用实际数据维度 input_dim={actual_input_dim}")
    else:
        print(f"✅ 数据维度匹配: {actual_input_dim} 维")

    # === Bug#17 check: 数据类别数检查 ===
    n_classes_actual = len(np.unique(y_train))
    output_dim_config = config.get("output_dim", 2)
    if n_classes_actual < 2:
        print(f"❌ 数据类别数异常! y_train 只有 {n_classes_actual} 个类别, 无法做分类!")
        print(f"   数据可能损坏或预处理有误, 请检查 {dataset}_train.npz 文件")
        print(f"   y_train 唯一值: {np.unique(y_train)}")
        raise ValueError(f"数据类别数={n_classes_actual}, 分类任务至少需要2个类别!")
    if n_classes_actual != output_dim_config:
        print(f"⚠️ 数据类别数({n_classes_actual})与配置output_dim({output_dim_config})不一致")
        print(f"   → 使用数据实际类别数 output_dim={n_classes_actual}")

    # Override config for this run
    run_config = config.copy()
    run_config["dataset"] = dataset
    run_config["alpha"] = alpha
    run_config["seed"] = seed
    # 动态覆盖 input_dim，确保模型与数据一致
    if dataset == "iotid20":
        run_config["input_dim_iotid20"] = actual_input_dim
    elif dataset == "ciciot2023":
        run_config["input_dim_ciciot2023"] = actual_input_dim
    else:
        run_config["input_dim_iotid20"] = actual_input_dim
    input_dim = actual_input_dim

    # === Bug#16 fix: 统一设备管理 — device 在函数开头定义，全局模型显式移到目标设备 ===
    device = config.get("device", "cpu")
    print(f"  🖥️ 运行设备: {device}")

    # Create model
    model = create_model(run_config)
    d = model.num_parameters()

    # Create Dirichlet partitions (re-partition for this seed/alpha)
    new_partitions = dirichlet_partition(y_train, N=config.get("N", 10), alpha=alpha, seed=seed)

    # Number of clients and rounds
    N = config.get("N", 10)
    T = config.get("T", 50)

    # Communication tracker
    comm_tracker = CommunicationTracker()

    # Results storage
    round_results = []
    final_metrics = {}

    print(f"\n{'='*60}")
    print(f"Experiment: {method} | Dataset: {dataset} | Seed: {seed} | α={alpha}")
    print(f"{'='*60}")

    # --- Simulated FL Loop (without Flower server for simplicity) ---
    # In production, use Flower simulation. Here we simulate for faster prototyping.

    global_model = create_model(run_config).to(device)  # Bug#16 fix: 显式移到目标设备
    global_state = global_model.state_dict()

    # Create clients
    clients = []
    for i in range(N):
        X_client, y_client = create_client_data(X_train, y_train, new_partitions, i)

        if method == "adagq":
            client = create_client(i, create_model(run_config), X_client, y_client,
                                   X_test, y_test, run_config)
        elif method == "fedavg":
            client = BaselineFedAvgClient(i, create_model(run_config), X_client, y_client,
                                          X_test, y_test, run_config)
        elif method == "fedprox":
            client = BaselineFedProxClient(i, create_model(run_config), X_client, y_client,
                                           X_test, y_test, run_config)
        elif method == "dp_fedavg":
            client = BaselineDPFedAvgClient(i, create_model(run_config), X_client, y_client,
                                           X_test, y_test, run_config)
        elif method == "top_k_only":
            client = BaselineTopKOnlyClient(i, create_model(run_config), X_client, y_client,
                                           X_test, y_test, run_config)
        elif method == "quant_only":
            client = BaselineQuantOnlyClient(i, create_model(run_config), X_client, y_client,
                                            X_test, y_test, run_config)
        elif method == "naive_combination":
            client = BaselineNaiveCombClient(i, create_model(run_config), X_client, y_client,
                                            X_test, y_test, run_config)
        else:
            raise ValueError(f"Unknown method: {method}")

        clients.append(client)

    # --- FL Training Loop ---
    for round_idx in range(T):
        # Set global model on all clients
        global_params_list = [val.detach().cpu().numpy() for val in global_model.state_dict().values()]

        # Client updates
        client_updates = []
        comm_bytes_round = 0
        epsilon_round = 0

        for client in clients:
            result = client.fit(global_params_list, {"current_round": round_idx, "total_rounds": T})
            client_updates.append(result)

            # Track communication
            if result[2]:  # metrics dict
                comm_bytes_round += result[2].get("comm_bytes", 0)
                epsilon_round = result[2].get("epsilon_current", epsilon_round)

        # Server aggregation (weighted average by sample count)
        total_samples = sum(r[1] for r in client_updates)
        aggregated_weights = {}

        for key in global_model.state_dict().keys():
            weighted_sum = None
            expected_shape = global_model.state_dict()[key].shape  # 模型期望的参数形状
            for result in client_updates:
                client_params = result[0]
                n_samples = result[1]
                # Find the corresponding array in client_params
                idx = list(global_model.state_dict().keys()).index(key)
                if idx < len(client_params):
                    param_arr = client_params[idx]
                    # Bug fix: 防御性reshape — 客户端可能返回flat 1D数组或正确形状的数组
                    # 如果是flat 1D数组(如 [1600]), reshape回模型期望的形状(如 [64,25])
                    param = torch.tensor(param_arr, device=device)
                    if param.shape != expected_shape and param.numel() == global_model.state_dict()[key].numel():
                        param = param.reshape(expected_shape)
                    weighted = param * (n_samples / total_samples)
                    if weighted_sum is None:
                        weighted_sum = weighted
                    else:
                        weighted_sum += weighted
            # 防御性检查：如果某个参数键在所有客户端中都缺失，使用原始全局模型参数
            if weighted_sum is None:
                aggregated_weights[key] = global_model.state_dict()[key]
            else:
                aggregated_weights[key] = weighted_sum

        # Update global model
        global_model.load_state_dict(aggregated_weights)

        # Evaluate on test set
        global_model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_test, dtype=torch.long).to(device)
            outputs = global_model(X_t)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            metrics = compute_metrics(y_t.cpu().numpy(), preds.cpu().numpy(),
                                      probs.cpu().numpy())

        # Record results
        round_result = {
            "round": round_idx,
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "alpha": alpha,
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "comm_bytes": comm_bytes_round,
            "epsilon": epsilon_round,
        }
        round_results.append(round_result)

        # Fix: comm_tracker.record 参数顺序匹配方法签名 record(round_id, client_id, method, upload_bytes)
        comm_tracker.record(round_id=round_idx, client_id=0, method=method,
                            upload_bytes=comm_bytes_round)

        if (round_idx + 1) % 10 == 0:
            print(f"  Round {round_idx+1}/{T}: F1={metrics['f1']:.4f}, "
                  f"comm={comm_bytes_round/1024:.1f}KB, ε={epsilon_round:.3f}")

    # Final metrics
    final_metrics = round_results[-1]
    # Fix: CommunicationTracker 没有 get_total_comm_kb / get_avg_comm_kb 方法
    # 使用实际存在的 total_upload / avg_upload_per_round 方法
    final_metrics["total_comm_kb"] = comm_tracker.total_upload(method) / 1024.0
    final_metrics["avg_comm_kb"] = comm_tracker.avg_upload_per_round(method) / 1024.0
    final_metrics["d"] = d

    print(f"\n  ✅ Final: F1={final_metrics['f1']:.4f}, "
          f"Comm={final_metrics['avg_comm_kb']:.1f}KB/round, "
          f"ε={final_metrics['epsilon']:.3f}")

    # Save results (JSON for programmatic use)
    results_dir = config.get("results_dir", "results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    result_filename = f"{method}_{dataset}_alpha{alpha}_seed{seed}"
    result_path = os.path.join(results_dir, f"{result_filename}.json")

    output = {
        "experiment": result_filename,
        "config": run_config,
        "round_results": round_results,
        "final_metrics": final_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Save txt format result (方便用户阅读和发送给AI) ---
    txt_path = os.path.join(results_dir, f"{result_filename}.txt")
    txt_lines = []
    txt_lines.append("=" * 60)
    txt_lines.append(f"AdaGQ-Matter 实验结果")
    txt_lines.append("=" * 60)
    txt_lines.append("")
    txt_lines.append(f"方法: {method}")
    txt_lines.append(f"数据集: {dataset}")
    txt_lines.append(f"种子: {seed}")
    txt_lines.append(f"α (Non-IID浓度): {alpha}")
    txt_lines.append(f"时间戳: {datetime.now().isoformat()}")
    txt_lines.append("")
    txt_lines.append("--- 环境诊断参数 ---")
    txt_lines.append(f"PyTorch版本: {torch.__version__}")
    txt_lines.append(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        txt_lines.append(f"GPU名称: {torch.cuda.get_device_name(0)}")
        txt_lines.append(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    txt_lines.append(f"设备: {config.get('device', 'cpu')}")
    txt_lines.append(f"input_dim (实际数据维度): {actual_input_dim}")
    txt_lines.append(f"input_dim (配置期望维度): {expected_input_dim}")
    txt_lines.append(f"模型参数总量 d: {d}")
    txt_lines.append(f"X_train形状: {X_train.shape}")
    txt_lines.append(f"X_test形状: {X_test.shape}")
    txt_lines.append(f"y_train类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    txt_lines.append(f"y_test类别分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    txt_lines.append("")
    txt_lines.append("--- FL配置参数 ---")
    txt_lines.append(f"T (总轮数): {T}")
    txt_lines.append(f"N (客户端数): {N}")
    txt_lines.append(f"E_local (本地迭代): {config.get('E_local', '?')}")
    txt_lines.append(f"batch_size: {config.get('batch_size', '?')}")
    txt_lines.append(f"eta (学习率): {config.get('eta', '?')}")
    txt_lines.append(f"epsilon (隐私预算): {config.get('epsilon', '?')}")
    txt_lines.append(f"delta (隐私参数): {config.get('delta', '?')}")
    txt_lines.append(f"s_max (量化级数): {config.get('s_max', '?')}")
    txt_lines.append(f"kappa_min/kappa_max/kappa_default: {config.get('kappa_min','?')}/{config.get('kappa_max','?')}/{config.get('kappa_default','?')}")
    txt_lines.append(f"dp_noise_multiplier: {config.get('dp_noise_multiplier', '?')}")
    txt_lines.append(f"dp_clipping_norm: {config.get('dp_clipping_norm', '?')}")
    txt_lines.append("")
    txt_lines.append("--- 最终结果 ---")
    txt_lines.append(f"F1: {final_metrics['f1']:.4f}")
    txt_lines.append(f"Accuracy: {final_metrics['accuracy']:.4f}")
    txt_lines.append(f"Loss: {final_metrics['loss']:.6f}")
    txt_lines.append(f"ε (隐私消耗): {final_metrics['epsilon']:.4f}")
    txt_lines.append(f"总通信量: {final_metrics['total_comm_kb']:.2f} KB")
    txt_lines.append(f"平均每轮通信量: {final_metrics['avg_comm_kb']:.2f} KB")
    txt_lines.append(f"模型参数量 d: {d}")
    txt_lines.append("")
    txt_lines.append("--- 每轮详细结果 ---")
    txt_lines.append(f"{'Round':>5} {'F1':>8} {'Acc':>8} {'Loss':>10} {'Comm(KB)':>10} {'ε':>8}")
    txt_lines.append("-" * 55)
    for rr in round_results:
        txt_lines.append(f"{rr['round']:>5} {rr['f1']:>8.4f} {rr['accuracy']:>8.4f} "
                         f"{rr['loss']:>10.6f} {rr['comm_bytes']/1024:>10.2f} {rr['epsilon']:>8.4f}")
    txt_lines.append("")
    txt_lines.append("=" * 60)

    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))

    print(f"  Results saved to {result_path}")
    print(f"  TXT report saved to {txt_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run AdaGQ-Matter FL experiments")
    parser.add_argument("--dataset", default="iotid20",
                        choices=["iotid20", "ciciot2023", "both"])
    parser.add_argument("--method", default="adagq",
                        choices=["adagq", "fedavg", "fedprox", "dp_fedavg",
                                 "top_k_only", "quant_only", "naive_combination", "all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--config", default="configs/base_cpu.yaml")
    parser.add_argument("--T", type=int, default=None,
                        help="Override number of FL rounds")
    parser.add_argument("--N", type=int, default=None,
                        help="Override number of clients")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override config from CLI args
    if args.T:
        config["T"] = args.T
    if args.N:
        config["N"] = args.N

    # Determine which methods to run
    if args.method == "all":
        methods = ["fedavg", "fedprox", "dp_fedavg", "top_k_only",
                   "quant_only", "naive_combination", "adagq"]
    else:
        methods = [args.method]

    # Determine which datasets to run
    datasets = [args.dataset] if args.dataset != "both" else ["iotid20", "ciciot2023"]

    # Run experiments (with proper error display)
    all_results = {}
    for dataset in datasets:
        for method in methods:
            try:
                result = run_single_experiment(config, dataset, method, args.seed, args.alpha)
                all_results[f"{method}_{dataset}"] = result
            except Exception as e:
                import traceback
                print(f"\n❌ 实验失败: {method}_{dataset}")
                print(f"   错误类型: {type(e).__name__}")
                print(f"   错误信息: {e}")
                traceback.print_exc()
                print(f"   ↑ 以上是完整的错误追踪信息 ↑\n")
                # 仍然继续下一个实验
                all_results[f"{method}_{dataset}"] = {"error": str(e), "traceback": traceback.format_exc()}

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    n_success = 0
    n_fail = 0
    for key, result in all_results.items():
        if "final_metrics" in result:
            fm = result["final_metrics"]
            print(f"  ✅ {key}: F1={fm['f1']:.4f}, Comm={fm['avg_comm_kb']:.1f}KB, ε={fm['epsilon']:.3f}")
            n_success += 1
        else:
            print(f"  ❌ {key}: {result.get('error', '未知错误')}")
            n_fail += 1
    print(f"  统计: 成功={n_success}, 失败={n_fail}, 总计={len(all_results)}")

    # Save summary (JSON — 需要处理 numpy 数组不可序列化的问题)
    summary_path = os.path.join(config.get("results_dir", "results"), "summary.json")
    # 用自定义 encoder 处理 numpy 类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    try:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        print(f"  Summary JSON saved to {summary_path}")
    except Exception as e:
        print(f"  ⚠️ Summary JSON 保存失败: {e}")

    # Save summary (TXT — 方便阅读和发送)
    summary_txt_path = os.path.join(config.get("results_dir", "results"), "summary.txt")
    txt_lines = []
    txt_lines.append("=" * 60)
    txt_lines.append("AdaGQ-Matter 实验汇总")
    txt_lines.append("=" * 60)
    txt_lines.append(f"时间戳: {datetime.now().isoformat()}")
    txt_lines.append("")
    for key, result in all_results.items():
        if "final_metrics" in result:
            fm = result["final_metrics"]
            txt_lines.append(f"✅ {key}: F1={fm['f1']:.4f}, Acc={fm['accuracy']:.4f}, "
                             f"Comm={fm['avg_comm_kb']:.1f}KB, ε={fm['epsilon']:.3f}, d={fm.get('d','?')}")
        elif "error" in result:
            txt_lines.append(f"❌ {key}: {result['error']}")
    txt_lines.append("")
    txt_lines.append("=" * 60)
    with open(summary_txt_path, "w") as f:
        f.write("\n".join(txt_lines))
    print(f"  Summary TXT saved to {summary_txt_path}")


if __name__ == "__main__":
    main()
