"""
诊断脚本 — 替代 run_T50.sh 中所有 python -c 内联命令。

功能：
1. GPU 检测 (名称、显存、利用率)
2. Config 关键参数打印
3. 数据维度检测
4. 数据完整性验证
5. 模型参数量计算

输出：纯文本，带前缀标记，方便 shell 脚本读取
"""

import sys
import os
import yaml
import numpy as np
import torch

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/base_gpu_T50.yaml"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "/root/datasets/processed"

    # === GPU 检测 ===
    print("[GPU]")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU名称: {gpu_name}")
        print(f"  GPU显存: {gpu_mem:.1f} GB")
        print(f"  CUDA版本: {torch.version.cuda}")
    else:
        print("  GPU: 不可用 (将使用CPU)")
    print(f"  PyTorch版本: {torch.__version__}")
    print()

    # === Config 参数 ===
    print("[CONFIG]")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        key_params = [
            "T", "N", "batch_size", "device", "epsilon", "delta", "eta",
            "E_local", "kappa_default", "b_default", "s_max",
            "input_dim_iotid20", "input_dim_ciciot2023", "output_dim",
            "model_type", "data_dir",
        ]
        for key in key_params:
            val = config.get(key, "?")
            print(f"  {key}: {val}")
        print(f"  hidden_dims: {config.get('hidden_dims', '?')}")
    except Exception as e:
        print(f"  ❌ 读取config失败: {e}")
    print()

    # === 数据维度 ===
    print("[DATA]")
    datasets_info = {}
    for ds_name in ["iotid20", "ciciot2023"]:
        train_path = os.path.join(data_dir, f"{ds_name}_train.npz")
        test_path = os.path.join(data_dir, f"{ds_name}_test.npz")
        partition_path = os.path.join(data_dir, f"{ds_name}_partitions.json")

        if not os.path.exists(train_path):
            print(f"  {ds_name}: ❌ 训练数据不存在 ({train_path})")
            continue

        try:
            train_data = np.load(train_path)
            X_train = train_data["X"]
            y_train = train_data["y"]
            n_classes = len(np.unique(y_train))

            if os.path.exists(test_path):
                test_data = np.load(test_path)
                X_test = test_data["X"]
                y_test = test_data["y"]
                n_classes_test = len(np.unique(y_test))
                print(f"  {ds_name}: ✅ 数据完整")
                print(f"    X_train: {X_train.shape} (float{X_train.dtype})")
                print(f"    y_train: {y_train.shape} (类别数={n_classes}, dtype={y_train.dtype})")
                if n_classes < 2:
                    print(f"    ⚠️ 类别数<2! 数据异常, y_train唯一值: {np.unique(y_train)}")
                    print(f"    ⚠️ 无法做分类, 请检查数据文件是否损坏或重新运行预处理")
                print(f"    X_test:  {X_test.shape}")
                print(f"    y_test:  {y_test.shape}")
            else:
                print(f"  {ds_name}: ⚠️ 训练数据存在但测试数据缺失")

            # 检查分区文件
            if os.path.exists(partition_path):
                import json
                with open(partition_path, "r") as f:
                    partitions = json.load(f)
                n_clients_in_partition = len(partitions)
                print(f"    partitions: {n_clients_in_partition} 个客户端")
            else:
                print(f"    partitions: ❌ 不存在")

            datasets_info[ds_name] = {
                "input_dim": X_train.shape[1],
                "n_classes": n_classes,
                "n_train": X_train.shape[0],
            }
        except Exception as e:
            print(f"  {ds_name}: ❌ 加载失败: {e}")
    print()

    # === 模型参数量 ===
    print("[MODEL]")
    try:
        # 动态 import
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.factory import create_model

        for ds_name, info in datasets_info.items():
            model_config = config.copy()
            model_config["dataset"] = ds_name
            model_config["input_dim_iotid20"] = info["input_dim"]
            model_config["input_dim_ciciot2023"] = info["input_dim"]
            # Bug#18 fix: output_dim 使用 config 的值（实验代码用config值），不是数据类别数
            # 如果数据类别数<2，仍然用config的output_dim=2来创建模型（因为实验代码就是这样做的）
            config_output_dim = int(config.get("output_dim", 2))
            model_config["output_dim"] = config_output_dim
            model = create_model(model_config)
            d = model.num_parameters()
            kb = d * 4 / 1024
            print(f"  {ds_name}: input_dim={info['input_dim']}, output_dim={config_output_dim}, d={d} 参数, ≈{kb:.1f} KB")
            if info["n_classes"] < 2:
                print(f"  ⚠️ {ds_name}: 数据类别数={info['n_classes']} < 2, 无法做分类! 请检查数据")
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
    print()

    # === 综合判断 ===
    print("[STATUS]")
    has_data = len(datasets_info) > 0
    print(f"  数据就绪: {'✅ YES' if has_data else '❌ NO'}")
    print(f"  GPU可用: {'✅ YES' if torch.cuda.is_available() else '⚠️ NO (CPU模式)'}")
    if has_data:
        for ds_name, info in datasets_info.items():
            expected_dim = config.get(f"input_dim_{ds_name}", "?")
            actual_dim = info["input_dim"]
            match = "✅ 匹配" if str(expected_dim) == str(actual_dim) else f"⚠️ 不匹配(配置={expected_dim}, 实际={actual_dim})"
            print(f"  {ds_name} 维度: {match}")

if __name__ == "__main__":
    main()
