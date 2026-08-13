#!/bin/bash
# ============================================================
# AdaGQ-Matter 一键安装脚本 (AutoDL GPU 版 v3)
# ============================================================
# v3: 数据集与代码分离！
#   - 数据集存放在 /root/datasets/ (独立目录)
#   - 代码存放在 /root/AdaGQ-Matter/ (不含数据)
#   - 更换代码zip时不会影响数据集
# ============================================================

# 自动切换到项目目录
cd "$(dirname "$0")" || { echo "❌ 无法切换到项目目录"; exit 1; }
echo "工作目录: $(pwd)"

DATASET_ROOT="/root/datasets"
RAW_DIR="$DATASET_ROOT/raw"
PROC_DIR="$DATASET_ROOT/processed"

# 创建数据目录
mkdir -p "$RAW_DIR" "$PROC_DIR"

echo ""
echo "=========================================="
echo "  AdaGQ-Matter 一键安装 (v3)"
echo "  数据目录: $DATASET_ROOT (与代码分离)"
echo "=========================================="

# --- Step 0: AutoDL 学术加速 ---
echo ""
echo ">>> [0/6] 开启 AutoDL 学术加速 ..."
if [ -f "/etc/network_turbo" ]; then
    source /etc/network_turbo
    echo "✅ AutoDL 学术加速已开启!"
    echo "   现在可以访问 HuggingFace、GitHub 等学术资源"
elif [ -f "/usr/local/bin/autodl-tmp" ]; then
    source /usr/local/bin/autodl-tmp
    echo "✅ AutoDL 加速已开启!"
else
    echo "⚠️ 未找到 AutoDL 加速脚本，尝试 HuggingFace 镜像 ..."
    export HF_ENDPOINT=https://hf-mirror.com
    echo "   已设置 HF_ENDPOINT=https://hf-mirror.com"
fi

# --- Step 1: 安装依赖 ---
echo ""
echo ">>> [1/6] 安装 pip 依赖 (约3分钟) ..."
pip install -r requirements.txt

# --- Step 2: 验证 GPU ---
echo ""
echo ">>> [2/6] 验证 GPU 环境 ..."
python -c "
import torch
if torch.cuda.is_available():
    print('✅ GPU 可用!')
    print('   GPU 名称:', torch.cuda.get_device_name(0))
    print('   GPU 显存:', round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1), 'GB')
else:
    print('❌ GPU 不可用! 请检查实例是否选了 GPU')
    exit(1)
"

# --- Step 3: 验证模型 ---
echo ""
echo ">>> [3/6] 验证模型 ..."
python -c "
import sys, os
sys.path.insert(0, os.getcwd())
from models.dnn import AnomalyDNN
m = AnomalyDNN(input_dim=79)
print('✅ 模型创建成功!')
"

# --- Step 4: 下载 CICIoT2023 ---
echo ""
echo ">>> [4/6] 下载 CICIoT2023 数据集 → $RAW_DIR ..."
if [ ! -f "$RAW_DIR/CICIoT2023.csv" ]; then
    python -m data.download_datasets --dataset ciciot2023 --output_dir "$RAW_DIR"
    if [ -f "$RAW_DIR/CICIoT2023.csv" ]; then
        echo "✅ CICIoT2023 下载完成! → $RAW_DIR/CICIoT2023.csv"
    else
        echo "⚠️ CICIoT2023 自动下载失败"
        echo "   请手动上传 CICIoT2023.csv 到 $RAW_DIR/ 目录"
    fi
else
    echo "✅ CICIoT2023 已存在: $RAW_DIR/CICIoT2023.csv,跳过下载"
fi

# --- Step 4b: 下载 IoTID20 ---
echo ""
echo ">>> [4b/6] 下载 IoTID20 数据集 → $RAW_DIR ..."
if [ ! -f "$RAW_DIR/IoTID20.csv" ] && [ ! -d "$RAW_DIR/iotid20_chunks" ]; then
    python -m data.download_datasets --dataset iotid20 --output_dir "$RAW_DIR"
    if [ -f "$RAW_DIR/IoTID20.csv" ]; then
        echo "✅ IoTID20 下载完成! → $RAW_DIR/IoTID20.csv"
    else
        echo "⚠️ IoTID20 自动下载失败,需要手动上传"
        echo ""
        echo "   === 手动上传 IoTID20 的3种方式 ==="
        echo "   方式1: 从 Mac 上传 CSV 文件"
        echo "     scp -P <端口> ~/Downloads/IoTID20*.csv root@<地址>:$RAW_DIR/"
        echo ""
        echo "   方式2: 在 AutoDL JupyterLab 文件管理里拖拽上传"
        echo "     把 IoTID20 CSV 文件拖到 $RAW_DIR/ 目录下"
        echo ""
        echo "   方式3: 在 AutoDL JupyterLab Terminal 里 wget 下载"
        echo "     wget -O $RAW_DIR/IoTID20.csv https://data.mendeley.com/..."
    fi
else
    echo "✅ IoTID20 已存在: $RAW_DIR/,跳过下载"
fi

# --- Step 5: 预处理数据 (输出到 /root/datasets/processed/) ---
echo ""
echo ">>> [5/6] 预处理数据集 → $PROC_DIR ..."
echo "   数据与代码分离: 处理结果存放在独立目录 $PROC_DIR"

# 不再删除旧预处理文件 — 数据集独立保存，不随代码更新被删除

# 预处理 CICIoT2023
if [ -f "$RAW_DIR/CICIoT2023.csv" ]; then
    echo "   预处理 CICIoT2023 ..."
    python -m data.preprocess --dataset ciciot2023 --raw_dir "$RAW_DIR" --output_dir "$PROC_DIR" --N 10 --alpha 0.5 --seed 1 --subsample_ratio 1.0
    if [ -f "$PROC_DIR/ciciot2023_train.npz" ]; then
        echo "✅ CICIoT2023 预处理完成! → $PROC_DIR/"
    else
        echo "❌ CICIoT2023 预处理失败"
    fi
else
    echo "⚠️ CICIoT2023 原始数据不存在,需要先下载"
fi

# 预处理 IoTID20 (单文件模式)
if [ -f "$RAW_DIR/IoTID20.csv" ]; then
    echo "   预处理 IoTID20 ..."
    python -m data.preprocess --dataset iotid20 --raw_dir "$RAW_DIR" --output_dir "$PROC_DIR" --N 10 --alpha 0.5 --seed 1
    if [ -f "$PROC_DIR/iotid20_train.npz" ]; then
        echo "✅ IoTID20 预处理完成! → $PROC_DIR/"
    else
        echo "❌ IoTID20 预处理失败"
    fi
elif [ -d "$RAW_DIR/iotid20_chunks" ]; then
    echo "   预处理 IoTID20 (chunk模式) ..."
    python -m data.preprocess --dataset iotid20 --chunk_dir "$RAW_DIR/iotid20_chunks" --output_dir "$PROC_DIR" --N 10 --alpha 0.5 --seed 1
    if [ -f "$PROC_DIR/iotid20_train.npz" ]; then
        echo "✅ IoTID20 预处理完成! → $PROC_DIR/"
    else
        echo "❌ IoTID20 预处理失败"
    fi
else
    echo "⚠️ IoTID20 原始数据不存在,需要先上传或下载"
fi

# --- Step 6: 最终检查 ---
echo ""
echo ">>> [6/6] 最终环境检查 ..."
echo ""
echo "  检查数据文件 ($PROC_DIR/):"
python -c "
import os
import numpy as np

proc_dir = '$PROC_DIR'
files = [
    f'{proc_dir}/ciciot2023_train.npz',
    f'{proc_dir}/ciciot2023_test.npz',
    f'{proc_dir}/iotid20_train.npz',
    f'{proc_dir}/iotid20_test.npz',
    f'{proc_dir}/ciciot2023_partitions.json',
    f'{proc_dir}/iotid20_partitions.json',
]
found = 0
for f in files:
    exists = os.path.exists(f)
    tag = '✅' if exists else '❌'
    print(f'  {tag} {f}')
    if exists:
        found += 1

# 检查数据维度
print('')
print('  检查数据维度:')
for ds in ['iotid20', 'ciciot2023']:
    train_f = f'{proc_dir}/{ds}_train.npz'
    if os.path.exists(train_f):
        d = np.load(train_f)
        X_shape = d['X'].shape
        y_shape = d['y'].shape
        print(f'  ✅ {ds}: X.shape={X_shape}, y.shape={y_shape}')
        print(f'     特征维度={X_shape[1]}, 样本数={X_shape[0]}')
    else:
        print(f'  ❌ {ds}: 数据文件不存在')

print(f'\n  数据文件就绪: {found}/{len(files)}')
if found >= 4:
    print('  至少一个数据集已准备完成,可以开始实验!')
else:
    print('  ⚠️ 数据集不完整,需要先完成数据准备')
"

# --- 完成 ---
echo ""
echo "=========================================="
echo "  安装 & 数据准备流程结束"
echo "=========================================="
echo ""
echo "  📁 数据目录: $DATASET_ROOT/"
echo "     原始数据: $RAW_DIR/"
echo "     处理数据: $PROC_DIR/"
echo ""
echo "  💡 数据与代码分离! 更换代码不影响数据集"
echo "     更换代码: scp上传zip → unzip -o 到 /root/AdaGQ-Matter/"
echo "     数据集始终保存在: /root/datasets/"
echo ""
echo "  如果所有数据集都 ✅，运行实验:"
echo "    bash run_T50.sh          # T=50 快速版"
echo ""
echo "  如果数据集有 ❌，请先手动上传数据文件到:"
echo "    $RAW_DIR/"
echo ""
