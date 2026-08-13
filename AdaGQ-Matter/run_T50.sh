#!/bin/bash
# ============================================================
# AdaGQ-Matter 实验运行脚本 (T=50 GPU 版 v7)
# ============================================================
# v7 修复内容：
#   - Bug#16: global_model 未移到GPU → 加 .to(device)，解决 CPU/CUDA 设备不匹配
#   - Bug#17: ciciot2023 类别数=1 数据异常 → 加类别数检查，<2类自动报错
#   - Bug#18: diagnose.py output_dim 逻辑 → 改用 config 的 output_dim
# v6 修复内容（已包含）：
#   - Bug#13: 用 diagnose.py 替代所有 python -c 内联命令
#   - Bug#14: GPU显存 total_mem → total_memory
#   - Bug#15: summary错误处理（失败实验不阻断summary生成）
#   - 数据集与代码分离 (数据在 /root/datasets/)
#   - AutoDL 学术加速自动开启
#   - GPU 状态监控 (每5分钟打印一次)
# ============================================================

# 自动切换到项目目录（防止在其他目录运行时出错）
cd "$(dirname "$0")" || { echo "❌ 无法切换到项目目录"; exit 1; }
echo "工作目录: $(pwd)"

LOG_FILE="experiment_log.txt"
DATA_DIR="/root/datasets/processed"
OLD_DATA_DIR="data/processed"
CONFIG="configs/base_gpu_T50.yaml"
SEEDS="1 2 3 4 5"

# --- 函数: 打印带时间戳的日志 ---
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# --- 函数: GPU 状态监控 ---
gpu_monitor() {
    if command -v nvidia-smi &> /dev/null; then
        log "📊 GPU状态: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'nvidia-smi不可用')"
    else
        log "📊 GPU监控: nvidia-smi 未安装"
    fi
}

# ============================================================
# Step 0: 开启 AutoDL 学术加速
# ============================================================
log "🚀 [0/9] 开启 AutoDL 学术加速 ..."
if [ -f "/etc/network_turbo" ]; then
    source /etc/network_turbo 2>/dev/null || true
    log "✅ AutoDL 学术加速已开启!"
elif [ -f "/usr/local/bin/autodl-tmp" ]; then
    source /usr/local/bin/autodl-tmp 2>/dev/null || true
    log "✅ AutoDL 加速已开启!"
else
    log "⚠️ 未找到 AutoDL 加速脚本, 设置 HF 镜像 ..."
    export HF_ENDPOINT=https://hf-mirror.com
    log "   已设置 HF_ENDPOINT=https://hf-mirror.com"
fi

# ============================================================
# Step 1: 环境诊断（用 diagnose.py 替代所有 python -c 内联命令）
# ============================================================
log "🔍 [1/9] 环境诊断 ..."
log "   Python版本: $(python --version 2>&1)"
log "   工作目录: $(pwd)"
log "   关键文件检查:"
for f in experiments/run_main.py fl/client.py core/dp.py core/compression.py configs/base_gpu_T50.yaml scripts/diagnose.py; do
    if [ -f "$f" ]; then
        log "   ✅ $f 存在"
    else
        log "   ❌ $f 缺失!"
    fi
done

# 用 Python 诊断脚本替代所有 python -c 内联命令（彻底避免 shell 引号问题）
log "📋 运行诊断脚本 ..."
python scripts/diagnose.py "$CONFIG" "$DATA_DIR" 2>&1 | while IFS= read -r line; do
    log "   $line"
done

gpu_monitor

# ============================================================
# Step 2: 数据准备（迁移或下载）
# ============================================================
log "📦 [2/9] 数据准备 ..."

mkdir -p "$DATA_DIR"

# 检查新位置是否有数据
HAS_DATA_IN_NEW=false
if [ -f "$DATA_DIR/iotid20_train.npz" ] || [ -f "$DATA_DIR/ciciot2023_train.npz" ]; then
    HAS_DATA_IN_NEW=true
fi

# 检查老位置是否有数据
HAS_DATA_IN_OLD=false
if [ -f "$OLD_DATA_DIR/iotid20_train.npz" ] || [ -f "$OLD_DATA_DIR/ciciot2023_train.npz" ]; then
    HAS_DATA_IN_OLD=true
fi

if [ "$HAS_DATA_IN_NEW" = true ]; then
    log "   ✅ 数据已在新位置 $DATA_DIR/ — 无需迁移"
elif [ "$HAS_DATA_IN_OLD" = true ]; then
    log "   🔄 发现数据在老位置 $OLD_DATA_DIR/, 正在迁移到 $DATA_DIR/ ..."
    cp -v "$OLD_DATA_DIR"/*.npz "$DATA_DIR/" 2>/dev/null || true
    cp -v "$OLD_DATA_DIR"/*.json "$DATA_DIR/" 2>/dev/null || true
    log "   ✅ 数据迁移完成! 以后更换代码不影响数据集"
else
    log "   ⚠️ 新旧位置都没有数据!"
    log "   🔄 自动运行 setup_autodl.sh 下载和预处理数据 ..."
    bash setup_autodl.sh 2>&1 | while IFS= read -r line; do log "   $line"; done
    if [ -f "$DATA_DIR/iotid20_train.npz" ] || [ -f "$DATA_DIR/ciciot2023_train.npz" ]; then
        log "   ✅ 数据准备完成! 继续实验 ..."
    else
        log "   ❌ 数据准备失败! 请手动检查"
        log "   手动步骤: bash setup_autodl.sh"
        exit 1
    fi
fi

# 设置 HAS_IOTID20 / HAS_CICIoT 标志
if [ -f "$DATA_DIR/iotid20_train.npz" ]; then
    HAS_IOTID20=true
else
    HAS_IOTID20=false
fi
if [ -f "$DATA_DIR/ciciot2023_train.npz" ]; then
    HAS_CICIoT=true
else
    HAS_CICIoT=false
fi

log "   HAS_IOTID20=$HAS_IOTID20  HAS_CICIoT=$HAS_CICIoT"

if [ "$HAS_IOTID20" = false ] && [ "$HAS_CICIoT" = false ]; then
    log "❌ 两个数据集都没有准备好!"
    log "   请先运行: bash setup_autodl.sh"
    exit 1
fi

# ============================================================
# Step 3: Sanity Test — 1轮快速验证（确保代码能跑）
# ============================================================
log ""
log "🧪 [3/9] Sanity Test — 1轮快速验证 ..."
log "   如果这步失败，说明代码有bug，需要修复后再跑全实验"

SANITY_DATASET="iotid20"
if [ "$HAS_IOTID20" = false ]; then
    SANITY_DATASET="ciciot2023"
fi

SANITY_START=$(date +%s)
python -m experiments.run_main --dataset $SANITY_DATASET --method fedavg --seed 1 --alpha 0.5 --config $CONFIG --T 1 --N 3 2>&1 | tee -a $LOG_FILE
SANITY_EXIT=${PIPESTATUS[0]}
SANITY_END=$(date +%s)
SANITY_TIME=$((SANITY_END-SANITY_START))

if [ $SANITY_EXIT -ne 0 ]; then
    log "   ❌ Sanity Test 失败! (exit_code=$SANITY_EXIT, 耗时=$SANITY_TIME秒)"
    log "   ↑ 上面有完整的错误 traceback ↑"
    log "   请检查错误信息，修复后再继续"
    log ""
    log "   常见原因:"
    log "   1. 数据路径不对 → 检查 config 的 data_dir 是否指向 $DATA_DIR"
    log "   2. 模型维度不匹配 → 检查诊断脚本输出的 input_dim 是否与数据实际维度一致"
    log "   3. CPU/GPU设备不匹配 → 检查 config 的 device=cuda，以及代码中 .to(device)"
    log "   4. 数据类别数异常 → 检查诊断脚本输出的 类别数 是否>=2"
    log "   5. import 错误 → 检查是否有模块缺失"
    log ""
    log "   快速排查命令:"
    log "     python scripts/diagnose.py $CONFIG $DATA_DIR"
    log "     python -m experiments.run_main --dataset $SANITY_DATASET --method fedavg --seed 1 --alpha 0.5 --config $CONFIG --T 1 --N 3"
    exit 1
else
    log "   ✅ Sanity Test 通过! (耗时=$SANITY_TIME秒) — 代码可正常运行"
fi

# ============================================================
# Step 4+: 正式实验
# ============================================================
TOTAL_START=$(date +%s)
RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

# --- GPU 后台监控 (每5分钟打印一次) ---
gpu_monitor_pid=""
(
    while true; do
        sleep 300  # 5分钟
        gpu_monitor
    done
) &
gpu_monitor_pid=$!

# ============================================================
# E1: IoTID20 主实验 (AdaGQ-Matter vs 6 baselines)
# ============================================================
if [ "$HAS_IOTID20" = true ]; then
    log ""
    log "=============================================="
    log "  [4/9] E1: IoTID20 主实验 (7方法 × 5种子 = 35次)"
    log "  预估时间: ~25-30 分钟"
    log "=============================================="

    for method in adagq fedavg fedprox dp_fedavg top_k_only quant_only naive_combination; do
        for seed in $SEEDS; do
            log ""
            log ">>> 运行: $method, seed=$seed, dataset=iotid20"
            START=$(date +%s)
            python -m experiments.run_main --dataset iotid20 --method $method --seed $seed --alpha 0.5 --config $CONFIG 2>&1 | tee -a $LOG_FILE
            EXIT_CODE=${PIPESTATUS[0]}
            END=$(date +%s)
            if [ $EXIT_CODE -ne 0 ]; then
                log "    ⚠️ 该实验失败 (exit_code=$EXIT_CODE), 继续下一个 (上方应有完整错误traceback)"
            else
                log "    ✅ 完成! 耗时: $((END-START))秒"
            fi
        done
    done

    log "✅ E1 IoTID20 主实验完成!"
else
    log "⏭️ 跳过 E1 (IoTID20 数据未准备)"
fi

# ============================================================
# E2: CICIoT2023 主实验
# ============================================================
if [ "$HAS_CICIoT" = true ]; then
    log ""
    log "=============================================="
    log "  [5/9] E2: CICIoT2023 主实验 (7方法 × 5种子 = 35次)"
    log "  预估时间: ~15-20 分钟"
    log "=============================================="

    for method in adagq fedavg fedprox dp_fedavg top_k_only quant_only naive_combination; do
        for seed in $SEEDS; do
            log ""
            log ">>> 运行: $method, seed=$seed, dataset=ciciot2023"
            START=$(date +%s)
            python -m experiments.run_main --dataset ciciot2023 --method $method --seed $seed --alpha 0.5 --config $CONFIG 2>&1 | tee -a $LOG_FILE
            EXIT_CODE=${PIPESTATUS[0]}
            END=$(date +%s)
            if [ $EXIT_CODE -ne 0 ]; then
                log "    ⚠️ 该实验失败 (exit_code=$EXIT_CODE), 继续下一个"
            else
                log "    ✅ 完成! 耗时: $((END-START))秒"
            fi
        done
    done

    log "✅ E2 CICIoT2023 主实验完成!"
else
    log "⏭️ 跳过 E2 (CICIoT2023 数据未准备)"
fi

# ============================================================
# E3: DP 权衡实验 (ε=1/3/5/8/10/∞)
# ============================================================
log ""
log "=============================================="
log "  [6/9] E3: DP 隐私预算权衡实验"
log "=============================================="

if [ "$HAS_IOTID20" = true ]; then
    log ">>> DP tradeoff: dataset=iotid20"
    python -m experiments.run_dp_tradeoff --dataset iotid20 --seed 1 --config $CONFIG 2>&1 | tee -a $LOG_FILE || log "⚠️ DP tradeoff IoTID20 失败"
fi

if [ "$HAS_CICIoT" = true ]; then
    log ">>> DP tradeoff: dataset=ciciot2023"
    python -m experiments.run_dp_tradeoff --dataset ciciot2023 --seed 1 --config $CONFIG 2>&1 | tee -a $LOG_FILE || log "⚠️ DP tradeoff CICIoT2023 失败"
fi

log "✅ E3 DP 权衡实验完成!"

# ============================================================
# E4: 消融实验 (9种配置)
# ============================================================
log ""
log "=============================================="
log "  [7/9] E4: 消融实验 (9种配置)"
log "=============================================="

if [ "$HAS_IOTID20" = true ]; then
    log ">>> 消融: dataset=iotid20"
    python -m experiments.run_ablation --dataset iotid20 --seed 1 --config configs/ablation.yaml 2>&1 | tee -a $LOG_FILE || log "⚠️ 消融 IoTID20 失败"
fi

if [ "$HAS_CICIoT" = true ]; then
    log ">>> 消融: dataset=ciciot2023"
    python -m experiments.run_ablation --dataset ciciot2023 --seed 1 --config configs/ablation.yaml 2>&1 | tee -a $LOG_FILE || log "⚠️ 消融 CICIoT2023 失败"
fi

log "✅ E4 消融实验完成!"

# ============================================================
# E5: Non-IID 稳健性 (α=0.1/0.5/1.0)
# ============================================================
if [ "$HAS_IOTID20" = true ]; then
    log ""
    log "=============================================="
    log "  [8/9] E5: Non-IID 稳健性实验"
    log "=============================================="

    for alpha in 0.1 0.5 1.0; do
        log ">>> Non-IID: dataset=iotid20, alpha=$alpha"
        python -m experiments.run_main --dataset iotid20 --method adagq --seed 1 --alpha $alpha --config $CONFIG 2>&1 | tee -a $LOG_FILE || log "⚠️ Non-IID alpha=$alpha 失败"
    done

    log "✅ E5 Non-IID 实验完成!"
else
    log "⏭️ 跳过 E5 (IoTID20 数据未准备)"
fi

# ============================================================
# E6: 隐私攻击评估 (MIA + DLG + InvGrad)
# ============================================================
log ""
log "=============================================="
log "  E6: 隐私攻击评估"
log "=============================================="

if [ "$HAS_IOTID20" = true ]; then
    log ">>> 攻击评估: dataset=iotid20, ε=3"
    python -m experiments.run_attack --dataset iotid20 --seed 1 --epsilon 3.0 --config $CONFIG 2>&1 | tee -a $LOG_FILE || log "⚠️ 攻击 IoTID20 失败"
fi

if [ "$HAS_CICIoT" = true ]; then
    log ">>> 攻击评估: dataset=ciciot2023, ε=3"
    python -m experiments.run_attack --dataset ciciot2023 --seed 1 --epsilon 3.0 --config $CONFIG 2>&1 | tee -a $LOG_FILE || log "⚠️ 攻击 CICIoT2023 失败"
fi

log "✅ E6 攻击评估完成!"

# ============================================================
# E7: 绘图 + 汇总
# ============================================================
log ""
log "=============================================="
log "  E7: 绘图 & 汇总"
log "=============================================="

python -m utils.visualization --results_dir $RESULTS_DIR 2>&1 | tee -a $LOG_FILE || log "⚠️ 绘图失败"

log "✅ 绘图完成!"

# --- 停止 GPU 监控 ---
if [ -n "$gpu_monitor_pid" ]; then
    kill $gpu_monitor_pid 2>/dev/null
fi

# ============================================================
# 打包结果
# ============================================================
log ""
log "=============================================="
log "  打包实验结果 ..."
log "=============================================="

cd $RESULTS_DIR
zip -r results_T50.zip ./ 2>/dev/null || tar czf results_T50.tar.gz ./
cd ..

# --- 生成 txt 格式结果汇总 ---
log ""
log "=============================================="
log "  生成 txt 格式结果汇总 ..."
log "=============================================="

python scripts/gen_summary.py "$RESULTS_DIR" 2>&1 | tee -a $LOG_FILE

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END-TOTAL_START))

log ""
log "=========================================="
log "  ✅✅✅ 全部实验完成!"
log "=========================================="
log ""
log "  总耗时: $((TOTAL_TIME/60))分钟 $((TOTAL_TIME%60))秒"
log ""
log "  结果文件位置: $(pwd)/$RESULTS_DIR/"
log "  结果汇总(txt): $(pwd)/$RESULTS_DIR/results_summary.txt"
log "  完整日志(txt): $(pwd)/$LOG_FILE"
log "  结果打包(zip): $(pwd)/$RESULTS_DIR/results_T50.zip"
log ""
log "  从 Mac 下载结果 (在 Mac Terminal 执行):"
log "    scp -P 43864 root@connect.cqa1.seetacloud.com:$(pwd)/$RESULTS_DIR/results_summary.txt ~/Downloads/"
log "    scp -P 43864 root@connect.cqa1.seetacloud.com:$(pwd)/$LOG_FILE ~/Downloads/"
log "    scp -P 43864 root@connect.cqa1.seetacloud.com:$(pwd)/$RESULTS_DIR/results_T50.zip ~/Downloads/"
log ""
log "  >>> 记得关机! (AutoDL网页 → 我的实例 → 关机)"
log ""
