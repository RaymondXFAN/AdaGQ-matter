#!/bin/bash
# ============================================================
# 重跑 dp_fedavg (Bug#19 修复版) — 标准 DP-SGD + RDP 追踪
# ============================================================
# 用途: 只重跑 dp_fedavg 5个种子, 不碰其他已完成实验
# 前置: 等当前 quant_only/naive_combination 循环跑完后执行
# 说明: 会先删除旧的 dp_fedavg 结果文件 (旧实现有bug, 结果不可用)
# ============================================================

cd "$(dirname "$0")" || { echo "❌ 无法切换到项目目录"; exit 1; }
echo "工作目录: $(pwd)"

LOG_FILE="experiment_log.txt"
CONFIG="configs/base_gpu_T50.yaml"

# --- 删除旧的 dp_fedavg 结果 (Bug#19 修复前的结果不可信) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🧹 删除旧 dp_fedavg 结果文件 (Bug#19 修复前)..."
rm -f results/dp_fedavg_iotid20_alpha0.5_seed*.txt
rm -f results/dp_fedavg_iotid20_alpha0.5_seed*.json
echo "✅ 旧结果已清理"

# --- 跑 dp_fedavg × 5 seeds ---
for seed in 1 2 3 4 5; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> 运行: dp_fedavg, seed=$seed, dataset=iotid20"
    START=$(date +%s)
    python -m experiments.run_main --dataset iotid20 --method dp_fedavg --seed $seed --alpha 0.5 --config $CONFIG 2>&1 | tee -a $LOG_FILE
    EXIT_CODE=${PIPESTATUS[0]}
    END=$(date +%s)
    if [ $EXIT_CODE -ne 0 ]; then
        echo "    ⚠️ dp_fedavg seed=$seed 失败 (exit_code=$EXIT_CODE), 继续下一个"
    else
        echo "    ✅ dp_fedavg seed=$seed 完成! 耗时: $((END-START))秒"
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ dp_fedavg 全部重跑完成!"
echo "结果文件:"
ls -la results/dp_fedavg_iotid20_alpha0.5_seed*.txt
