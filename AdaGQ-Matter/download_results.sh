#!/bin/bash
# ============================================================
# 从 AutoDL 云主机下载实验结果到 Mac (端口 43864)
# ============================================================
# 使用方法: 在 Mac Terminal 中运行 bash download_results.sh
# 
# 下载的文件说明:
#   experiment_log.txt — 完整运行日志(包含所有打印输出)
#   results/results_summary.txt — 所有实验结果汇总(txt格式)
#   results/*.txt — 每个实验的详细结果(txt格式)
#   results/*.json — 每个实验的完整数据(json格式)
#   results/results_T50.zip — 打包全部结果(zip)
# ============================================================

HOST="connect.cqa1.seetacloud.com"
PORT="43864"
USER="root"
REMOTE_DIR="/root/AdaGQ-Matter"
LOCAL_DIR="$HOME/Downloads/AdaGQ-Matter-results"

echo "=========================================="
echo "  从 AutoDL 下载实验结果"
echo "=========================================="
echo ""
echo "  主机: $USER@$HOST"
echo "  端口: $PORT"
echo "  远程路径: $REMOTE_DIR"
echo "  本地保存: $LOCAL_DIR"
echo ""

# 创建本地目录
mkdir -p "$LOCAL_DIR"
mkdir -p "$LOCAL_DIR/results"

# --- 1. 下载运行日志 (txt) ---
echo "📥 [1/5] 下载运行日志 experiment_log.txt ..."
scp -P $PORT $USER@$HOST:$REMOTE_DIR/experiment_log.txt "$LOCAL_DIR/" 
if [ $? -eq 0 ]; then
    echo "   ✅ 已下载到 $LOCAL_DIR/experiment_log.txt"
else
    echo "   ⚠️ experiment_log.txt 不存在或下载失败"
fi

# --- 2. 下载结果汇总 (txt) ---
echo "📥 [2/5] 下载结果汇总 results_summary.txt ..."
scp -P $PORT $USER@$HOST:$REMOTE_DIR/results/results_summary.txt "$LOCAL_DIR/results/"
if [ $? -eq 0 ]; then
    echo "   ✅ 已下载到 $LOCAL_DIR/results/results_summary.txt"
else
    echo "   ⚠️ results_summary.txt 不存在或下载失败"
fi

# --- 3. 下载所有txt结果 ---
echo "📥 [3/5] 下载所有 .txt 结果文件 ..."
scp -P $PORT $USER@$HOST:$REMOTE_DIR/results/*.txt "$LOCAL_DIR/results/"
if [ $? -eq 0 ]; then
    echo "   ✅ txt 结果文件已下载"
else
    echo "   ⚠️ .txt 结果文件下载失败(可能实验尚未完成)"
fi

# --- 4. 下载json结果 (可选) ---
echo "📥 [4/5] 下载 .json 结果文件 (可选) ..."
scp -P $PORT $USER@$HOST:$REMOTE_DIR/results/*.json "$LOCAL_DIR/results/"
if [ $? -eq 0 ]; then
    echo "   ✅ json 结果文件已下载"
else
    echo "   ⚠️ .json 结果文件下载失败"
fi

# --- 5. 下载打包结果 (zip) ---
echo "📥 [5/6] 下载打包结果 results_T50.zip ..."
scp -P $PORT $USER@$HOST:$REMOTE_DIR/results/results_T50.zip "$LOCAL_DIR/results/"
if [ $? -eq 0 ]; then
    echo "   ✅ 已下载到 $LOCAL_DIR/results/results_T50.zip"
else
    echo "   ⚠️ results_T50.zip 不存在(实验可能尚未打包)"
fi

# --- 6. 下载实验日志中最后的诊断信息 ---
echo "📥 [6/6] 下载实验诊断摘要 ..."
scp -P $PORT $USER@$HOST:$REMOTE_DIR/experiment_log.txt "$LOCAL_DIR/"
echo "   ✅ 完整日志已下载"

echo ""
echo "=========================================="
echo "  ✅ 下载完成!"
echo "=========================================="
echo ""
echo "  📁 文件保存在: $LOCAL_DIR"
echo ""
echo "  💡 数据集在 AutoDL 上独立保存于: /root/datasets/"
echo "     更换代码不影响数据集，下次只需上传新代码zip覆盖即可"
echo ""
echo "  快速查看结果汇总:"
echo "    cat $LOCAL_DIR/results/results_summary.txt"
echo ""
echo "  快速查看运行日志(最后50行):"
echo "    tail -50 $LOCAL_DIR/experiment_log.txt"
echo ""
echo "  把txt内容发给AI (复制粘贴):"
echo "    cat $LOCAL_DIR/results/results_summary.txt"
echo "    cat $LOCAL_DIR/results/adagq_iotid20_alpha0.5_seed1.txt"
echo ""
echo "  >>> 记得关机! (AutoDL网页 → 我的实例 → 关机)"
echo ""
