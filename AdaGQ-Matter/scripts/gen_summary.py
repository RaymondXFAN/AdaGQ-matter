"""
结果汇总生成脚本 — 扫描 results 目录，生成 txt 格式汇总。

用于 run_T50.sh 最后一步，生成方便阅读和传输的 txt 汇总。
"""

import os
import sys
import json
import glob
from datetime import datetime

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    txt_lines = []
    txt_lines.append("=" * 70)
    txt_lines.append("AdaGQ-Matter 实验结果汇总")
    txt_lines.append("=" * 70)
    txt_lines.append(f"生成时间: {datetime.now().isoformat()}")
    txt_lines.append(f"结果目录: {results_dir}")
    txt_lines.append("")
    
    # 扫描所有 json 结果文件
    json_files = glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
    
    n_success = 0
    n_fail = 0
    
    for jf in sorted(json_files):
        basename = os.path.basename(jf)
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            
            # 判断是单个结果还是 summary
            if "final_metrics" in data:
                # 单个结果文件
                fm = data["final_metrics"]
                method = data.get("method", "?")
                dataset = data.get("dataset", "?")
                seed = data.get("seed", "?")
                
                txt_lines.append(f"✅ {basename}: {method}/{dataset}/seed={seed}")
                txt_lines.append(f"   F1={fm.get('f1', '?'):.4f if isinstance(fm.get('f1'), float) else fm.get('f1', '?')}")
                txt_lines.append(f"   Acc={fm.get('accuracy', '?')}")
                txt_lines.append(f"   Comm={fm.get('avg_comm_kb', '?')} KB")
                txt_lines.append(f"   ε={fm.get('epsilon', '?')}")
                txt_lines.append(f"   d={fm.get('d', '?')} 参数")
                txt_lines.append("")
                n_success += 1
            elif "error" in data:
                # 失败的结果
                txt_lines.append(f"❌ {basename}: {data.get('error', '未知错误')}")
                txt_lines.append("")
                n_fail += 1
            else:
                # 可能是 summary.json — 里面是嵌套的多个实验结果
                for key, result in data.items():
                    if isinstance(result, dict):
                        if "final_metrics" in result:
                            fm = result["final_metrics"]
                            txt_lines.append(f"✅ {key}: F1={fm.get('f1', '?')}, Acc={fm.get('accuracy', '?')}, "
                                             f"Comm={fm.get('avg_comm_kb', '?')}KB, ε={fm.get('epsilon', '?')}")
                            n_success += 1
                        elif "error" in result:
                            txt_lines.append(f"❌ {key}: {result['error']}")
                            n_fail += 1
                txt_lines.append("")
        except Exception as e:
            txt_lines.append(f"⚠️ {basename}: 解析失败 - {e}")
            txt_lines.append("")
    
    # 统计
    txt_lines.append("-" * 70)
    txt_lines.append(f"总计: 成功={n_success}, 失败={n_fail}, 文件数={len(json_files)}")
    txt_lines.append("=" * 70)
    
    # 写入文件
    summary_path = os.path.join(results_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(txt_lines))
    
    print(f"✅ 结果汇总已保存到: {summary_path}")
    print(f"   成功={n_success}, 失败={n_fail}")
    
    # 同时打印内容
    print("\n" + "\n".join(txt_lines))

if __name__ == "__main__":
    main()
