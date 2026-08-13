"""
AdaGQ-Matter 论文绘图脚本
生成4张关键figure：
1. 收敛曲线 (F1 vs Round, 7方法对比)
2. F1对比柱状图 (最终F1 ± std)
3. 通信量对比柱状图
4. 隐私-效用-通信三方权衡散点图
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# ===== 全局配置 =====
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# 颜色方案（按隐私+压缩类别）
COLORS = {
    'fedavg':            '#4C72B0',  # 蓝
    'fedprox':           '#55A868',  # 绿
    'top_k_only':        '#C44E52',  # 红
    'quant_only':        '#8172B3',  # 紫
    'naive_combination': '#937860',  # 棕
    'adagq':             '#D62728',  # 深红（本文方法，突出）
    'dp_fedavg':         '#FF8C00',  # 橙
}

# 方法显示名称
DISPLAY_NAMES = {
    'fedavg':            'FedAvg',
    'fedprox':           'FedProx',
    'top_k_only':        'Top-K Only',
    'quant_only':        'Quant Only',
    'naive_combination': 'Naive Combination',
    'adagq':             'AdaGQ-Matter (Ours)',
    'dp_fedavg':         'DP-FedAvg',
}

# 方法顺序（先无DP后有DP，最后突出本文）
METHOD_ORDER = ['fedavg', 'fedprox', 'top_k_only', 'quant_only',
                'naive_combination', 'dp_fedavg', 'adagq']

# ===== 加载数据 =====
def load_all_results(results_dir='results'):
    """加载所有实验的final_metrics和round_results"""
    data = {}
    for method in METHOD_ORDER:
        files = sorted(glob(os.path.join(results_dir, f'{method}_iotid20_alpha0.5_seed*.json')))
        seeds_data = []
        for f in files:
            with open(f) as fp:
                d = json.load(fp)
            fm = d.get('final_metrics', {})
            rr = d.get('round_results', [])
            seeds_data.append({'final': fm, 'rounds': rr, 'seed': fm.get('seed')})
        data[method] = seeds_data
    return data


# ===== Figure 1: 收敛曲线 =====
def fig1_convergence(data, save_path='results/figures/fig1_convergence.png'):
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHOD_ORDER:
        seeds_data = data[method]
        if not seeds_data:
            continue
        # 收集所有种子的rounds
        all_rounds = []
        for s in seeds_data:
            rounds = s['rounds']
            f1s = [r['f1'] for r in rounds]
            rounds_axis = [r['round'] for r in rounds]
            all_rounds.append((rounds_axis, f1s))

        if not all_rounds:
            continue

        # 取最长序列
        max_len = max(len(r[0]) for r in all_rounds)
        padded = np.full((len(all_rounds), max_len), np.nan)
        for i, (x, y) in enumerate(all_rounds):
            padded[i, :len(y)] = y

        mean_f1 = np.nanmean(padded, axis=0)
        rounds_axis = all_rounds[0][0]

        # 标准差阴影
        std_f1 = np.nanstd(padded, axis=0)
        x = np.array(rounds_axis)

        lw = 2.5 if method == 'adagq' else 1.5
        ax.plot(x, mean_f1, color=COLORS[method], lw=lw,
                label=DISPLAY_NAMES[method])
        if method in ['adagq', 'dp_fedavg', 'fedavg']:
            ax.fill_between(x, mean_f1 - std_f1, mean_f1 + std_f1,
                            color=COLORS[method], alpha=0.15)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('F1 Score')
    ax.set_title('Convergence Curves on IoTID20 (α=0.5, 5 seeds)')
    ax.legend(loc='lower right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.80, 1.005])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===== Figure 2: F1 对比柱状图 =====
def fig2_f1_comparison(data, save_path='results/figures/fig2_f1_comparison.png'):
    fig, ax = plt.subplots(figsize=(9, 5))

    methods = METHOD_ORDER
    means = []
    stds = []
    for m in methods:
        f1s = [s['final']['f1'] for s in data[m] if 'final' in s and s['final']]
        means.append(np.mean(f1s))
        stds.append(np.std(f1s, ddof=1))

    x = np.arange(len(methods))
    colors = [COLORS[m] for m in methods]
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.8, capsize=5, error_kw={'lw': 1.5})

    # 突出AdaGQ
    for i, m in enumerate(methods):
        if m == 'adagq':
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2.5)

    # 数值标注
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.005, f'{mean:.4f}',
                ha='center', va='bottom', fontsize=9)

    labels = [DISPLAY_NAMES[m] for m in methods]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Final F1 Score')
    ax.set_title('F1 Comparison on IoTID20 (T=50, N=10, 5 seeds)')
    ax.set_ylim([0.90, 1.005])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 标记DP方法
    ymin = ax.get_ylim()[0]
    for i, m in enumerate(methods):
        if m in ['adagq', 'dp_fedavg']:
            ax.text(i, ymin + 0.005, 'DP', ha='center', va='bottom',
                    fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===== Figure 3: 通信量对比 =====
def fig3_communication(data, save_path='results/figures/fig3_communication.png'):
    fig, ax = plt.subplots(figsize=(9, 5))

    methods = METHOD_ORDER
    comms = []
    for m in methods:
        if data[m]:
            comms.append(data[m][0]['final']['avg_comm_kb'])
        else:
            comms.append(0)

    x = np.arange(len(methods))
    colors = [COLORS[m] for m in methods]
    bars = ax.bar(x, comms, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.8)

    # 突出AdaGQ
    for i, m in enumerate(methods):
        if m == 'adagq':
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2.5)

    # 数值标注
    for i, c in enumerate(comms):
        ax.text(i, c + 2, f'{c:.1f}KB',
                ha='center', va='bottom', fontsize=9)

    labels = [DISPLAY_NAMES[m] for m in methods]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Communication per Round (KB)')
    ax.set_title('Communication Cost Comparison')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 添加节省百分比的注释
    baseline = comms[methods.index('fedavg')]
    for i, m in enumerate(methods):
        if m not in ['fedavg', 'fedprox'] and comms[i] < baseline:
            save_pct = (1 - comms[i]/baseline) * 100
            ax.text(i, comms[i]/2, f'−{save_pct:.0f}%',
                    ha='center', va='center', fontsize=10,
                    color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===== Figure 4: 隐私-效用-通信三方权衡 =====
def fig4_tradeoff(data, save_path='results/figures/fig4_tradeoff.png'):
    fig, ax = plt.subplots(figsize=(8, 6))

    for m in METHOD_ORDER:
        if not data[m]:
            continue
        f1s = [s['final']['f1'] for s in data[m]]
        comm = data[m][0]['final']['avg_comm_kb']
        eps = data[m][0]['final'].get('epsilon', 0)
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s, ddof=1)

        # 气泡大小表示通信量（越大=越多通信）
        size = comm * 30
        marker = '*' if m == 'adagq' else 'o'
        s = size if m != 'adagq' else size * 1.5

        ax.scatter(eps, mean_f1, s=s, c=COLORS[m],
                   alpha=0.7, edgecolor='black', marker=marker,
                   linewidth=1.5 if m == 'adagq' else 1.0,
                   label=DISPLAY_NAMES[m])

        # 标注ε值
        if m in ['adagq', 'dp_fedavg']:
            ax.annotate(f'ε={eps:.2f}', (eps, mean_f1),
                        textcoords='offset points', xytext=(8, 8),
                        fontsize=9, color='red', fontweight='bold')

    ax.set_xlabel('Privacy Budget ε (δ=1e-5)')
    ax.set_ylabel('Final F1 Score')
    ax.set_title('Privacy-Utility Trade-off (IoTID20, 5 seeds)')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', framealpha=0.9)

    # 标注气泡大小含义
    ax.text(0.98, 0.02, 'Bubble size ∝ Communication cost',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===== 主程序 =====
if __name__ == '__main__':
    print("📊 Loading experimental results...")
    data = load_all_results('results')

    print(f"\n🎨 Generating 4 figures...")
    fig1_convergence(data)
    fig2_f1_comparison(data)
    fig3_communication(data)
    fig4_tradeoff(data)

    print("\n✅ All figures generated!")
    print("📁 Output: results/figures/")