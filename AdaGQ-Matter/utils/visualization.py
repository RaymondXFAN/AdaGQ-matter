"""
Experimental Results Visualization Module.

Produces publication-quality figures for AdaGQ-Matter evaluation:
1. F1 vs. FL rounds curve (multi-method comparison)
2. ε vs. F1 privacy-utility tradeoff
3. Communication overhead comparison (bar chart)
4. Convergence curves (loss vs. rounds, multi-method)
5. Ablation study bar chart
6. Non-IID degree comparison
7. Membership inference attack (MIA) success rate comparison

All plots use matplotlib + seaborn with academic styling:
- Font: serif (Times-like)
- Grid: subtle
- Colors: colorblind-friendly palette
- Figure size: single-column (3.5in) or double-column (7in)

Reference: AdaGQ-Matter, Section 5 (Experimental Results)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union

# ============================================================
# Academic Plot Style Configuration
# ============================================================

# Colorblind-friendly palette (Wong, 2011)
CB_PALETTE = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Light Blue
    "#E69F00",  # Orange
    "#000000",  # Black
]


def setup_academic_style() -> None:
    """
    Configure matplotlib/seaborn for publication-quality plots.

    Settings:
    - Font: serif (Times New Roman fallback)
    - Font size: 10pt base, 12pt title
    - Grid: subtle gray dashes
    - Tight layout
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })
    sns.set_palette(CB_PALETTE)


setup_academic_style()


# ============================================================
# Plot Functions
# ============================================================

def plot_f1_vs_rounds(
    results_dict: Dict[str, List[float]],
    output_path: str,
    title: str = "F1 Score vs. FL Rounds",
    xlabel: str = "Round",
    ylabel: str = "F1 Score (macro)",
    figsize: tuple = (7, 4),
) -> None:
    """
    Plot F1 score evolution across FL rounds for multiple methods.

    Args:
        results_dict: Dict mapping method_name → list of F1 values per round.
                      Example: {"AdaGQ": [0.82, 0.85, ...], "FedAvg": [0.78, 0.80, ...]}
        output_path: File path to save the figure (e.g., "figs/f1_vs_rounds.pdf")
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size in inches
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, (method, f1_values) in enumerate(results_dict.items()):
        rounds = range(1, len(f1_values) + 1)
        ax.plot(rounds, f1_values, label=method,
                color=CB_PALETTE[i % len(CB_PALETTE)],
                marker="o" if i == 0 else "s",
                markevery=max(1, len(f1_values) // 10))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[Visualization] Saved F1 vs rounds plot → {output_path}")


def plot_dp_tradeoff(
    epsilon_list: List[float],
    f1_list: List[float],
    output_path: str,
    title: str = "Privacy–Utility Tradeoff (ε vs. F1)",
    xlabel: str = "Privacy Budget ε",
    ylabel: str = "F1 Score (macro)",
    figsize: tuple = (3.5, 3),
) -> None:
    """
    Plot privacy-utility tradeoff: ε vs. F1 score.

    Args:
        epsilon_list: List of ε values (x-axis)
        f1_list: Corresponding F1 scores (y-axis)
        output_path: File path to save the figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (single-column width by default)
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(epsilon_list, f1_list, color=CB_PALETTE[0],
            marker="D", linewidth=2, markersize=6)

    # Shade region where ε is too small (excessive noise)
    if len(epsilon_list) > 1:
        ax.fill_between(epsilon_list, f1_list, alpha=0.1, color=CB_PALETTE[0])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale("log" if max(epsilon_list) / min(epsilon_list) > 10 else "linear")
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[Visualization] Saved DP tradeoff plot → {output_path}")


def plot_communication_comparison(
    methods_dict: Dict[str, float],
    output_path: str,
    title: str = "Communication Overhead Comparison",
    xlabel: str = "Method",
    ylabel: str = "Total Upload (KB)",
    figsize: tuple = (3.5, 3),
) -> None:
    """
    Bar chart comparing communication overhead across methods.

    Args:
        methods_dict: Dict mapping method_name → total upload in KB.
                      Example: {"FedAvg": 1534.0, "AdaGQ": 153.4, "SignSGD": 76.7}
        output_path: File path to save the figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(methods_dict.keys())
    values = list(methods_dict.values())
    colors = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(methods))]

    bars = ax.bar(methods, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(values),
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[Visualization] Saved communication comparison → {output_path}")


def plot_convergence_curve(
    results_dict: Dict[str, List[float]],
    output_path: str,
    title: str = "Training Loss Convergence",
    xlabel: str = "Round",
    ylabel: str = "Cross-Entropy Loss",
    figsize: tuple = (7, 4),
) -> None:
    """
    Plot convergence curves (loss vs. rounds) for multiple methods.

    Args:
        results_dict: Dict mapping method_name → list of loss values per round.
                      Example: {"AdaGQ": [2.3, 1.8, 0.5, ...], "FedAvg": [2.3, 2.0, 0.6, ...]}
        output_path: File path to save the figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, (method, loss_values) in enumerate(results_dict.items()):
        rounds = range(1, len(loss_values) + 1)
        ax.plot(rounds, loss_values, label=method,
                color=CB_PALETTE[i % len(CB_PALETTE)],
                linestyle="-" if i == 0 else "--",
                marker="o" if i == 0 else "s",
                markevery=max(1, len(loss_values) // 10))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[Visualization] Saved convergence curve → {output_path}")


def plot_ablation_bar(
    ablation_results: Dict[str, float],
    output_path: str,
    title: str = "Ablation Study",
    xlabel: str = "Configuration",
    ylabel: str = "F1 Score (macro)",
    baseline_key: str = "Full (AdaGQ)",
    figsize: tuple = (7, 3.5),
) -> None:
    """
    Bar chart for ablation study results.

    The baseline (full configuration) is highlighted with a distinct color.

    Args:
        ablation_results: Dict mapping config_name → metric value (F1).
                          Example: {"Full (AdaGQ)": 0.92, "No Sparsity": 0.88, "No Quant": 0.90, ...}
        output_path: File path to save the figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        baseline_key: Key name for the baseline configuration (highlighted)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    configs = list(ablation_results.keys())
    values = list(ablation_results.values())

    # Highlight baseline with blue, others with gray
    colors = []
    for cfg in configs:
        if cfg == baseline_key:
            colors.append(CB_PALETTE[0])  # Blue for baseline
        else:
            colors.append("#CCCCCC")  # Gray for ablated variants

    bars = ax.bar(configs, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Add a dashed line at baseline level
    baseline_val = ablation_results.get(baseline_key, 0)
    ax.axhline(y=baseline_val, color=CB_PALETTE[0], linestyle="--", alpha=0.5, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=max(0, baseline_val - 0.15))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[Visualization] Saved ablation bar chart → {output_path}")


def plot_noniid_comparison(
    alpha_list: List[float],
    f1_list: List[float],
    output_path: str,
    title: str = "Performance vs. Non-IID Degree (Dirichlet α)",
    xlabel: str = "Dirichlet α (lower = more Non-IID)",
    ylabel: str = "F1 Score (macro)",
    figsize: tuple = (3.5, 3),
) -> None:
    """
    Plot F1 score as a function of Non-IID degree (Dirichlet α parameter).

    Lower α = more heterogeneous data distribution.

    Args:
        alpha_list: List of Dirichlet α values (x-axis)
        f1_list: Corresponding F1 scores (y-axis)
        output_path: File path to save the figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(alpha_list, f1_list, color=CB_PALETTE[0],
            marker="D", linewidth=2, markersize=6)

    # Annotate key points
    if len(alpha_list) > 0:
        # Highlight worst Non-IID point
        min_alpha_idx = np.argmin(alpha_list)
        ax.annotate(f"α={alpha_list[min_alpha_idx]:.1f}",
                    xy=(alpha_list[min_alpha_idx], f1_list[min_alpha_idx]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=CB_PALETTE[1])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[Visualization] Saved Non-IID comparison → {output_path}")


def plot_attack_mia(
    mia_results: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Membership Inference Attack Success Rate",
    xlabel: str = "Method",
    ylabel: str = "Attack Success Rate (AUC)",
    figsize: tuple = (3.5, 3),
) -> None:
    """
    Bar chart comparing MIA attack success rates across defense methods.

    Args:
        mia_results: Dict mapping method_name → {"attack_auc": float, "baseline_auc": float}.
                      Example: {"No Defense": {"attack_auc": 0.72, "baseline_auc": 0.5},
                                "AdaGQ-DP": {"attack_auc": 0.52, "baseline_auc": 0.5}}
        output_path: File path to save the figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(mia_results.keys())
    attack_aucs = [mia_results[m]["attack_auc"] for m in methods]
    baseline_aucs = [mia_results[m].get("baseline_auc", 0.5) for m in methods]

    x_pos = np.arange(len(methods))
    bar_width = 0.35

    # Attack AUC bars
    bars1 = ax.bar(x_pos - bar_width / 2, attack_aucs, bar_width,
                   label="Attack AUC", color=CB_PALETTE[1], edgecolor="black", linewidth=0.5)
    # Baseline bars
    bars2 = ax.bar(x_pos + bar_width / 2, baseline_aucs, bar_width,
                   label="Random Guess", color="#CCCCCC", edgecolor="black", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars1, attack_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, baseline_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[Visualization] Saved MIA attack plot → {output_path}")


# ============================================================
# Multi-Plot Dashboard (convenience function)
# ============================================================

def create_experiment_dashboard(
    results_dict: Dict[str, Dict[str, List[float]]],
    output_dir: str,
) -> List[str]:
    """
    Generate all standard experiment figures from a unified results dict.

    Args:
        results_dict: Nested dict with structure:
            {
                "f1_per_round": {"AdaGQ": [...], "FedAvg": [...]},
                "loss_per_round": {"AdaGQ": [...], "FedAvg": [...]},
                ...
            }
        output_dir: Directory to save all figures

    Returns:
        List of saved figure file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # F1 vs rounds
    if "f1_per_round" in results_dict:
        path = os.path.join(output_dir, "f1_vs_rounds.pdf")
        plot_f1_vs_rounds(results_dict["f1_per_round"], path)
        saved_paths.append(path)

    # Convergence curve
    if "loss_per_round" in results_dict:
        path = os.path.join(output_dir, "convergence.pdf")
        plot_convergence_curve(results_dict["loss_per_round"], path)
        saved_paths.append(path)

    return saved_paths


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":
    import os
    test_dir = "/tmp/adagq_viz_test"
    os.makedirs(test_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    n_rounds = 30

    print("=== F1 vs Rounds Test ===")
    f1_adagq = np.clip(rng.normal(0.85, 0.05, n_rounds), 0.6, 0.95)
    f1_adagq = np.sort(f1_adagq)  # Make it look like convergence
    f1_fedavg = np.clip(rng.normal(0.80, 0.05, n_rounds), 0.6, 0.90)
    f1_fedavg = np.sort(f1_fedavg)
    f1_signsgd = np.clip(rng.normal(0.75, 0.06, n_rounds), 0.5, 0.85)
    f1_signsgd = np.sort(f1_signsgd)

    plot_f1_vs_rounds(
        {"AdaGQ-Matter": list(f1_adagq), "FedAvg": list(f1_fedavg), "SignSGD": list(f1_signsgd)},
        os.path.join(test_dir, "f1_vs_rounds.pdf"),
    )

    print("\n=== DP Tradeoff Test ===")
    eps_list = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    f1_dp = [0.55, 0.72, 0.85, 0.90, 0.92, 0.93, 0.93]
    plot_dp_tradeoff(eps_list, f1_dp, os.path.join(test_dir, "dp_tradeoff.pdf"))

    print("\n=== Communication Comparison Test ===")
    comm_dict = {"FedAvg": 1534.0, "AdaGQ": 153.4, "SignSGD": 76.7, "Top-k": 306.8}
    plot_communication_comparison(comm_dict, os.path.join(test_dir, "comm_comparison.pdf"))

    print("\n=== Convergence Curve Test ===")
    loss_adagq = 2.3 - 1.8 * np.linspace(0, 1, n_rounds) + rng.normal(0, 0.05, n_rounds)
    loss_fedavg = 2.3 - 1.7 * np.linspace(0, 1, n_rounds) + rng.normal(0, 0.06, n_rounds)
    plot_convergence_curve(
        {"AdaGQ-Matter": list(loss_adagq), "FedAvg": list(loss_fedavg)},
        os.path.join(test_dir, "convergence.pdf"),
    )

    print("\n=== Ablation Bar Chart Test ===")
    ablation = {
        "Full (AdaGQ)": 0.92,
        "No Sparsity": 0.88,
        "No Quant": 0.90,
        "No DP": 0.93,
        "No Error Comp": 0.89,
    }
    plot_ablation_bar(ablation, os.path.join(test_dir, "ablation.pdf"))

    print("\n=== Non-IID Comparison Test ===")
    alpha_list = [0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
    f1_noniid = [0.78, 0.85, 0.88, 0.91, 0.92, 0.93]
    plot_noniid_comparison(alpha_list, f1_noniid, os.path.join(test_dir, "noniid.pdf"))

    print("\n=== MIA Attack Plot Test ===")
    mia = {
        "No Defense": {"attack_auc": 0.72, "baseline_auc": 0.50},
        "FedAvg-DP": {"attack_auc": 0.58, "baseline_auc": 0.50},
        "AdaGQ-DP": {"attack_auc": 0.52, "baseline_auc": 0.50},
    }
    plot_attack_mia(mia, os.path.join(test_dir, "mia_attack.pdf"))

    print("\n=== Dashboard Test ===")
    results = {
        "f1_per_round": {"AdaGQ": list(f1_adagq), "FedAvg": list(f1_fedavg)},
        "loss_per_round": {"AdaGQ": list(loss_adagq), "FedAvg": list(loss_fedavg)},
    }
    paths = create_experiment_dashboard(results, os.path.join(test_dir, "dashboard"))
    print(f"  Saved figures: {paths}")

    # List all generated files
    print(f"\n=== All test figures in {test_dir} ===")
    for f in sorted(os.listdir(test_dir)):
        fpath = os.path.join(test_dir, f)
        print(f"  {f} ({os.path.getsize(fpath)} bytes)")
