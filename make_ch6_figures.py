# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


OUT_DIR = Path(r"C:\hithesis\examples\hitbook\chinese\figures")


def setup_style() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font_candidates = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    font_name = next((name for name in font_candidates if name in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.edgecolor"] = "#3f4652"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.color"] = "#333333"
    plt.rcParams["ytick.color"] = "#333333"
    plt.rcParams["text.color"] = "#222222"


def save_tamas_overall() -> None:
    modes = ["Baseline", "Guardian-only", "Sentinel-only", "Guardian+Sentinel"]
    colors = ["#8a8f98", "#3b6fb6", "#d9822b", "#2f8f5b"]
    resistance = [57.0, 98.7, 97.0, 100.0]
    malicious_tools = [267, 2, 9, 0]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d9dee7", linewidth=0.7, alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelrotation=18)

    bars = axes[0].bar(modes, resistance, color=colors, width=0.62)
    axes[0].set_ylabel("攻击抵抗率（%）")
    axes[0].set_ylim(0, 105)
    axes[0].set_yticks(np.arange(0, 101, 20))
    for bar, val in zip(bars, resistance):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    bars = axes[1].bar(modes, malicious_tools, color=colors, width=0.62)
    axes[1].set_ylabel("恶意工具调用数")
    axes[1].set_ylim(0, 285)
    axes[1].set_yticks(np.arange(0, 281, 50))
    for bar, val in zip(bars, malicious_tools):
        y = val + 7 if val > 0 else 4
        axes[1].text(bar.get_x() + bar.get_width() / 2, y, str(val), ha="center", va="bottom", fontsize=9)

    fig.savefig(OUT_DIR / "ch6_tamas_overall_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_attack_type_heatmap() -> None:
    modes = ["Baseline", "Guardian-only", "Sentinel-only", "Guardian+Sentinel"]
    attack_types = ["Byzantine", "Colluding", "Contradicting", "DPI", "IPI", "Impersonation"]
    data = np.array([
        [78.0, 100.0, 98.0, 100.0],
        [50.0, 100.0, 96.0, 100.0],
        [74.0, 100.0, 94.0, 100.0],
        [20.0, 100.0, 98.0, 100.0],
        [92.0, 100.0, 96.0, 100.0],
        [28.0, 92.0, 100.0, 100.0],
    ])

    fig, ax = plt.subplots(figsize=(9.2, 5.3), constrained_layout=True)
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(modes)), labels=modes)
    ax.set_yticks(np.arange(len(attack_types)), labels=attack_types)
    ax.tick_params(axis="x", labelrotation=18)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val >= 75 else "#1e2a35"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(np.arange(-0.5, len(modes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(attack_types), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("攻击抵抗率（%）")
    fig.savefig(OUT_DIR / "ch6_attack_type_resistance.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_sentinel_distribution() -> None:
    actions = ["Allow", "Alert", "Block", "Quarantine", "HITL"]
    colors = ["#7d8790", "#4f79bd", "#d47a2a", "#b25159", "#5c6bc0"]
    gaia_labels = ["GAIA\nSentinel-only", "GAIA\nGuardian+Sentinel"]
    gaia_data = np.array([
        [832, 15, 3, 1, 11],
        [1024, 30, 5, 4, 38],
    ])
    tamas_labels = ["TAMAS\nSentinel-only", "TAMAS\nGuardian+Sentinel"]
    tamas_data = np.array([
        [10174, 901, 919, 334, 258],
        [11465, 797, 765, 534, 814],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)
    for ax, labels, data, ymax in [
        (axes[0], gaia_labels, gaia_data, 1160),
        (axes[1], tamas_labels, tamas_data, 15000),
    ]:
        bottom = np.zeros(data.shape[0])
        x = np.arange(data.shape[0])
        for idx, (action, color) in enumerate(zip(actions, colors)):
            ax.bar(x, data[:, idx], bottom=bottom, label=action, color=color, width=0.55)
            bottom += data[:, idx]
        ax.set_xticks(x, labels)
        ax.set_ylabel("事件数量")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#d9dee7", linewidth=0.7, alpha=0.85)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for xi, total in zip(x, bottom):
            ax.text(xi, total + ymax * 0.018, f"{int(total)}", ha="center", va="bottom", fontsize=9)

    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.12, 1.16), ncol=5, frameon=False)
    fig.savefig(OUT_DIR / "ch6_sentinel_action_distribution.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    setup_style()
    save_tamas_overall()
    save_attack_type_heatmap()
    save_sentinel_distribution()
    for name in [
        "ch6_tamas_overall_comparison.png",
        "ch6_attack_type_resistance.png",
        "ch6_sentinel_action_distribution.png",
    ]:
        path = OUT_DIR / name
        print(f"{path} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
