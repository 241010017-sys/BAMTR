"""Plot actual logs and prediction metrics; never synthesize loss curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_run(run_dir: str | Path, output: str | Path | None = None, show: bool = False):
    run_dir = Path(run_dir)
    history = pd.read_csv(run_dir / "history.csv")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 13, "axes.labelsize": 15, "axes.linewidth": 1.1})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], color="#007398", linewidth=2, label="Training")
    axes[0].plot(history["epoch"], history["val_loss"], color="#AA2163", linewidth=2, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Total loss")
    axes[0].legend(frameon=False)
    axes[1].plot(history["epoch"], history["val_mae_m"], color="#313695", linewidth=2, marker="o", markersize=4)
    best_index = history["val_mae_m"].idxmin()
    best = history.loc[best_index]
    axes[1].scatter([best["epoch"]], [best["val_mae_m"]], color="#F14D43", edgecolor="black", zorder=5,
                    label=f"Best MAE: {best['val_mae_m']:.2f} m")
    axes[1].set(xlabel="Epoch", ylabel="Validation MAE (m)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_recall(metrics_path: str | Path, output: str | Path):
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    pairs = sorted((float(threshold), value * 100) for threshold, value in metrics["recall_at_m"].items())
    x, y = zip(*pairs)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(x, y, color="#313695", marker="*", linewidth=2)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{value:g}" for value in x])
    ax.set(xlabel="Spatial tolerance (m)", ylabel="Location Recall (%)", ylim=(0, 100))
    ax.grid(alpha=0.25)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
