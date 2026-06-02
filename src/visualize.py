from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

FIGURES_DIR = Path("results/figures")


def ensure_figures_dir(directory: Path | str = FIGURES_DIR) -> Path:
    """Create the figures output directory if it does not exist."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    save_path = Path(save_path)
    ensure_figures_dir(save_path.parent)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(
    results_dict: dict[str, float],
    metric_name: str,
    save_path: str | Path,
) -> None:
    """Bar plot comparing Baseline vs SNN for a given metric."""
    save_path = Path(save_path)
    ensure_figures_dir(save_path.parent)

    models = list(results_dict.keys())
    values = list(results_dict.values())

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, values, color=["#4C72B0", "#55A868"])
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Comparison")
    ax.set_ylim(0, max(max(values) * 1.15, 0.1))

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_task_confusion_matrices(
    results: dict[str, Any],
    *,
    label_names: list[str],
    label_ids: list[int],
    prefix: str,
    figures_dir: Path,
) -> None:
    baseline_cm = confusion_matrix(
        results["baseline"]["y_test"],
        results["baseline"]["y_pred"],
        labels=label_ids,
    )
    plot_confusion_matrix(
        baseline_cm,
        label_names,
        f"{prefix} Baseline Confusion Matrix",
        figures_dir / f"{prefix.lower()}_baseline_cm.png",
    )

    snn_cm = confusion_matrix(
        results["snn"]["y_test"],
        results["snn"]["y_pred"],
        labels=label_ids,
    )
    plot_confusion_matrix(
        snn_cm,
        label_names,
        f"{prefix} Tuned SNN Confusion Matrix",
        figures_dir / f"{prefix.lower()}_snn_cm.png",
    )


def generate_all_figures(
    binary_results: dict[str, Any],
    multi_results: dict[str, Any],
    *,
    binary_label_names: list[str],
    multi_label_names: list[str],
    figures_dir: Path | str = FIGURES_DIR,
) -> None:
    """Generate all Step 14 evaluation figures."""
    figures_dir = ensure_figures_dir(figures_dir)

    _save_task_confusion_matrices(
        binary_results,
        label_names=binary_label_names,
        label_ids=[0, 1],
        prefix="Binary",
        figures_dir=figures_dir,
    )
    _save_task_confusion_matrices(
        multi_results,
        label_names=multi_label_names,
        label_ids=list(range(len(multi_label_names))),
        prefix="Multi",
        figures_dir=figures_dir,
    )

    plot_metric_comparison(
        {
            "Baseline": multi_results["baseline"]["acc"],
            "SNN": multi_results["snn"]["acc"],
        },
        "Accuracy",
        figures_dir / "accuracy_comparison.png",
    )
    plot_metric_comparison(
        {
            "Baseline": multi_results["baseline"]["macro_f1"],
            "SNN": multi_results["snn"]["macro_f1"],
        },
        "Macro F1",
        figures_dir / "macrof1_comparison.png",
    )
