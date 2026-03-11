"""Plotting utilities."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_learning_curves(
    train_loss: List[float],
    val_loss: List[float],
    save_path: Optional[Path] = None,
):
    """
    Plot training and validation loss curves.

    Args:
        train_loss: List of training losses
        val_loss: List of validation losses
        save_path: Path to save figure (if None, returns figure)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss", linewidth=2)
    plt.plot(val_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        return plt.gcf()


def plot_reconstruction_error_histogram(
    errors: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
):
    """
    Plot histogram of reconstruction errors.

    Args:
        errors: (n_samples,) array of reconstruction errors
        labels: (n_samples,) array of labels (optional, for separating normal/anomaly)
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    if labels is not None:
        # Separate normal and anomaly
        normal_errors = errors[labels == 0]
        anomaly_errors = errors[labels == 1]

        plt.hist(
            normal_errors,
            bins=30,
            alpha=0.6,
            label=f"Normal (n={len(normal_errors)})",
            color="blue",
        )
        if len(anomaly_errors) > 0:
            plt.hist(
                anomaly_errors,
                bins=30,
                alpha=0.6,
                label=f"Anomaly (n={len(anomaly_errors)})",
                color="red",
            )
    else:
        plt.hist(errors, bins=30, alpha=0.7, color="blue")

    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Reconstruction Errors")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        return plt.gcf()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    save_path: Optional[Path] = None,
):
    """
    Plot ROC curve for anomaly detection.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auroc: Area under ROC curve
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUROC={auroc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        return plt.gcf()


def plot_day1_vs_day2_histogram(
    day1_errors: np.ndarray,
    day2_errors: np.ndarray,
    title: str = "Reconstruction Error: Day 1 vs Day 2",
    bins: int = 40,
    save_path: Optional[Path] = None,
):
    """Plot overlapping histograms of Day 1 and Day 2 reconstruction errors.

    Args:
        day1_errors: Per-window errors for in-distribution test data.
        day2_errors: Per-window errors for drifted test data.
        title: Plot title.
        bins: Number of histogram bins.
        save_path: Path to save figure.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(
        day1_errors, bins=bins, alpha=0.6,
        label=f"Day 1 (n={len(day1_errors)})", color="steelblue",
    )
    plt.hist(
        day2_errors, bins=bins, alpha=0.6,
        label=f"Day 2 (n={len(day2_errors)})", color="coral",
    )
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        return plt.gcf()


def plot_drift_bar(
    drift_metrics: dict,
    model_name: str = "",
    save_path: Optional[Path] = None,
):
    """Bar plot comparing Day 1 vs Day 2 mean MSE and degradation %.

    Args:
        drift_metrics: Dict returned by ``compute_drift_metrics``.
        model_name: Label for the model.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Mean MSE ---
    labels = ["Day 1", "Day 2"]
    values = [drift_metrics["mse_day1_mean"], drift_metrics["mse_day2_mean"]]
    colors = ["steelblue", "coral"]
    axes[0].bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Mean MSE")
    axes[0].set_title(f"Mean Reconstruction Error{' – ' + model_name if model_name else ''}")
    axes[0].grid(True, alpha=0.3, axis="y")

    # --- Degradation % ---
    deg = drift_metrics["degradation_pct"]
    bar_color = "coral" if deg > 0 else "mediumseagreen"
    axes[1].bar(["Degradation"], [deg], color=bar_color, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Degradation (%)")
    axes[1].set_title(f"MSE Degradation Day1→Day2{' – ' + model_name if model_name else ''}")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


# ---------------------------------------------------------------------------
# Phase 1.5 plots
# ---------------------------------------------------------------------------


def plot_subtle_drift_curves(
    model_results: Dict[str, Dict[str, List[float]]],
    save_path: Optional[Path] = None,
):
    """Bar chart of Day 1 vs Day 2 mean MSE per model with error bars across seeds.

    Args:
        model_results: ``{model_name: {"day1": [mse_per_seed], "day2": [...]}}``
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(model_results.keys())
    x = np.arange(len(model_names))
    width = 0.35

    day1_means = [np.mean(model_results[m]["day1"]) for m in model_names]
    day1_stds = [np.std(model_results[m]["day1"]) for m in model_names]
    day2_means = [np.mean(model_results[m]["day2"]) for m in model_names]
    day2_stds = [np.std(model_results[m]["day2"]) for m in model_names]

    ax.bar(x - width / 2, day1_means, width, yerr=day1_stds, label="Day 1",
           color="steelblue", edgecolor="black", linewidth=0.5, capsize=4)
    ax.bar(x + width / 2, day2_means, width, yerr=day2_stds, label="Day 2",
           color="coral", edgecolor="black", linewidth=0.5, capsize=4)

    ax.set_ylabel("Mean MSE")
    ax.set_title("Subtle Drift: Day 1 vs Day 2 Reconstruction Error")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


def plot_corruption_robustness(
    model_corruption: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
):
    """Line plot of corruption degradation % vs sigma for each model.

    Args:
        model_corruption: ``{model_name: {"0.02": deg%, "0.05": ..., "0.1": ...}}``
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "D", "^", "v"]
    for i, (name, levels) in enumerate(model_corruption.items()):
        sigmas = sorted(levels.keys(), key=float)
        degs = [levels[s] for s in sigmas]
        ax.plot(
            [float(s) for s in sigmas], degs,
            marker=markers[i % len(markers)], linewidth=2, markersize=8, label=name,
        )

    ax.set_xlabel("Corruption σ")
    ax.set_ylabel("Degradation %")
    ax.set_title("Corruption Robustness: MSE Degradation vs Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


def plot_corruption_error_histograms(
    model_errors: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[Path] = None,
):
    """Histograms: clean vs corrupted reconstruction errors for each model.

    Args:
        model_errors: ``{model_name: {"clean": errors, "corrupted": errors}}``
        save_path: Path to save figure.
    """
    n_models = len(model_errors)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
    axes = axes[0]

    for ax, (name, errs) in zip(axes, model_errors.items()):
        ax.hist(errs["clean"], bins=30, alpha=0.6, color="steelblue", label="Clean")
        ax.hist(errs["corrupted"], bins=30, alpha=0.6, color="coral", label="Corrupted")
        ax.set_title(name)
        ax.set_xlabel("Reconstruction Error (MSE)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


def plot_auroc_boxplots(
    model_aurocs: Dict[str, List[float]],
    title: str = "AUROC Distribution (5 seeds)",
    save_path: Optional[Path] = None,
):
    """Box-plot of AUROC across seeds for each model.

    Args:
        model_aurocs: ``{model_name: [auroc_seed1, auroc_seed2, ...]}``
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    names = list(model_aurocs.keys())
    data = [model_aurocs[n] for n in names]

    bp = ax.boxplot(data, labels=names, patch_artist=True, widths=0.5)
    colors = sns.color_palette("Set2", len(names))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")

    # Overlay individual points
    for i, d in enumerate(data, start=1):
        ax.scatter([i] * len(d), d, color="black", s=30, zorder=3, alpha=0.7)

    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig
