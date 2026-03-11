"""Plotting utilities."""

from pathlib import Path
from typing import List, Optional

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
    save_path: Optional[Path] = None,
):
    """Plot overlapping histograms of Day 1 and Day 2 reconstruction errors.

    Args:
        day1_errors: Per-window errors for in-distribution test data.
        day2_errors: Per-window errors for drifted test data.
        title: Plot title.
        save_path: Path to save figure.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(
        day1_errors, bins=40, alpha=0.6,
        label=f"Day 1 (n={len(day1_errors)})", color="steelblue",
    )
    plt.hist(
        day2_errors, bins=40, alpha=0.6,
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
