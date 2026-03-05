"""Evaluation metrics."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader


def compute_reconstruction_errors(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: str = "mse",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reconstruction errors for all samples.

    Args:
        model: Autoencoder model (should be in eval mode)
        dataloader: Data loader
        device: Device to compute on
        loss_fn: Loss function ('mse' or 'mae')

    Returns:
        errors: (n_samples,) array of reconstruction errors
        labels: (n_samples,) array of labels (if available)
    """
    model.eval()
    errors = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            x_recon = model(x)

            # Compute error
            if loss_fn == "mse":
                error = torch.mean((x - x_recon) ** 2, dim=[1, 2])  # Per-sample MSE
            elif loss_fn == "mae":
                error = torch.mean(torch.abs(x - x_recon), dim=[1, 2])
            else:
                raise ValueError(f"Unknown loss_fn: {loss_fn}")

            errors.append(error.cpu().numpy())
            if y is not None and not torch.all(y == -1):
                labels.append(y.numpy())

    errors = np.concatenate(errors)
    labels = np.concatenate(labels) if labels else None

    return errors, labels


def compute_anomaly_metrics(
    errors: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute anomaly detection metrics.

    Assumes labels are binary (0 = normal, 1 = anomaly).

    Args:
        errors: (n_samples,) array of reconstruction errors
        labels: (n_samples,) array of binary labels

    Returns:
        Dict with keys: auroc, auprc, f1_opt, threshold_opt
    """
    if labels is None:
        return {}

    # AUROC
    auroc = roc_auc_score(labels, errors)

    # AUPRC
    precision, recall, _ = precision_recall_curve(labels, errors)
    auprc = auc(recall, precision)

    # Optimal F1 threshold
    fpr, tpr, thresholds = roc_curve(labels, errors)
    f1_scores = []
    for threshold in thresholds:
        predictions = (errors >= threshold).astype(int)
        f1 = f1_score(labels, predictions)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    f1_opt = f1_scores[best_idx]
    threshold_opt = thresholds[best_idx]

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1_opt": float(f1_opt),
        "threshold_opt": float(threshold_opt),
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Dict with test metrics
    """
    model.eval()
    criterion = torch.nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)

    return {
        "test_loss": float(avg_test_loss),
    }
