"""Evaluation metrics."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
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
    day1_errors: np.ndarray,
    day2_errors: np.ndarray,
) -> Dict[str, float]:
    """
    Compute anomaly detection metrics treating Day 1 as normal and Day 2 as drift.

    Builds binary labels: 0 (Day 1 / normal), 1 (Day 2 / drift) and uses the
    per-window reconstruction error as the anomaly score.

    Args:
        day1_errors: (n_day1,) array of reconstruction errors for in-distribution data.
        day2_errors: (n_day2,) array of reconstruction errors for drifted data.

    Returns:
        Dict with keys: auroc, auprc
    """
    errors = np.concatenate([day1_errors, day2_errors])
    labels = np.concatenate([
        np.zeros(len(day1_errors), dtype=int),
        np.ones(len(day2_errors), dtype=int),
    ])

    # AUROC
    auroc = roc_auc_score(labels, errors)

    # AUPRC
    precision, recall, _ = precision_recall_curve(labels, errors)
    auprc = auc(recall, precision)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
    }


def compute_drift_metrics(
    model: nn.Module,
    day1_loader: DataLoader,
    day2_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compare reconstruction quality on in-distribution vs drifted data.

    Args:
        model: Trained autoencoder (will be set to eval mode).
        day1_loader: In-distribution test set (Session 1 / Day 1).
        day2_loader: Drifted test set (Session 2 / Day 2).
        device: Torch device.

    Returns:
        Dict with keys:
            - mse_day1_mean
            - mse_day2_mean
            - degradation_pct = 100 * ((mse_day2_mean / mse_day1_mean) - 1)
            - q95_day1  (95th percentile of per-window errors)
            - q95_day2
    """
    day1_errors, _ = compute_reconstruction_errors(model, day1_loader, device, loss_fn="mse")
    day2_errors, _ = compute_reconstruction_errors(model, day2_loader, device, loss_fn="mse")

    mse_day1_mean = float(np.mean(day1_errors))
    mse_day2_mean = float(np.mean(day2_errors))

    degradation_pct = 100.0 * ((mse_day2_mean / mse_day1_mean) - 1) if mse_day1_mean > 0 else 0.0

    return {
        "mse_day1_mean": mse_day1_mean,
        "mse_day2_mean": mse_day2_mean,
        "degradation_pct": float(degradation_pct),
        "q95_day1": float(np.percentile(day1_errors, 95)),
        "q95_day2": float(np.percentile(day2_errors, 95)),
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
