"""Evaluation metrics."""

from typing import Dict, List, Tuple

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


def corruption_robustness_test(
    model: nn.Module,
    clean_test_loader: DataLoader,
    device: torch.device,
    corruption_levels: List[float] = None,
) -> Dict[str, float]:
    """Test model robustness to synthetic input corruption.

    Adds Gaussian noise at several levels to the clean test data and measures
    how much the reconstruction error degrades compared to clean inputs.

    Args:
        model: Trained autoencoder (set to eval internally).
        clean_test_loader: DataLoader with clean in-distribution test data.
        device: Torch device.
        corruption_levels: List of noise standard deviations to evaluate.

    Returns:
        Dict with keys:
            - ``clean_mse``: mean MSE on the original clean data.
            - ``corrupted_mse_<sigma>``: mean MSE at each corruption level.
            - ``degradation_pct_<sigma>``: percentage increase over clean MSE.
            - ``auroc_clean_vs_<sigma>``: AUROC treating clean as label 0 and
              corrupted as label 1 (based on reconstruction error as score).
    """
    if corruption_levels is None:
        corruption_levels = [0.02, 0.05, 0.1]

    model.eval()

    # --- Collect clean windows ---
    all_clean = []
    for x, _ in clean_test_loader:
        all_clean.append(x)
    all_clean = torch.cat(all_clean, dim=0)

    # --- Clean errors ---
    clean_errors = _compute_errors_from_tensor(model, all_clean, device)
    clean_mse = float(np.mean(clean_errors))

    results: Dict[str, float] = {"clean_mse": clean_mse}

    for sigma in corruption_levels:
        torch.manual_seed(0)  # deterministic noise per level
        noise = sigma * torch.randn_like(all_clean)
        corrupted = all_clean + noise

        corr_errors = _compute_errors_from_tensor(model, corrupted, device)
        corr_mse = float(np.mean(corr_errors))
        degradation = 100.0 * (corr_mse / clean_mse - 1) if clean_mse > 0 else 0.0

        # AUROC: can reconstruction error separate clean from corrupted?
        combined_errors = np.concatenate([clean_errors, corr_errors])
        labels = np.concatenate([
            np.zeros(len(clean_errors), dtype=int),
            np.ones(len(corr_errors), dtype=int),
        ])
        try:
            auroc = float(roc_auc_score(labels, combined_errors))
        except ValueError:
            auroc = 0.5

        key = f"{sigma}"
        results[f"corrupted_mse_{key}"] = corr_mse
        results[f"degradation_pct_{key}"] = degradation
        results[f"auroc_clean_vs_{key}"] = auroc

    return results


def _compute_errors_from_tensor(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute per-sample MSE from a raw tensor (no DataLoader needed)."""
    model.eval()
    errors = []
    n = data.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = data[i : i + batch_size].to(device)
            recon = model(batch)
            err = torch.mean((batch - recon) ** 2, dim=[1, 2])
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)
