"""Evaluation metrics."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader


def compute_reconstruction_errors_with_noise(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    noise_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reconstruction errors with optional noise injection.
    
    Used for synthetic corruption testing.
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            
            # Add noise if specified
            if noise_sigma > 0:
                x_corrupted = x + torch.randn_like(x) * noise_sigma
            else:
                x_corrupted = x
            
            x_recon = model(x_corrupted)
            error = torch.mean((x - x_recon) ** 2, dim=[1, 2])
            errors.append(error.cpu().numpy())
    
    return np.concatenate(errors)


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


def compute_per_channel_mse(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute MSE per channel (average across samples and time).

    Returns:
        Dict with keys: channel_0_mse, channel_1_mse, ..., mean_channel_mse, std_channel_mse
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_recon = model(x)
            # Per-channel error: (batch, channels, time)
            errors = torch.mean((x - x_recon) ** 2, dim=2)  # (batch, channels)
            all_errors.append(errors.cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)  # (n_samples, n_channels)
    channel_mses = np.mean(all_errors, axis=0)  # (n_channels,)

    result = {
        f"channel_{i}_mse": float(mse)
        for i, mse in enumerate(channel_mses)
    }
    result["mean_channel_mse"] = float(np.mean(channel_mses))
    result["std_channel_mse"] = float(np.std(channel_mses))

    return result


def compute_error_quantiles(
    errors: np.ndarray,
) -> Dict[str, float]:
    """
    Compute quantiles of reconstruction errors.

    Returns:
        Dict with keys: q05, q25, q50 (median), q75, q95
    """
    return {
        "error_q05": float(np.quantile(errors, 0.05)),
        "error_q25": float(np.quantile(errors, 0.25)),
        "error_median": float(np.quantile(errors, 0.50)),
        "error_q75": float(np.quantile(errors, 0.75)),
        "error_q95": float(np.quantile(errors, 0.95)),
        "error_min": float(np.min(errors)),
        "error_max": float(np.max(errors)),
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
    }


def compute_spectral_metrics(
    x: np.ndarray,
    x_recon: np.ndarray,
    fs: float = 250.0,
) -> Dict[str, float]:
    """
    Compute spectral domain metrics (requires 3D arrays).

    Args:
        x: Original signals (n_samples, n_channels, time)
        x_recon: Reconstructed signals (same shape)
        fs: Sampling frequency

    Returns:
        Dict with spectral metrics
    """
    # Compute FFT magnitude
    fft_orig = np.abs(np.fft.rfft(x, axis=-1))
    fft_recon = np.abs(np.fft.rfft(x_recon, axis=-1))

    # Spectral MSE (in frequency domain)
    spectral_mse = np.mean((fft_orig - fft_recon) ** 2)

    # Spectral correlation (mean across samples and channels)
    correlations = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            corr = np.corrcoef(fft_orig[i, j], fft_recon[i, j])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    mean_spectral_corr = np.mean(correlations) if correlations else 0.0

    return {
        "spectral_mse": float(spectral_mse),
        "mean_spectral_correlation": float(mean_spectral_corr),
    }


def compute_signal_to_noise_ratio(
    x: np.ndarray,
    error: np.ndarray,
) -> Dict[str, float]:
    """
    Compute SNR and related metrics.

    Args:
        x: Original signals (n_samples, n_channels, time)
        error: Reconstruction error (n_samples, n_channels, time)

    Returns:
        Dict with SNR metrics
    """
    # Signal power (mean squared)
    signal_power = np.mean(x ** 2)

    # Error/noise power
    noise_power = np.mean(error ** 2)

    # SNR in dB
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Per-channel SNR
    signal_power_per_channel = np.mean(x ** 2, axis=(0, 2))
    noise_power_per_channel = np.mean(error ** 2, axis=(0, 2))
    snr_per_channel = 10 * np.log10(
        signal_power_per_channel / (noise_power_per_channel + 1e-10)
    )

    return {
        "snr_db": float(snr_db),
        "snr_db_mean_per_channel": float(np.mean(snr_per_channel)),
        "snr_db_std_per_channel": float(np.std(snr_per_channel)),
        "snr_db_min_channel": float(np.min(snr_per_channel)),
        "snr_db_max_channel": float(np.max(snr_per_channel)),
    }


def compute_comprehensive_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute all available metrics on test set.

    Returns:
        Dict with all test metrics
    """
    model.eval()
    all_x = []
    all_x_recon = []
    all_errors = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_recon = model(x)
            all_x.append(x.cpu().numpy())
            all_x_recon.append(x_recon.cpu().numpy())
            errors = x - x_recon
            all_errors.append(errors.cpu().numpy())

    x_np = np.concatenate(all_x, axis=0)
    x_recon_np = np.concatenate(all_x_recon, axis=0)
    errors_np = np.concatenate(all_errors, axis=0)
    errors_flat = np.mean(np.mean(errors_np ** 2, axis=2), axis=1)

    # Compute all metrics
    metrics = {}

    # Basic test loss
    metrics.update(evaluate_model(model, test_loader, device))

    # Per-channel MSE
    metrics.update(compute_per_channel_mse(model, test_loader, device))

    # Error quantiles
    metrics.update(compute_error_quantiles(errors_flat))

    # SNR metrics
    metrics.update(compute_signal_to_noise_ratio(x_np, errors_np))

    # Spectral metrics
    metrics.update(compute_spectral_metrics(x_np, x_recon_np))

    return metrics


def compute_drift_metrics(
    model: nn.Module,
    session1_loader: DataLoader,
    session2_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute drift detection metrics using reconstruction error.
    
    Session 1: "normal" (training distribution)
    Session 2: "anomalous" (drift/new distribution)
    
    Args:
        model: Trained model
        session1_loader: Session 1 (normal) test loader
        session2_loader: Session 2 (drift) test loader
        device: Device to compute on
    
    Returns:
        Dict with drift metrics: auroc, auprc, normal_mse, anomaly_mse, degradation %
    """
    model.eval()
    
    # Get reconstruction errors for both sessions
    errors_s1, _ = compute_reconstruction_errors(model, session1_loader, device)
    errors_s2, _ = compute_reconstruction_errors(model, session2_loader, device)
    
    # Create labels: 0 = normal (S1), 1 = anomaly/drift (S2)
    labels = np.concatenate([np.zeros(len(errors_s1)), np.ones(len(errors_s2))])
    errors = np.concatenate([errors_s1, errors_s2])
    
    # Compute AUROC and AUPRC
    auroc = roc_auc_score(labels, errors)
    precision, recall, _ = precision_recall_curve(labels, errors)
    auprc = auc(recall, precision)
    
    # Compute degradation
    s1_mse_mean = float(np.mean(errors_s1))
    s2_mse_mean = float(np.mean(errors_s2))
    degradation_pct = 100.0 * (s2_mse_mean / s1_mse_mean - 1.0)
    
    return {
        "auroc_drift": float(auroc),
        "auprc_drift": float(auprc),
        "session1_mse_mean": s1_mse_mean,
        "session1_mse_q95": float(np.quantile(errors_s1, 0.95)),
        "session2_mse_mean": s2_mse_mean,
        "session2_mse_q95": float(np.quantile(errors_s2, 0.95)),
        "degradation_pct": degradation_pct,
        "s1_error_std": float(np.std(errors_s1)),
        "s2_error_std": float(np.std(errors_s2)),
    }


def compute_synthetic_corruption_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    corruption_sigma: float = 0.05,
) -> Dict[str, float]:
    """
    Test model robustness to synthetic corruption.
    
    This tests whether the model can handle external noise/corruption.
    For denoising autoencoders, they should be more robust than plain AEs.
    
    Args:
        model: Trained model
        test_loader: Test data loader (clean data)
        device: Device to compute on
        corruption_sigma: Standard deviation of noise to add (external corruption)
    
    Returns:
        Dict with keys:
        - clean_mse_mean: MSE on clean data
        - clean_mse_std: Std of MSE on clean data
        - corrupted_mse_mean: MSE on corrupted data
        - corrupted_mse_std: Std of MSE on corrupted data
        - degradation_pct: % increase in MSE from clean to corrupted
        - sensitivity: Ratio (corrupted_MSE / clean_MSE), lower is more robust
    """
    model.eval()
    
    # Test 1: Clean data
    errors_clean = compute_reconstruction_errors_with_noise(
        model, test_loader, device, noise_sigma=0.0
    )
    
    # Test 2: Data with external corruption
    errors_corrupted = compute_reconstruction_errors_with_noise(
        model, test_loader, device, noise_sigma=corruption_sigma
    )
    
    clean_mse_mean = float(np.mean(errors_clean))
    clean_mse_std = float(np.std(errors_clean))
    corrupted_mse_mean = float(np.mean(errors_corrupted))
    corrupted_mse_std = float(np.std(errors_corrupted))
    
    corruption_degradation_pct = 100.0 * (corrupted_mse_mean / clean_mse_mean - 1.0)
    sensitivity = corrupted_mse_mean / clean_mse_mean if clean_mse_mean > 1e-8 else np.inf
    
    return {
        "clean_mse_mean": clean_mse_mean,
        "clean_mse_std": clean_mse_std,
        "corrupted_mse_mean": corrupted_mse_mean,
        "corrupted_mse_std": corrupted_mse_std,
        "corruption_degradation_pct": corruption_degradation_pct,
        "corruption_sensitivity": float(sensitivity),
    }
