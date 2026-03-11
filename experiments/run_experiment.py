"""Main experiment runner."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from white_noise_experimentation.config import load_config
from white_noise_experimentation.data.loaders import load_eeg_dataset_session_split
from white_noise_experimentation.evaluation.metrics import (
    compute_anomaly_metrics,
    compute_drift_metrics,
    compute_reconstruction_errors,
    evaluate_model,
)
from white_noise_experimentation.evaluation.plots import (
    plot_day1_vs_day2_histogram,
    plot_drift_bar,
    plot_learning_curves,
    plot_reconstruction_error_histogram,
)
from white_noise_experimentation.models.ann_autoencoder import ANNAutoencoder
from white_noise_experimentation.models.denoising_autoencoder import DenoisingAutoencoder
from white_noise_experimentation.training.trainer import train_autoencoder
from white_noise_experimentation.utils.logging import TimestampedLogger, set_seed


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run white_noise_experimentation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config_dict = config.to_dict()

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = TimestampedLogger(run_dir)
    logger.save_config(config_dict)
    logger.info(f"Starting experiment: {args.config}")
    logger.info(f"Run directory: {run_dir}")

    # Set seed
    set_seed(config.train.seed)

    # Device
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load data with session-aware Day 1 / Day 2 split
    # ------------------------------------------------------------------
    logger.info("Loading EEG dataset with session-aware splits...")
    data_loaders = load_eeg_dataset_session_split(config)

    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_day1_loader = data_loaders["test_day1"]
    test_day2_loader = data_loaders["test_day2"]

    logger.info(
        f"Data loaded: {len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val, "
        f"{len(test_day1_loader.dataset)} test_day1, "
        f"{len(test_day2_loader.dataset)} test_day2 samples"
    )

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------
    logger.info(f"Creating model: {config.model.type}")
    if config.model.type == "ann_autoencoder":
        model = ANNAutoencoder(
            n_channels=config.model.n_channels,
            window_size=config.model.window_size,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims,
        )
    elif config.model.type == "denoising_autoencoder":
        model = DenoisingAutoencoder(
            n_channels=config.model.n_channels,
            window_size=config.model.window_size,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims,
            sigma=config.noise.sigma,
            noise_where=config.noise.where,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting training...")
    train_results = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_dict,
        device=device,
        run_dir=run_dir,
    )

    model = train_results["model"]
    train_loss = train_results["train_loss_history"]
    val_loss = train_results["val_loss_history"]

    logger.info(f"Training complete. Best val loss: {min(val_loss):.6f}")

    # ------------------------------------------------------------------
    # Evaluate on Day 1 (in-distribution) test set
    # ------------------------------------------------------------------
    logger.info("Evaluating on Day 1 test set (in-distribution)...")
    test_day1_metrics = evaluate_model(model, test_day1_loader, device)
    logger.info(f"Day 1 test loss: {test_day1_metrics['test_loss']:.6f}")

    # ------------------------------------------------------------------
    # Drift metrics: Day 1 vs Day 2
    # ------------------------------------------------------------------
    logger.info("Computing drift metrics (Day 1 vs Day 2)...")
    drift_metrics = compute_drift_metrics(model, test_day1_loader, test_day2_loader, device)
    logger.info(
        f"Drift: Day1 MSE={drift_metrics['mse_day1_mean']:.6f}, "
        f"Day2 MSE={drift_metrics['mse_day2_mean']:.6f}, "
        f"Degradation={drift_metrics['degradation_pct']:.1f}%"
    )

    # ------------------------------------------------------------------
    # Anomaly detection: Day 1 (normal=0) vs Day 2 (drift=1)
    # ------------------------------------------------------------------
    logger.info("Computing anomaly metrics (Day 1 normal vs Day 2 drift)...")
    day1_errors, _ = compute_reconstruction_errors(model, test_day1_loader, device)
    day2_errors, _ = compute_reconstruction_errors(model, test_day2_loader, device)
    anomaly_metrics = compute_anomaly_metrics(day1_errors, day2_errors)
    logger.info(
        f"Anomaly metrics – AUROC: {anomaly_metrics['auroc']:.3f}, "
        f"AUPRC: {anomaly_metrics['auprc']:.3f}"
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "config": config_dict,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_day1_metrics": test_day1_metrics,
        "drift_metrics": drift_metrics,
        "anomaly_metrics": anomaly_metrics,
    }

    results_file = run_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    all_metrics = {**test_day1_metrics, **drift_metrics, **anomaly_metrics}
    logger.save_metrics(all_metrics)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    logger.info("Generating plots...")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    plot_learning_curves(train_loss, val_loss, plot_dir / "learning_curves.png")
    logger.info(f"Learning curves saved to {plot_dir / 'learning_curves.png'}")

    plot_day1_vs_day2_histogram(
        day1_errors, day2_errors,
        title=f"Reconstruction Error – {config.model.type}",
        save_path=plot_dir / "day1_vs_day2_errors.png",
    )
    logger.info(f"Day1 vs Day2 histogram saved to {plot_dir / 'day1_vs_day2_errors.png'}")

    plot_drift_bar(
        drift_metrics,
        model_name=config.model.type,
        save_path=plot_dir / "drift_bar.png",
    )
    logger.info(f"Drift bar chart saved to {plot_dir / 'drift_bar.png'}")

    # Reconstruction error histogram (combined with labels)
    all_errors = np.concatenate([day1_errors, day2_errors])
    all_labels = np.concatenate([
        np.zeros(len(day1_errors), dtype=int),
        np.ones(len(day2_errors), dtype=int),
    ])
    plot_reconstruction_error_histogram(
        all_errors, all_labels, plot_dir / "reconstruction_errors.png"
    )
    logger.info(f"Error histogram saved to {plot_dir / 'reconstruction_errors.png'}")

    logger.info(f"Experiment complete! Results saved to {run_dir}")


if __name__ == "__main__":
    main()
