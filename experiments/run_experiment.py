"""Main experiment runner."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from white_noise_experimentation.config import load_config
from white_noise_experimentation.data.loaders import load_eeg_dataset
from white_noise_experimentation.evaluation.metrics import (
    compute_anomaly_metrics,
    compute_reconstruction_errors,
    evaluate_model,
)
from white_noise_experimentation.evaluation.plots import (
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

    # Load data
    logger.info("Loading EEG dataset...")
    data_loaders = load_eeg_dataset(
        data_root=config.data.data_root,
        window_size=config.data.window_size,
        window_stride=config.data.window_stride,
        normalize=config.data.normalize,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        n_channels=config.model.n_channels,
    )

    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]

    logger.info(
        f"Data loaded: {len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples"
    )

    # Create model
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
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
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

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    logger.info(f"Test loss: {test_metrics['test_loss']:.6f}")

    # Compute reconstruction errors
    logger.info("Computing reconstruction errors...")
    errors, labels = compute_reconstruction_errors(model, test_loader, device)

    # Compute anomaly metrics (if labels available and binary)
    anomaly_metrics = {}
    if labels is not None and len(np.unique(labels)) == 2:
        anomaly_metrics = compute_anomaly_metrics(errors, labels)
        logger.info(f"Anomaly metrics - AUROC: {anomaly_metrics['auroc']:.3f}, AUPRC: {anomaly_metrics['auprc']:.3f}")

    # Save results
    results = {
        "config": config_dict,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_metrics": test_metrics,
        "anomaly_metrics": anomaly_metrics,
    }

    results_file = run_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Save metrics
    all_metrics = {**test_metrics, **anomaly_metrics}
    logger.save_metrics(all_metrics)

    # Plot learning curves
    logger.info("Generating plots...")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    plot_learning_curves(train_loss, val_loss, plot_dir / "learning_curves.png")
    logger.info(f"Learning curves saved to {plot_dir / 'learning_curves.png'}")

    plot_reconstruction_error_histogram(
        errors, labels, plot_dir / "reconstruction_errors.png"
    )
    logger.info(f"Error histogram saved to {plot_dir / 'reconstruction_errors.png'}")

    logger.info(f"Experiment complete! Results saved to {run_dir}")


if __name__ == "__main__":
    import numpy as np

    main()
