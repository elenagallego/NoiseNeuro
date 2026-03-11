#!/usr/bin/env python
"""Run all Phase 1 experiments and produce a Markdown comparison table.

Executes four experiment configurations:
  1. ANN Autoencoder baseline (no noise)
  2. Denoising AE with σ = 0.05
  3. Denoising AE with σ = 0.1
  4. Drift test (ANN baseline evaluated on Day 1 vs Day 2)

All results are written to ``EXPERIMENT_RESULTS.md`` at the repository root
as an easy-to-compare table.
"""

import json
import sys
import textwrap
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running from the experiments/ directory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

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
from white_noise_experimentation.utils.logging import set_seed

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"

EXPERIMENTS = OrderedDict(
    [
        ("ANN Baseline", "ann_baseline.yaml"),
        ("DAE σ=0.05", "denoising_ae_sigma_0_05.yaml"),
        ("DAE σ=0.1", "denoising_ae_sigma_0_1.yaml"),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _create_model(config):
    """Instantiate model from config."""
    if config.model.type == "ann_autoencoder":
        return ANNAutoencoder(
            n_channels=config.model.n_channels,
            window_size=config.model.window_size,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims,
        )
    elif config.model.type == "denoising_autoencoder":
        return DenoisingAutoencoder(
            n_channels=config.model.n_channels,
            window_size=config.model.window_size,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims,
            sigma=config.noise.sigma,
            noise_where=config.noise.where,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")


def run_single_experiment(name: str, config_file: str, run_root: Path, device):
    """Train one model and return its metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Running experiment: {name}")
    print(f"  Config: {config_file}")
    print(f"{'='*60}\n")

    config = load_config(str(CONFIGS_DIR / config_file))
    config_dict = config.to_dict()

    # Reproducibility
    set_seed(config.train.seed)

    # Data
    loaders = load_eeg_dataset_session_split(config)

    # Model
    model = _create_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Train
    run_dir = run_root / name.replace(" ", "_").replace("σ=", "sigma_")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_results = train_autoencoder(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config_dict,
        device=device,
        run_dir=run_dir,
    )
    model = train_results["model"]

    # --- Metrics ---
    best_val_loss = train_results["best_val_loss"]

    # Day-1 test
    test_day1_metrics = evaluate_model(model, loaders["test_day1"], device)

    # Drift
    drift_metrics = compute_drift_metrics(
        model, loaders["test_day1"], loaders["test_day2"], device
    )

    # Anomaly
    day1_errors, _ = compute_reconstruction_errors(model, loaders["test_day1"], device)
    day2_errors, _ = compute_reconstruction_errors(model, loaders["test_day2"], device)
    anomaly_metrics = compute_anomaly_metrics(day1_errors, day2_errors)

    # --- Plots ---
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_learning_curves(
        train_results["train_loss_history"],
        train_results["val_loss_history"],
        plot_dir / "learning_curves.png",
    )
    plot_day1_vs_day2_histogram(
        day1_errors,
        day2_errors,
        title=f"Reconstruction Error – {name}",
        save_path=plot_dir / "day1_vs_day2_errors.png",
    )
    plot_drift_bar(
        drift_metrics,
        model_name=name,
        save_path=plot_dir / "drift_bar.png",
    )
    all_errors = np.concatenate([day1_errors, day2_errors])
    all_labels = np.concatenate(
        [np.zeros(len(day1_errors), dtype=int), np.ones(len(day2_errors), dtype=int)]
    )
    plot_reconstruction_error_histogram(
        all_errors, all_labels, plot_dir / "reconstruction_errors.png"
    )

    # Combine
    result = {
        "name": name,
        "config_file": config_file,
        "n_params": n_params,
        "epochs_trained": len(train_results["train_loss_history"]),
        "best_val_loss": best_val_loss,
        "test_day1_loss": test_day1_metrics["test_loss"],
        **drift_metrics,
        **anomaly_metrics,
    }

    # Persist per-experiment JSON
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Markdown table generation
# ---------------------------------------------------------------------------
def _fmt(value, decimals=4):
    """Format a float for the table."""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def generate_markdown_table(all_results: list) -> str:
    """Return a Markdown string with the comparison table."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Phase 1 – Experiment Results\n")
    lines.append(f"_Generated on {timestamp}_\n")
    lines.append(
        "This table compares three autoencoder configurations trained on "
        "synthetic EEG data (Session 1 / Day 1) and evaluated for "
        "reconstruction quality and drift robustness on both Day 1 "
        "(in-distribution) and Day 2 (distribution-shifted) test sets.\n"
    )

    # --- Main comparison table ---
    lines.append("## Metrics Comparison\n")

    # Column definitions
    cols = [
        ("Metric", 34),
        *[(r["name"], 18) for r in all_results],
    ]

    header = "| " + " | ".join(c[0].ljust(c[1]) for c in cols) + " |"
    sep = "| " + " | ".join("-" * c[1] for c in cols) + " |"

    rows_data = [
        ("**Model type**",          [r["config_file"].replace(".yaml", "") for r in all_results]),
        ("**Parameters**",          [f'{r["n_params"]:,}' for r in all_results]),
        ("**Epochs trained**",      [str(r["epochs_trained"]) for r in all_results]),
        ("**Best val loss (MSE)**", [_fmt(r["best_val_loss"]) for r in all_results]),
        ("**Test Day 1 loss (MSE)**", [_fmt(r["test_day1_loss"]) for r in all_results]),
        ("**MSE Day 1 (mean)**",    [_fmt(r["mse_day1_mean"]) for r in all_results]),
        ("**MSE Day 2 (mean)**",    [_fmt(r["mse_day2_mean"]) for r in all_results]),
        ("**Degradation %**",       [_fmt(r["degradation_pct"], 2) + "%" for r in all_results]),
        ("**MSE Day 1 (95th %ile)**", [_fmt(r["q95_day1"]) for r in all_results]),
        ("**MSE Day 2 (95th %ile)**", [_fmt(r["q95_day2"]) for r in all_results]),
        ("**AUROC**",               [_fmt(r["auroc"]) for r in all_results]),
        ("**AUPRC**",               [_fmt(r["auprc"]) for r in all_results]),
    ]

    lines.append(header)
    lines.append(sep)
    for label, values in rows_data:
        cells = [label.ljust(cols[0][1])] + [
            v.ljust(cols[i + 1][1]) for i, v in enumerate(values)
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # --- Key definitions ---
    lines.append("## Metric Definitions\n")
    lines.append(
        "| Metric | Description |\n"
        "| ------ | ----------- |\n"
        "| Best val loss | Lowest validation MSE during training (in-distribution) |\n"
        "| Test Day 1 loss | MSE on held-out Day 1 test set (in-distribution) |\n"
        "| MSE Day 1 / Day 2 (mean) | Average per-window reconstruction error |\n"
        "| Degradation % | 100 × (MSE_day2 / MSE_day1 − 1); higher = worse under drift |\n"
        "| 95th %ile | 95th percentile of per-window errors (tail behaviour) |\n"
        "| AUROC | Area under ROC curve treating Day 2 as anomalous |\n"
        "| AUPRC | Area under Precision-Recall curve (Day 2 = positive class) |\n"
    )

    # --- Interpretation ---
    lines.append("## Interpretation\n")

    # Find best model per metric
    best_val = min(all_results, key=lambda r: r["best_val_loss"])
    best_day1 = min(all_results, key=lambda r: r["test_day1_loss"])
    lowest_deg = min(all_results, key=lambda r: r["degradation_pct"])
    best_auroc = max(all_results, key=lambda r: r["auroc"])

    lines.append(
        f"- **Best in-distribution reconstruction**: {best_val['name']} "
        f"(val loss = {_fmt(best_val['best_val_loss'])})\n"
    )
    lines.append(
        f"- **Most robust to drift** (lowest degradation): {lowest_deg['name']} "
        f"(degradation = {_fmt(lowest_deg['degradation_pct'], 2)}%)\n"
    )
    lines.append(
        f"- **Best drift/anomaly detection** (highest AUROC): {best_auroc['name']} "
        f"(AUROC = {_fmt(best_auroc['auroc'])})\n"
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_root = REPO_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_batch")
    run_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for name, config_file in EXPERIMENTS.items():
        result = run_single_experiment(name, config_file, run_root, device)
        all_results.append(result)
        print(f"\n  ✓ {name} complete")

    # Save combined JSON
    with open(run_root / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate Markdown
    md = generate_markdown_table(all_results)
    md_path = REPO_ROOT / "EXPERIMENT_RESULTS.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\n{'='*60}")
    print(f"  Results table written to: {md_path}")
    print(f"  Run artifacts in: {run_root}")
    print(f"{'='*60}")
    print(md)


if __name__ == "__main__":
    main()
