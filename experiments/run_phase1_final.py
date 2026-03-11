#!/usr/bin/env python
"""Phase 1.5: Prove denoising actually works.

Runs three experiments across multiple seeds and produces a comparison table:

  1. **Subtle drift** – realistic BCI session drift (AUROC should drop from 1.0).
  2. **Corruption robustness** – add Gaussian noise at test time; DAE should
     degrade less than plain AE.
  3. **Multi-seed statistics** – 5 seeds with mean ± std and a simple
     significance test.

Usage::

    python experiments/run_phase1_final.py
    # or
    python experiments/run_experiment.py --config experiments/configs/phase1_final.yaml
"""

import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from scipy import stats

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from white_noise_experimentation.config import load_config
from white_noise_experimentation.data.loaders import load_eeg_dataset_session_split
from white_noise_experimentation.evaluation.metrics import (
    compute_anomaly_metrics,
    compute_drift_metrics,
    compute_reconstruction_errors,
    corruption_robustness_test,
    evaluate_model,
)
from white_noise_experimentation.evaluation.plots import (
    plot_auroc_boxplots,
    plot_corruption_error_histograms,
    plot_corruption_robustness,
    plot_subtle_drift_curves,
)
from white_noise_experimentation.models.ann_autoencoder import ANNAutoencoder
from white_noise_experimentation.models.denoising_autoencoder import DenoisingAutoencoder
from white_noise_experimentation.training.trainer import train_autoencoder
from white_noise_experimentation.utils.logging import set_seed

# ---------------------------------------------------------------------------
CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"
SEEDS = [42, 123, 456, 777, 999]

# Model specs: (display name, config file)
MODELS = OrderedDict([
    ("ANN Baseline", "ann_baseline.yaml"),
    ("DAE σ=0.05", "denoising_ae_sigma_0_05.yaml"),
    ("DAE σ=0.1", "denoising_ae_sigma_0_1.yaml"),
])

CORRUPTION_LEVELS = [0.02, 0.05, 0.1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _create_model(config):
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


def _load_config_with_overrides(config_file, seed, drift_mode="subtle", drift_strength=0.1):
    """Load a per-model YAML and override seed + drift settings."""
    config = load_config(str(CONFIGS_DIR / config_file))
    # Override seed
    config.data.seed = seed
    config.train.seed = seed
    # Override drift
    config.data.drift_mode = drift_mode
    config.data.drift_strength = drift_strength
    # Force CPU + 0 workers for reproducibility
    config.train.device = "cpu"
    config.data.num_workers = 0
    return config


def train_and_evaluate(config, run_dir, device):
    """Train a model and return all metrics + the trained model."""
    set_seed(config.train.seed)

    loaders = load_eeg_dataset_session_split(config)
    model = _create_model(config)
    config_dict = config.to_dict()

    train_results = train_autoencoder(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config_dict,
        device=device,
        run_dir=run_dir,
    )
    model = train_results["model"]

    # Day-1 test
    test_day1_metrics = evaluate_model(model, loaders["test_day1"], device)

    # Drift metrics
    drift_metrics = compute_drift_metrics(
        model, loaders["test_day1"], loaders["test_day2"], device
    )

    # Anomaly metrics
    day1_errors, _ = compute_reconstruction_errors(model, loaders["test_day1"], device)
    day2_errors, _ = compute_reconstruction_errors(model, loaders["test_day2"], device)
    anomaly_metrics = compute_anomaly_metrics(day1_errors, day2_errors)

    # Corruption robustness (Experiment 2)
    corruption_metrics = corruption_robustness_test(
        model, loaders["test_day1"], device, corruption_levels=CORRUPTION_LEVELS
    )

    return {
        "best_val_loss": train_results["best_val_loss"],
        "test_day1_loss": test_day1_metrics["test_loss"],
        "mse_day1_mean": drift_metrics["mse_day1_mean"],
        "mse_day2_mean": drift_metrics["mse_day2_mean"],
        "degradation_pct": drift_metrics["degradation_pct"],
        "q95_day1": drift_metrics["q95_day1"],
        "q95_day2": drift_metrics["q95_day2"],
        "auroc": anomaly_metrics["auroc"],
        "auprc": anomaly_metrics["auprc"],
        **corruption_metrics,
    }, model


# ---------------------------------------------------------------------------
# Summary table + statistical test
# ---------------------------------------------------------------------------
def _fmt(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def _mean_std(vals, d=4):
    m, s = np.mean(vals), np.std(vals)
    return f"{m:.{d}f} ± {s:.{d}f}"


def generate_phase15_markdown(all_seed_results, run_root):
    """Generate the full Phase 1.5 results markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# Phase 1.5 – Proving Denoising Works\n")
    lines.append(f"_Generated on {timestamp} | {len(SEEDS)} seeds per model_\n")
    lines.append(
        "Three experiments stress-test whether denoising autoencoders (DAE) "
        "provide measurable benefits over a plain ANN autoencoder on synthetic "
        "EEG data with **subtle, realistic BCI drift**.\n"
    )

    model_names = list(all_seed_results.keys())

    # ---- Experiment 1: Subtle Drift ----
    lines.append("## Experiment 1: Subtle Drift (Realistic BCI Session Shift)\n")
    lines.append(
        "| Metric | " + " | ".join(model_names) + " |"
    )
    lines.append(
        "| --- | " + " | ".join(["---"] * len(model_names)) + " |"
    )

    for key, label in [
        ("best_val_loss", "Best val loss"),
        ("test_day1_loss", "Test Day 1 MSE"),
        ("mse_day1_mean", "MSE Day 1 (mean)"),
        ("mse_day2_mean", "MSE Day 2 (mean)"),
        ("degradation_pct", "Degradation %"),
        ("q95_day1", "Day 1 95th %ile"),
        ("q95_day2", "Day 2 95th %ile"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
    ]:
        vals = []
        for m in model_names:
            seed_vals = [r[key] for r in all_seed_results[m]]
            vals.append(_mean_std(seed_vals, 4))
        lines.append(f"| **{label}** | " + " | ".join(vals) + " |")

    lines.append("")

    # ---- Experiment 2: Corruption Robustness ----
    lines.append("## Experiment 2: Corruption Robustness Test\n")
    lines.append(
        "| Corruption σ | " + " | ".join(
            [f"{m} degradation %" for m in model_names]
        ) + " |"
    )
    lines.append(
        "| --- | " + " | ".join(["---"] * len(model_names)) + " |"
    )

    for sigma in CORRUPTION_LEVELS:
        key = f"degradation_pct_{sigma}"
        vals = []
        for m in model_names:
            seed_vals = [r[key] for r in all_seed_results[m]]
            vals.append(_mean_std(seed_vals, 2))
        lines.append(f"| **σ = {sigma}** | " + " | ".join(vals) + " |")

    lines.append("")

    # AUROC for corruption
    lines.append("### Corruption Detection AUROC (clean vs corrupted)\n")
    lines.append(
        "| Corruption σ | " + " | ".join(model_names) + " |"
    )
    lines.append(
        "| --- | " + " | ".join(["---"] * len(model_names)) + " |"
    )
    for sigma in CORRUPTION_LEVELS:
        key = f"auroc_clean_vs_{sigma}"
        vals = []
        for m in model_names:
            seed_vals = [r[key] for r in all_seed_results[m]]
            vals.append(_mean_std(seed_vals, 4))
        lines.append(f"| **σ = {sigma}** | " + " | ".join(vals) + " |")

    lines.append("")

    # ---- Experiment 3: Statistical Test ----
    lines.append("## Experiment 3: Multi-Seed Statistical Comparison\n")

    baseline_name = model_names[0]
    baseline_degs = [r["degradation_pct"] for r in all_seed_results[baseline_name]]
    baseline_aurocs = [r["auroc"] for r in all_seed_results[baseline_name]]

    for m in model_names[1:]:
        m_degs = [r["degradation_pct"] for r in all_seed_results[m]]
        m_aurocs = [r["auroc"] for r in all_seed_results[m]]

        # One-sided t-test: DAE degradation < baseline degradation
        t_deg, p_deg = stats.ttest_ind(m_degs, baseline_degs, alternative="less")
        # One-sided t-test: DAE AUROC > baseline AUROC (higher is better)
        t_aur, p_aur = stats.ttest_ind(m_aurocs, baseline_aurocs, alternative="greater")

        lines.append(f"### {m} vs {baseline_name}\n")
        lines.append(f"- Degradation %: {_mean_std(m_degs, 2)} vs {_mean_std(baseline_degs, 2)} "
                      f"(t={t_deg:.3f}, p={p_deg:.4f})")
        lines.append(f"- AUROC: {_mean_std(m_aurocs, 4)} vs {_mean_std(baseline_aurocs, 4)} "
                      f"(t={t_aur:.3f}, p={p_aur:.4f})")
        sig_deg = "✅ significant" if p_deg < 0.1 else "❌ not significant"
        sig_aur = "✅ significant" if p_aur < 0.1 else "❌ not significant"
        lines.append(f"- Degradation improvement: {sig_deg} (p < 0.1)")
        lines.append(f"- AUROC improvement: {sig_aur} (p < 0.1)")
        lines.append("")

    # ---- Corruption improvement test ----
    lines.append("### Corruption Robustness Improvement (σ = 0.1)\n")
    corr_key = f"degradation_pct_{CORRUPTION_LEVELS[-1]}"
    baseline_corr = [r[corr_key] for r in all_seed_results[baseline_name]]
    for m in model_names[1:]:
        m_corr = [r[corr_key] for r in all_seed_results[m]]
        t_c, p_c = stats.ttest_ind(m_corr, baseline_corr, alternative="less")
        lines.append(f"- **{m}**: {_mean_std(m_corr, 2)} vs {_mean_std(baseline_corr, 2)} "
                      f"(t={t_c:.3f}, p={p_c:.4f}) "
                      f"{'✅' if p_c < 0.1 else '❌'}")

    lines.append("")

    # ---- Success criteria ----
    lines.append("## Success Criteria Checklist\n")

    # Check AUROC dropped
    all_aurocs = []
    for m in model_names:
        for r in all_seed_results[m]:
            all_aurocs.append(r["auroc"])
    mean_auroc = np.mean(all_aurocs)
    auroc_dropped = mean_auroc < 0.95
    lines.append(
        f"- [{'x' if auroc_dropped else ' '}] Subtle drift AUROC drops below 0.95 "
        f"(actual mean: {mean_auroc:.4f})"
    )

    # Check any DAE beats baseline on degradation or AUROC
    any_dae_wins_deg = False
    any_dae_wins_corr = False
    for m in model_names[1:]:
        m_degs = np.mean([r["degradation_pct"] for r in all_seed_results[m]])
        b_degs = np.mean(baseline_degs)
        if m_degs < b_degs:
            any_dae_wins_deg = True
        m_corr_mean = np.mean([r[corr_key] for r in all_seed_results[m]])
        b_corr_mean = np.mean(baseline_corr)
        if m_corr_mean < b_corr_mean:
            any_dae_wins_corr = True

    lines.append(
        f"- [{'x' if any_dae_wins_deg else ' '}] At least one DAE beats plain AE on degradation %"
    )
    lines.append(
        f"- [{'x' if any_dae_wins_corr else ' '}] DAE shows less degradation under corruption"
    )
    lines.append("- [x] 5-seed statistics with mean ± std")
    lines.append(f"- [x] Plots saved to `{run_root.relative_to(REPO_ROOT)}/`")
    lines.append("- [x] Summary table with statistical tests")

    lines.append("")

    # ---- Interpretation ----
    lines.append("## Interpretation\n")

    # Find best model for each metric
    best_deg_model = min(
        model_names,
        key=lambda m: np.mean([r["degradation_pct"] for r in all_seed_results[m]]),
    )
    best_corr_model = min(
        model_names,
        key=lambda m: np.mean([r[corr_key] for r in all_seed_results[m]]),
    )
    best_auroc_model = max(
        model_names,
        key=lambda m: np.mean([r["auroc"] for r in all_seed_results[m]]),
    )

    lines.append(f"- **Most robust to drift**: {best_deg_model}")
    lines.append(f"- **Best corruption robustness**: {best_corr_model}")
    lines.append(f"- **Highest drift AUROC**: {best_auroc_model}")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cpu")  # Synthetic data — CPU is fine
    print(f"Using device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Models: {list(MODELS.keys())}")

    run_root = REPO_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_phase15")
    run_root.mkdir(parents=True, exist_ok=True)

    # Collect all results: {model_name: [result_per_seed]}
    all_seed_results = OrderedDict()

    for model_name, config_file in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")
        seed_results = []
        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            config = _load_config_with_overrides(config_file, seed)
            run_dir = run_root / model_name.replace(" ", "_").replace("σ=", "s") / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            result, model = train_and_evaluate(config, run_dir, device)
            seed_results.append(result)

            # Save per-run JSON
            with open(run_dir / "results.json", "w") as f:
                json.dump(result, f, indent=2)

            print(
                f"    degradation={result['degradation_pct']:.2f}%, "
                f"auroc={result['auroc']:.4f}, "
                f"corr_deg_0.1={result.get('degradation_pct_0.1', 0):.2f}%"
            )

        all_seed_results[model_name] = seed_results
        print(f"\n  ✓ {model_name} complete ({len(SEEDS)} seeds)")

    # --- Generate plots ---
    print("\nGenerating Phase 1.5 plots...")
    plot_dir = run_root / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Plot 1: Subtle drift curves
    drift_data = {}
    for m in all_seed_results:
        drift_data[m] = {
            "day1": [r["mse_day1_mean"] for r in all_seed_results[m]],
            "day2": [r["mse_day2_mean"] for r in all_seed_results[m]],
        }
    plot_subtle_drift_curves(drift_data, save_path=plot_dir / "subtle_drift_curves.png")

    # Plot 2: Corruption robustness
    corr_data = {}
    for m in all_seed_results:
        corr_data[m] = {}
        for sigma in CORRUPTION_LEVELS:
            key = f"degradation_pct_{sigma}"
            corr_data[m][str(sigma)] = np.mean([r[key] for r in all_seed_results[m]])
    plot_corruption_robustness(corr_data, save_path=plot_dir / "corruption_robustness.png")

    # Plot 3: Error histograms (using first seed for visualization)
    # We need to rebuild models for this — use cached results instead
    # Just show the corruption degradation summary
    # (Full histograms require re-running inference; skip for clean output)

    # Plot 4: AUROC boxplots
    auroc_data = {m: [r["auroc"] for r in all_seed_results[m]] for m in all_seed_results}
    plot_auroc_boxplots(auroc_data, save_path=plot_dir / "auroc_boxplots.png")

    # Also make boxplots for corruption AUROC
    for sigma in CORRUPTION_LEVELS:
        key = f"auroc_clean_vs_{sigma}"
        corr_auroc = {m: [r[key] for r in all_seed_results[m]] for m in all_seed_results}
        plot_auroc_boxplots(
            corr_auroc,
            title=f"Corruption Detection AUROC (σ={sigma})",
            save_path=plot_dir / f"corruption_auroc_sigma_{sigma}.png",
        )

    # --- Save combined JSON ---
    with open(run_root / "all_results.json", "w") as f:
        json.dump(
            {m: results for m, results in all_seed_results.items()},
            f, indent=2,
        )

    # --- Generate Markdown ---
    md = generate_phase15_markdown(all_seed_results, run_root)
    md_path = REPO_ROOT / "EXPERIMENT_RESULTS.md"
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n{'='*60}")
    print(f"  PHASE 1.5 COMPLETE")
    print(f"  Results: {md_path}")
    print(f"  Plots: {plot_dir}")
    print(f"{'='*60}")
    print(md)


if __name__ == "__main__":
    main()
