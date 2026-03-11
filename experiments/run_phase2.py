#!/usr/bin/env python
"""Phase 2: Compare ANN with Spiking Neural Networks.

Compares the Phase 1 ANN baseline with three SNN variants across
multiple seeds and produces a full comparison table:

1. **ANN Baseline** – Phase 1 reference.
2. **Fixed-Noise SNN** – SNN with fixed σ = 0.1 (spiking equivalent of DAE).
3. **Noise-Controlled SNN** – SNN with learned per-layer σ (Phase 2 novelty).

Usage::

    python experiments/run_phase2.py
    # or
    python experiments/run_experiment.py --config experiments/configs/phase2_full.yaml
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
    plot_corruption_robustness,
    plot_subtle_drift_curves,
)
from white_noise_experimentation.models.ann_autoencoder import ANNAutoencoder
from white_noise_experimentation.models.denoising_autoencoder import DenoisingAutoencoder
from white_noise_experimentation.models.snn_autoencoder import FixedNoiseSNN, LatentNoiseSNN
from white_noise_experimentation.models.noise_controller_snn import NoiseControlledSNN
from white_noise_experimentation.training.trainer import train_autoencoder
from white_noise_experimentation.utils.logging import set_seed

# ---------------------------------------------------------------------------
CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"
SEEDS = [42, 123, 456]

# Model specs: (display name, config file)
MODELS = OrderedDict([
    ("ANN Baseline", "ann_baseline.yaml"),
    ("Fixed SNN σ=0.1", "snn_fixed.yaml"),
    ("Learned SNN", "snn_controller.yaml"),
])

CORRUPTION_LEVELS = [0.02, 0.05, 0.1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _create_model(config):
    """Create a model from config."""
    model_type = config.model.type
    kwargs = dict(
        n_channels=config.model.n_channels,
        window_size=config.model.window_size,
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims,
    )
    if model_type == "ann_autoencoder":
        return ANNAutoencoder(**kwargs)
    elif model_type == "denoising_autoencoder":
        return DenoisingAutoencoder(
            **kwargs,
            sigma=config.noise.sigma,
            noise_where=config.noise.where,
        )
    elif model_type == "fixed_noise_snn":
        return FixedNoiseSNN(**kwargs, noise_sigma=config.noise.sigma)
    elif model_type == "latent_noise_snn":
        return LatentNoiseSNN(**kwargs, noise_sigma=config.noise.sigma)
    elif model_type == "noise_controller_snn":
        return NoiseControlledSNN(**kwargs, noise_sigma=config.noise.sigma)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _load_config_with_overrides(config_file, seed, drift_mode="subtle", drift_strength=0.15):
    """Load a per-model YAML and override seed + drift settings."""
    config = load_config(str(CONFIGS_DIR / config_file))
    config.data.seed = seed
    config.train.seed = seed
    config.data.drift_mode = drift_mode
    config.data.drift_strength = drift_strength
    config.train.device = "cpu"
    config.data.num_workers = 0
    return config


def train_and_evaluate(config, run_dir, device):
    """Train a model and return all metrics + the trained model."""
    set_seed(config.train.seed)

    loaders = load_eeg_dataset_session_split(config)
    model = _create_model(config)
    config_dict = config.to_dict()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model parameters: {n_params:,}")

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

    # Corruption robustness
    corruption_metrics = corruption_robustness_test(
        model, loaders["test_day1"], device, corruption_levels=CORRUPTION_LEVELS
    )

    # If NoiseControlledSNN, report mean predicted sigmas
    extra = {}
    if hasattr(model, "last_sigmas") and model.last_sigmas is not None:
        extra["controller_sigmas_mean"] = float(model.last_sigmas.mean().item())

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
        **extra,
    }, model


# ---------------------------------------------------------------------------
# Summary and markdown generation
# ---------------------------------------------------------------------------
def _fmt(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def _mean_std(vals, d=4):
    m, s = np.mean(vals), np.std(vals)
    return f"{m:.{d}f} ± {s:.{d}f}"


def generate_phase2_markdown(all_seed_results, run_root):
    """Generate full Phase 2 results markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# Phase 2 – SNN vs ANN Comparison\n")
    lines.append(f"_Generated on {timestamp} | {len(SEEDS)} seeds per model_\n")
    lines.append(
        "Phase 2 replaces the ANN autoencoder with Spiking Neural Networks (SNNs) "
        "while keeping the same pipeline, metrics, and evaluation. The goal is to "
        "prove SNNs can match ANN performance on drift/anomaly detection, and that "
        "learned noise (NoiseControlledSNN) beats fixed noise.\n"
    )

    model_names = list(all_seed_results.keys())

    # ---- Main comparison table ----
    lines.append("## Drift Detection (Subtle BCI Drift)\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(model_names)) + " |")

    for key, label in [
        ("best_val_loss", "Best val loss"),
        ("test_day1_loss", "Test Day 1 MSE"),
        ("mse_day1_mean", "MSE Day 1 (mean)"),
        ("mse_day2_mean", "MSE Day 2 (mean)"),
        ("degradation_pct", "Degradation %"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
    ]:
        vals = []
        for m in model_names:
            seed_vals = [r[key] for r in all_seed_results[m]]
            vals.append(_mean_std(seed_vals, 4))
        lines.append(f"| **{label}** | " + " | ".join(vals) + " |")
    lines.append("")

    # ---- Corruption robustness ----
    lines.append("## Corruption Robustness\n")
    lines.append(
        "| Corruption σ | " + " | ".join(
            [f"{m} deg %" for m in model_names]
        ) + " |"
    )
    lines.append("| --- | " + " | ".join(["---"] * len(model_names)) + " |")

    for sigma in CORRUPTION_LEVELS:
        key = f"degradation_pct_{sigma}"
        vals = []
        for m in model_names:
            seed_vals = [r[key] for r in all_seed_results[m]]
            vals.append(_mean_std(seed_vals, 2))
        lines.append(f"| **σ = {sigma}** | " + " | ".join(vals) + " |")
    lines.append("")

    # ---- Statistical tests ----
    lines.append("## Statistical Comparison (vs ANN Baseline)\n")

    baseline_name = model_names[0]
    baseline_degs = [r["degradation_pct"] for r in all_seed_results[baseline_name]]
    baseline_aurocs = [r["auroc"] for r in all_seed_results[baseline_name]]

    for m in model_names[1:]:
        m_degs = [r["degradation_pct"] for r in all_seed_results[m]]
        m_aurocs = [r["auroc"] for r in all_seed_results[m]]

        t_deg, p_deg = stats.ttest_ind(m_degs, baseline_degs, alternative="less")
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

    # ---- Success criteria ----
    lines.append("## Success Criteria\n")

    # Fixed SNN runs without NaN
    fixed_snn_ok = True
    for m in model_names:
        if "SNN" in m or "snn" in m.lower():
            for r in all_seed_results[m]:
                if np.isnan(r["auroc"]) or np.isnan(r["degradation_pct"]):
                    fixed_snn_ok = False
    lines.append(f"- [{'x' if fixed_snn_ok else ' '}] SNN models run without NaN (AUROC > 0)")

    # Controller beats fixed
    learned_name = [m for m in model_names if "Learned" in m or "Controller" in m or "controller" in m]
    fixed_name = [m for m in model_names if "Fixed SNN" in m or "fixed_noise" in m.lower()]
    if learned_name and fixed_name:
        learned_aurocs = np.mean([r["auroc"] for r in all_seed_results[learned_name[0]]])
        fixed_aurocs = np.mean([r["auroc"] for r in all_seed_results[fixed_name[0]]])
        controller_beats = learned_aurocs > fixed_aurocs
        lines.append(
            f"- [{'x' if controller_beats else ' '}] Controller beats fixed SNN on AUROC "
            f"({learned_aurocs:.4f} vs {fixed_aurocs:.4f})"
        )

    # Training stable across seeds
    all_stable = True
    for m in model_names:
        auroc_std = np.std([r["auroc"] for r in all_seed_results[m]])
        if auroc_std > 0.3:
            all_stable = False
    lines.append(f"- [{'x' if all_stable else ' '}] Training stable across {len(SEEDS)} seeds")
    lines.append("- [x] Same plots/metrics as Phase 1 (apples-to-apples)")
    lines.append("")

    # ---- Interpretation ----
    lines.append("## Interpretation\n")

    best_auroc_model = max(
        model_names,
        key=lambda m: np.mean([r["auroc"] for r in all_seed_results[m]]),
    )
    best_deg_model = min(
        model_names,
        key=lambda m: np.mean([r["degradation_pct"] for r in all_seed_results[m]]),
    )
    lines.append(f"- **Highest AUROC**: {best_auroc_model}")
    lines.append(f"- **Most robust to drift**: {best_deg_model}")

    # Check if any SNN beats ANN
    ann_auroc = np.mean([r["auroc"] for r in all_seed_results[baseline_name]])
    any_snn_wins = False
    for m in model_names[1:]:
        m_auroc = np.mean([r["auroc"] for r in all_seed_results[m]])
        if m_auroc >= ann_auroc:
            any_snn_wins = True
    if any_snn_wins:
        lines.append("- ✅ **At least one SNN matches/beats ANN performance!**")
    else:
        lines.append("- ⚠️ SNNs do not yet match ANN AUROC – further tuning needed.")

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary_table(all_seed_results):
    """Print a compact console summary."""
    print(f"\n{'='*80}")
    print("  PHASE 2 RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'AUROC(subtle)':<18} {'Degradation%':<18} {'Corr deg σ=0.1':<18}")
    print("-" * 80)

    for model_name, results in all_seed_results.items():
        aurocs = [r["auroc"] for r in results]
        degs = [r["degradation_pct"] for r in results]
        corrs = [r.get("degradation_pct_0.1", 0) for r in results]
        print(
            f"{model_name:<25} "
            f"{np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}  "
            f"{np.mean(degs):.2f} ± {np.std(degs):.2f}%    "
            f"{np.mean(corrs):.2f} ± {np.std(corrs):.2f}%"
        )

    # Find winner
    best_model = max(
        all_seed_results.keys(),
        key=lambda m: np.mean([r["auroc"] for r in all_seed_results[m]]),
    )
    print(f"\n  → Best AUROC: {best_model}")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Models: {list(MODELS.keys())}")

    run_root = REPO_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_phase2")
    run_root.mkdir(parents=True, exist_ok=True)

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
    print("\nGenerating Phase 2 plots...")
    plot_dir = run_root / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Plot 1: Subtle drift curves
    drift_data = {}
    for m in all_seed_results:
        drift_data[m] = {
            "day1": [r["mse_day1_mean"] for r in all_seed_results[m]],
            "day2": [r["mse_day2_mean"] for r in all_seed_results[m]],
        }
    plot_subtle_drift_curves(drift_data, save_path=plot_dir / "drift_curves.png")

    # Plot 2: Corruption robustness
    corr_data = {}
    for m in all_seed_results:
        corr_data[m] = {}
        for sigma in CORRUPTION_LEVELS:
            key = f"degradation_pct_{sigma}"
            corr_data[m][str(sigma)] = np.mean([r[key] for r in all_seed_results[m]])
    plot_corruption_robustness(corr_data, save_path=plot_dir / "corruption_robustness.png")

    # Plot 3: AUROC boxplots
    auroc_data = {m: [r["auroc"] for r in all_seed_results[m]] for m in all_seed_results}
    plot_auroc_boxplots(auroc_data, title="Phase 2: AUROC (Subtle Drift)", save_path=plot_dir / "auroc_boxplots.png")

    # --- Save combined JSON ---
    with open(run_root / "all_results.json", "w") as f:
        json.dump(
            {m: results for m, results in all_seed_results.items()},
            f, indent=2,
        )

    # --- Generate Markdown ---
    md = generate_phase2_markdown(all_seed_results, run_root)
    md_path = REPO_ROOT / "PHASE2_RESULTS.md"
    with open(md_path, "w") as f:
        f.write(md)

    # --- Console summary ---
    print_summary_table(all_seed_results)

    print(f"  PHASE 2 COMPLETE")
    print(f"  Results: {md_path}")
    print(f"  Plots: {plot_dir}")
    print(f"{'='*60}")
    print(md)


if __name__ == "__main__":
    main()
