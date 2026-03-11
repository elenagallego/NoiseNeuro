#!/usr/bin/env python
"""Debug sweep for Phase 1 - test lower noise levels."""

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

def main():
    """Run comprehensive debug sweep."""
    # Change to project root
    os.chdir("/Users/elena/NoiseNeuro")
    
    configs = [
        ("drift_test.yaml", "ANN Baseline (σ=0.0)"),
        ("drift_test_denoising.yaml", "Denoising AE (σ=0.1, input)"),
        ("drift_denoise_001.yaml", "Denoising AE (σ=0.01, input)"),
        ("drift_denoise_002.yaml", "Denoising AE (σ=0.02, input)"),
        ("drift_denoise_005.yaml", "Denoising AE (σ=0.05, input)"),
    ]
    
    seeds = [42, 123, 777]
    results = []
    
    print("\n" + "="*100)
    print("DRIFT TEST SWEEP: Debug Phase - Lower σ Sweep + Latent Noise")
    print("="*100)
    print(f"Models: {len(configs)} | Seeds: {len(seeds)} | Total runs: {len(configs)*len(seeds)}")
    print("="*100 + "\n")
    
    for cfg_idx, (cfg_file, model_name) in enumerate(configs, 1):
        print(f"\n{'='*100}")
        print(f"[{cfg_idx}/{len(configs)}] {model_name}")
        print(f"{'='*100}")
        
        model_runs = []
        for seed_idx, seed in enumerate(seeds, 1):
            print(f"  Seed {seed_idx}/{len(seeds)} (seed={seed})...", end=" ", flush=True)
            
            cmd = [
                "python", "experiments/run_experiment.py",
                "--config", f"experiments/configs/{cfg_file}",
                "--seed", str(seed)
            ]
            
            # Get current time to find the right run later
            before_time = datetime.now()
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            after_time = datetime.now()
            
            if result.returncode != 0:
                print(f"❌ FAILED (exit code {result.returncode})")
                if result.stderr:
                    # Show last error line
                    err_lines = result.stderr.split('\n')
                    for line in reversed(err_lines):
                        if "error" in line.lower() or "exception" in line.lower():
                            print(f"      Error: {line[:80]}")
                            break
                continue
            
            # Find the run directory by looking for the most recent one created
            runs_dir = Path("runs")
            run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
            
            run_dir = None
            for d in run_dirs[:5]:  # Check 5 most recent
                # Check if created within our time window
                mtime = datetime.fromtimestamp(d.stat().st_mtime)
                if before_time <= mtime <= after_time + __import__('datetime').timedelta(seconds=5):
                    run_dir = d
                    break
            
            if not run_dir or not run_dir.exists():
                print(f"❌ Run dir not found")
                continue
            
            # Load metrics
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                print(f"❌ No metrics.json")
                continue
            
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"❌ Failed to load metrics: {e}")
                continue
            
            # Extract drift metrics
            s1_mse = metrics.get("session1_mse_mean")
            s2_mse = metrics.get("session2_mse_mean")
            degrad = metrics.get("degradation_pct")
            auroc = metrics.get("auroc_drift")
            
            # Extract corruption metrics (synthetic noise test)
            corr_clean = metrics.get("clean_mse_mean")
            corr_corrupt = metrics.get("corrupted_mse_mean")
            corr_degrad = metrics.get("corruption_degradation_pct")
            sensitivity = metrics.get("corruption_sensitivity", 1.0)
            
            rec = {
                "Model": model_name,
                "Seed": seed,
                "Run": run_dir.name,
                "S1_MSE": s1_mse,
                "S2_MSE": s2_mse,
                "Drift_Degrad%": degrad,
                "AUROC": auroc,
                "Clean_MSE": corr_clean,
                "Corrupted_MSE": corr_corrupt,
                "Corruption_Sensitivity": sensitivity,
            }
            
            results.append(rec)
            model_runs.append(rec)
            print(f"✅ Drift:{degrad:5.1f}% AUROC:{auroc:.3f} | Corr_Sens:{sensitivity:.3f}")
    
    # Print summary
    if results:
        print("\n\n" + "="*100)
        print("SWEEP SUMMARY - AGGREGATED STATISTICS")
        print("="*100 + "\n")
        
        # Group by model
        for model_name in [cfg[1] for cfg in configs]:
            model_results = [r for r in results if r["Model"] == model_name]
            if not model_results:
                print(f"{model_name}:")
                print(f"  ❌ No results\n")
                continue
            
            # Aggregate drift metrics
            degrad_values = [r["Drift_Degrad%"] for r in model_results if r["Drift_Degrad%"] is not None]
            auroc_values = [r["AUROC"] for r in model_results if r["AUROC"] is not None]
            s1_mse_values = [r["S1_MSE"] for r in model_results if r["S1_MSE"] is not None]
            s2_mse_values = [r["S2_MSE"] for r in model_results if r["S2_MSE"] is not None]
            sensitivity_values = [r["Corruption_Sensitivity"] for r in model_results if r["Corruption_Sensitivity"] is not None]
            
            degrad_mean = np.mean(degrad_values) if degrad_values else np.nan
            degrad_std = np.std(degrad_values) if degrad_values else np.nan
            auroc_mean = np.mean(auroc_values) if auroc_values else np.nan
            auroc_std = np.std(auroc_values) if auroc_values else np.nan
            s1_mean = np.mean(s1_mse_values) if s1_mse_values else np.nan
            s2_mean = np.mean(s2_mse_values) if s2_mse_values else np.nan
            sensitivity_mean = np.mean(sensitivity_values) if sensitivity_values else np.nan
            sensitivity_std = np.std(sensitivity_values) if sensitivity_values else np.nan
            
            print(f"{model_name}:")
            print(f"  Drift Test:")
            print(f"    S1 MSE: {s1_mean:.4f}")
            print(f"    S2 MSE: {s2_mean:.4f}")
            print(f"    Degradation: {degrad_mean:.1f}% ± {degrad_std:.1f}%")
            print(f"    AUROC: {auroc_mean:.3f} ± {auroc_std:.3f}")
            print(f"  Synthetic Corruption Test (σ=0.05):")
            print(f"    Sensitivity (lower=more robust): {sensitivity_mean:.3f} ± {sensitivity_std:.3f}")
            print()
    
    print("="*100 + "\n")
    
    # Final comparison
    print("FINAL COMPARISON: Which model is best?")
    print("-" * 100)
    
    baseline = next((r for r in results if "ANN Baseline" in r["Model"] and r["Seed"] == 42), None)
    if baseline:
        baseline_degrad = baseline["Drift_Degrad%"]
        baseline_sensitivity = baseline["Corruption_Sensitivity"]
        
        for model_name in [cfg[1] for cfg in configs[1:]]:
            model_results = [r for r in results if r["Model"] == model_name]
            if not model_results:
                continue
            
            degrad_values = [r["Drift_Degrad%"] for r in model_results if r["Drift_Degrad%"] is not None]
            sensitivity_values = [r["Corruption_Sensitivity"] for r in model_results if r["Corruption_Sensitivity"] is not None]
            
            degrad_mean = np.mean(degrad_values) if degrad_values else np.nan
            sensitivity_mean = np.mean(sensitivity_values) if sensitivity_values else np.nan
            
            print(f"\n{model_name}:")
            
            # Drift comparison
            if not np.isnan(degrad_mean) and not np.isnan(baseline_degrad):
                improvement = baseline_degrad - degrad_mean
                if improvement > 0:
                    print(f"  ✅ BETTER on drift: {improvement:.1f}% less degradation")
                else:
                    print(f"  ❌ WORSE on drift: {-improvement:.1f}% more degradation")
            
            # Sensitivity comparison
            if not np.isnan(sensitivity_mean) and not np.isnan(baseline_sensitivity):
                if sensitivity_mean < baseline_sensitivity:
                    print(f"  ✅ BETTER on corruption: {(1-sensitivity_mean/baseline_sensitivity)*100:.1f}% more robust")
                else:
                    print(f"  ❌ WORSE on corruption: {(sensitivity_mean/baseline_sensitivity-1)*100:.1f}% less robust")
    
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()
