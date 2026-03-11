"""Multi-seed debug sweep for denoising hypothesis failure.

Tests 3 hypotheses for why denoising AE performs WORSE on drift:
1. σ=0.1 too high → test σ=[0.01, 0.02, 0.05]
2. Wrong noise location → test latent injection
3. Architecture too simple → test synthetic corruption baseline
"""

import json
import subprocess
from pathlib import Path
import sys


def run_sweep():
    """Run debug experiments with multiple seeds and models."""
    
    # Test 1: Lower noise levels
    configs_lower_sigma = [
        ("drift_denoise_001.yaml", "Denoising AE (σ=0.01)"),
        ("drift_denoise_002.yaml", "Denoising AE (σ=0.02)"),
        ("drift_denoise_005.yaml", "Denoising AE (σ=0.05)"),
    ]
    
    # Baseline for comparison
    configs_baseline = [
        ("drift_test.yaml", "ANN Baseline (no noise)"),
        ("drift_test_denoising.yaml", "Denoising AE (σ=0.1, input)"),
    ]
    
    seeds = [42, 123, 777]
    results = []
    
    print("\n" + "="*90)
    print("DENOISING HYPOTHESIS DEBUG SWEEP")
    print("="*90)
    print("Phase 1 Hypothesis: Denoising AE should be MORE ROBUST to drift than plain AE")
    print("Current Status: FAILED - Denoising makes drift WORSE (18% vs 16.6% baseline)")
    print()
    print("Debug Plan:")
    print("  Test 1: Lower noise levels σ=[0.01, 0.02, 0.05] (hypothesis: σ=0.1 too high)")
    print("  Test 2: Latent noise injection (hypothesis: wrong noise location)")
    print("  Test 3: Synthetic corruption baseline (proves denoising works on clean data)")
    print("="*90 + "\n")
    
    # Run baselines first
    print(f"\n{'='*90}")
    print("BASELINE MODELS (for comparison)")
    print(f"{'='*90}\n")
    
    for cfg_file, model_name in configs_baseline:
        print(f"\n{model_name}...")
        
        for seed in seeds:
            cmd = [
                "python", "experiments/run_experiment.py",
                "--config", f"experiments/configs/{cfg_file}",
                "--seed", str(seed)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  Seed {seed}: ❌ FAILED")
                continue
            
            # Parse run directory from output
            lines = result.stderr.split('\n')
            run_dir = None
            for line in lines:
                if "Run directory:" in line:
                    run_dir = Path(line.split("Run directory:")[-1].strip())
                    break
            
            if not run_dir or not run_dir.exists():
                print(f"  Seed {seed}: ❌ Run dir not found")
                continue
            
            # Load metrics
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                print(f"  Seed {seed}: ❌ No metrics.json")
                continue
            
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"  Seed {seed}: ❌ Failed to load metrics")
                continue
            
            rec = {
                "Model": model_name,
                "Seed": seed,
                "Test": "BASELINE",
                "S1_MSE": metrics.get("session1_mse_mean"),
                "S2_MSE": metrics.get("session2_mse_mean"),
                "Degradation%": metrics.get("degradation_pct"),
                "AUROC": metrics.get("auroc_drift"),
                "Sensitivity": metrics.get("sensitivity"),
            }
            
            results.append(rec)
            print(f"  Seed {seed}: ✅ Degrad={rec['Degradation%']:.1f}% | "
                  f"AUROC={rec['AUROC']:.3f} | Sens={rec['Sensitivity']:.3f}")
    
    # Run Test 1: Lower noise levels
    print(f"\n{'='*90}")
    print("TEST 1: LOWER NOISE LEVELS (σ=[0.01, 0.02, 0.05])")
    print(f"{'='*90}\n")
    
    for cfg_file, model_name in configs_lower_sigma:
        print(f"\n{model_name}...")
        
        for seed in seeds:
            cmd = [
                "python", "experiments/run_experiment.py",
                "--config", f"experiments/configs/{cfg_file}",
                "--seed", str(seed)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  Seed {seed}: ❌ FAILED")
                continue
            
            lines = result.stderr.split('\n')
            run_dir = None
            for line in lines:
                if "Run directory:" in line:
                    run_dir = Path(line.split("Run directory:")[-1].strip())
                    break
            
            if not run_dir or not run_dir.exists():
                print(f"  Seed {seed}: ❌ Run dir not found")
                continue
            
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                print(f"  Seed {seed}: ❌ No metrics.json")
                continue
            
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"  Seed {seed}: ❌ Failed to load metrics")
                continue
            
            rec = {
                "Model": model_name,
                "Seed": seed,
                "Test": "LOWER_SIGMA",
                "S1_MSE": metrics.get("session1_mse_mean"),
                "S2_MSE": metrics.get("session2_mse_mean"),
                "Degradation%": metrics.get("degradation_pct"),
                "AUROC": metrics.get("auroc_drift"),
                "Sensitivity": metrics.get("sensitivity"),
            }
            
            results.append(rec)
            print(f"  Seed {seed}: ✅ Degrad={rec['Degradation%']:.1f}% | "
                  f"AUROC={rec['AUROC']:.3f} | Sens={rec['Sensitivity']:.3f}")
    
    # Print comprehensive summary
    print("\n\n" + "="*90)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*90 + "\n")
    
    if results:
        # Summary by model
        models_baseline = set(r["Model"] for r in results if r["Test"] == "BASELINE")
        models_lower = set(r["Model"] for r in results if r["Test"] == "LOWER_SIGMA")
        
        print("BASELINE COMPARISON:")
        print("-" * 90)
        print(f"{'Model':<35} {'Seed':>6} {'S1_MSE':>10} {'S2_MSE':>10} {'Degrad%':>10} {'AUROC':>8}")
        print("-" * 90)
        
        for model_name in sorted(models_baseline):
            model_results = [r for r in results if r["Model"] == model_name and r["Test"] == "BASELINE"]
            for r in sorted(model_results, key=lambda x: x["Seed"]):
                print(f"{r['Model']:<35} {r['Seed']:>6} {r['S1_MSE']:>10.4f} "
                      f"{r['S2_MSE']:>10.4f} {r['Degradation%']:>10.1f} {r['AUROC']:>8.3f}")
            
            if len(model_results) > 1:
                degrad = [r["Degradation%"] for r in model_results]
                auroc = [r["AUROC"] for r in model_results]
                print(f"{'  Mean':<35} {'':<6} {'':<10} {'':<10} "
                      f"{sum(degrad)/len(degrad):>10.1f} {sum(auroc)/len(auroc):>8.3f}")
                print("-" * 90)
        
        print("\nLOWER NOISE SWEEP:")
        print("-" * 90)
        print(f"{'Model':<35} {'Seed':>6} {'S1_MSE':>10} {'S2_MSE':>10} {'Degrad%':>10} {'AUROC':>8}")
        print("-" * 90)
        
        for model_name in sorted(models_lower):
            model_results = [r for r in results if r["Model"] == model_name and r["Test"] == "LOWER_SIGMA"]
            for r in sorted(model_results, key=lambda x: x["Seed"]):
                print(f"{r['Model']:<35} {r['Seed']:>6} {r['S1_MSE']:>10.4f} "
                      f"{r['S2_MSE']:>10.4f} {r['Degradation%']:>10.1f} {r['AUROC']:>8.3f}")
            
            if len(model_results) > 1:
                degrad = [r["Degradation%"] for r in model_results]
                auroc = [r["AUROC"] for r in model_results]
                print(f"{'  Mean':<35} {'':<6} {'':<10} {'':<10} "
                      f"{sum(degrad)/len(degrad):>10.1f} {sum(auroc)/len(auroc):>8.3f}")
                print("-" * 90)
        
        # Analysis
        print("\nKEY FINDINGS:")
        print("-" * 90)
        
        baseline_degrad = [r["Degradation%"] for r in results if "Baseline" in r["Model"]]
        if baseline_degrad:
            print(f"ANN Baseline mean degradation: {sum(baseline_degrad)/len(baseline_degrad):.1f}%")
        
        lower_degrad = [r["Degradation%"] for r in results if r["Test"] == "LOWER_SIGMA"]
        if lower_degrad:
            print(f"Lower sigma denoising mean degradation: {sum(lower_degrad)/len(lower_degrad):.1f}%")
        
        print("\nNEXT STEPS:")
        if baseline_degrad and lower_degrad:
            baseline_mean = sum(baseline_degrad) / len(baseline_degrad)
            lower_mean = sum(lower_degrad) / len(lower_degrad)
            
            if lower_mean < baseline_mean - 1.0:
                print("✅ HYPOTHESIS 1 CONFIRMED: Lower noise helps!")
                print("   → Move to Test 2: Latent noise injection")
            elif lower_mean > baseline_mean + 1.0:
                print("❌ HYPOTHESIS 1 FAILED: Lower noise makes it worse!")
                print("   → Root cause is NOT noise level")
                print("   → Proceed to Test 2: Latent noise injection + synthetic corruption")
            else:
                print("⚠️  INCONCLUSIVE: Similar performance across noise levels")
                print("   → Proceed to Test 2: Latent noise injection + synthetic corruption")


if __name__ == "__main__":
    run_sweep()
