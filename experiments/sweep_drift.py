"""Multi-seed drift test sweep for phase 1 validation."""

import json
import subprocess
from pathlib import Path
import sys

def run_sweep():
    """Run drift experiments with multiple seeds and models."""
    configs = [
        ("drift_test.yaml", "ANN Baseline (no noise)"),
        ("drift_test_denoising.yaml", "Denoising AE (σ=0.1)"),
    ]
    
    seeds = [42, 123, 777]
    results = []
    
    print("\n" + "="*80)
    print("DRIFT TEST SWEEP: Phase 1 Validation")
    print("="*80)
    print(f"Models: {len(configs)} | Seeds: {len(seeds)} | Total runs: {len(configs)*len(seeds)}")
    print("="*80 + "\n")
    
    for cfg_file, model_name in configs:
        print(f"\n{'='*80}")
        print(f"Model: {model_name} ({cfg_file})")
        print(f"{'='*80}")
        
        for i, seed in enumerate(seeds, 1):
            print(f"\nSeed {i}/{len(seeds)} (seed={seed})...", flush=True)
            
            cmd = [
                "python", "experiments/run_experiment.py",
                "--config", f"experiments/configs/{cfg_file}",
                "--seed", str(seed)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  ❌ FAILED")
                if "error" in result.stderr.lower():
                    print(f"  Error: {result.stderr.split('Error')[-1][:200]}")
                continue
            
            # Parse stderr to find run directory
            # The logger output contains "Results saved to runs/TIMESTAMP/..."
            lines = result.stderr.split('\n')
            run_dir = None
            for line in lines:
                if "Results saved to runs/" in line:
                    # Extract: "Results saved to runs/20260309_135856/results.json"
                    parts = line.split("runs/")
                    if len(parts) > 1:
                        run_id = parts[1].split("/")[0]
                        run_dir = Path("runs") / run_id
                        break
            
            if not run_dir or not run_dir.exists():
                print(f"  ❌ Run dir not found in output")
                continue
            
            # Load metrics
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                print(f"  ❌ No metrics.json")
                continue
            
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"  ❌ Failed to load metrics: {e}")
                continue
            
            # Extract key metrics
            rec = {
                "Model": model_name,
                "Seed": seed,
                "Run": run_dir.name,
                "S1_MSE": metrics.get("session1_mse_mean"),
                "S2_MSE": metrics.get("session2_mse_mean"),
                "Degradation%": metrics.get("degradation_pct"),
                "AUROC": metrics.get("auroc_drift"),
                "AUPRC": metrics.get("auprc_drift"),
            }
            
            results.append(rec)
            print(f"  ✅ S1={rec['S1_MSE']:.4f} | S2={rec['S2_MSE']:.4f} | "
                  f"Degrad={rec['Degradation%']:.1f}% | AUROC={rec['AUROC']:.3f}")
    
    # Print summary
    if results:
        print("\n\n" + "="*80)
        print("SWEEP SUMMARY")
        print("="*80 + "\n")
        
        # Group by model
        for model_name in [cfg[1] for cfg in configs]:
            model_results = [r for r in results if r["Model"] == model_name]
            if not model_results:
                continue
            
            print(f"{model_name}:")
            for r in model_results:
                print(f"  Seed {r['Seed']:3d}: S1={r['S1_MSE']:.4f} S2={r['S2_MSE']:.4f} "
                      f"Degrad={r['Degradation%']:5.1f}% AUROC={r['AUROC']:.3f}")
            
            # Compute stats
            s1_vals = [r['S1_MSE'] for r in model_results if r['S1_MSE'] is not None]
            s2_vals = [r['S2_MSE'] for r in model_results if r['S2_MSE'] is not None]
            degrad_vals = [r['Degradation%'] for r in model_results if r['Degradation%'] is not None]
            auroc_vals = [r['AUROC'] for r in model_results if r['AUROC'] is not None]
            
            if s1_vals:
                import statistics
                print(f"  Mean S1 MSE:    {statistics.mean(s1_vals):.4f} ± {statistics.stdev(s1_vals) if len(s1_vals)>1 else 0:.4f}")
            if s2_vals:
                import statistics
                print(f"  Mean S2 MSE:    {statistics.mean(s2_vals):.4f} ± {statistics.stdev(s2_vals) if len(s2_vals)>1 else 0:.4f}")
            if degrad_vals:
                import statistics
                print(f"  Mean Degrad:    {statistics.mean(degrad_vals):.1f}% ± {statistics.stdev(degrad_vals) if len(degrad_vals)>1 else 0:.1f}%")
            if auroc_vals:
                import statistics
                print(f"  Mean AUROC:     {statistics.mean(auroc_vals):.3f} ± {statistics.stdev(auroc_vals) if len(auroc_vals)>1 else 0:.3f}")
            print()
    
    print("="*80)
    print("SWEEP COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_sweep()
