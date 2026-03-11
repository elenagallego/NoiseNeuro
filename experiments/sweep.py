#!/usr/bin/env python
"""Parameter sweep runner for noise level experiments."""

import json
import subprocess
from pathlib import Path
from datetime import datetime

# Configurations to sweep
CONFIGS = [
    ("ANN Baseline", "experiments/configs/ann_baseline.yaml"),
    ("Denoising σ=0.05", "experiments/configs/denoising_ae_sigma_005.yaml"),
    ("Denoising σ=0.1", "experiments/configs/denoising_ae.yaml"),
    ("Denoising σ=0.2", "experiments/configs/denoising_ae_sigma_020.yaml"),
]


def run_sweep():
    """Run all experiments in sweep."""
    results = {}
    sweep_dir = Path("sweeps") / datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Starting Parameter Sweep")
    print(f"Sweep Directory: {sweep_dir}")
    print(f"{'='*80}\n")

    for name, config_path in CONFIGS:
        print(f"\n{'─'*80}")
        print(f"Running: {name}")
        print(f"Config: {config_path}")
        print(f"{'─'*80}")

        # Run experiment
        cmd = [
            "python",
            "experiments/run_experiment.py",
            "--config",
            config_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {name} completed successfully")
                
                # Extract run directory from output
                for line in result.stderr.split("\n"):
                    if "Run directory:" in line:
                        run_dir = line.split("Run directory:")[-1].strip()
                        results[name] = {
                            "status": "success",
                            "run_dir": run_dir,
                            "config": config_path,
                        }
                        break
            else:
                print(f"❌ {name} failed")
                results[name] = {
                    "status": "failed",
                    "config": config_path,
                    "error": result.stderr,
                }
        except Exception as e:
            print(f"❌ {name} error: {e}")
            results[name] = {
                "status": "error",
                "config": config_path,
                "error": str(e),
            }

    # Save sweep summary
    summary_file = sweep_dir / "sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Sweep Complete!")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")

    # Print results summary
    for name, result in results.items():
        status = result["status"]
        symbol = "✅" if status == "success" else "❌"
        print(f"{symbol} {name}: {status}")
        if "run_dir" in result:
            print(f"   Output: {result['run_dir']}")


if __name__ == "__main__":
    run_sweep()
