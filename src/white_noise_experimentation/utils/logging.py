"""Logging utilities with timestamps and configuration saving."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


class TimestampedLogger:
    """Simple timestamped logger that also saves configs and metrics."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Set up basic logging to stdout and file
        self.logger = logging.getLogger("white_noise_experimentation")
        self.logger.setLevel(logging.INFO)

        # File handler
        log_file = self.run_dir / "run.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def save_config(self, config: Dict[str, Any]):
        """Save configuration as YAML."""
        config_file = self.run_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        self.info(f"Config saved to {config_file}")

    def save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics as JSON."""
        metrics_file = self.run_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        self.info(f"Metrics saved to {metrics_file}")


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
