"""Callbacks for training."""

from pathlib import Path
from typing import Optional

import torch


class EarlyStoppingCallback:
    """Early stopping callback based on validation loss."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait_count = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Check if should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if should stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            self.should_stop = True

        return self.should_stop


class CheckpointCallback:
    """Save best model checkpoint."""

    def __init__(self, save_dir: Path, monitor: str = "val_loss"):
        """
        Initialize checkpointing.

        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor ('val_loss' or 'train_loss')
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.best_value = float("inf")
        self.best_model_path = None

    def step(self, model: torch.nn.Module, current_value: float) -> Optional[str]:
        """
        Save checkpoint if current value is better.

        Args:
            model: Model to save
            current_value: Current value of monitored metric

        Returns:
            Path to saved checkpoint or None
        """
        if current_value < self.best_value:
            self.best_value = current_value
            checkpoint_path = self.save_dir / "best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            self.best_model_path = str(checkpoint_path)
            return str(checkpoint_path)
        return None

    def load_best(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Load best model.

        Args:
            model: Model to load weights into

        Returns:
            Model with best weights loaded
        """
        if self.best_model_path:
            model.load_state_dict(torch.load(self.best_model_path))
        return model
