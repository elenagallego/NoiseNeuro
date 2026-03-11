"""Configuration management with YAML support and dataclasses."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    data_root: str
    window_size: int
    window_stride: int
    normalize: bool = True
    batch_size: int = 64
    num_workers: int = 4
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    split_mode: str = "random_split"  # 'random_split' or 'session_split'


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    type: str  # 'ann_autoencoder' or 'denoising_autoencoder'
    n_channels: int = 64  # EEG channels
    window_size: int = 256
    latent_dim: int = 64
    hidden_dims: list = None

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


@dataclass
class NoiseConfig:
    """Noise injection configuration."""

    enabled: bool = False
    sigma: float = 0.1
    location: str = "input"  # 'input' or 'latent'



@dataclass
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    device: str = "cuda"
    seed: int = 42


@dataclass
class Config:
    """Master configuration class."""

    data: DataConfig
    model: ModelConfig
    noise: NoiseConfig
    train: TrainConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "noise": asdict(self.noise),
            "train": asdict(self.train),
        }


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file with support for includes.

    Supports 'include: base.yaml' to inherit from another config.
    """
    config_path = Path(config_path)

    # Load the main config file
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Handle includes
    if "include" in config_dict:
        include_path = config_path.parent / config_dict.pop("include")
        if include_path.exists():
            with open(include_path, "r") as f:
                base_config = yaml.safe_load(f)
            # Merge: base_config is updated with config_dict
            for key, value in config_dict.items():
                if isinstance(value, dict) and key in base_config:
                    base_config[key].update(value)
                else:
                    base_config[key] = value
            config_dict = base_config

    # Construct Config object
    data_cfg = DataConfig(**config_dict.get("data", {}))
    model_cfg = ModelConfig(**config_dict.get("model", {}))
    noise_cfg = NoiseConfig(**config_dict.get("noise", {}))
    train_cfg = TrainConfig(**config_dict.get("train", {}))

    return Config(data=data_cfg, model=model_cfg, noise=noise_cfg, train=train_cfg)
