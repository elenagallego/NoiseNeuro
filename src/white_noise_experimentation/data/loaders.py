"""EEG data loading with session-aware splits for drift evaluation.

Provides synthetic EEG-like data structured to simulate multi-session
BCI recordings (Day 1 vs Day 2/7). The Day 2 data is shifted to
simulate distribution drift (amplitude scaling + additive low-frequency
drift), enabling evaluation of model robustness under non-stationary
conditions.

When a real dataset (e.g., BCI Competition IV 2a) is available, replace
the ``_generate_synthetic_eeg`` helper while keeping the public API
(``EEGWindowDataset`` and ``load_eeg_dataset_session_split``) unchanged.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EEGWindowDataset(Dataset):
    """Holds windowed EEG-like data with optional labels and session IDs.

    Args:
        windows: Array of shape ``(n_windows, n_channels, window_size)``.
        labels: Optional integer labels per window (e.g., class or session).
        session_ids: Optional session identifier per window.
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: Optional[np.ndarray] = None,
        session_ids: Optional[np.ndarray] = None,
    ):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = (
            torch.tensor(labels, dtype=torch.long)
            if labels is not None
            else torch.full((len(windows),), -1, dtype=torch.long)
        )
        self.session_ids = session_ids

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Synthetic EEG generation
# ---------------------------------------------------------------------------

def _generate_synthetic_eeg(
    n_channels: int,
    n_samples: int,
    fs: float = 256.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic multi-channel EEG-like data.

    Produces a mixture of sine waves at typical EEG band frequencies
    (alpha ~10 Hz, beta ~20 Hz) with pink-ish noise.

    Args:
        n_channels: Number of EEG channels.
        n_samples: Total number of time-points.
        fs: Sampling frequency in Hz.
        seed: Random seed for reproducibility.

    Returns:
        data: Array of shape ``(n_channels, n_samples)``.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / fs

    data = np.zeros((n_channels, n_samples), dtype=np.float32)
    for ch in range(n_channels):
        # Channel-specific random phase offsets
        phase_alpha = rng.uniform(0, 2 * np.pi)
        phase_beta = rng.uniform(0, 2 * np.pi)

        alpha = 1.0 * np.sin(2 * np.pi * 10 * t + phase_alpha)
        beta = 0.5 * np.sin(2 * np.pi * 20 * t + phase_beta)
        noise = 0.3 * rng.randn(n_samples)

        data[ch] = (alpha + beta + noise).astype(np.float32)

    return data


def _apply_drift(
    data: np.ndarray,
    amplitude_scale: float = 1.15,
    drift_frequency: float = 0.5,
    drift_amplitude: float = 0.3,
    seed: int = 123,
) -> np.ndarray:
    """Apply controlled distribution shift to simulate Day 2 / session drift.

    Two transformations are applied:
      1. **Amplitude scaling** – multiplies all channels by a constant factor.
      2. **Additive low-frequency drift** – adds a slow sine wave per channel,
         with random phase, to mimic electrode drift or environment change.

    Args:
        data: Array of shape ``(n_channels, n_samples)``.
        amplitude_scale: Multiplicative scale factor (>1 = amplified).
        drift_frequency: Frequency of the additive drift wave (Hz).
        drift_amplitude: Amplitude of the additive drift wave.
        seed: Random seed for reproducibility.

    Returns:
        shifted: Shifted copy of ``data`` with the same shape.
    """
    rng = np.random.RandomState(seed)
    n_channels, n_samples = data.shape
    t = np.arange(n_samples) / 256.0  # assume 256 Hz

    shifted = data.copy() * amplitude_scale
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        shifted[ch] += drift_amplitude * np.sin(2 * np.pi * drift_frequency * t + phase)

    return shifted.astype(np.float32)


# ---------------------------------------------------------------------------
# Windowing & normalisation
# ---------------------------------------------------------------------------

def _create_windows(
    data: np.ndarray,
    window_size: int,
    window_stride: int,
) -> np.ndarray:
    """Slice continuous data into overlapping windows.

    Args:
        data: Array of shape ``(n_channels, n_samples)``.
        window_size: Number of time-points per window.
        window_stride: Step between consecutive windows.

    Returns:
        windows: Array of shape ``(n_windows, n_channels, window_size)``.
    """
    n_channels, n_samples = data.shape
    starts = range(0, n_samples - window_size + 1, window_stride)
    windows = np.stack([data[:, s : s + window_size] for s in starts])
    return windows


def _normalize_per_channel(windows: np.ndarray) -> np.ndarray:
    """Per-channel z-scoring (mean 0, std 1) computed across the window axis.

    Args:
        windows: Array of shape ``(n_windows, n_channels, window_size)``.

    Returns:
        normalised: Same shape, each channel independently standardised.
    """
    # Compute stats over (n_windows, window_size) for each channel
    mean = windows.mean(axis=(0, 2), keepdims=True)
    std = windows.std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1.0  # guard against zero-variance channels
    return ((windows - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_eeg_dataset_session_split(config) -> Dict[str, DataLoader]:
    """Load (or generate) EEG data and return session-aware DataLoaders.

    Session 1 ("Day 1") is used for train / val / test_day1.
    Session 2 ("Day 2") is a distribution-shifted version used for test_day2.

    Args:
        config: A ``Config`` object (see ``config.py``) with at least:
            - ``config.data.window_size``
            - ``config.data.window_stride``
            - ``config.data.normalize``
            - ``config.data.batch_size``
            - ``config.data.num_workers``
            - ``config.data.val_split``
            - ``config.data.test_split``
            - ``config.data.seed``
            - ``config.model.n_channels``

    Returns:
        Dictionary with keys:
            - ``'train'``: DataLoader — Session 1, training split.
            - ``'val'``: DataLoader — Session 1, validation split.
            - ``'test_day1'``: DataLoader — Session 1, held-out test (in-distribution).
            - ``'test_day2'``: DataLoader — Session 2, drifted / shifted.
    """
    seed = config.data.seed
    n_channels = config.model.n_channels
    window_size = config.data.window_size
    window_stride = config.data.window_stride

    # --- Generate synthetic EEG for two sessions ---
    # Session 1 (Day 1): ~60 s of data at 256 Hz
    n_samples_day1 = 256 * 60
    day1_raw = _generate_synthetic_eeg(n_channels, n_samples_day1, seed=seed)

    # Session 2 (Day 2): same length, with controlled drift applied
    day2_raw = _generate_synthetic_eeg(n_channels, n_samples_day1, seed=seed)
    day2_raw = _apply_drift(day2_raw, seed=seed + 1)

    # --- Window ---
    day1_windows = _create_windows(day1_raw, window_size, window_stride)
    day2_windows = _create_windows(day2_raw, window_size, window_stride)

    # --- Normalise (per-channel z-scoring) ---
    if config.data.normalize:
        day1_windows = _normalize_per_channel(day1_windows)
        day2_windows = _normalize_per_channel(day2_windows)

    # --- Split Day 1 into train / val / test_day1 ---
    rng = np.random.RandomState(seed)
    n_day1 = len(day1_windows)
    indices = rng.permutation(n_day1)

    n_test = max(1, int(n_day1 * config.data.test_split))
    n_val = max(1, int(n_day1 * config.data.val_split))
    n_train = n_day1 - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    # Labels: 0 = Day 1 (normal), used consistently for anomaly metrics
    train_ds = EEGWindowDataset(day1_windows[train_idx], labels=np.zeros(n_train, dtype=np.int64))
    val_ds = EEGWindowDataset(day1_windows[val_idx], labels=np.zeros(n_val, dtype=np.int64))
    test_day1_ds = EEGWindowDataset(
        day1_windows[test_idx], labels=np.zeros(n_test, dtype=np.int64)
    )
    # Day 2: label = 1 (drifted)
    test_day2_ds = EEGWindowDataset(
        day2_windows, labels=np.ones(len(day2_windows), dtype=np.int64)
    )

    batch_size = config.data.batch_size
    num_workers = config.data.num_workers

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test_day1": DataLoader(
            test_day1_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test_day2": DataLoader(
            test_day2_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }

    return loaders


# Backward-compatible alias used by the existing run_experiment.py
load_eeg_dataset = load_eeg_dataset_session_split
