# white_noise_experimentation

A clean, reproducible research codebase for exploring noise-modulated STDP in spiking neural networks for BCI applications.

## Phase 1: Baseline Models

Phase 1 focuses on building a solid foundation with classical (non-spiking) autoencoder baselines on EEG data:

- **ANN Autoencoder**: A standard 1D convolutional autoencoder for unsupervised feature learning
- **Denoising Autoencoder**: An autoencoder trained with additive Gaussian noise for robustness
- **Reconstruction Error-based Anomaly Detection**: Using the reconstruction error as an anomaly score

## Project Structure

```
white_noise_experimentation/
├── README.md
├── pyproject.toml                 # Project metadata and dependencies
├── .gitignore
├── src/
│   └── white_noise_experimentation/
│       ├── __init__.py
│       ├── config.py              # Configuration management (YAML + dataclasses)
│       ├── data/
│       │   ├── __init__.py
│       │   └── loaders.py         # EEG data loading and preprocessing
│       ├── models/
│       │   ├── __init__.py
│       │   ├── ann_autoencoder.py
│       │   └── denoising_autoencoder.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py         # Training loop
│       │   └── callbacks.py       # Early stopping, checkpointing
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py         # AUROC, AUPRC, F1, reconstruction errors
│       │   └── plots.py           # Learning curves, histograms, etc.
│       └── utils/
│           ├── __init__.py
│           └── logging.py         # Timestamped logging, seed setting
├── experiments/
│   ├── configs/
│   │   ├── base.yaml              # Base config (inherited by others)
│   │   ├── ann_baseline.yaml      # ANN autoencoder config
│   │   └── denoising_ae.yaml      # Denoising AE config
│   └── run_experiment.py          # Main experiment runner
├── notebooks/
│   └── 00_exploration.ipynb       # Exploratory analysis
└── runs/                          # Output directory (generated)
    └── YYYYMMDD_HHMMSS/
        ├── config.yaml            # Config used for this run
        ├── metrics.json           # Final metrics
        ├── results.json           # Detailed results
        ├── run.log                # Log file
        ├── checkpoints/
        │   └── best_model.pth
        └── plots/
            ├── learning_curves.png
            └── reconstruction_errors.png
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/white_noise_experimentation.git
   cd white_noise_experimentation
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install in editable mode**:
   ```bash
   pip install -e .
   ```

4. **Verify installation**:
   ```bash
   python -c "import white_noise_experimentation; print('Success!')"
   ```

## Usage

### Running Experiments

Train the ANN baseline:
```bash
python experiments/run_experiment.py --config experiments/configs/ann_baseline.yaml
```

Train the denoising autoencoder:
```bash
python experiments/run_experiment.py --config experiments/configs/denoising_ae.yaml
```

Results, plots, and metrics are saved to `runs/YYYYMMDD_HHMMSS/`.

### Configuration

All hyperparameters are managed via YAML configs in `experiments/configs/`. Configs support inheritance:

```yaml
# denoising_ae.yaml
include: base.yaml   # Inherits from base.yaml

model:
  type: "denoising_autoencoder"
  sigma: 0.1

noise:
  enabled: true
```

Key config sections:
- **data**: Dataset paths, window sizes, normalization
- **model**: Model type, architecture parameters
- **train**: Optimizer, learning rate, early stopping
- **noise**: Noise injection settings (for denoising AE)

## Design Principles

1. **Modularity**: Each component (data, models, training, evaluation) is independent and reusable.
2. **Reproducibility**: Configs are saved with results; seeds are set globally; all hyperparameters are config-driven.
3. **Clarity**: Code is well-documented and follows PEP 8.
4. **Extensibility**: Easy to add new models, datasets, or training strategies in later phases.

## Data

Phase 1 uses the **MNE sample dataset** (motor imagery EEG) by default. In later phases, we can swap to:
- BCI Competition IV 2a
- Custom datasets
- Real-time streaming data

Data is automatically downloaded and preprocessed (bandpass filtering, z-score normalization).

## Notebooks

- `notebooks/00_exploration.ipynb`: Exploratory data analysis and sanity checks

## Dependencies

Core:
- `torch` (model training)
- `numpy`, `scipy` (numerical computing)
- `pandas` (data handling)
- `scikit-learn` (metrics, preprocessing)
- `mne` (EEG data loading)
- `matplotlib`, `seaborn` (plotting)
- `pyyaml` (config management)
- `tqdm` (progress bars)

Dev (optional):
- `pytest` (testing)
- `black` (code formatting)
- `flake8` (linting)

## Future Phases

- **Phase 2**: Spiking neural networks (SNNs) with Leaky Integrate-and-Fire (LIF) neurons
- **Phase 3**: Spike-Timing-Dependent Plasticity (STDP) learning rules
- **Phase 4**: Learned noise controller module
- **Phase 5**: Experiments with BCI Competition data and real-time adaptation

## License

MIT

## Contact

Elena Gallego Mauleon (elenagallegomauleon@github.com)
