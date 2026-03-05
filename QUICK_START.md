# Quick Start Guide - Phase 1 Complete ✅

## Environment Setup

```bash
# Activate the conda environment
conda activate white_noise

# Install package in editable mode (one-time)
pip install -e .
```

## Running Experiments

### ANN Baseline (Standard Autoencoder)
```bash
python experiments/run_experiment.py --config experiments/configs/ann_baseline.yaml
```
- No noise injection
- 59 input channels
- Latent dimension: 64
- Training time: ~2-3 minutes on CPU

### Denoising Autoencoder (Gaussian Noise)
```bash
python experiments/run_experiment.py --config experiments/configs/denoising_ae.yaml
```
- Additive Gaussian noise (σ=0.1) during training
- Same architecture as ANN baseline
- Training time: ~2-3 minutes on CPU

## Output Structure

Each experiment creates a timestamped directory in `runs/`:

```
runs/20260305_231017/
├── config.yaml                          # Configuration used
├── metrics.json                         # Final metrics
├── results.json                         # Detailed results with history
├── run.log                              # Full experiment log
├── checkpoints/
│   └── best_model.pth                   # Best model weights
└── plots/
    ├── learning_curves.png              # Training & validation loss
    └── reconstruction_errors.png        # Error distribution histogram
```

## Creating Custom Configurations

1. **Copy a base config**:
   ```bash
   cp experiments/configs/ann_baseline.yaml experiments/configs/my_config.yaml
   ```

2. **Edit parameters**:
   - `data`: window_size, window_stride, batch_size, etc.
   - `model`: n_channels, latent_dim, hidden_dims
   - `noise`: enabled, sigma
   - `train`: epochs, lr, weight_decay, early_stopping_patience

3. **Run with custom config**:
   ```bash
   python experiments/run_experiment.py --config experiments/configs/my_config.yaml
   ```

## Key Files Overview

### Data & Models
- `src/data/loaders.py`: MNE data loading, preprocessing, windowing
- `src/models/ann_autoencoder.py`: Standard 1D conv autoencoder
- `src/models/denoising_autoencoder.py`: Autoencoder with noise injection

### Training & Evaluation
- `src/training/trainer.py`: Main training loop
- `src/training/callbacks.py`: Early stopping, checkpointing
- `src/evaluation/metrics.py`: Reconstruction error, AUROC, AUPRC, F1
- `src/evaluation/plots.py`: Visualization functions

### Configuration & Utilities
- `src/config.py`: YAML config loading with inheritance
- `src/utils/logging.py`: Timestamped logging, seed management

## Analyzing Results

```bash
# View latest experiment results
LATEST=$(ls -t runs/ | head -1)
cat runs/$LATEST/metrics.json

# View configuration
cat runs/$LATEST/config.yaml

# View log
tail -50 runs/$LATEST/run.log

# Compare two runs
diff runs/<run1>/metrics.json runs/<run2>/metrics.json
```

## Development Tips

### Adding a New Model
1. Create `src/models/my_model.py`
2. Implement a `nn.Module` with `forward(x)` method
3. Update `src/config.py` with model config class
4. Update `experiments/run_experiment.py` to instantiate it
5. Create corresponding YAML in `experiments/configs/`

### Adding a New Dataset
1. Modify `src/data/loaders.py` with a new function
2. Ensure it returns `{"train": DataLoader, "val": DataLoader, "test": DataLoader}`
3. Update config with new data parameters
4. Test with a small experiment first

### Debugging
- Add print statements in `src/training/trainer.py` for training details
- Check `runs/<timestamp>/run.log` for full execution trace
- Use `torch.backends.cudnn.benchmark = False` for deterministic behavior

## Performance Notes

- **CPU training**: ~2-3 minutes per experiment (50 epochs)
- **GPU training**: Much faster (if available via CUDA)
- **Data**: MNE sample data is ~167 seconds of EEG
- **Windows**: ~324 total windows after preprocessing

## GPU Usage

If GPU is available:
1. Ensure CUDA is installed and PyTorch was built with CUDA support
2. Set `device: "cuda"` in YAML config (it will auto-detect and use CPU fallback)
3. Training will be significantly faster (~10-30x speedup)

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: white_noise_experimentation` | Run `pip install -e .` in repo root |
| `CUDA out of memory` | Reduce batch_size in config or use CPU |
| `Early stopping at epoch X` | Increase early_stopping_patience if desired |
| `No channels selected` | MNE data has 59 EEG channels; don't change n_channels |

## Citation

```bibtex
@software{white_noise_experimentation,
  title={white_noise_experimentation: Noise-Modulated STDP in SNNs for BCI},
  author={Gallego, Elena},
  year={2026},
  url={https://github.com/elenagallego/NoiseNeuro}
}
```

---

**Phase 1 Status**: ✅ Complete  
**Last Updated**: 2026-03-05  
**Python Version**: 3.11+
