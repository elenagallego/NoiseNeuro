# Phase 2 – SNN vs ANN Comparison

_Generated on 2026-03-11 14:02:45 | 3 seeds per model_

Phase 2 replaces the ANN autoencoder with Spiking Neural Networks (SNNs) while keeping the same pipeline, metrics, and evaluation. The goal is to prove SNNs can match ANN performance on drift/anomaly detection, and that learned noise (NoiseControlledSNN) beats fixed noise.

## Drift Detection (Subtle BCI Drift)

| Metric | ANN Baseline | Fixed SNN σ=0.1 | Learned SNN |
| --- | --- | --- | --- |
| **Best val loss** | 0.1283 ± 0.0004 | 0.9987 ± 0.0016 | 0.9987 ± 0.0016 |
| **Test Day 1 MSE** | 0.1286 ± 0.0009 | 1.0015 ± 0.0005 | 1.0015 ± 0.0005 |
| **MSE Day 1 (mean)** | 0.1286 ± 0.0009 | 1.0015 ± 0.0005 | 1.0015 ± 0.0005 |
| **MSE Day 2 (mean)** | 0.1448 ± 0.0045 | 1.0054 ± 0.0089 | 1.0054 ± 0.0089 |
| **Degradation %** | 12.5544 ± 3.7224 | 0.3971 ± 0.9205 | 0.3971 ± 0.9205 |
| **AUROC** | 1.0000 ± 0.0000 | 0.6305 ± 0.2824 | 0.6305 ± 0.2824 |
| **AUPRC** | 1.0000 ± 0.0000 | 0.9371 ± 0.0625 | 0.9371 ± 0.0625 |

## Corruption Robustness

| Corruption σ | ANN Baseline deg % | Fixed SNN σ=0.1 deg % | Learned SNN deg % |
| --- | --- | --- | --- |
| **σ = 0.02** | 0.36 ± 0.08 | 0.06 ± 0.01 | 0.06 ± 0.01 |
| **σ = 0.05** | 2.07 ± 0.20 | 0.29 ± 0.03 | 0.29 ± 0.03 |
| **σ = 0.1** | 8.01 ± 0.36 | 1.08 ± 0.05 | 1.08 ± 0.05 |

## Statistical Comparison (vs ANN Baseline)

### Fixed SNN σ=0.1 vs ANN Baseline

- Degradation %: 0.40 ± 0.92 vs 12.55 ± 3.72 (t=-4.484, p=0.0055)
- AUROC: 0.6305 ± 0.2824 vs 1.0000 ± 0.0000 (t=-1.850, p=0.9310)
- Degradation improvement: ✅ significant (p < 0.1)
- AUROC improvement: ❌ not significant (p < 0.1)

### Learned SNN vs ANN Baseline

- Degradation %: 0.40 ± 0.92 vs 12.55 ± 3.72 (t=-4.484, p=0.0055)
- AUROC: 0.6305 ± 0.2824 vs 1.0000 ± 0.0000 (t=-1.850, p=0.9310)
- Degradation improvement: ✅ significant (p < 0.1)
- AUROC improvement: ❌ not significant (p < 0.1)

## Success Criteria

- [x] SNN models run without NaN (AUROC > 0)
- [ ] Controller beats fixed SNN on AUROC (0.6305 vs 0.6305)
- [x] Training stable across 3 seeds
- [x] Same plots/metrics as Phase 1 (apples-to-apples)

## Interpretation

- **Highest AUROC**: ANN Baseline
- **Most robust to drift**: Learned SNN
- ⚠️ SNNs do not yet match ANN AUROC – further tuning needed.

