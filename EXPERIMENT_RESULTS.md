# Phase 1.5 – Proving Denoising Works

_Generated on 2026-03-11 11:36:07 | 5 seeds per model_

Three experiments stress-test whether denoising autoencoders (DAE) provide measurable benefits over a plain ANN autoencoder on synthetic EEG data with **subtle, realistic BCI drift**.

## Experiment 1: Subtle Drift (Realistic BCI Session Shift)

| Metric | ANN Baseline | DAE σ=0.05 | DAE σ=0.1 |
| --- | --- | --- | --- |
| **Best val loss** | 0.1282 ± 0.0008 | 0.1282 ± 0.0009 | 0.1282 ± 0.0008 |
| **Test Day 1 MSE** | 0.1288 ± 0.0008 | 0.1288 ± 0.0009 | 0.1288 ± 0.0009 |
| **MSE Day 1 (mean)** | 0.1288 ± 0.0008 | 0.1288 ± 0.0009 | 0.1288 ± 0.0009 |
| **MSE Day 2 (mean)** | 0.1347 ± 0.0044 | 0.1347 ± 0.0045 | 0.1348 ± 0.0044 |
| **Degradation %** | 4.5221 ± 3.3285 | 4.5687 ± 3.3381 | 4.6031 ± 3.3394 |
| **Day 1 95th %ile** | 0.1323 ± 0.0009 | 0.1323 ± 0.0009 | 0.1323 ± 0.0010 |
| **Day 2 95th %ile** | 0.1388 ± 0.0046 | 0.1389 ± 0.0048 | 0.1389 ± 0.0048 |
| **AUROC** | 0.8510 ± 0.2093 | 0.8547 ± 0.2083 | 0.8556 ± 0.2068 |
| **AUPRC** | 0.9762 ± 0.0381 | 0.9762 ± 0.0387 | 0.9765 ± 0.0382 |

## Experiment 2: Corruption Robustness Test

| Corruption σ | ANN Baseline degradation % | DAE σ=0.05 degradation % | DAE σ=0.1 degradation % |
| --- | --- | --- | --- |
| **σ = 0.02** | 0.34 ± 0.07 | 0.34 ± 0.07 | 0.34 ± 0.07 |
| **σ = 0.05** | 2.02 ± 0.17 | 2.01 ± 0.17 | 2.01 ± 0.16 |
| **σ = 0.1** | 7.91 ± 0.33 | 7.90 ± 0.31 | 7.88 ± 0.31 |

### Corruption Detection AUROC (clean vs corrupted)

| Corruption σ | ANN Baseline | DAE σ=0.05 | DAE σ=0.1 |
| --- | --- | --- | --- |
| **σ = 0.02** | 0.6000 ± 0.0134 | 0.5901 ± 0.0170 | 0.6050 ± 0.0191 |
| **σ = 0.05** | 0.8066 ± 0.0329 | 0.8116 ± 0.0340 | 0.8050 ± 0.0337 |
| **σ = 0.1** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

## Experiment 3: Multi-Seed Statistical Comparison

### DAE σ=0.05 vs ANN Baseline

- Degradation %: 4.57 ± 3.34 vs 4.52 ± 3.33 (t=0.020, p=0.5076)
- AUROC: 0.8547 ± 0.2083 vs 0.8510 ± 0.2093 (t=0.025, p=0.5096)
- Degradation improvement: ❌ not significant (p < 0.1)
- AUROC improvement: ❌ not significant (p < 0.1)

### DAE σ=0.1 vs ANN Baseline

- Degradation %: 4.60 ± 3.34 vs 4.52 ± 3.33 (t=0.034, p=0.5133)
- AUROC: 0.8556 ± 0.2068 vs 0.8510 ± 0.2093 (t=0.031, p=0.5120)
- Degradation improvement: ❌ not significant (p < 0.1)
- AUROC improvement: ❌ not significant (p < 0.1)

### Corruption Robustness Improvement (σ = 0.1)

- **DAE σ=0.05**: 7.90 ± 0.31 vs 7.91 ± 0.33 (t=-0.043, p=0.4834) ❌
- **DAE σ=0.1**: 7.88 ± 0.31 vs 7.91 ± 0.33 (t=-0.107, p=0.4587) ❌

## Success Criteria Checklist

- [x] Subtle drift AUROC drops below 0.95 (actual mean: 0.8538)
- [ ] At least one DAE beats plain AE on degradation %
- [x] DAE shows less degradation under corruption
- [x] 5-seed statistics with mean ± std
- [x] Plots saved to `runs/20260311_113526_phase15/`
- [x] Summary table with statistical tests

## Interpretation

- **Most robust to drift**: ANN Baseline
- **Best corruption robustness**: DAE σ=0.1
- **Highest drift AUROC**: DAE σ=0.1

