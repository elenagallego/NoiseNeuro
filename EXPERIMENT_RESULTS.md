# Phase 1.5 – Proving Denoising Works

_Generated on 2026-03-11 11:39:25 | 5 seeds per model_

Three experiments stress-test whether denoising autoencoders (DAE) provide measurable benefits over a plain ANN autoencoder on synthetic EEG data with **subtle, realistic BCI drift**.

## Experiment 1: Subtle Drift (Realistic BCI Session Shift)

| Metric | ANN Baseline | DAE σ=0.05 | DAE σ=0.1 |
| --- | --- | --- | --- |
| **Best val loss** | 0.1282 ± 0.0008 | 0.1282 ± 0.0009 | 0.1282 ± 0.0008 |
| **Test Day 1 MSE** | 0.1288 ± 0.0008 | 0.1288 ± 0.0009 | 0.1288 ± 0.0009 |
| **MSE Day 1 (mean)** | 0.1288 ± 0.0008 | 0.1288 ± 0.0009 | 0.1288 ± 0.0009 |
| **MSE Day 2 (mean)** | 0.1346 ± 0.0044 | 0.1347 ± 0.0045 | 0.1347 ± 0.0044 |
| **Degradation %** | 4.4745 ± 3.3308 | 4.5212 ± 3.3403 | 4.5558 ± 3.3415 |
| **Day 1 95th %ile** | 0.1323 ± 0.0009 | 0.1323 ± 0.0009 | 0.1323 ± 0.0010 |
| **Day 2 95th %ile** | 0.1388 ± 0.0046 | 0.1388 ± 0.0047 | 0.1388 ± 0.0048 |
| **AUROC** | 0.8484 ± 0.2102 | 0.8521 ± 0.2098 | 0.8532 ± 0.2098 |
| **AUPRC** | 0.9757 ± 0.0385 | 0.9757 ± 0.0393 | 0.9755 ± 0.0400 |

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

- Degradation %: 4.52 ± 3.34 vs 4.47 ± 3.33 (t=0.020, p=0.5077)
- AUROC: 0.8521 ± 0.2098 vs 0.8484 ± 0.2102 (t=0.025, p=0.4905)
- Degradation improvement: ❌ not significant (p < 0.1)
- AUROC improvement: ❌ not significant (p < 0.1)

### DAE σ=0.1 vs ANN Baseline

- Degradation %: 4.56 ± 3.34 vs 4.47 ± 3.33 (t=0.034, p=0.5133)
- AUROC: 0.8532 ± 0.2098 vs 0.8484 ± 0.2102 (t=0.032, p=0.4877)
- Degradation improvement: ❌ not significant (p < 0.1)
- AUROC improvement: ❌ not significant (p < 0.1)

### Corruption Robustness Improvement (σ = 0.1)

- **DAE σ=0.05**: 7.90 ± 0.31 vs 7.91 ± 0.33 (t=-0.043, p=0.4834) ❌
- **DAE σ=0.1**: 7.88 ± 0.31 vs 7.91 ± 0.33 (t=-0.107, p=0.4587) ❌

## Success Criteria Checklist

- [x] Subtle drift AUROC drops below 0.95 (actual mean: 0.8512)
- [ ] At least one DAE beats plain AE on degradation %
- [x] DAE shows less degradation under corruption
- [x] 5-seed statistics with mean ± std
- [x] Plots saved to `runs/20260311_113844_phase15/`
- [x] Summary table with statistical tests

## Interpretation

- **Most robust to drift**: ANN Baseline
- **Best corruption robustness**: DAE σ=0.1
- **Highest drift AUROC**: DAE σ=0.1

