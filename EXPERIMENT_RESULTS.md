# Phase 1 – Experiment Results

_Generated on 2026-03-11 11:01:32_

This table compares three autoencoder configurations trained on synthetic EEG data (Session 1 / Day 1) and evaluated for reconstruction quality and drift robustness on both Day 1 (in-distribution) and Day 2 (distribution-shifted) test sets.

## Metrics Comparison

| Metric                             | ANN Baseline       | DAE σ=0.05         | DAE σ=0.1          |
| ---------------------------------- | ------------------ | ------------------ | ------------------ |
| **Model type**                     | ann_baseline       | denoising_ae_sigma_0_05 | denoising_ae_sigma_0_1 |
| **Parameters**                     | 578,262            | 578,262            | 578,262            |
| **Epochs trained**                 | 50                 | 50                 | 50                 |
| **Best val loss (MSE)**            | 0.1287             | 0.1288             | 0.1289             |
| **Test Day 1 loss (MSE)**          | 0.1299             | 0.1299             | 0.1299             |
| **MSE Day 1 (mean)**               | 0.1299             | 0.1299             | 0.1299             |
| **MSE Day 2 (mean)**               | 0.1670             | 0.1667             | 0.1668             |
| **Degradation %**                  | 28.56%             | 28.41%             | 28.32%             |
| **MSE Day 1 (95th %ile)**          | 0.1333             | 0.1335             | 0.1336             |
| **MSE Day 2 (95th %ile)**          | 0.1717             | 0.1716             | 0.1718             |
| **AUROC**                          | 1.0000             | 1.0000             | 1.0000             |
| **AUPRC**                          | 1.0000             | 1.0000             | 1.0000             |

## Metric Definitions

| Metric | Description |
| ------ | ----------- |
| Best val loss | Lowest validation MSE during training (in-distribution) |
| Test Day 1 loss | MSE on held-out Day 1 test set (in-distribution) |
| MSE Day 1 / Day 2 (mean) | Average per-window reconstruction error |
| Degradation % | 100 × (MSE_day2 / MSE_day1 − 1); higher = worse under drift |
| 95th %ile | 95th percentile of per-window errors (tail behaviour) |
| AUROC | Area under ROC curve treating Day 2 as anomalous |
| AUPRC | Area under Precision-Recall curve (Day 2 = positive class) |

## Interpretation

- **Best in-distribution reconstruction**: ANN Baseline (val loss = 0.1287)

- **Most robust to drift** (lowest degradation): DAE σ=0.1 (degradation = 28.32%)

- **Best drift/anomaly detection** (highest AUROC): ANN Baseline (AUROC = 1.0000)

