# Four-Seed MF-DeepONet Ensemble Report

Date: 2026-06-21

This report documents a stronger UQ baseline for the reproduced Lu 2022 BTE
multifidelity DeepONet experiment: a strict four-seed ensemble of full
MF residual DeepONets.

## 1. Training Setup

Each member is trained with the original Lu 2022 MF residual DeepONet
configuration:

- official BTE `mf_train.npz` / `mf_test.npz`;
- width = 512;
- batch size = 65,536;
- learning rate = 1e-4;
- epochs = 400,000;
- stage = MF residual only;
- residual target = `y_H - y_low_x`;
- trunk input includes `y_low_x`;
- only random seed differs.

Training used at most two GPUs:

```text
GPU 2: seeds 2026062101 -> 2026062102
GPU 3: seeds 2026062103 -> 2026062104
```

The ensemble training harness is:

```text
run_mf_ensemble.sh
```

The harness has been updated to propagate failures correctly.

## 2. Individual Member Results

All four members completed successfully.

| Member | MSE | Relative L2 | Best step |
| --- | ---: | ---: | ---: |
| seed 2026062101 | 5.9413e-5 | 2.7158% | 399000 |
| seed 2026062102 | 5.8571e-5 | 2.6965% | 397000 |
| seed 2026062103 | 5.9129e-5 | 2.7093% | 397000 |
| seed 2026062104 | 5.9413e-5 | 2.7158% | 396000 |

These are all close to the reproduced deterministic MF baseline and better
than the original paper's reported MF relative L2 of 3.34%.

Note: the ensemble evaluator restores the latest checkpoint in each member
directory (`model.ckpt-400000.ckpt`). The per-member `metrics.json` records the
best-step metrics from training. The final-step ensemble is what is reported
below.

## 3. Ensemble Test Metrics

Official `mf_test.npz`:

| Metric | Best single member | 4-member ensemble |
| --- | ---: | ---: |
| MSE | 5.8571e-5 | **5.3972e-5** |
| Relative L2 | 2.6965% | **2.5885%** |
| NLL | — | -3.4614 |
| 95% coverage | — | 89.90% |
| Pointwise uncertainty-error Spearman | — | 0.0838 |
| Sample-level uncertainty-error Spearman | — | **0.5254** |

Compared with earlier Bayesian prototypes:

| Method | MSE | Relative L2 | Notes |
| --- | ---: | ---: | --- |
| Deterministic reproduced MF baseline | 5.9957e-5 | 2.7282% | single checkpoint |
| Bayesian last-layer exact-low | 5.8674e-5 | 2.6989% | post-hoc linear readout |
| Best ensemble member | 5.8571e-5 | 2.6965% | full MF model |
| **4-member ensemble** | **5.3972e-5** | **2.5885%** | strongest point metric |

Conclusion: the ensemble gives the strongest point prediction so far, reducing
relative L2 from the reproduced single-checkpoint baseline 2.7282% to 2.5885%.

## 4. UQ Behavior

The ensemble provides materially stronger epistemic uncertainty than the
last-layer Bayesian head.

Positive:

- sample-level uncertainty-error Spearman = 0.5254;
- ensemble objective std is much larger than last-layer posterior std;
- ensemble mean significantly improves MSE/relative L2.

Limitations:

- 95% coverage is only 89.90%, so the ensemble is under-dispersed;
- pointwise uncertainty-error Spearman is only 0.0838;
- aleatoric variance was calibrated on a training subset, not on a fully
  held-out calibration split;
- only four ensemble members were trained.

Thus the ensemble is a stronger UQ baseline than the last-layer posterior, but
it is not yet a fully calibrated predictive distribution.

## 5. Inverse-Design Candidate Uncertainty

The ensemble was evaluated on the same existing GA/TO candidates used in the
previous Bayesian reports.

### Center Objective

| Quantity | Value |
| --- | ---: |
| Number of unique candidates | 18 |
| Best ensemble mean objective | 2.4440 |
| Std of best candidate objective | 0.00784 |
| Most uncertain objective std | 0.01847 |

### Two-Points Objective

| Quantity | Value |
| --- | ---: |
| Number of unique candidates | 114 |
| Best ensemble mean objective | 1.8791 |
| Std of best candidate objective | 0.01018 |
| Most uncertain objective std | 0.02149 |

Compared with the last-layer Bayesian candidate uncertainty:

| Objective | Last-layer best std | Ensemble best std | Increase |
| --- | ---: | ---: | ---: |
| center | ~0.0010 | ~0.00784 | ~7.8× |
| two-points | ~0.00048 | ~0.01018 | ~21× |

Conclusion: ensemble disagreement provides much more meaningful candidate
uncertainty than the last-layer posterior. However, best-by-mean and
best-by-risk still select the same candidates in this candidate set. Therefore
we should not claim improved inverse-design ranking yet without high-fidelity
validation of candidate errors.

## 6. Safe Claims

Safe claims:

1. A strict four-seed MF-DeepONet ensemble, trained under original Lu 2022
   conditions, gives the best point prediction among our tested variants.
2. The ensemble improves official test MSE and relative L2 compared with both
   the reproduced single deterministic baseline and the Bayesian last-layer
   prototype.
3. The ensemble provides substantially larger and more useful candidate
   objective uncertainty than last-layer Bayesian UQ.
4. Sample-level uncertainty-error correlation is meaningfully positive.

Unsafe claims:

1. Do not claim the ensemble is well calibrated; coverage remains below nominal.
2. Do not claim strong pointwise error detection.
3. Do not claim inverse-design ranking improvement without high-fidelity
   validation or active learning.
4. Do not claim this is a full Bayesian posterior; it is an ensemble
   approximation.

## 7. Summary

The four-member ensemble is the strongest result so far:

```text
Single deterministic MF rel-L2: 2.7282%
Bayesian last-layer rel-L2:     2.6989%
Best ensemble member rel-L2:    2.6965%
4-member ensemble rel-L2:       2.5885%
```

It also improves uncertainty behavior relative to the last-layer method:

```text
sample-level uncertainty-error Spearman: 0.5254
candidate objective std: 7x-20x larger than last-layer posterior
```

The best next step is to use the ensemble uncertainty in an active-learning
loop: select high-objective/high-uncertainty BTE candidates, run high-fidelity
OpenBTE validation, and test whether uncertainty predicts surrogate ranking
errors.

