# Locked Results Manifest

Timestamp: 2026-06-21 01:52 UTC+8

The following result directories are treated as immutable baselines for the
next stronger-UQ experiments. New experiments must write to separate result
directories and must not overwrite these files.

## Deterministic Reproduction Baseline

- `results/bte_real/`
  - Reproduced Lu 2022 BTE LF/MF DeepONet checkpoints and deterministic metrics.
  - Official BTE data under `data/bte_real/`.
  - Key deterministic MF metric recomputed in Bayesian scripts:
    - MSE: `5.995699521181382e-05`
    - relative L2: `0.02728232751695476`

## Bayesian Last-Layer Residual UQ

- `results/bte_bayesian_last_layer/`
  - Code: `src/bte_bayesian/run_bayesian_last_layer.py`
  - Report: `results/bte_bayesian_last_layer/REPORT_bayesian_last_layer.md`
  - Key metrics:
    - MSE: `5.8674303621902794e-05`
    - relative L2: `0.026988917158485008`
    - NLL with validation-calibrated variance: `-3.42310277049217`
    - 95% coverage: `0.9088274730998105`

## Propagated LF+Residual Multifidelity UQ

- `results/bte_multifidelity_uq/`
  - Code: `src/bte_bayesian/run_multifidelity_uq.py`
  - Report: `results/bte_multifidelity_uq/REPORT_multifidelity_uq.md`
  - Key strict model-low baseline:
    - MSE: `8.924486082840415e-05`
    - relative L2: `0.03328534759906071`
  - Key propagated MF-UQ metrics:
    - MSE: `8.694516686559686e-05`
    - relative L2: `0.032853694242066196`
    - NLL: `-3.2344713333981736`
    - 95% coverage: `0.9208419676200397`

## Next Experiment Namespace

The stronger ensemble/UQ experiments will use new directories:

- `results/bte_mf_ensemble/`
- `logs/bte_mf_ensemble/`

