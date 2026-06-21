# Failed / Weak Attempts and Cleanup Log

Timestamp: 2026-06-21 22:18 UTC+8

This file records weak or failed experiment branches before removing their
large/verbose artifacts from the clean handoff directory. The goal is to keep
the workspace focused on the successful and still-active research direction.

## Retained Core Assets

These are retained and should not be deleted:

- Official BTE data:
  - `multifidelity-deeponet/data/bte_real/`
- Reproduced deterministic Lu 2022 baseline:
  - `multifidelity-deeponet/results/bte_real/`
- Successful four-member MF-DeepONet ensemble:
  - `multifidelity-deeponet/results/bte_mf_ensemble/`
  - `multifidelity-deeponet/results/bte_mf_ensemble_eval/`
- OpenBTE high-fidelity validation summaries:
  - `openbte_validation/all_validation_rounds_with_local_summary.json`
  - `openbte_validation/REPORT_full_inverse_design_validation.md`
- Clean study directory:
  - `bayesian-mf-deeponet-study/`

## Weak / Failed Branches

### 1. Bayesian Last-Layer Only

Status: weak positive prototype.

Result:

- MSE improved slightly: `5.9957e-5 -> 5.8674e-5`
- rel-L2 improved slightly: `2.7282% -> 2.6989%`
- UQ remained too narrow and weak for inverse-design ranking.

Reason for cleanup:

- Useful as a first prototype but not the main path to a strong inverse-design
  result.

### 2. Propagated LF+Residual MF-UQ

Status: weak positive prototype.

Result:

- rel-L2 improved slightly in strict model-low setting:
  `3.3285% -> 3.2854%`
- Coverage remained below nominal and uncertainty-error ranking was weak.

Reason for cleanup:

- Demonstrates feasibility but does not solve inverse design.

### 3. One-Shot UCB / Risk Ranking

Status: failed to improve high-fidelity objective.

Result:

- Center: best remained old candidate `1.9108`.
- Two-points: best remained old candidate `1.6739`.
- UCB candidates did not exceed prior best.

Reason for cleanup:

- One-shot uncertainty acquisition did not improve validated objective.

### 4. Objective-Level Ridge Correction

Status: failed to improve high-fidelity objective.

Result:

- Corrected candidates did not exceed existing best for either objective.

Reason for cleanup:

- Simple post-hoc correction over a small validated set was insufficient.

### 5. One-Bit Local OpenBTE Search

Status: failed to improve high-fidelity objective.

Result:

- Center local best: `1.9017 < 1.9108`
- Two-points local best: `1.6683 < 1.6739`

Reason for cleanup:

- Valuable diagnostic but not a successful method branch.

## Active Direction After Cleanup

Continue with a true closed-loop inverse-design strategy:

1. Use all OpenBTE-validated samples as objective-level supervision.
2. Fit a stronger uncertainty-aware objective correction model.
3. Search a larger candidate pool with corrected objective and diversity.
4. Validate only high-promise candidates with OpenBTE.
5. Iterate until a true objective improvement is found.

