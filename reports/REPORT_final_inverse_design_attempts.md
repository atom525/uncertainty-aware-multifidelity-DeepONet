# Final Inverse-Design Attempts Summary

Date: 2026-06-21

This report summarizes all inverse-design attempts after improving the BTE
field-prediction task with a 4-member MF-DeepONet ensemble.

## 1. Starting Point

The ensemble improved the BTE field-prediction task:

| Method | Relative L2 |
| --- | ---: |
| Lu 2022 full MF-DeepONet | 3.34% |
| reproduced full model-low MF-UQ | 3.285% |
| Lu 2022 exact-low reference | 2.72% |
| reproduced exact-low deterministic | 2.728% |
| 4-member ensemble exact-low | **2.588%** |

The question was whether this improvement and the ensemble uncertainty could
also improve the high-fidelity inverse-design objective after OpenBTE
validation.

## 2. OpenBTE Validation Setup

We reconstructed the OpenBTE validation path and calibrated it against official
raw BTE data:

| Quantity | Official raw sample 0 | Re-generated OpenBTE |
| --- | ---: | ---: |
| elements | 900 | 908 |
| nodes | 530 | 534 |
| center objective proxy | 1.2173 | 1.1948 |
| relative diff | — | 1.84% |

This is sufficient for within-study candidate comparison, but not bit-exact to
the authors' original environment.

## 3. Attempts Made

We validated multiple rounds of candidates with real OpenBTE solves.

### Round 0: Existing GA/TO candidates

Baseline candidates from reproduced GA/TO and ensemble candidate tables.

### Round 1: Ensemble-UCB GA

Acquisition:

```text
UCB(v) = ensemble_mean(v) + 2 * ensemble_std(v)
```

Search sizes:

- center: 916 unique designs;
- two-points: 1031 unique designs.

### Round 2: Objective-level ridge correction

Fit a small correction model on existing OpenBTE-validated data:

```text
true objective ~= f(ensemble mean, ensemble std, no_pores, geometry features)
```

Then selected new candidates for validation.

### Round 3: One-bit local OpenBTE search

Directly evaluated one-bit neighbors around current best designs.

### Round 4: Objective-forest Bayesian optimization style selection

Fit an ExtraTrees objective surrogate on all OpenBTE-validated candidates and
searched a larger design pool. Selected new candidates by:

```text
predicted OpenBTE mean + kappa * tree std
```

## 4. Final OpenBTE Results

### Center Objective

Across 41 validated center candidates:

| Rank | Round | True OpenBTE objective | Pores |
| ---: | --- | ---: | ---: |
| 1 | existing | **1.9108** | 20 |
| 2 | existing | 1.9042 | 19 |
| 3 | existing | 1.9019 | 19 |
| 4 | local 1-bit | 1.9017 | 19 |
| 5 | local 1-bit | 1.8929 | 19 |
| 6 | existing | 1.8915 | 19 |
| 7 | UCB-GA | 1.8908 | 19 |
| 8 | local 1-bit | 1.8867 | 19 |
| 9 | existing | 1.8847 | 18 |
| 10 | local 1-bit | 1.8838 | 19 |

No new method exceeded the existing best `1.9108`.

### Two-Points Objective

Across 36 validated two-points candidates:

| Rank | Round | True OpenBTE objective | Pores |
| ---: | --- | ---: | ---: |
| 1 | existing | **1.6739** | 11 |
| 2 | objective forest | 1.6731 | 11 |
| 3 | local 1-bit | 1.6683 | 10 |
| 4 | UCB-GA | 1.6645 | 11 |
| 5 | local 1-bit | 1.6607 | 10 |
| 6 | objective forest | 1.6561 | 11 |
| 7 | corrected ridge | 1.6544 | 9 |
| 8 | UCB-GA | 1.6492 | 11 |
| 9 | existing | 1.6479 | 11 |
| 10 | existing | 1.6453 | 11 |

The objective-forest method came very close (`1.6731` vs `1.6739`) but still
did not exceed the existing best.

## 5. Conclusion

The inverse-design objective has **not yet been improved**.

What succeeded:

- Field prediction improved strongly.
- OpenBTE validation pipeline works.
- Several methods found candidates close to the existing best.
- Objective-forest surrogate got very close on the two-points task.

What failed:

- One-shot UQ acquisition did not improve the final objective.
- Simple objective correction did not improve the final objective.
- Local one-bit search did not improve the final objective.
- Objective-forest BO-style selection did not improve the final objective.

## 6. Key Lesson

The main lesson is:

> Better field prediction does not automatically imply better inverse-design
> ranking.

The ensemble reduces exact-low BTE field relative L2 to 2.588% and the full learned-LF pipeline reaches about 3.285%, but objective values are
still systematically overestimated and ranking remains imperfect.

## 7. Next Step If Continuing

The only promising path left is a **true model-updating loop**, not more
one-shot selection:

```text
OpenBTE validated candidates
-> augment HF training data
-> fine-tune objective-specific head or residual model
-> regenerate candidates
-> validate again
```

The most direct option is to train an objective-specific surrogate only for the
scalar inverse-design objective, using all validated OpenBTE candidates, then
continue Bayesian optimization. But a convincing paper-level improvement will
require demonstrating a better true OpenBTE objective under a fixed validation
budget.

