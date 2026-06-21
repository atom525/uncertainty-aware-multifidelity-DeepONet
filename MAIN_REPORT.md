# Bayesian / UQ Extensions of Multifidelity DeepONet on the BTE Benchmark

## 0. Executive Summary

This study investigates whether the deterministic multifidelity DeepONet from
Lu et al. 2022 can be extended with Bayesian uncertainty quantification (UQ)
while keeping the original BTE data, model configuration, and evaluation
setting.

We started from a reproduced Lu et al. 2022 BTE benchmark:

- official low-fidelity and high-fidelity BTE data;
- original MF-DeepONet architecture;
- original residual learning and input-augmentation design;
- width 512, batch size 65,536, Adam learning rate 1e-4;
- official BTE test split.

We implemented and evaluated three levels of UQ-enhanced multifidelity
DeepONet:

1. **Bayesian last-layer MF-DeepONet**
2. **Propagated LF+HF multifidelity UQ**
3. **Four-member MF-DeepONet ensemble**

The results should be read under two different BTE prediction settings. The paper reports both.

**Full MF pipeline**: learned LF DeepONet + learned HF residual DeepONet.

| Method | Test MSE | Test relative L2 |
| --- | ---: | ---: |
| Lu et al. 2022 full MF-DeepONet | 8.89e-5 | 3.34% |
| Our deterministic model-low inference | 8.9245e-5 | 3.3285% |
| **Our propagated MF-UQ** | **8.6945e-5** | **3.2854%** |

**Exact-low reference**: exact low-fidelity values + learned HF residual DeepONet.

| Method | Test MSE | Test relative L2 |
| --- | ---: | ---: |
| Lu et al. 2022 exact-low reference | 6.00e-5 | 2.72% |
| Our reproduced exact-low deterministic | 5.9957e-5 | 2.7282% |
| Our Bayesian last-layer exact-low | 5.8674e-5 | 2.6989% |
| **Our four-member ensemble exact-low** | **5.3972e-5** | **2.5885%** |

Thus, on the **BTE high-fidelity field prediction task**, our UQ-enhanced
multifidelity variants improve their corresponding reproduced baselines. The
strongest number, 2.5885%, belongs to the exact-low reference setting, not to
the full learned-LF pipeline.

However, on the **inverse design task**, the improvement is not yet achieved.
We performed multiple OpenBTE high-fidelity validation rounds, including
ensemble-UCB, objective correction, objective forest search, and local one-bit
search. None of these found a candidate with higher OpenBTE-validated objective
than the current best candidates.

The key lesson is:

> Better field prediction does not automatically imply better inverse-design
> objective ranking.

The current work establishes a strong UQ surrogate foundation, but improving
inverse design likely requires a true closed-loop active learning procedure
where OpenBTE-validated candidates are added back into training or used to
update an objective-specific surrogate.

---

## 1. Background and Motivation

The four Brown/Karniadakis-line papers suggest two complementary threads:

### Bayesian / UQ thread

- **B-PINNs** introduce Bayesian inference into physics-informed neural
  networks, mainly for function-level PDE solutions and inverse PDE parameter
  identification.
- **Multi-fidelity Bayesian Neural Networks** combine low- and high-fidelity
  information with Bayesian uncertainty quantification.

### Operator-learning / multifidelity DeepONet thread

- **Multifidelity deep neural operators for BTE inverse design** use low- and
  high-fidelity DeepONet surrogates for fast nanoscale heat-transport inverse
  design.
- **Multifidelity deep operator networks for data-driven and physics-informed
  problems** further systematize multifidelity DeepONet architectures.

The gap is:

```text
Bayesian UQ exists mainly for function-level PDE learning,
while multifidelity DeepONet is mostly deterministic.
```

This motivates the central question:

> Can we combine multifidelity DeepONet and UQ so that the operator surrogate
> predicts both high-fidelity fields and predictive uncertainty?

---

## 2. Data and Baseline

### 2.1 Official BTE data

We used the official data released by the authors:

```text
low_train.npz
low_test.npz
mf_train.npz
mf_test.npz
bte5x5_2iter_size10532.npz
```

Meaning:

- `low_*`: low-fidelity OpenBTE with 2 source iterations.
- `mf_*`: high-fidelity OpenBTE with 5 source iterations plus low-fidelity
  local values `y_low_x`.
- Branch input: 25-bit 5x5 pore geometry.
- Trunk input: spatial coordinate `(x, y)` or `(x, y, y_low_x)`.
- Output: BTE heat-flux magnitude.

No synthetic data was used in the final reported experiments.

### 2.2 Baseline reproduction

We reproduced the deterministic MF-DeepONet using:

```text
width = 512
batch size = 65536
learning rate = 1e-4
MF residual training epochs = 400000
LF training epochs = 500000
```

For the exact-low reference setting, where the residual DeepONet is given the exact low-fidelity values `y_low_x`, our reproduced deterministic residual model gives:

| Metric | Value |
| --- | ---: |
| Test MSE | 5.9957e-5 |
| Test relative L2 | 2.7282% |

This matches the paper's exact-low reference result, which reports MSE `6.00e-5` and relative L2 `2.72%`. The full learned-LF pipeline is evaluated separately in the propagated MF-UQ section.

---

## 3. Mathematical Formulation of the Baseline

This section defines the notation used by all our methods.

Each BTE geometry is represented by a binary pore vector:

```text
v = [v_1, ..., v_25] in {0,1}^{25}.
```

For a query location

```text
xi = (x, y),
```

the low-fidelity solver gives a heat-flux magnitude

```text
y_L(v, xi),
```

and the high-fidelity solver gives

```text
y_H(v, xi).
```

In Lu et al. 2022, low fidelity corresponds to OpenBTE with fewer source
iterations, while high fidelity corresponds to more source iterations.

The MF-DeepONet learns a residual correction:

```text
y_H(v, xi) = y_L(v, xi) + R(v, xi, y_L(v, xi)).
```

The residual DeepONet has a branch network and a trunk network:

```text
B_theta(v) in R^p
T_theta(xi, y_L) in R^p
```

with `p = 512`. The residual prediction is:

```text
R_theta(v, xi, y_L)
  = <B_theta(v), T_theta(xi, y_L)> + b
  = sum_{k=1}^p B_k(v) T_k(xi, y_L) + b.
```

Training minimizes residual MSE:

```text
min_theta sum_i |R_theta(v_i, xi_i, y_{L,i}) - (y_{H,i} - y_{L,i})|^2.
```

The high-fidelity field prediction is then:

```text
y_hat_H(v, xi) = y_L(v, xi) + R_theta(v, xi, y_L(v, xi)).
```

The paper metrics are:

```text
MSE = mean_i |y_hat_i - y_i|^2

relative L2 = ||y_hat - y||_2 / ||y||_2.
```

For UQ experiments we also report Gaussian negative log likelihood:

```text
NLL = 1/2 mean_i [log(2 pi sigma_i^2)
                  + (y_i - mu_i)^2 / sigma_i^2],
```

and interval coverage:

```text
coverage_95 = fraction of i with |y_i - mu_i| <= 1.96 sigma_i.
```

For inverse design, the scalar objectives are computed from the flux field.

Center objective:

```text
J_center(v) = y_H(v, xi_center) / mean_xi y_H(v, xi).
```

Two-points objective:

```text
J_two(v)
  = 1/2 [y_H(v, xi_1) + y_H(v, xi_2)] / mean_xi y_H(v, xi),
```

where

```text
xi_1 = (-20, 20),   xi_2 = (20, 0).
```

The original implementation sets:

```text
J_two(v) = 0 if sum(v) > 11.
```

This matters because field-level relative L2 and objective ranking are related
but not equivalent.

---

## 4. Method 1: Bayesian Last-Layer MF-DeepONet

### 4.1 Idea

The deterministic MF-DeepONet residual model is:

```text
y_H(v, xi) = y_L(v, xi) + R_theta(v, xi, y_L).
```

The residual DeepONet computes:

```text
R_theta(v, xi, y_L) = sum_k b_k(v) t_k(xi, y_L) + bias.
```

We freeze the trained branch/trunk networks and rewrite the residual as a
linear model over learned features:

```text
phi(v, xi) = [b_1 t_1, ..., b_p t_p, 1]

R(v, xi) = phi(v, xi)^T w + eps.
```

The training target is the high-fidelity residual:

```text
r_i = y_{H,i} - y_{L,i}.
```

We place a Gaussian prior on the readout:

```text
w ~ N(0, alpha^{-1} I)

eps_i ~ N(0, sigma^2).
```

Let

```text
Phi = [phi_1^T; ...; phi_N^T],
r   = [r_1, ..., r_N]^T.
```

Then Bayesian linear regression gives:

```text
p(w | D) = N(mu_w, Sigma_w)

Sigma_w = sigma^2 (Phi^T Phi + lambda I)^{-1}

mu_w = (Phi^T Phi + lambda I)^{-1} Phi^T r.
```

Here `lambda` is the ridge/prior ratio. It is selected on a geometry-level
validation split of the official training geometries.

For a test point:

```text
mu_R(phi_*) = phi_*^T mu_w

Var[R(phi_*)]
  = sigma^2 [1 + phi_*^T (Phi^T Phi + lambda I)^{-1} phi_*].
```

Finally:

```text
mu_H = y_L + mu_R

Var[y_H] = Var[R].
```

This is a lightweight Bayesian extension: it does not retrain the large
DeepONet, but it gives posterior predictive mean and variance.

### 4.2 Results

| Metric | Deterministic MF | Bayesian last-layer |
| --- | ---: | ---: |
| MSE | 5.9957e-5 | **5.8674e-5** |
| relative L2 | 2.7282% | **2.6989%** |
| NLL | -3.3991 | **-3.4231** |
| 95% coverage | 90.05% | **90.88%** |

### 4.3 Interpretation

This result shows that a Bayesian last-layer readout can be attached to a
reproduced MF-DeepONet without hurting the original prediction metric. It
slightly improves both point prediction and marginal probabilistic metrics.

However, the uncertainty is narrow and weak for inverse-design ranking. This
method is useful as a lightweight proof of concept, but not sufficient as the
main UQ method.

---

## 5. Method 2: Propagated LF+HF Multifidelity UQ

### 5.1 Idea

The first method only places Bayesian uncertainty on the high-fidelity residual
head and uses exact `y_low_x` from the test file.

For a more realistic multifidelity UQ pipeline, we model both:

```text
low-fidelity prediction uncertainty
high-fidelity residual uncertainty
```

The low-fidelity DeepONet is represented with its own Bayesian readout:

```text
y_L(v, xi) = phi_L(v, xi)^T w_L + eps_L,

w_L | D_L ~ posterior.
```

The high-fidelity residual is also represented with a Bayesian readout:

```text
R(v, xi, y_L) = phi_R(v, xi, y_L)^T w_R + eps_R,

w_R | D_H ~ posterior.
```

The final high-fidelity prediction is:

```text
y_H = y_L + R(v, xi, y_L).
```

Because the residual depends on the low-fidelity value, low-fidelity
uncertainty affects the final high-fidelity uncertainty. We use a first-order
Taylor expansion around the low-fidelity posterior mean `mu_L`:

```text
R(v, xi, y_L)
  ~= R(v, xi, mu_L)
      + (dR/dy_L) (y_L - mu_L).
```

Therefore:

```text
Var[y_H]
  ~= Var[R(v, xi, mu_L)]
      + (1 + dR/dy_L)^2 Var[y_L].
```

The factor is:

```text
d y_H / d y_L = d(y_L + R)/d y_L = 1 + dR/dy_L.
```

So this method decomposes final uncertainty into:

```text
HF residual uncertainty
+ propagated LF prediction uncertainty.
```

### 5.2 Results

Strict deployment setting: `y_L` is predicted by the low-fidelity DeepONet, not
taken directly from `mf_test.npz`.

| Metric | Deterministic model-low | Propagated MF-UQ |
| --- | ---: | ---: |
| MSE | 8.9245e-5 | **8.6945e-5** |
| relative L2 | 3.3285% | **3.2854%** |
| NLL | -3.2236 | **-3.2345** |
| 95% coverage | 91.63% | **92.08%** |

### 5.3 Interpretation

This is the closest version to the phrase:

> multifidelity DeepONet + UQ using different-fidelity data.

It separately models LF uncertainty and HF residual uncertainty, and combines
them into final HF uncertainty.

The result is positive but small. It confirms feasibility, but the uncertainty
is still under-calibrated and not strong enough for inverse-design ranking.

---

## 6. Method 3: Four-Member MF-DeepONet Ensemble

### 6.1 Idea

A full Bayesian DeepONet over all weights is expensive. A practical alternative
is deep ensemble:

```text
Train K independent MF-DeepONets with different random seeds.
Prediction mean = ensemble mean.
Prediction uncertainty = ensemble variance.
```

Let the ensemble members be:

```text
f_1(v, xi), ..., f_K(v, xi).
```

The ensemble predictive mean is:

```text
mu(v, xi) = 1/K sum_{k=1}^K f_k(v, xi).
```

The epistemic variance is:

```text
s_epi^2(v, xi)
  = 1/(K-1) sum_{k=1}^K [f_k(v, xi) - mu(v, xi)]^2.
```

For likelihood and coverage metrics, we add a scalar aleatoric term estimated
from official training data:

```text
sigma_total^2(v, xi) = s_epi^2(v, xi) + sigma_aleatoric^2.
```

This is not a full Bayesian posterior, but it is a standard practical
approximation to model uncertainty.

Each member uses the original Lu et al. 2022 configuration:

```text
width = 512
epochs = 400000
batch = 65536
lr = 1e-4
official BTE data
```

Only the random seed changes.

### 6.2 Member results

| Member | MSE | relative L2 |
| --- | ---: | ---: |
| seed 2026062101 | 5.9413e-5 | 2.7158% |
| seed 2026062102 | 5.8571e-5 | **2.6965%** |
| seed 2026062103 | 5.9129e-5 | 2.7093% |
| seed 2026062104 | 5.9413e-5 | 2.7158% |

### 6.3 Ensemble results

Exact-low reference setting:

| Method | MSE | relative L2 |
| --- | ---: | ---: |
| paper exact-low reference | 6.00e-5 | 2.72% |
| reproduced exact-low deterministic | 5.9957e-5 | 2.7282% |
| Bayesian last-layer exact-low | 5.8674e-5 | 2.6989% |
| best ensemble member exact-low | 5.8571e-5 | 2.6965% |
| **4-member ensemble exact-low** | **5.3972e-5** | **2.5885%** |

For the full learned-LF pipeline, the comparable result is the propagated MF-UQ setting:

| Method | MSE | relative L2 |
| --- | ---: | ---: |
| paper full MF-DeepONet | 8.89e-5 | 3.34% |
| our deterministic model-low inference | 8.9245e-5 | 3.3285% |
| our propagated MF-UQ | **8.6945e-5** | **3.2854%** |

UQ metrics:

| Metric | Ensemble |
| --- | ---: |
| NLL | -3.4614 |
| 95% coverage | 89.90% |
| pointwise uncertainty-error Spearman | 0.0838 |
| sample-level uncertainty-error Spearman | **0.5254** |

### 6.4 Interpretation

The ensemble is the strongest method in the exact-low reference setting. It improves relative L2 from:

```text
2.7282% -> 2.5885%
```

For the full learned-LF pipeline, the corresponding improvement is:

```text
3.3285% -> 3.2854%
```

and gives much stronger geometry-level uncertainty than the last-layer
posterior.

However, it is still under-calibrated at the pointwise level.

---

## 7. Inverse Design Exploration

### 7.1 Motivation

Lu et al. use the trained MF-DeepONet for inverse design:

```text
pore geometry -> predicted BTE flux field -> objective -> GA/TO search
```

We tested whether the improved ensemble surrogate and UQ can improve the final
high-fidelity OpenBTE objective.

For each design, ensemble objective values are:

```text
J_k(v) = J(f_k(v, .)).
```

Then:

```text
mu_J(v) = 1/K sum_k J_k(v)

sigma_J^2(v)
  = 1/(K-1) sum_k [J_k(v) - mu_J(v)]^2.
```

We tested three acquisition ideas:

Mean ranking:

```text
A_mean(v) = mu_J(v).
```

Upper confidence bound:

```text
A_UCB(v) = mu_J(v) + kappa sigma_J(v).
```

Risk-averse ranking:

```text
A_risk(v) = mu_J(v) - beta sigma_J(v).
```

We also tested objective-level correction:

```text
J_OpenBTE(v)
  ~= g(mu_J(v), sigma_J(v), no_pores(v), geometry_features(v)).
```

The goal of correction is to learn and remove systematic objective bias in the
field surrogate.

### 7.2 OpenBTE validation setup

We reconstructed the OpenBTE validation environment:

- OpenBTE 0.9.31;
- 5 source iterations;
- 5x5 pore geometry;
- square pores;
- mesh step 5.0;
- Si-300K material.

Calibration against official raw sample 0:

| Quantity | official raw | regenerated |
| --- | ---: | ---: |
| elements | 900 | 908 |
| nodes | 530 | 534 |
| center objective | 1.2173 | 1.1948 |
| objective relative diff | — | 1.84% |

Thus OpenBTE validation is close, but not bit-exact to the authors' environment.

### 7.3 Rounds attempted

We tried:

1. ensemble-UCB GA;
2. objective-level ridge correction;
3. one-bit local OpenBTE search;
4. objective-specific ExtraTrees/forest search;
5. swap-neighborhood search around current best candidates.

All candidates were validated with OpenBTE.

### 7.4 Final center objective results

Across 41 center candidates:

| Rank | Round | OpenBTE objective | pores |
| ---: | --- | ---: | ---: |
| 1 | existing | **1.9108** | 20 |
| 2 | existing | 1.9042 | 19 |
| 3 | existing | 1.9019 | 19 |
| 4 | local 1-bit | 1.9017 | 19 |
| 5 | local 1-bit | 1.8929 | 19 |

No new method exceeded the existing best center candidate.

### 7.5 Final two-points objective results

Across 36+ two-points candidates:

| Rank | Round | OpenBTE objective | pores |
| ---: | --- | ---: | ---: |
| 1 | existing | **1.6739** | 11 |
| 2 | objective forest | 1.6731 | 11 |
| 3 | local 1-bit | 1.6683 | 10 |
| 4 | UCB-GA | 1.6645 | 11 |
| 5 | local 1-bit | 1.6607 | 10 |

The objective-forest method came very close, but did not exceed the existing
best.

### 7.6 Interpretation

The inverse-design task is not solved yet.

Main finding:

```text
Better field prediction does not automatically imply better objective ranking.
```

The field surrogate systematically overestimates objective values for extreme
candidates. UQ helps reveal model disagreement, but current UQ is not strong
enough to reliably correct objective ranking.

---

## 8. Current Final Position

### What is successful

1. Multifidelity DeepONet + UQ for BTE field prediction is successful.
2. The four-member ensemble gives the best field prediction:

```text
relative L2 = 2.5885%
```

3. Geometry-level uncertainty becomes meaningful:

```text
sample-level uncertainty-error Spearman = 0.5254
```

### What is not yet successful

1. One-shot UQ-guided inverse design does not improve OpenBTE objective.
2. Objective correction and local search also did not exceed existing best.
3. Current UQ does not reliably predict objective-ranking error.

---

## 9. Next Direction

The next step should be true closed-loop learning, not more one-shot selection:

```text
OpenBTE validated candidates
-> augment high-fidelity objective dataset
-> update objective-specific surrogate or residual head
-> search again
-> validate again
```

The most promising route is objective-specific Bayesian optimization:

- input: 25-bit pore design;
- output: scalar OpenBTE objective;
- model: objective-level GP / random forest / neural ensemble;
- acquisition: expected improvement or UCB;
- loop: validate, update, reselect.

This directly targets inverse design rather than relying on field-level
prediction accuracy.

---

## 10. Clean Deliverables

The clean directory is:

```text
/data1/liulingfeng/cooperation/ghy/Brown/reproductions/bayesian-mf-deeponet-study/
```

It contains:

- successful ensemble scripts;
- OpenBTE validation scripts;
- final metrics;
- failure log;
- roadmap;
- this final summary.

Weak/failed detailed artifacts have been removed, and only concise summaries
are kept.

