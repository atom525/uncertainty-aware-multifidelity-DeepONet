# Bayesian Multifidelity DeepONet Study

This is the cleaned handoff package for the current successful research line:

> improve the Lu et al. 2022 BTE multifidelity DeepONet field surrogate with ensemble-based uncertainty, then test whether that helps inverse design.

Start with:

```text
MAIN_REPORT.md
```

## What Is Kept

```text
README.md
MAIN_REPORT.md
FAILED_ATTEMPTS_AND_CLEANUP_LOG.md
ROADMAP.md

reports/
├── REPORT_mf_ensemble.md
├── REPORT_final_inverse_design_attempts.md
└── RESULTS_LOCKED_MANIFEST.md

metrics/
├── mf_ensemble_metrics.json
├── mf_ensemble_candidate_summary.json
├── deterministic_lf_metrics.json
└── all_validation_rounds_final_summary.json

scripts/
├── run_mf_ensemble.sh
├── evaluate_mf_ensemble.py
├── analyze_ensemble_candidates.py
├── run_openbte_candidate.py
├── calibrate_openbte_mesh.py
└── match_candidates_in_raw_bte.py
```

Weak or failed branches were removed from this clean directory and documented in:

```text
FAILED_ATTEMPTS_AND_CLEANUP_LOG.md
```

## Main Successful Result

The strongest retained method is a strict four-member MF-DeepONet ensemble.

| Method | MSE | Relative L2 |
| --- | ---: | ---: |
| Lu et al. paper MF-DeepONet | 8.89e-5 | 3.34% |
| reproduced deterministic MF | 5.9957e-5 | 2.7282% |
| **4-member ensemble** | **5.3972e-5** | **2.5885%** |

The ensemble also gives useful geometry-level uncertainty:

```text
sample-level uncertainty-error Spearman = 0.5254
```

## Inverse Design Status

We reconstructed and calibrated an OpenBTE validation path. Across multiple candidate-selection strategies, no method has yet exceeded the existing OpenBTE-validated best candidates:

| Objective | Current best OpenBTE value |
| --- | ---: |
| center | 1.9108 |
| two-points | 1.6739 |

The key finding is:

```text
better field prediction does not automatically imply better inverse-design ranking
```

See:

```text
reports/REPORT_final_inverse_design_attempts.md
```

## Reproduction Commands

Install/activate the MF-DeepONet environment:

```bash
conda create -n mfdeeponet-uq python=3.9 -y
conda activate mfdeeponet-uq
pip install -r envs/requirements-mfdeeponet.txt
source env_setup.sh
```

Then run from this repository root:

```bash
cd /path/to/bayesian-mf-deeponet-study
```

Train ensemble:

```bash
bash run_mf_ensemble.sh
```

Evaluate ensemble:

```bash
CUDA_VISIBLE_DEVICES=2 python src/bte_bayesian/evaluate_mf_ensemble.py --batch_size 32768
CUDA_VISIBLE_DEVICES=2 python src/bte_bayesian/analyze_ensemble_candidates.py
```

Activate OpenBTE validation environment:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda create -n openbte-validation python=3.9 -y
conda activate openbte-validation
conda install -c conda-forge gmsh openmpi mpi4py -y
pip install -r envs/requirements-openbte.txt
```

Run one OpenBTE validation candidate:

```bash
python /data1/liulingfeng/cooperation/ghy/Brown/reproductions/bayesian-mf-deeponet-study/scripts/run_openbte_candidate.py --design_json <candidate_json> --design_index 0 --objective two_points
```

## Recommended Next Step

Move from one-shot candidate ranking to true closed-loop learning:

```text
OpenBTE validated candidates
-> augment high-fidelity objective dataset
-> update objective-specific surrogate or residual model
-> search again
-> validate again
```

The current evidence suggests that optimizing the scalar objective directly may be more effective than relying only on field-level relative L2.

## GitHub Upload

This directory is ready to be initialized as a standalone Git repository. The
large official data files and heavy generated artifacts are excluded by
`.gitignore`.

```bash
cd /data1/liulingfeng/cooperation/ghy/Brown/reproductions/bayesian-mf-deeponet-study
git init
git add .
git commit -m "Add Bayesian multifidelity DeepONet study"
git branch -M main
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```
