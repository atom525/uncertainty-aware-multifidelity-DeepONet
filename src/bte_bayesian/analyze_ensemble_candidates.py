"""Evaluate ensemble uncertainty on existing GA/TO inverse-design candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

os.environ.setdefault("DDEBACKEND", "tensorflow.compat.v1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import deepxde as dde  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC_BAYES = ROOT / "src" / "bte_bayesian"
SRC_BTE = ROOT / "src" / "bte"
for p in (SRC_BAYES, SRC_BTE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_candidate_uncertainty import (  # noqa: E402
    center_quad_points,
    load_candidates,
    not_in_pore_mask,
    objective_from_flux,
    two_points_quad_points,
    write_csv,
)
from evaluate_mf_ensemble import build_member_model  # noqa: E402
from run_bayesian_last_layer import find_latest_checkpoint  # noqa: E402
from run_bte_pipeline import build_lf_net, load_lf_dataset, periodic_phase_factory  # noqa: E402


def set_tf_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


set_tf_memory_growth()


def build_low_model(data_dir: Path, results_dir: Path, width: int) -> dde.Model:
    graph = tf.Graph()
    with graph.as_default():
        data = load_lf_dataset(str(data_dir))
        net = build_lf_net(25, width)
        net.apply_feature_transform(periodic_phase_factory())
        mu = float(np.mean(data.train_y))
        sd = float(np.std(data.train_y))
        net.apply_output_transform(lambda _, y: y * sd + mu)
        model = dde.Model(data, net)
        model.compile("adam", lr=0.0)
        ckpt = find_latest_checkpoint(results_dir / "low_ckpt")
        model.restore(str(ckpt), verbose=1)
    return model


def eval_candidate_objectives(
    low_model: dde.Model,
    member_models: Sequence[dde.Model],
    design: Sequence[int],
    objective: str,
) -> Tuple[np.ndarray, float]:
    coords = center_quad_points() if objective == "center" else two_points_quad_points()
    design_arr = np.asarray(design, dtype=np.float32)
    x_branch = np.tile(design_arr, (len(coords), 1)).astype(np.float32)
    y_low = low_model.predict((x_branch, coords)).reshape(-1).astype(np.float64)
    trunk = np.hstack((coords, y_low[:, None].astype(np.float32))).astype(np.float32)
    mask = not_in_pore_mask(design_arr, coords)
    vals = []
    for model in member_models:
        resid = model.predict((x_branch, trunk)).reshape(-1).astype(np.float64)
        flux = (y_low + resid) * mask
        vals.append(float(objective_from_flux(flux[:, None], objective)[0]))
    vals = np.asarray(vals, dtype=np.float64)
    return vals, float(np.mean(vals))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data" / "bte_real")
    parser.add_argument("--results_dir", type=Path, default=ROOT / "results" / "bte_real")
    parser.add_argument("--ensemble_dir", type=Path, default=ROOT / "results" / "bte_mf_ensemble")
    parser.add_argument("--inverse_results_dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "bte_mf_ensemble_eval")
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.inverse_results_dir)
    print(f"Loaded {len(candidates)} candidates")
    low_model = build_low_model(args.data_dir, args.results_dir, args.width)
    member_dirs = sorted(
        p for p in args.ensemble_dir.glob("member_seed_*") if (p / "metrics.json").exists()
    )
    member_models = []
    member_names = []
    for d in member_dirs:
        model, ckpt = build_member_model(args.data_dir, d, args.width)
        member_models.append(model)
        member_names.append(d.name)
        print(f"Restored {d.name}: {ckpt}")

    rows = []
    for i, cand in enumerate(candidates):
        vals, mean_val = eval_candidate_objectives(
            low_model, member_models, cand.design, cand.objective
        )
        feasible = cand.constraint_n is None or cand.no_pores <= cand.constraint_n
        adj_vals = vals if feasible else np.zeros_like(vals)
        row = {
            "source": cand.source,
            "objective": cand.objective,
            "design": list(cand.design),
            "no_pores": cand.no_pores,
            "constraint_n": cand.constraint_n,
            "feasible": feasible,
            "stored_objective": cand.stored_objective,
            "ensemble_objective_mean_raw": float(np.mean(vals)),
            "ensemble_objective_std_raw": float(np.std(vals, ddof=1)),
            "ensemble_objective_mean": float(np.mean(adj_vals)),
            "ensemble_objective_std": float(np.std(adj_vals, ddof=1)),
            "ensemble_q05": float(np.quantile(adj_vals, 0.05)),
            "ensemble_q50": float(np.quantile(adj_vals, 0.50)),
            "ensemble_q95": float(np.quantile(adj_vals, 0.95)),
            "risk_score_beta_1": float(np.mean(adj_vals) - np.std(adj_vals, ddof=1)),
            "explore_score_kappa_1": float(np.mean(adj_vals) + np.std(adj_vals, ddof=1)),
            "relative_objective_std": float(np.std(adj_vals, ddof=1) / max(abs(np.mean(adj_vals)), 1e-18))
            if feasible
            else 0.0,
        }
        for name, val in zip(member_names, vals):
            row[f"member_{name}"] = float(val)
        rows.append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(candidates):
            print(f"evaluated {i+1}/{len(candidates)}")

    json.dump(rows, open(args.out_dir / "candidate_uncertainty_ensemble.json", "w"), indent=2)
    write_csv(args.out_dir / "candidate_uncertainty_ensemble.csv", rows)

    summary = {"members": member_names}
    for objective in ("center", "two_points"):
        sub = [r for r in rows if r["objective"] == objective]
        if not sub:
            continue
        summary[objective] = {
            "n_candidates": len(sub),
            "best_by_mean": max(sub, key=lambda r: r["ensemble_objective_mean"]),
            "best_by_risk_beta_1": max(sub, key=lambda r: r["risk_score_beta_1"]),
            "most_uncertain": max(sub, key=lambda r: r["ensemble_objective_std"]),
            "mean_relative_objective_std": float(np.mean([r["relative_objective_std"] for r in sub])),
            "max_relative_objective_std": float(np.max([r["relative_objective_std"] for r in sub])),
        }
    json.dump(summary, open(args.out_dir / "candidate_uncertainty_ensemble_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

