"""Evaluate a strict MF-DeepONet deep ensemble on the official BTE test set.

The ensemble members are independently trained MF residual DeepONets using the
original Lu 2022 configuration (width=512, batch=65536, epochs=400k), differing
only by random seed.

This evaluator:
  1. restores each member checkpoint,
  2. predicts on official `mf_test.npz`,
  3. computes ensemble mean and epistemic variance,
  4. calibrates a homoscedastic aleatoric noise term on official `mf_train.npz`
     using the trained ensemble (not using test labels),
  5. reports paper metrics and UQ metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("DDEBACKEND", "tensorflow.compat.v1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import deepxde as dde  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC_BTE = ROOT / "src" / "bte"
SRC_BAYES = ROOT / "src" / "bte_bayesian"
for p in (SRC_BTE, SRC_BAYES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_bte_pipeline import (  # noqa: E402
    build_mf_net,
    load_mf_dataset,
    periodic_phase_factory,
)
from run_bayesian_last_layer import (  # noqa: E402
    DatasetFlat,
    compute_metrics,
    find_latest_checkpoint,
    iter_slices,
    json_dump,
    load_flat_mf_npz,
)


def set_tf_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


set_tf_memory_growth()


def build_member_model(
    data_dir: Path,
    member_dir: Path,
    width: int,
) -> Tuple[dde.Model, Path]:
    graph = tf.Graph()
    with graph.as_default():
        data, _, _ = load_mf_dataset(str(data_dir), lf_model=None)
        net = build_mf_net(25, width, lf_in_trunk=1)
        net.apply_feature_transform(periodic_phase_factory())
        mu = float(np.mean(data.train_y))
        sd = float(np.std(data.train_y))
        net.apply_output_transform(lambda _, y: y * sd + mu)
        model = dde.Model(data, net)
        model.compile("adam", lr=0.0)
        ckpt = find_latest_checkpoint(member_dir / "mf_ckpt")
        model.restore(str(ckpt), verbose=1)
    return model, ckpt


def predict_member_residual(
    model: dde.Model,
    dataset: DatasetFlat,
    batch_size: int,
) -> np.ndarray:
    preds = []
    for sl in iter_slices(dataset.x_branch.shape[0], batch_size):
        preds.append(
            model.predict((dataset.x_branch[sl], dataset.x_trunk[sl]))
            .reshape(-1)
            .astype(np.float64)
        )
    return np.concatenate(preds)


def ensemble_predict(
    member_dirs: List[Path],
    data_dir: Path,
    dataset: DatasetFlat,
    width: int,
    batch_size: int,
) -> Tuple[np.ndarray, List[dict]]:
    residual_predictions = []
    member_records = []
    for member_dir in member_dirs:
        print(f"Restoring ensemble member: {member_dir.name}")
        model, ckpt = build_member_model(data_dir, member_dir, width)
        resid = predict_member_residual(model, dataset, batch_size)
        full = dataset.y_low + resid
        residual_predictions.append(resid)
        mse = float(np.mean((full - dataset.y_high) ** 2))
        rel = float(np.linalg.norm(full - dataset.y_high) / np.linalg.norm(dataset.y_high))
        member_records.append(
            {
                "member": member_dir.name,
                "checkpoint": str(ckpt),
                "mse": mse,
                "relative_l2": rel,
            }
        )
        # Explicitly close TF1 session/graph resources before next member.
        try:
            model.sess.close()
        except Exception:
            pass
    return np.vstack(residual_predictions), member_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data" / "bte_real")
    parser.add_argument("--ensemble_dir", type=Path, default=ROOT / "results" / "bte_mf_ensemble")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "bte_mf_ensemble_eval")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--max_train_points_for_noise", type=int, default=250000)
    args = parser.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    member_dirs = sorted(
        p for p in args.ensemble_dir.glob("member_seed_*") if (p / "metrics.json").exists()
    )
    if len(member_dirs) < 2:
        raise RuntimeError(f"Need at least 2 ensemble members in {args.ensemble_dir}")
    print(f"Evaluating {len(member_dirs)} ensemble members")

    test = load_flat_mf_npz(args.data_dir / "mf_test.npz")
    test_resids, member_test_records = ensemble_predict(
        member_dirs, args.data_dir, test, args.width, args.batch_size
    )
    test_full_members = test.y_low[None, :] + test_resids
    test_mean = np.mean(test_full_members, axis=0)
    test_epistemic_var = np.var(test_full_members, axis=0, ddof=1)

    # Calibrate aleatoric noise on train labels using a deterministic subset of
    # official mf_train. This avoids test-label leakage while limiting runtime.
    train = load_flat_mf_npz(args.data_dir / "mf_train.npz")
    if len(train.y_high) > args.max_train_points_for_noise:
        idx = np.linspace(0, len(train.y_high) - 1, args.max_train_points_for_noise).astype(int)
        train_sub = DatasetFlat(
            x_branch=train.x_branch[idx],
            x_trunk=train.x_trunk[idx],
            y_high=train.y_high[idx],
            y_low=train.y_low[idx],
            residual=train.residual[idx],
            lengths=np.asarray([len(idx)], dtype=np.int64),
        )
    else:
        train_sub = train
    train_resids, _ = ensemble_predict(member_dirs, args.data_dir, train_sub, args.width, args.batch_size)
    train_full = train_sub.y_low[None, :] + train_resids
    train_mean = np.mean(train_full, axis=0)
    train_epistemic = np.var(train_full, axis=0, ddof=1)
    # Non-negative homoscedastic aleatoric term after accounting for epistemic.
    train_sqerr = (train_mean - train_sub.y_high) ** 2
    aleatoric_sigma2 = float(max(np.mean(train_sqerr - train_epistemic), 1e-18))

    total_var = np.maximum(test_epistemic_var + aleatoric_sigma2, 1e-18)
    metrics = compute_metrics(test.y_high, test_mean, total_var, test.lengths)

    # Deterministic single best for comparison from member records.
    best_member = min(member_test_records, key=lambda r: r["relative_l2"])

    result = {
        "n_members": len(member_dirs),
        "members": member_test_records,
        "best_single_member": best_member,
        "ensemble": asdict(metrics),
        "aleatoric_sigma2_from_train_subset": aleatoric_sigma2,
        "train_subset_points_for_noise": int(len(train_sub.y_high)),
        "elapsed_sec": time.time() - t0,
    }
    json_dump(args.out_dir / "metrics.json", result)
    np.savez_compressed(
        args.out_dir / "predictions_test.npz",
        y_high=test.y_high.astype(np.float32),
        y_low=test.y_low.astype(np.float32),
        member_predictions=test_full_members.astype(np.float32),
        ensemble_mean=test_mean.astype(np.float32),
        ensemble_epistemic_std=np.sqrt(test_epistemic_var).astype(np.float32),
        ensemble_total_std=np.sqrt(total_var).astype(np.float32),
        lengths=test.lengths,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

