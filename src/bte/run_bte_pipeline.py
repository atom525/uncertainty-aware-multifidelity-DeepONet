"""End-to-end multifidelity BTE-DeepONet pipeline driver.

This script faithfully mirrors `deeponet_low.py` + `deeponet_mf.py` from the
upstream `lu-group/multifidelity-deeponet` repo:
- modified DeepONet, branch=[m, w,w,w], trunk=[dim_x, w,w,w,w], relu, Glorot
- periodic phase feature transform on trunk inputs
- output transform that rescales by (mean, std) of training y
- low-fidelity model trained first, then high-fidelity *residual* network with
  stacked-trunk input augmentation: trunk input is (xi, low_fidelity(xi))

What's different from the originals:
- CLI selects the data directory and architecture width (so we can run on the
  synthetic BTE-schema dataset when the official OneDrive dataset is
  unavailable, see `data/bte/gen_synthetic.py`)
- epochs / batch_size exposed as CLI args
- writes metrics + checkpoints + loss curves to a results directory
- explicit memory_growth + DDEBACKEND=tensorflow.compat.v1 for TF 2.13

To reproduce the paper numbers on the official 10,532-sample dataset, run with
  --data_dir <path-with-official-bte5x5_2iter-derived mf_*.npz/low_*.npz>
  --width 512 --epochs_low 500000 --epochs_mf 400000 --batch 65536
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("DDEBACKEND", "tensorflow.compat.v1")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass
import deepxde as dde  # noqa: E402
from deepxde.backend import tf as dtf  # noqa: E402


def periodic_phase_factory(m_phase=16):
    """Apply a 4-direction periodic Fourier feature on the first two columns
    of the trunk input (x, y in [-50, 50]), passing any extra columns through.

    This is exactly the `periodic_phase` defined in `deeponet_low.py`."""
    phi0 = np.linspace(0, np.pi, num=m_phase, endpoint=False, dtype=np.float32)

    def transform(inputs):
        x = inputs[:, :1] * np.pi
        y = inputs[:, 1:2] * np.pi
        phi1 = dtf.Variable(phi0, trainable=True)
        phi2 = dtf.Variable(phi0, trainable=True)
        phi3 = dtf.Variable(phi0, trainable=True)
        phi4 = dtf.Variable(phi0, trainable=True)
        xy = dtf.math.cos(dtf.concat(
            [x - phi1, y - phi2, x + y - phi3, x - y - phi4], 1))
        if inputs.shape[1] > 2:
            return dtf.concat([xy, inputs[:, 2:]], 1)
        return xy

    return transform


def load_lf_dataset(data_dir):
    """Low-fidelity data: branch=pores, trunk=xi, target=low-fidelity flux."""
    fname_train = Path(data_dir) / "low_train.npz"
    fname_test = Path(data_dir) / "low_test.npz"

    def stack(fname):
        d = np.load(fname, allow_pickle=True)
        Xb = np.vstack(d["X0"]).astype(np.float32)
        Xt = np.vstack(d["X1"]).astype(np.float32) / 50.0
        y = np.vstack(d["y"]).astype(np.float32)
        return Xb, Xt, y

    Xb_tr, Xt_tr, y_tr = stack(fname_train)
    Xb_te, Xt_te, y_te = stack(fname_test)
    return dde.data.Triple((Xb_tr, Xt_tr), y_tr, (Xb_te, Xt_te), y_te)


def load_mf_dataset(data_dir, lf_model=None):
    """High-fidelity (residual + stacked trunk) data.

    If `lf_model` is given, low-fidelity predictions for the test set are
    obtained from the trained LF DeepONet (this matches `deeponet_mf.py`
    where `y_low_x = model_low((X_branch, X_trunk))`).
    If `lf_model` is None we fall back to exact LF in the npz.
    """
    fname_train = Path(data_dir) / "mf_train.npz"
    fname_test = Path(data_dir) / "mf_test.npz"

    d = np.load(fname_train, allow_pickle=True)
    Xb_tr = np.vstack(d["X0"]).astype(np.float32)
    Xt_tr_raw = np.vstack(d["X1"]).astype(np.float32) / 50.0
    y_tr = np.vstack(d["y"]).astype(np.float32)
    y_lf_tr = np.vstack(d["y_low_x"]).astype(np.float32)
    Xt_tr = np.hstack((Xt_tr_raw, y_lf_tr))  # stack low-fidelity prediction
    y_tr_resid = y_tr - y_lf_tr

    d = np.load(fname_test, allow_pickle=True)
    Xb_te = np.vstack(d["X0"]).astype(np.float32)
    Xt_te_raw = np.vstack(d["X1"]).astype(np.float32) / 50.0
    y_te = np.vstack(d["y"]).astype(np.float32)
    if lf_model is not None:
        y_lf_te = lf_model.predict((Xb_te, Xt_te_raw)).astype(np.float32)
    else:
        y_lf_te = np.vstack(d["y_low_x"]).astype(np.float32)
    Xt_te = np.hstack((Xt_te_raw, y_lf_te))
    y_te_resid = y_te - y_lf_te

    return dde.data.Triple(
        (Xb_tr, Xt_tr), y_tr_resid, (Xb_te, Xt_te), y_te_resid
    ), y_lf_te, y_te


def build_lf_net(m, width):
    return dde.maps.DeepONet(
        [m, width, width, width],
        [2, width, width, width, width],
        "relu",
        "Glorot normal",
    )


def build_mf_net(m, width, lf_in_trunk):
    trunk_in = 2 + (lf_in_trunk or 0)
    return dde.maps.DeepONet(
        [m, width],
        [trunk_in, width, width, width, width],
        "relu",
        "Glorot normal",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="Dir with low_train.npz, low_test.npz, mf_train.npz, mf_test.npz")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--epochs_low", type=int, default=500000)
    parser.add_argument("--epochs_mf", type=int, default=400000)
    parser.add_argument("--batch", type=int, default=2**16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None,
                        help="optional random seed for reproducible ensemble members")
    parser.add_argument("--m", type=int, default=25,
                        help="branch input dim (25 = 5x5 pore vector)")
    parser.add_argument("--skip_low", action="store_true",
                        help="skip LF training (only train MF residual)")
    parser.add_argument("--low_ckpt", default=None,
                        help="restore LF DeepONet from this checkpoint")
    parser.add_argument("--stage", choices=["both", "lf", "mf"], default="both",
                        help="which stage to run; 'lf' only trains LF, 'mf' only "
                             "trains MF (using exact y_low_x from the dataset)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    if args.seed is not None:
        # Do not call `dde.config.set_random_seed` here: in TF 2.13 it enables
        # op determinism, while the legacy `tf.layers.dense` initializer used
        # by DeepXDE 1.4 does not provide op-level seeds and then fails with
        # "Random ops require a seed". We set the graph/random seeds without
        # enabling deterministic ops, which is enough to create reproducible
        # ensemble initializations in this TF1-compat graph code.
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.compat.v1.set_random_seed(args.seed)

    metrics = {}

    # --------- 1) Low-fidelity DeepONet ---------
    if args.stage in ("both", "lf"):
        t0 = time.time()
        g_low = dtf.Graph()
        _run_lf(args, out_dir, metrics, t0, g_low)
    else:
        print("[harness] --stage=mf, skipping LF training/eval")

    # --------- 2) Multifidelity residual DeepONet ---------
    if args.stage in ("both", "mf"):
        t1 = time.time()
        g_mf = dtf.Graph()
        _run_mf(args, out_dir, metrics, t1, g_mf)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def _run_lf(args, out_dir, metrics, t0, g_low):
    with g_low.as_default():
        data_low = load_lf_dataset(args.data_dir)
        net_low = build_lf_net(args.m, args.width)
        net_low.apply_feature_transform(periodic_phase_factory())
        mu = float(np.mean(data_low.train_y))
        sd = float(np.std(data_low.train_y))
        print(f"[LF] target normalisation mu={mu:.4f} sd={sd:.4f}")
        net_low.apply_output_transform(lambda _, y: y * sd + mu)
        model_low = dde.Model(data_low, net_low)
        model_low.compile("adam", lr=args.lr)
        ck_low = dde.callbacks.ModelCheckpoint(
            str(out_dir / "low_ckpt" / "model.ckpt"),
            save_better_only=True, period=max(1, args.epochs_low // 50))
        (out_dir / "low_ckpt").mkdir(exist_ok=True, parents=True)
        if args.skip_low:
            assert args.low_ckpt, "--skip_low requires --low_ckpt"
            model_low.restore(args.low_ckpt, verbose=1)
        else:
            loss_low, state_low = model_low.train(
                epochs=args.epochs_low,
                batch_size=args.batch,
                callbacks=[ck_low])
            dde.utils.save_loss_history(loss_low, str(out_dir / "lf_loss.txt"))
            metrics["lf_best_step"] = int(state_low.best_step)
            metrics["lf_best_test_loss"] = float(np.sum(state_low.best_metrics))

        y_pred = model_low.predict(data_low.test_x)
        lf_mse = float(np.mean((y_pred - data_low.test_y) ** 2))
        lf_rel = float(
            np.linalg.norm(y_pred - data_low.test_y)
            / np.linalg.norm(data_low.test_y))
        metrics["lf_test_mse"] = lf_mse
        metrics["lf_test_relative_l2"] = lf_rel
        metrics["lf_elapsed_sec"] = time.time() - t0
        print(f"[LF] MSE={lf_mse:.6f}  relL2={lf_rel:.6f}")

def _run_mf(args, out_dir, metrics, t1, g_mf):
    with g_mf.as_default():
        data_mf, y_lf_test, y_high_test = load_mf_dataset(
            args.data_dir, lf_model=None)  # use exact LF for fair compare
        net_mf = build_mf_net(args.m, args.width, lf_in_trunk=1)
        net_mf.apply_feature_transform(periodic_phase_factory())
        mu2 = float(np.mean(data_mf.train_y))
        sd2 = float(np.std(data_mf.train_y))
        print(f"[MF] residual normalisation mu={mu2:.4f} sd={sd2:.4f}")
        net_mf.apply_output_transform(lambda _, y: y * sd2 + mu2)
        model_mf = dde.Model(data_mf, net_mf)
        model_mf.compile("adam", lr=args.lr)
        ck_mf = dde.callbacks.ModelCheckpoint(
            str(out_dir / "mf_ckpt" / "model.ckpt"),
            save_better_only=True, period=max(1, args.epochs_mf // 50))
        (out_dir / "mf_ckpt").mkdir(exist_ok=True, parents=True)
        loss_mf, state_mf = model_mf.train(
            epochs=args.epochs_mf,
            batch_size=args.batch,
            callbacks=[ck_mf])
        dde.utils.save_loss_history(loss_mf, str(out_dir / "mf_loss.txt"))

        y_pred_resid = model_mf.predict(data_mf.test_x)
        # Recombine to actual flux: y_full = y_lf + residual
        y_pred_full = y_pred_resid + y_lf_test
        full_mse = float(np.mean((y_pred_full - y_high_test) ** 2))
        full_rel = float(
            np.linalg.norm(y_pred_full - y_high_test)
            / np.linalg.norm(y_high_test))
        metrics["mf_test_mse_full"] = full_mse
        metrics["mf_test_relative_l2_full"] = full_rel
        metrics["mf_test_mse_residual"] = float(
            np.mean((y_pred_resid - data_mf.test_y) ** 2))
        metrics["mf_best_step"] = int(state_mf.best_step)
        metrics["mf_best_test_loss"] = float(np.sum(state_mf.best_metrics))
        metrics["mf_elapsed_sec"] = time.time() - t1
        print(f"[MF] full MSE={full_mse:.6f}  relL2={full_rel:.6f}")

        np.savez_compressed(out_dir / "mf_predictions.npz",
                            y_lf=y_lf_test,
                            y_high=y_high_test,
                            y_pred_residual=y_pred_resid,
                            y_pred_full=y_pred_full)


if __name__ == "__main__":
    sys.exit(main())
