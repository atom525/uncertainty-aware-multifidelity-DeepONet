#!/bin/bash
# Train a strict original-config MF-DeepONet ensemble on the official BTE data.
#
# Each member uses the Lu 2022 reproduced MF residual configuration:
#   width=512, epochs_mf=400000, batch=65536, lr=1e-4
#
# Only random seeds differ. Two GPU lanes are used at most: GPU 2 and GPU 3.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f "$ROOT/env_setup.sh" ]; then source "$ROOT/env_setup.sh"; fi

mkdir -p "$ROOT/logs/bte_mf_ensemble"
mkdir -p "$ROOT/results/bte_mf_ensemble"

run_member() {
    local seed="$1"
    local gpu="$2"
    local out_dir="$ROOT/results/bte_mf_ensemble/member_seed_${seed}"
    local log_file="$ROOT/logs/bte_mf_ensemble/member_seed_${seed}.log"
    echo "[$(date '+%F %T')] START seed=${seed} gpu=${gpu}" | tee -a "$ROOT/logs/bte_mf_ensemble/driver.log"
    CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/src/bte/run_bte_pipeline.py" \
        --data_dir "$ROOT/data/bte_real" \
        --out_dir "$out_dir" \
        --stage mf \
        --width 512 \
        --epochs_mf 400000 \
        --batch 65536 \
        --lr 1e-4 \
        --seed "$seed" \
        > "$log_file" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] DONE  seed=${seed} gpu=${gpu} exit=${rc}" | tee -a "$ROOT/logs/bte_mf_ensemble/driver.log"
    return "$rc"
}

(
    run_member 2026062101 2
    run_member 2026062102 2
) &
PID_A=$!

(
    run_member 2026062103 3
    run_member 2026062104 3
) &
PID_B=$!

echo "[$(date '+%F %T')] launched lane_gpu2=${PID_A} lane_gpu3=${PID_B}" | tee -a "$ROOT/logs/bte_mf_ensemble/driver.log"
wait "$PID_A" "$PID_B"
RC=$?
echo "[$(date '+%F %T')] ALL_DONE exit=${RC}" | tee -a "$ROOT/logs/bte_mf_ensemble/driver.log"
exit "$RC"

