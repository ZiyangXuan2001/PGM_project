#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/data/diving48_embeddings/clip_vit_b16/train.pt}"
VAL_FILE="${VAL_FILE:-/workspace/data/diving48_embeddings/clip_vit_b16/test.pt}"
DEVICE="${DEVICE:-auto}"

mkdir -p logs/p_series outputs/p_series

RUN_IDS=(
  P0
  P1
  P2
  P3
  P4
  P5
  P6
  P7
  P8
  P9
  P10
  P11
  P12
  P13
)

CONFIGS=(
  configs/p_series/P0_trajectory_matrix_noPGM.yaml
  configs/p_series/P1_trajectory_matrix_pgm_lam005.yaml
  configs/p_series/P2_trajectory_matrix_pgm_lam010.yaml
  configs/p_series/P3_trajectory_matrix_linear_noPGM.yaml
  configs/p_series/P4_trajectory_matrix_linear_pgm_lam005.yaml
  configs/p_series/P5_trajectory_matrix_linear_pgm_lam010.yaml
  configs/p_series/P6_prePGM_lam005_trajectory_matrix_linear.yaml
  configs/p_series/P7_prePGM_lam010_trajectory_matrix_linear.yaml
  configs/p_series/P8_trajectory_matrix_linear_pgm_lam020.yaml
  configs/p_series/P9_trajectory_matrix_linear_pgm_lam030.yaml
  configs/p_series/P10_trajectory_matrix_linear_pgm_lam040.yaml
  configs/p_series/P11_prePGM_lam020_trajectory_matrix_linear.yaml
  configs/p_series/P12_prePGM_lam030_trajectory_matrix_linear.yaml
  configs/p_series/P13_prePGM_lam040_trajectory_matrix_linear.yaml
)

HEADS=(
  trajectory_matrix
  trajectory_matrix
  trajectory_matrix
  trajectory_matrix_linear
  trajectory_matrix_linear
  trajectory_matrix_linear
  trajectory_matrix_linear
  trajectory_matrix_linear
  trajectory_matrix_linear
  trajectory_matrix_linear
  trajectory_matrix_linear
  trajectory_matrix_linear
  trajectory_matrix_linear
  trajectory_matrix_linear
)

USE_PGMS=(
  false
  true
  true
  false
  true
  true
  false
  false
  true
  true
  true
  false
  false
  false
)

LAMBDAS=(
  none
  0.05
  0.10
  none
  0.05
  0.10
  none
  none
  0.20
  0.30
  0.40
  none
  none
  none
)

PRE_USE_PGMS=(
  false
  false
  false
  false
  false
  false
  true
  true
  false
  false
  false
  true
  true
  true
)

PRE_LAMBDAS=(
  none
  none
  none
  none
  none
  none
  0.05
  0.10
  none
  none
  none
  0.20
  0.30
  0.40
)

RUN_DIRS=(
  outputs/p_series/P0_trajectory_matrix_noPGM
  outputs/p_series/P1_trajectory_matrix_pgm_lam005
  outputs/p_series/P2_trajectory_matrix_pgm_lam010
  outputs/p_series/P3_trajectory_matrix_linear_noPGM
  outputs/p_series/P4_trajectory_matrix_linear_pgm_lam005
  outputs/p_series/P5_trajectory_matrix_linear_pgm_lam010
  outputs/p_series/P6_prePGM_lam005_trajectory_matrix_linear
  outputs/p_series/P7_prePGM_lam010_trajectory_matrix_linear
  outputs/p_series/P8_trajectory_matrix_linear_pgm_lam020
  outputs/p_series/P9_trajectory_matrix_linear_pgm_lam030
  outputs/p_series/P10_trajectory_matrix_linear_pgm_lam040
  outputs/p_series/P11_prePGM_lam020_trajectory_matrix_linear
  outputs/p_series/P12_prePGM_lam030_trajectory_matrix_linear
  outputs/p_series/P13_prePGM_lam040_trajectory_matrix_linear
)

LOGS=(
  logs/p_series/P0_trajectory_matrix_noPGM.log
  logs/p_series/P1_trajectory_matrix_pgm_lam005.log
  logs/p_series/P2_trajectory_matrix_pgm_lam010.log
  logs/p_series/P3_trajectory_matrix_linear_noPGM.log
  logs/p_series/P4_trajectory_matrix_linear_pgm_lam005.log
  logs/p_series/P5_trajectory_matrix_linear_pgm_lam010.log
  logs/p_series/P6_prePGM_lam005_trajectory_matrix_linear.log
  logs/p_series/P7_prePGM_lam010_trajectory_matrix_linear.log
  logs/p_series/P8_trajectory_matrix_linear_pgm_lam020.log
  logs/p_series/P9_trajectory_matrix_linear_pgm_lam030.log
  logs/p_series/P10_trajectory_matrix_linear_pgm_lam040.log
  logs/p_series/P11_prePGM_lam020_trajectory_matrix_linear.log
  logs/p_series/P12_prePGM_lam030_trajectory_matrix_linear.log
  logs/p_series/P13_prePGM_lam040_trajectory_matrix_linear.log
)

is_completed() {
  local head="$1"
  local use_pgm="$2"
  local lambda_frame="$3"
  local use_pre_pgm="$4"
  local pre_lambda_frame="$5"
  "${PYTHON_BIN}" - "$head" "$use_pgm" "$lambda_frame" "$use_pre_pgm" "$pre_lambda_frame" <<'PY'
import json
import sys
from pathlib import Path

head, use_pgm, lambda_frame, use_pre_pgm, pre_lambda_frame = sys.argv[1:6]
use_pgm = use_pgm.lower() == "true"
use_pre_pgm = use_pre_pgm.lower() == "true"
lambda_value = None if lambda_frame == "none" else round(float(lambda_frame), 4)
pre_lambda_value = None if pre_lambda_frame == "none" else round(float(pre_lambda_frame), 4)

roots = [Path("outputs/runs"), Path("outputs/p_series")]

def norm_lambda(value):
    if value is None:
        return None
    return round(float(value), 4)

def run_matches(path):
    metrics_path = path / "metrics.json"
    best = path / "checkpoints" / "best.pt"
    last = path / "checkpoints" / "last.pt"
    if not (metrics_path.is_file() and best.is_file() and last.is_file()):
        return False
    name = path.name
    config = {}
    resolved = path / "config_resolved.yaml"
    if resolved.is_file():
        try:
            import yaml
            config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        except Exception:
            config = {}

    cfg_model = config.get("model") or {}
    cfg_head = (config.get("classifier") or {}).get("type")
    cfg_use = cfg_model.get("use_pgm")
    cfg_lambda = (config.get("pgm") or {}).get("lambda_frame")
    if cfg_lambda is None:
        cfg_lambda = cfg_model.get("lambda_frame")
    cfg_pre_use = bool(cfg_model.get("use_pre_pgm", False))
    cfg_pre_lambda = (config.get("pre_pgm") or {}).get("lambda_frame")
    if cfg_pre_lambda is None:
        cfg_pre_lambda = cfg_model.get("pre_lambda_frame")
    if cfg_head is not None and cfg_use is not None:
        return (
            str(cfg_head) == head
            and bool(cfg_use) == use_pgm
            and norm_lambda(cfg_lambda if use_pgm else None) == lambda_value
            and cfg_pre_use == use_pre_pgm
            and norm_lambda(cfg_pre_lambda if use_pre_pgm else None) == pre_lambda_value
        )

    if "prePGM" in name or "pre_pgm" in name or "p_traj_pre_pgm" in name:
        detected_pre_use = True
        detected_use = False
        detected_lambda = None
        if "lam005" in name or "preframefixed0.05" in name:
            detected_pre_lambda = 0.05
        elif "lam010" in name or "preframefixed0.1" in name or "preframefixed0.10" in name:
            detected_pre_lambda = 0.10
        else:
            return False
    elif "P-noPGM" in name:
        detected_pre_use = False
        detected_pre_lambda = None
        detected_use = False
        detected_lambda = None
    elif "P-PGM" in name and "framefixed0.05" in name:
        detected_pre_use = False
        detected_pre_lambda = None
        detected_use = True
        detected_lambda = 0.05
    elif "P-PGM" in name and ("framefixed0.1" in name or "framefixed0.10" in name):
        detected_pre_use = False
        detected_pre_lambda = None
        detected_use = True
        detected_lambda = 0.10
    else:
        return False

    if "trajectory_matrix_linear" in name:
        detected_head = "trajectory_matrix_linear"
    elif "trajectory_matrix" in name:
        detected_head = "trajectory_matrix"
    else:
        return False

    return (
        detected_head == head
        and detected_use == use_pgm
        and norm_lambda(detected_lambda) == lambda_value
        and detected_pre_use == use_pre_pgm
        and norm_lambda(detected_pre_lambda) == pre_lambda_value
    )

for root in roots:
    if not root.exists():
        continue
    for path in root.iterdir():
        if path.is_dir() and run_matches(path):
            print(path)
            raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "P-series missing-run trainer"
echo "Started: $(date -Is)"
echo "Python: ${PYTHON_BIN}"
echo "Train file: ${TRAIN_FILE}"
echo "Val file: ${VAL_FILE}"
echo "Device: ${DEVICE}"
echo

declare -a TRAINED=()
declare -a SKIPPED=()

for i in "${!RUN_IDS[@]}"; do
  run_id="${RUN_IDS[$i]}"
  config="${CONFIGS[$i]}"
  head="${HEADS[$i]}"
  use_pgm="${USE_PGMS[$i]}"
  lambda_frame="${LAMBDAS[$i]}"
  use_pre_pgm="${PRE_USE_PGMS[$i]}"
  pre_lambda_frame="${PRE_LAMBDAS[$i]}"
  run_dir="${RUN_DIRS[$i]}"
  log_path="${LOGS[$i]}"

  echo "[$(date -Is)] Checking ${run_id}: head=${head} post_pgm=${use_pgm} post_lambda=${lambda_frame} pre_pgm=${use_pre_pgm} pre_lambda=${pre_lambda_frame}"
  if evidence="$(is_completed "$head" "$use_pgm" "$lambda_frame" "$use_pre_pgm" "$pre_lambda_frame")"; then
    echo "[$(date -Is)] SKIP ${run_id}: completed at ${evidence}"
    SKIPPED+=("${run_id}:${evidence}")
    continue
  fi

  echo "[$(date -Is)] START ${run_id}"
  echo "Command: ${PYTHON_BIN} -u scripts/train_embeddings.py --config ${config} --train_file ${TRAIN_FILE} --val_file ${VAL_FILE} --device ${DEVICE} --run_dir ${run_dir}"
  {
    echo "Started: $(date -Is)"
    echo "Run ID: ${run_id}"
    echo "Command: ${PYTHON_BIN} -u scripts/train_embeddings.py --config ${config} --train_file ${TRAIN_FILE} --val_file ${VAL_FILE} --device ${DEVICE} --run_dir ${run_dir}"
    "${PYTHON_BIN}" -u scripts/train_embeddings.py \
      --config "${config}" \
      --train_file "${TRAIN_FILE}" \
      --val_file "${VAL_FILE}" \
      --device "${DEVICE}" \
      --run_dir "${run_dir}"
    echo "Finished: $(date -Is)"
  } 2>&1 | tee "${log_path}"
  TRAINED+=("${run_id}:${run_dir}")
done

echo
echo "Finished: $(date -Is)"
echo "Skipped completed runs:"
printf '  %s\n' "${SKIPPED[@]:-none}"
echo "Newly trained runs:"
printf '  %s\n' "${TRAINED[@]:-none}"
