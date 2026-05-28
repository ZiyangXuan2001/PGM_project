#!/usr/bin/env bash
set -e

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/data/diving48_embeddings/clip_vit_b16/train.pt}"
VAL_FILE="${VAL_FILE:-/workspace/data/diving48_embeddings/clip_vit_b16/test.pt}"
DEVICE="${DEVICE:-auto}"

P_SERIES_CONFIGS=(
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

for config in "${P_SERIES_CONFIGS[@]}"; do
  "${PYTHON_BIN}" -u scripts/train_embeddings.py \
  --config "$config" \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE" \
  --device "$DEVICE"
done
