#!/bin/bash
# Stage-2 HC-RF: Reflow baseline (paper baseline)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="./data/OASIS3-longitudinal-v2.h5"
SEG_WEIGHTS="./seg_head_weights.pt"

COMMON_ARGS=(
  --data_path "$DATA_PATH"
  --model hc_rf
  --rf_sampling_steps 20
  --rf_use_ode_solver heun
  --rf_use_reflow
  --rf_reflow_steps 12
  --rf_reflow_weight 0.1
  --rf_reflow_ratio 0.2
  --epochs 30
  --batch_size 16
  --num_workers 4
  --eval_every 1
  --save_every 5
  --lr 1e-5
  --lr_policy cosine_warmup
  --warmup_epochs 10
  --eta_min 1e-7
  --cf_input_channels 3
  --cf_seg_head_weights "$SEG_WEIGHTS"
  --cf_seg_head_mobilenetv4_name mobilenetv4_conv_medium
  --cf_seg_head_lr 1e-7
  --cf_aux_weight 0.1
  --cf_roi_focus_weight 5.0
  --cf_reflow_seg_weight 1.0
  --cf_reflow_roi_lpips_weight 0.5
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

BASE_EXP_NAME="CFlow_release_baseline"
EXP_NAME="Reflow_abla_a_seg_lpips"

BASE_DIR=$(ls -1d logs/*-hc_rf-${BASE_EXP_NAME} 2>/dev/null | sort | tail -n 1)
if [ -z "$BASE_DIR" ]; then
  echo "[ERROR] Cannot locate stage-1 log dir for ${BASE_EXP_NAME}" >&2
  exit 1
fi

log "Running: $EXP_NAME (load_from: $BASE_DIR)"

python main.py "${COMMON_ARGS[@]}" --exp_name "$EXP_NAME" --load_from "$BASE_DIR"

log "Done: $EXP_NAME"
