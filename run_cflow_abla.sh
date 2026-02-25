#!/bin/bash
# Stage-1 HC-RF: CFlow training (release baseline)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="./data/OASIS3-longitudinal-v2.h5"

COMMON_ARGS=(
  --data_path "$DATA_PATH"
  --model hc_rf
  --rf_sampling_steps 10
  --rf_use_ode_solver dopri5
  --epochs 30
  --batch_size 16
  --num_workers 4
  --eval_every 1
  --save_every 10
  --lr 1e-4
  --lr_policy cosine_warmup
  --warmup_epochs 10
  --eta_min 1e-6
  --cf_input_channels 3
  --cf_aux_weight 0.1
  --cf_roi_focus_weight 5.0
  --cf_reflow_seg_weight 1.0
  --cf_reflow_roi_lpips_weight 0.5
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

EXP_NAME="CFlow_release_baseline"
log "Running: $EXP_NAME"

python main.py "${COMMON_ARGS[@]}" --exp_name "$EXP_NAME"

log "Done: $EXP_NAME"