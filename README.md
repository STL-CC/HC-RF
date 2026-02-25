# HC-RF (Pre-Publication Release)

This repository contains the official **pre-publication research release** of HC-RF for longitudinal MRI prediction. This git repo is cleaned up from our original codebase by removing functions and modules irrelevant to the paper. This repo is pending further review by the authors to ensure accuracy.

## License Notice
This code is released under a **restrictive pre-publication research license**.

- Research-only, non-commercial use
- No redistribution without written permission
- License may be replaced after formal publication

See [LICENSE](LICENSE) for full terms.

## Repository Layout

- `main.py` — Training entry point and experiment loop
- `args.py` — Argument management
- `dataloader.py` — Dataset building and train/val/test split
- `utils.py` — Logging, metrics, schedulers, serialization
- `models/hc_rf.py` — HC-RF model implementation
- `modules/dit.py` — DiT backbone
- `modules/seg_heads.py` — MobileNetV4 segmentation head
- `data/README.md` — Dataset placement instruction

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is not redistributed in this repository because they are controlled-access datasets and require appropriate authorization. After obtaining access, place the processed file at:

- `./data/ADNI-longitudinal.h5`
- `./data/OASIS3-longitudinal.h5`

## Training

### Stage 1 (CFlow)

```bash
python main.py \
  --data_path "./data/OASIS3-longitudinal.h5" \
  --exp_name "HC-RF_stage1" \
  --rf_sampling_steps 10 \
  --rf_use_ode_solver dopri5 \
  --save_every 10 \
  --lr 1e-4 \
  --eta_min 1e-6
```

### Stage 2 (Reflow baseline)

```bash
python main.py \
  --data_path "./data/OASIS3-longitudinal.h5" \
  --exp_name "HC-RF_stage2" \
  --load_from "logs/YOUR_STAGE_1_LOG_DIR" \
  --rf_sampling_steps 20 \
  --rf_use_ode_solver heun \
  --rf_use_reflow \
  --rf_reflow_steps 12 \
  --save_every 5 \
  --lr 1e-5 \
  --eta_min 1e-7 \
  --cf_seg_head_lr 1e-7
```


