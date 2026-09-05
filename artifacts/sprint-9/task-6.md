# Sprint 9 Task 6 — Training ablations (PARTIAL: 3/6 cells done, grid running)

## Design (trainer unchanged — config-only knobs)

- Script `experiments/20260905/05/train_ablation.py` (server, commit `d0af9e5`) mirrors
  `notebooks/train_v1_representation.ipynb` on the medium dataset (B=128, 40 epochs, 7840 steps,
  `val_stationary_joint_loss` selection, own 8192-row bank per cell). Only `V1Config` values vary;
  `RepresentationTrainer` source untouched. Runner `experiments/20260905/05/run_grid.sh` (detached
  `nohup`, master log `grid_master.log`).
- Grid: `baseline` (λ=0.1, τ=0.2, aug×1.0) | `lambda0` (λ=0.0) | `lambda05` (λ=0.5) | `tau007` (τ=0.07) |
  `aug0` (aug×0.0) | `aug2` (aug×2.0).

## Completed cells (exit 0)

- `baseline` exit 0, 1571 s → `ckpt_baseline.pt` (step 7840, reproduces prod-checkpoint regime).
  Final: train pred 0.1283 / cont 0.7287, val pred 0.0568 / cont 0.6994, val stationary joint 0.1267.
- `lambda0` exit 0, 1524 s → `ckpt_lambda0.pt`. `lambda05` exit 0, 1545 s → `ckpt_lambda05.pt`.
- Reference loss splits on the production checkpoint (`loss_split.py`, medium val 2500/2500):
  normal pred 0.072703 / cont 1.371591 vs abnormal pred 0.078364 / cont 1.367435
  (prediction gap +7.8%, contrastive identical — consistent with Tasks 4–5).

## Still running (detached on server, needs no supervision)

- `tau007` started 15:42:40Z (~epoch 20/40 at last poll), then `aug0`, `aug2` (~25 min/cell).
- Handoff to finish: poll `grid_master.log` until `GRID DONE`; per checkpoint run
  `V1_CKPT=experiments/20260905/05/ckpt_<name>.pt .venv/bin/python experiments/20260905/05/loss_split.py`
  (loss splits, ~15 s) and
  `V1_CHECKPOINT_PATH=experiments/20260905/05/ckpt_<name>.pt V1_OUT=experiments/20260905/05/<name>.npz
  .venv/bin/python experiments/20260905/04/dump_scores.py` (~61 s) + local AUROC/F1;
  then replace this file with the full loss-split/detection-delta table and verdict.
