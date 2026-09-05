# Sprint 9 Task 6 — Training ablations (FINAL: 6/6 cells)

## Design (trainer unchanged — config-only knobs)

- `experiments/20260905/05/train_ablation.py` (server commit `d0af9e5`) mirrors
  `notebooks/train_v1_representation.ipynb` on medium (B=128, 40 epochs = 7840 steps,
  `val_stationary_joint_loss` selection, own 8192-row bank per cell). Only `V1Config` values vary;
  `RepresentationTrainer` source untouched. Detached runner `run_grid.sh`, master log `grid_master.log`.
- Grid: `baseline` (λ=0.1, τ=0.2, aug×1.0) | `lambda0` (λ=0.0, prediction-only) | `lambda05` (λ=0.5) |
  `tau007` (τ=0.07) | `aug0` (aug×0.0: gain/offset/noise 0, shift 0) | `aug2` (aug×2.0).

## Training runs (all exit 0, RTX 4060 Ti)

- `baseline` exit 0, 1571 s → step 7840 (reproduces prod-checkpoint regime).
- `lambda0` exit 0, 1524 s. `lambda05` exit 0, 1545 s. `tau007` exit 0, 1537 s.
  `aug0` exit 0, 1437 s. `aug2` exit 0, 1519 s. `GRID DONE 2026-09-05T16:57:33Z`.
- Final train/val losses: baseline pred 0.1283/0.0568, cont 0.7287/0.6994 (val stat 0.1267);
  lambda0 pred 0.1365/0.0570, cont 2.3124/2.2051 (untrained head, expected at λ=0; best val stat 0.0350);
  lambda05 pred 0.1284/0.0585, cont 0.7066/0.6836 (val stat 0.4003);
  tau007 pred 0.1561/0.0692, cont 0.0027/0.0022 (sharp τ collapses contrastive loss; sim 0.9409 vs ~0.99
  elsewhere — positives harder but uninformative); aug0 pred 0.1219/0.0488 (easiest task, no aug),
  cont 0.7043/0.6769; aug2 pred 0.1319/0.0620, cont 0.7348/0.7038.

## Loss splits by normal/anomaly (`loss_split.py`, medium val 2500/2500, eval mode)

- baseline: normal pred 0.054230 / cont 0.416882 vs abnormal pred 0.062670 (+15.6%) / cont 0.418548 (+0.4%).
- lambda0: 0.053975/1.566770 vs 0.061331 (+13.6%) / 1.568148 (+0.1%).
- lambda05: 0.055719/0.407022 vs 0.065889 (+18.3%) / 0.408712 (+0.4%).
- tau007: 0.066708/0.001015 vs 0.075992 (+13.9%) / 0.001229 (+21% of ~nothing).
- aug0: 0.046920/0.401313 vs 0.053548 (+14.1%) / 0.403189 (+0.5%).
- aug2: 0.059748/0.421961 vs 0.068749 (+15.1%) / 0.423296 (+0.3%).
- Pattern: prediction loss carries a small, stable normal/abnormal gap (+13–18%) in every cell;
  contrastive loss never separates (gaps ≤0.5% except the collapsed tau007 head).

## Detection deltas (`dump_scores.py` per checkpoint, exit 0, ~75 s each; MAD×2.5 val-calibrated)

- prod-ckpt: F1 0.1950, AUROC S_pred 0.5235 / S_pop 0.5034 (reproduces Task 3 within 1 file of 20000).
- baseline: F1 **0.3096**, aucP 0.5410 / aucO 0.5242 (TP=2159 FP=1789 FN=7841 TN=8211).
- lambda0: F1 0.2816, aucP 0.5390 / aucO 0.5017.
- lambda05: F1 **0.3261**, aucP 0.5421 / aucO 0.5194 (best F1).
- tau007: F1 0.3117, aucP 0.5466 / aucO 0.5147.
- aug0: F1 0.2973, aucP **0.5552** (best S_pred) / aucO 0.4985.
- aug2: F1 0.3204, aucP 0.5418 / aucO 0.5233.

## Verdict: retrain (hyperparameter tuning insufficient)

No cell escapes near-chance: F1 spans 0.28–0.33, AUROCs 0.54–0.56/0.50–0.52 — λ/τ/augmentation move
detection by ±0.03 F1 at most. Two side-observations: (1) every freshly trained cell beats the production
checkpoint (0.28+ vs 0.20), so the prod checkpoint itself is a below-par draw, yet still far from usable;
(2) the prediction-loss normal/abnormal gap is real but tiny in every cell, matching the Task 4 Cohen's d.
With the trainer fixed, this grid exhausts the cheap axes — production-grade detection needs
representation-level changes (objective/architecture/signal), not further λ/τ/aug tuning.
