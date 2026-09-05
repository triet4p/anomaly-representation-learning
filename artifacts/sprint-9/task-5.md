# Sprint 9 Task 5 — Supervised probe ceiling

## Source

- Same dump as Task 4: `experiments/20260905/04/scores_medium.npz` (server commit `d0af9e5`,
  dump exit 0 in 61 s). Frozen 128-d file embeddings: train probes on medium **val**
  (5000 files, 2500 abnormal), evaluate on medium **test** (20000 files, 10000 abnormal).
  Probes fit locally on CPU in < 2 s each.

## Results (test)

- kNN on frozen embeddings: k=1 AUROC **0.5043** F1 0.4965; k=5 AUROC **0.5074** F1 0.4993;
  k=15 AUROC 0.5014; k=50 AUROC 0.4986. Chance — nonparametric heads find no neighborhood structure.
- Centroid-distance (normal-mean) probe: AUROC **0.5065**, AP 0.5042. Same as `S_pop` (0.5034):
  the population geometry carries no anomaly direction.
- Linear (logistic, standardized) probes: C=0.01 AUROC **0.5318** F1 0.5232; C=0.1 AUROC **0.5390**;
  C=1.0 AUROC **0.5409** F1 0.5228; C=10.0 AUROC **0.5410** F1 0.5233. The supervised ceiling sits at
  AUROC ≈ 0.54 / F1 ≈ 0.52 — a +0.02 lift over the raw scores, still ~chance on balanced data.
- Per-family linear-probe AUROC (C=1.0, one-vs-normal): duration_anomaly **0.6619** (only real signal —
  gross duration shifts leak into pooled statistics), missing_event 0.5610, over_regularity 0.5532,
  all others 0.505–0.525.

## Verdict: retrain

The frozen representation does not encode anomaly separability: not even a supervised linear head
trained with labels beats AUROC 0.55 overall. Combined with Task 4 (no score/threshold/fusion separates),
the conclusion is **retrain the representation** — no scoring fix on these embeddings can reach
production-grade detection. (Task 6 ablations test whether training hyperparameters move this ceiling
with the trainer unchanged.)
