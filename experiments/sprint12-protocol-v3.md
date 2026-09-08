# Sprint 12 Experiment Protocol v3 — risk-role amendment (frozen)

**Status:** FROZEN by explicit user selection. v1 (`experiments/sprint12-protocol-v1.md`)
and v2 (`experiments/sprint12-protocol-v2.md`) stay immutable; this amendment
supersedes ONLY the risk-fit roles and the G-rank event definition below. Any
further change needs v4 + explicit user approval. Frozen BEFORE any corrected
Task 12 execution.

## 0. Why v3 exists

Batch F review showed the Task 12 warning comparison mixed chronological roles
(dev fit across all 90 days), leaked the program-03 cold-start holdout into
training, applied maintenance asymmetrically, used unstandardized features
under L2, and evaluated file-level AUROC against an event-level gate. v3 fixes
the roles and the metric without touching validated base physics, seeds,
quarantine, or the sealed histories.

REJECTED (do not implement): training risk/calibration on dev_train-healthy
only. Supervised warning needs failure outcomes; healthy-only restriction
removes all positive outcomes. Post-cutoff precursor rows stay IN the
supervised fit (with the exclusions below).

## 1. Risk roles: split by development history (frozen assignment)

- RISK-FIT = eligible temporal rows from H-DEV-1 (seed 100), H-DEV-2 (101),
  H-DEV-3 (102). Verified by manifest `seeds.temporal`.
- RISK-VAL = eligible temporal rows from H-DEV-4 (seed 103). Thresholds and
  operating points are selected ONLY here.
- EVAL = sealed temporal views H-SEAL-1..4 (seeds 200–203). Evaluation only;
  sealed static views are never opened.
- "Eligible temporal row": file in the manifest `test_temporal` view, with
  `future_targets` present and not censored, program != `program-03`, and no
  maintenance-window overlap.
- History reset at maintenance: `usage_h` counts operating hours SINCE the
  last recommissioning (last maintenance end ≤ file start, else history
  start); trailing 7-day file history is truncated at the last maintenance
  end (no cross-maintenance lookback).

## 2. Frozen-from-FIT / selected-from-VAL

- Fit ONLY on RISK-FIT eligible rows: file-score q90, feature Standardizer,
  logistic coefficients/intercepts, constant rate.
- Select ONLY on RISK-VAL: per-arm alert threshold at FPR ≤ 0.10 via tie-safe
  operating points (threshold sweeps distinct scores; tied blocks move as one
  unit; constant arms yield only (0,0)→(1,1)).
- Sealed data never enters fitting or selection. Exclusion counts
  (non-temporal, censored, program-03, maintenance) are recorded, not silent.

## 3. Event-level G-rank (frozen definition)

Per sealed history, per scored arm (constant arm included as reference):

- Positives: ONE score per failure event = max file score among eligible eval
  files of the same robot ending in the causal window [T−7d, T]
  (T = failure start). Events with no eligible file in-window are counted as
  unevaluable (coverage reported, never imputed).
- Negatives: same-history, same-robot control windows of equal 7d duration.
  Deterministic tiling anchored at eligible file end-times (non-maintenance,
  non-quarantined files only), greedy non-overlapping in end-time order; a
  candidate window [e−7d, e] is kept iff no failure starts in [e−7d, e+7d)
  and no member file overlaps maintenance. Score = max file score in window.
- Matching is by construction (same history + same robot); negatives pool
  across robots within one history for a single event AUC.
- Ties use standard ROC (sklearn convention); constant baseline event
  AUROC = .5 by construction.
- UNAVAILABLE (not fail, never substituted): a history with <2 evaluable
  positives or <6 negatives reports G-rank UNAVAILABLE; file-level AUROC must
  NOT be substituted for it.
- PASS per history: event AUROC ≥ 0.70 AND ≥ constant + 0.10.
- Overall G-rank PASS requires all four sealed histories PASS.

## 4. Descriptive companions (not gates)

Per sealed history and arm: file-level AUROC, event recall at the frozen VAL
threshold, first-alert lead-time median over recalled events, persistence
(median flagged files per recalled event), false-alert runs per robot-day
(alert episodes grouped at ≤2d gaps, episodes with no alert inside any
failure [T−7d, T] window count false; denominator = evaluated robot-days).
1-day horizon reported only, no gate.

## 5. Preserved invariants

program-03 cold-start holdout from FIT+VAL (evaluated descriptively only);
reserved whole mechanisms untouched; sealed static never opened; learned
geometry excluded (no upstream standing); quarantine 7 d exact; no
performance seed replacement; bulk data server-side uncommitted.
