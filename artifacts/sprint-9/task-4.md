# Sprint 9 Task 4 — Score analysis and fusion

## Source

- Scores: `experiments/20260905/04/scores_medium.npz` (12,085,731 bytes), dumped on the server at
  commit `d0af9e5` via `experiments/20260905/04/dump_scores.py`
  (`.venv/bin/python experiments/20260905/04/dump_scores.py` → **exit 0, 61 s**, RTX 4060 Ti):
  restored 8192-row bank (`bank rows 8192 source restored step 7840`), deterministic seeds 5/6/7,
  medium val (5000 files, 2500 abnormal) + test (20000 files, 10000 abnormal), all 9 anomaly families.
  Means reproduce Task 3 exactly (val `S_pred 0.0745 / S_pop 4.5313`, test `0.0746 / 4.5422`).
- Analysis: offline (local CPU, scikit-learn 1.9.0) PR/threshold/fusion sweep over the dumped arrays.

## PR curves (test; chance AP = 0.5000)

- `S_pred`: AUROC **0.5235**, AP **0.5205**. `S_pop`: AUROC **0.5034**, AP **0.5033**. Both curves hug chance.
- Per-family one-vs-normal AUROC (`S_pred` / `S_pop`): contextual_replacement 0.5098/0.5095,
  cross_channel_inconsistency 0.5156/0.5094, **duration_anomaly 0.5998/0.5320** (only family above 0.55),
  freq_phase_mismatch 0.4944/0.5157, missing_event 0.5449/0.4834, over_regularity 0.5144/0.4981,
  realistic_stuck 0.5049/0.4944, subtle_drift 0.5138/0.4883, wrong_transition 0.5140/0.4997.
  No operating point separates; only gross-duration anomalies leak weak signal into `S_pred`.

## Fβ threshold sweep (threshold picked on val, reported on test)

- `S_pred`: β=1.0 → val thr 0.0692, test **F1 0.5152** (P 0.5154 R 0.5151); β=2.0 → test F 0.5152;
  β=0.5 → test F 0.5151. (Contrast: the fixed MAD×2.5 operating point from Task 3 scored F1 0.1952 —
  the MAD threshold sits far in the tail and destroys recall; a calibrated threshold recovers the
  maximum the score carries, which is still ~chance.)
- `S_pop`: β=1.0 → test **F1 0.5035**; β=2.0 → 0.5044; β=0.5 → 0.5022. Pure chance.
- Separation (Cohen's d, test): `S_pred` **0.077**, `S_pop` **0.013** — negligible.

## Fusion `S_pred` + `S_pop` (standardized on val, evaluated on test)

- Sum-fusion: AUROC **0.5137**, AP 0.5158; val-picked F1 point → test **F1 0.5107**.
- Logistic fusion (val-fit): coef `[0.1509, -0.0098]` (S_pop weight ≈ 0), AUROC **0.5244**, AP 0.5209 —
  identical to `S_pred` alone. The two scores carry no complementary signal.

## Verdict: retrain (from this analysis)

No threshold, β-weighting, or fusion lifts detection off chance: best test F1 ≈ 0.515 on balanced data
(where a coin flip scores 0.50), AUROCs ≤ 0.524, Cohen's d ≤ 0.08. Scoring-side fixes are exhausted;
whether any supervised head can do better is Task 5's probe ceiling.
