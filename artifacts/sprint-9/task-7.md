# Sprint 9 Task 7 — Production mixed-sample evaluation

## Method (deterministic balanced pair accounting)

- Script `experiments/20260905/06/task7_eval.py` on the server at commit `d0af9e5`:
  `.venv/bin/python experiments/20260905/06/task7_eval.py` → **exit 0, 88 s** (RTX 4060 Ti).
- Production test scan (in-order, all 100000 files): **50000 normal + 50000 abnormal**;
  family totals ~5555 each (contextual_replacement 5556, cross_channel_inconsistency 5555,
  duration_anomaly 5555, freq_phase_mismatch 5555, missing_event 5555, over_regularity 5556,
  realistic_stuck 5556, subtle_drift 5556, wrong_transition 5556).
- Sample: first **400 per family** (3600 abnormal) + first **3600 normal** in dataset order
  (exact 1:1 balance asserted in-script), then a fixed-seed interleave (labels retained).
- Model: production checkpoint step 7840 with `bank rows 8192 source restored`.
- Thresholds calibrated on production val head (5000 files, MAD ×2.5 — same convention as Task 3):
  `th_pred=0.1051, th_pop=5.8177` (val means `S_pred 0.0715 / S_pop 4.5074`).
- Artifacts: `experiments/20260905/06/task7_mixed.npz` + `task7_summary.json` (server-side, untracked).

## Results (mixed n=7200: 3600 normal + 3600 abnormal)

- Confusion (MAD×2.5 OR rule): **TP=528, FP=406, FN=3072, TN=3194**;
  **Precision 0.5653, Recall 0.1467, F1 0.2329**; **AUROC S_pred=0.5248, S_pop=0.4991**.
- Margins (non-null, tiny): `S_pred` abn-mean 0.0758 vs nor-mean 0.0729, **d=0.0917**;
  `S_pop` 4.5469 vs 4.5419, **d=0.0070**.
- Per-family recall: contextual_replacement 52/400 (0.1300), cross_channel_inconsistency 47/400 (0.1175),
  **duration_anomaly 105/400 (0.2625)**, freq_phase_mismatch 45/400 (0.1125), missing_event 66/400 (0.1650),
  over_regularity 52/400 (0.1300), realistic_stuck 52/400 (0.1300), subtle_drift 61/400 (0.1525),
  wrong_transition 48/400 (0.1200) — flat except duration_anomaly, matching Tasks 3–5.
- Localization vs ground-truth `anomaly_mask` (3600 abnormal files): **argmax-in-mask 0.1933**,
  **top-10% score mass in mask 0.3318**; traces non-degenerate (3600/3600, mean nonzero frac 0.60,
  mean length 505.3). Localization is weak: peaks land outside the true mask ~80% of the time.

## Verdict: retrain

Production mixed evaluation reproduces the medium-dataset picture (F1 0.23 vs 0.20, AUROCs ~0.52/0.50,
flat families, weak localization). The failure generalizes across datasets — scoring fixes cannot close
it; the representation must be retrained (pending Task 6 ablation deltas).
