# Sprint 9 results — V1 detection improvement (all real executions)

Consolidated summary of Sprint 9: the reference-bank fix, full-bank server reruns, score/threshold/fusion
analysis, probe ceiling, six-cell training ablations, and production mixed evaluation. No notebook was
re-executed for this summary; every number below cites its exact artifact
(`artifacts/sprint-9/task-1.md` … `task-7.md`). Source commits: `261f6f9` (Tasks 1–2) and `d0af9e5`
(Task 2 source fix); all server compute at `d0af9e5` on RTX 4060 Ti / CUDA unless noted.

Layout mapping: `02/` is the superseded predecessor partial (run at `261f6f9`, kept on server, ignored by
rule — never use its numbers); `03/` holds the authoritative reruns at the exact commit; `04/` the score
dump script; `05/` ablation training/eval scripts, per-cell histories and master logs; `06/` the Task 7
eval script and summary. Raw score archives (`*.npz`), checkpoint binaries (`ckpt_*.pt`), and long tqdm
logs stay server-side, ignored by rule.

## Contents

- `03/infer-medium.ipynb` / `03/infer-medium.executed.ipynb` — Task 3 medium rerun (exit 0, 62 s).
- `03/infer-production-head.ipynb` / `03/infer-production-head.executed.ipynb` — Task 3 production-head
  rerun (`V1_MAX_SAMPLES=5000`, exit 0, 25 s).
- `03/medium_stdout.log`, `03/production_head_stdout.log` — nbconvert exit-0 records.
- `04/dump_scores.py` — per-file S_pred/S_pop/label/family/embedding dump (exit 0, 61 s).
- `05/train_ablation.py`, `05/run_grid.sh`, `05/grid_master.log` — six-cell grid (all exit 0, GRID DONE).
- `05/loss_split.py`, `05/eval_grid.sh`, `05/eval_all.log`, `05/summarize_histories.py` — per-cell evals.
- `05/ckpt_*_history.json` (6×) — full per-epoch train/val loss histories; `05/smoke_history.json`.
- `06/task7_eval.py`, `06/task7_summary.json` — production mixed eval (exit 0, 88 s).
- No figures were produced by the Sprint 9 analyses (tables only — stated explicitly, none claimed).

## 1. Training provenance (read from actual artifacts)

- Production checkpoint `checkpoints/v1_representation_20260904_01.pt`, step **7840**, restored bank flag
  true: `reference bank restored: True`, 8192×128 rows (Tasks 3/7 outputs). Config: 6 channels, patch
  32 / stride 16, `d_model=128`, 4 sequence layers, 4 heads (`task-3.md`).
- Medium dataset `data/generated/medium`: train 25000 / val 5000 / test 20000; production dataset:
  train 100000 / val 20000 / test 100000 (`config_hash ee1ac4ba7f66`, generator 2.0.0) (`task-3.md`, `task-7.md`).
- Ablation baseline retrained from scratch on medium (B=128, 40 epochs, λ=0.1, τ=0.2, aug×1.0) reached
  step **7840** with val stationary joint loss **0.1267**, reproducing the production-checkpoint regime
  (`task-6.md`, `05/ckpt_baseline_history.json`).

## 2. Source fixes (Tasks 1–2; 19 + 17 focused tests green)

- Task 1: `prepare_reference_bank` keeps the restored checkpoint bank by default (`refit=False` returns
  `"restored"`); every rerun prints `normal train reference rows 8192 (bank source: restored)`
  (`task-1.md`, `task-3.md`). Prior runs scored against a 64-row refit bank.
- Task 2: `patch_to_timestep_scores` fail-fast coverage validation; synthetic-spike localization test
  passes; the validation never fired on live paths (coverage well-formed) (`task-2.md`, `task-3.md`).
- Provenance separation (do not mix): `S_pred` = EMA latent-prediction error (context consistency);
  `S_pop` = mean distance to up to k=5 nearest rows of the **checkpoint bank** (population distance);
  the Sprint 6 geometry `01/` within-cache distances are a different population and incomparable.

## 3. Full-bank reruns (Task 3, commit `d0af9e5`, exit 0)

- Medium test (10000 normal + 10000 abnormal), val-calibrated MAD×2.5 (`th_pred=0.1086, th_pop=5.8832`;
  score means val `S_pred 0.0745 / S_pop 4.5313`, test `0.0746 / 4.5422`): **TP=1191, FP=1013, FN=8809,
  TN=8987 → precision 0.5404, recall 0.1191, F1 0.1952; AUROC S_pred 0.5235, S_pop 0.5034** (`task-3.md`,
  `03/infer-medium.executed.ipynb`).
- Per-family recall (of ~1111 each): duration_anomaly 236/1111 (21.2%) best; all others 103–145 (~10–13%)
  (`task-3.md`). Timestep traces sparse-by-construction with trailing constant runs on saturated files.
- Production head (5000 files, all normal): same operating range (`S_pred 0.0732 / S_pop 4.5448`),
  evaluation skipped by design — mixed production eval is §7 (`task-3.md`).

## 4. Threshold sweep and fusion (Task 4; 25000 dumped files, balanced val/test)

- PR: `S_pred` AUROC 0.5235 / AP 0.5205; `S_pop` 0.5034 / 0.5033 (chance AP 0.50). Per-family one-vs-normal:
  only duration_anomaly exceeds 0.55 (`S_pred` 0.5998); all others 0.48–0.54 (`task-4.md`).
- Val-picked thresholds on test: `S_pred` F1 **0.5152** (β=1.0/2.0), `S_pop` F1 0.5035 — the fixed MAD×2.5
  point (F1 0.1952) sits far in the tail, but even the optimal point is ~coin-flip on balanced data.
  Cohen's d: `S_pred` 0.077, `S_pop` 0.013 (`task-4.md`).
- Fusion: standardized sum AUROC 0.5137 / F1 0.5107; logistic coef [0.1509, −0.0098] (≈ zero weight on
  `S_pop`), AUROC 0.5244 — no complementary signal. Verdict: scoring fixes exhausted (`task-4.md`).

## 5. Probe ceiling (Task 5; frozen 128-d embeddings, train val → eval test)

- kNN: k=1/5/15/50 AUROC 0.5043/0.5074/0.5014/0.4986 (chance); centroid-distance 0.5065 (`task-5.md`).
- Logistic (standardized): C=0.01/0.1/1.0/10.0 → AUROC 0.5318/0.5390/0.5409/**0.5410**, F1 ≈ 0.523.
  Per-family (C=1.0): duration_anomaly 0.6619, missing_event 0.5610, over_regularity 0.5532, rest ≤0.525.
- Verdict: the frozen representation does not encode anomaly separability — **retrain** (`task-5.md`).

## 6. Training ablations (Task 6; trainer unchanged, 6/6 exit 0, 1437–1571 s each)

| cell | config | train pred/cont | val pred/cont | split N pred/cont | split A pred/cont | F1 | aucP/aucO |
|---|---|---|---|---|---|---|---|
| prod-ckpt | (as shipped) | — | — | 0.0727/1.3716 | 0.0784/1.3674 | 0.1950 | 0.5235/0.5034 |
| baseline | λ=0.1 τ=0.2 aug×1 | 0.1283/0.7287 | 0.0568/0.6994 | 0.0542/0.4169 | 0.0627/0.4185 | 0.3096 | 0.5410/0.5242 |
| lambda0 | λ=0.0 | 0.1365/2.3124 | 0.0570/2.2051 | 0.0540/1.5668 | 0.0613/1.5681 | 0.2816 | 0.5390/0.5017 |
| lambda05 | λ=0.5 | 0.1284/0.7066 | 0.0585/0.6836 | 0.0557/0.4070 | 0.0659/0.4087 | 0.3261 | 0.5421/0.5194 |
| tau007 | τ=0.07 | 0.1561/0.0027 | 0.0692/0.0022 | 0.0667/0.0010 | 0.0760/0.0012 | 0.3117 | 0.5466/0.5147 |
| aug0 | aug×0 | 0.1219/0.7043 | 0.0488/0.6769 | 0.0469/0.4013 | 0.0535/0.4032 | 0.2973 | 0.5552/0.4985 |
| aug2 | aug×2 | 0.1319/0.7348 | 0.0620/0.7038 | 0.0597/0.4220 | 0.0687/0.4233 | 0.3204 | 0.5418/0.5233 |

Detection: MAD×2.5 val-calibrated (same convention as §3); full confusion per cell in `task-6.md`.
Prediction loss keeps a +13–18% normal/abnormal gap in every cell; contrastive never separates (≤0.5%).
No cell escapes near-chance (F1 0.28–0.33): λ/τ/augmentation move detection ±0.03. Every fresh cell beats
the shipped checkpoint (0.28+ vs 0.20) yet stays unusable — tuning is insufficient (`task-6.md`,
`05/ckpt_*_history.json`, `05/eval_all.log`).

## 7. Production mixed evaluation (Task 7, exit 0, 88 s)

- Production test scan: 50000 normal + 50000 abnormal (~5555/family). Deterministic sample: first 400 per
  family (3600 abnormal) + first 3600 normal, fixed-seed interleave; thresholds from production val
  (`th_pred=0.1051, th_pop=5.8177`) (`task-7.md`, `06/task7_summary.json`).
- Mixed n=7200: **TP=528, FP=406, FN=3072, TN=3194 → precision 0.5653, recall 0.1467, F1 0.2329;
  AUROC S_pred 0.5248, S_pop 0.4991**; margins d=0.0917/0.0070 (non-null, tiny) (`task-7.md`).
- Per-family recall: duration_anomaly 105/400 (26.3%) best; others 45–66 (11–17%). Localization vs
  ground-truth masks: argmax-in-mask **0.1933**, top-10% mass **0.3318** (weak; traces non-degenerate,
  mean nonzero frac 0.60) (`task-7.md`).

## Verdict: retrain (unanimous across Tasks 4–7)

Restored-bank inference is correct end-to-end, but detection is near-chance on medium and production,
under every threshold/fusion, under supervised probes (ceiling AUROC 0.54), and across all six training
ablations with the trainer fixed. Production-grade detection needs representation-level changes
(objective/architecture/signal) — recorded for the next sprint. Figures: Sprint 9 analyses produced no
PNGs; embedding-geometry context lives in `01/assets/` (12 figures, different provenance — see `01/README.md`).

## Limitations (explicit)

- One run per ablation cell, single seed family (train seeds 3/4, scoring 5/6/7), one GPU type for
  compute (RTX 4060 Ti; analysis on local CPU sklearn).
- Production val head is normal-only; thresholds transfer the medium MAD×2.5 convention.
- Task 3 production head and Task 5 probes use head/medium samples, not full production test (full scan
  only for Task 7 accounting).
- The 1-file confusion difference between Task 3 notebook (TP 1191) and Task 6 dump rescoring (TP 1190)
  is GPU floating-point nondeterminism across batching paths — immaterial (ΔF1 0.0002).
- Superseded `02/` numbers must never be cited; raw `*.npz`/`*.pt` remain server-side and uncommitted.
