# Sprint 10 Task 2 — Sprint 9 summary README

## What was written

- `experiments/20260905/SPRINT9-RESULTS.md` (9116 bytes) in `01/README.md` style: contents inventory,
  §§1–7 (provenance, fixes, reruns, threshold/fusion, probes, six-cell ablation table, mixed eval),
  unanimous retrain verdict, figures statement, explicit limitations, layout mapping (`02/` superseded,
  raw `*.npz`/`*.pt` server-side).
- Every number traces to `artifacts/sprint-9/task-1..7.md` or the committed `03/05/06` records;
  `S_pred`/`S_pop`/within-cache provenance separated (§2 + Limitations); no new figures — analyses
  produced none (stated in Contents + Verdict); geometry context linked to committed `01/assets/`
  (different provenance, per `01/README.md`).
- Verified at write time: ablation table cells match `task-6.md` + `eval_all.log` values; Task 3/4/5/7
  headline numbers match their evidence files; the 1-file Task 3 vs Task 6 rescoring difference
  (TP 1191 vs 1190) disclosed as GPU nondeterminism in Limitations.

## Commit/push/pull

- Committed with this evidence as `5c3767d`, pushed; server `git pull --ff-only` to the same
  hash; `git status --porcelain` clean on both sides except the pre-existing unrelated entries recorded
  in `task-1.md` (re-verified after pull).
