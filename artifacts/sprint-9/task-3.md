# Sprint 9 Task 3 — Server inference rerun with the restored bank

## Resume state (what was found, what was adopted vs redone)

- Local `HEAD == origin/master == 261f6f9` (Tasks 1–2 committed and pushed — verified, no push needed).
- Gap found: `261f6f9` committed the Task 2 *test* (`tests/synth/test_patchify.py`) but **not** the
  Task 2 *source fix* (`src/synth/patchify.py` validation was still an uncommitted workdir change;
  `git show 261f6f9 --stat` lists no `src/synth/` file). Focused tests on the workdir passed (17 passed).
- Fix: committed **only** `src/synth/patchify.py` as `d0af9e5` ("fix(synth): validate patch-to-timestep
  coverage…"), pushed (`261f6f9..d0af9e5 master -> master`). All other dirty/untracked local work
  (CHANGELOG, PLAN, sprint-6 docs, `artifacts/task_*` deletions, other agents' `artifacts/sprint-*/`
  and `experiments/` trees) was left untouched.
- Server `trietlm@192.168.30.244:~/anomaly-representation-learning` was already at `261f6f9`.
  Predecessor's partial state `experiments/20260905/02/` (medium + production-head execution copies and
  executed notebooks, no `task-3.md`) was inspected: both executed notebooks already used
  `prepare_reference_bank` with `normal train reference rows 8192 (bank source: restored)` — adopted as
  valid, then **redone at the exact new commit** (see below) so evidence cites one exact hash.
- Server pulled fast-forward to `d0af9e5` (`git pull --ff-only`: `src/synth/patchify.py 25 +++---`).
  Execution copies `experiments/20260905/02/infer-{medium,production-head}.ipynb` match canonical
  `notebooks/infer_v1_representation.ipynb` (same `prepare_reference_bank`/`REFIT_REFERENCE_BANK` cells;
  server-path + dataset-root cells differ by design). Reruns went to `experiments/20260905/03/`
  (copies of the `02/` execution notebooks) to keep the predecessor record intact.

## Server reruns (exact commit `d0af9e5`, exit 0)

- Medium (full medium dataset, no sample cap):
  `.venv/bin/jupyter nbconvert --to notebook --execute experiments/20260905/03/infer-medium.ipynb
  --output-dir experiments/20260905/03 --output infer-medium.executed.ipynb
  --ExecutePreprocessor.timeout=3600` → **exit 0, 62 s**
  (one earlier attempt exited 1 on a bad `--output` path only — notebook body had executed fine;
  rerun with `--output-dir` fixed it; no source change).
- Production head (`V1_MAX_SAMPLES=5000`, same head as predecessor):
  `V1_MAX_SAMPLES=5000 .venv/bin/jupyter nbconvert --to notebook --execute
  experiments/20260905/03/infer-production-head.ipynb --output-dir experiments/20260905/03
  --output infer-production-head.executed.ipynb --ExecutePreprocessor.timeout=3600` → **exit 0, 25 s**.
- GPU: NVIDIA GeForce RTX 4060 Ti, CUDA. Checkpoint `checkpoints/v1_representation_20260904_01.pt`
  (step 7840, `reference bank restored: True`) in both runs.

## Results — medium (test: 10000 normal + 10000 abnormal, val-calibrated MAD ×2.5)

- Bank: `normal train reference rows 8192 (bank source: restored)` — the Task 1 fix holds on-server.
- Score means: val `S_pred mean=0.0745 / S_pop mean=4.5313`, test `S_pred mean=0.0746 / S_pop mean=4.5422`;
  thresholds `th_pred=0.1086, th_pop=5.8832`. Numbers are bit-identical to the `02/` run (fixed seeds 5/6/7).
- Confusion (OR fusion of the two MAD decisions): **TP=1191, FP=1013, FN=8809, TN=8987**;
  **Precision 0.5404, Recall 0.1191, F1 0.1952**; **AUROC S_pred=0.5235, S_pop=0.5034**.
- Per-family recall (detected/total): contextual_replacement 126/1112, cross_channel_inconsistency 116/1111,
  duration_anomaly 236/1111, freq_phase_mismatch 103/1111, missing_event 145/1111, over_regularity 120/1111,
  realistic_stuck 109/1111, subtle_drift 122/1111, wrong_transition 114/1111 — flat across families
  (~10–21%), no family separates.
- Timestep traces on real files still show sparse leading zeros + trailing constant runs (by construction
  only masked patches score nonzero; long constant runs = saturated predictions/targets). The new
  `patch_to_timestep_scores` validation did **not** raise on the live path — coverage is well-formed.

## Results — production head (V1_MAX_SAMPLES=5000)

- Bank: `8192 (bank source: restored)`. Score means: val `S_pred 0.0721 / S_pop 4.5105`,
  test `S_pred 0.0732 / S_pop 4.5448` — same operating range as medium.
- Test head sampled **5000 normal, 0 abnormal** → `evaluation metrics skipped` (by notebook design).
  Mixed production evaluation with balanced pair accounting is Task 7's job.

## Verdict / handoff

- Restored-bank inference is confirmed end-to-end on-server at `d0af9e5` with provenance in outputs.
- Detection with the restored bank is still near-chance (F1 0.1952, AUROCs ~0.52/0.50, flat family recall):
  the bank overwrite was real but **not** the detection bottleneck. Tasks 4–5 decide scoring-fix vs retrain.
