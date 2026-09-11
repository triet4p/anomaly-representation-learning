# Sprint 9 — V1 Detection Improvement

**Goal:** Lift V1 anomaly detection from near-chance (F1 0.323, AUROC ~0.54) by fixing the reference-bank overwrite, analyzing scores and probe ceilings honestly, repairing timestep localization, ablating training, and evaluating on production mixed data — all through the Git-synchronized server workflow.

**Status:** Complete — all tracks executed, evidence-only gate passed, verdict: retrain

---

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] **Task 1 — Preserve the restored reference bank.** Change inference/extraction so a restored checkpoint bank is used as-is by default with an explicit opt-in refit flag, instead of unconditionally overwriting it with a 64-row batch fit; keep `S_pop` provenance recorded (bank rows, source) in outputs. Add focused tests proving restore-then-score uses the restored bank and the flag path refits. Record evidence in `artifacts/sprint-9/task-1.md`.
- [x] **Task 2 — Diagnose and repair timestep localization.** Trace why patch-level score traces are degenerate (near-zero with trailing constant run), fix at the source boundary, add focused tests with non-degenerate synthetic anomalies, and verify traces localize on the mixed rerun in Task 4. Record evidence in `artifacts/sprint-9/task-2.md`.
- [x] **Task 3 — Rerun server inference with the full bank.** Commit and push Tasks 1–2, pull on `trietlm@192.168.30.244`, and rerun inference directly on medium and production-head samples with the restored 8192-row bank; retain executed notebooks, logs, confusion, AUROC, per-family detection, and score means. Record evidence in `artifacts/sprint-9/task-3.md`.
- [x] **Task 4 — Score analysis and fusion.** On the Task 3 outputs compute PR curves per score and per family, sweep thresholds (Fβ operating points, not fixed MAD), and evaluate a simple `S_pred`+`S_pop` fusion; report whether any operating point separates or the curves are flat. Record evidence in `artifacts/sprint-9/task-4.md`.
- [x] **Task 5 — Supervised probe ceiling.** Freeze embeddings from the Task 3 runs and fit kNN/linear probes with the full bank on mixed data; report probe AUROC/F1 as the representation ceiling that decides scoring-fix versus retrain. Record evidence in `artifacts/sprint-9/task-5.md`.
- [x] **Task 6 — Training ablations.** Using the existing train pipeline unchanged (no Sprint 4 stabilization scope), run a small server-side ablation grid over `lambda_max`, temperature, and augmentation strength; record prediction-vs-contrastive loss splits by normal/anomaly and resulting detection deltas. All six cells exit 0 (pushed `978370b`); F1 spans 0.28–0.33, AUROC 0.54–0.56/0.50–0.52 — no cell escapes near-chance. Record evidence in `artifacts/sprint-9/task-6.md`.
- [x] **Task 7 — Production mixed-sample evaluation.** Evaluate detection on a production mixed sample with deterministic balanced pair accounting, reporting full confusion, per-family recall, margin metrics, and localization quality. Record evidence in `artifacts/sprint-9/task-7.md`.
- [x] **Task 8 — Review and close.** Run an `evidence-reviewer` gate over source fixes, tests, server reruns, analyses, and result claims; correct all actionable findings before marking the sprint complete. No `reviewer` stage is used.

## Acceptance Criteria

- Restored bank is the default scoring reference; every `S_pop` states its bank row count and source.
- Timestep traces are non-degenerate on synthetic anomalies with focused coverage.
- Server reruns execute exact pushed commits with exit 0; confusion, AUROC, family recall, and score means are reported from real outputs.
- PR/threshold/fusion analysis and the probe ceiling each give a verdict: scoring-fix or retrain.
- Ablations report loss splits and detection deltas without changing the trainer.
- Mixed production evaluation uses balanced pair accounting with non-null margins where both classes exist.
- Credentials, checkpoints, and large generated data are never committed.

## Review Gates

- [x] Tasks 1–7 completed by `hard-task` workers (predecessor Tasks 1–3 partial, fresh worker Tasks 3–7; commits `261f6f9`, `d0af9e5`, `b471dc4`, `978370b`).
- [x] Tasks 1–7 `evidence-reviewer` passed — PASS (`Sprint9ImprovementEvidence`; confidence 1.0, zero actionable findings, advance true). All verdicts converge: scoring fixes exhausted, representation must be retrained.

## Notes / Constraints

- Sprint 4 stays paused: use the trainer as-is for ablations; no stabilization changes.
- Local → GitHub push → server pull loop for every source fix before reruns; never reset/clean/force-push or touch unrelated work.
- Real notebook/training execution on the server is explicitly authorized; local focused tests use synthetic CPU fixtures.
- Tasks 3–7 verdicts converge: full-bank inference still near-chance, no threshold/fusion lifts off chance, probe ceiling AUROC ≈0.54, production mixed eval reproduces the picture, six-cell ablation grid moves F1 by ±0.03 at most — scoring fixes exhausted, representation must be retrained.
