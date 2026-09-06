# Sprint 6 — Server-Native Geometry Execution and Inference Result Review

**Goal:** Replace the Kaggle export workflow with a Git-synchronized server workflow that executes the geometry notebooks against repository source, `checkpoints/`, and `experiments/` on `trietlm@192.168.30.244` and returns real run artifacts for review; then read and interpret the executed inference notebook in the same run directory.

**Status:** Complete — inference result review passed evidence-only gate

---

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] **Task 1 — Remove Kaggle export workflow.** Delete all `exports/kaggle-*` directories and ZIP files, remove source-ZIP packaging and active documentation/test assumptions, and preserve unrelated exports or user files. Record evidence in `artifacts/sprint-6/task-1.md`.
- [x] **Task 2 — Convert active notebooks to server paths.** Make canonical train, inference, geometry extraction, and geometry analysis notebooks use explicit editable repository `src/`, local materialized-data, `checkpoints/`, cache, and `experiments/` paths with fail-fast validation and no Kaggle mount/source-archive logic. Keep notebook cells clean until the server run copies execute. Record evidence in `artifacts/sprint-6/task-2.md`.
- [x] **Task 3 — Push a coherent source commit.** Stage only the intended Sprint 5/6 source, notebook, test, plan, and Kaggle-removal changes; preserve unrelated user work; commit them on the current branch; and push the exact commit to the configured GitHub remote. Record branch, commit, remote, and evidence in `artifacts/sprint-6/task-3.md` without recording credentials.
- [x] **Task 4 — Pull the commit on the server.** From Windows PowerShell, SSH to `trietlm@192.168.30.244`, locate the existing repository without destructive search/reset, preserve remote user changes, and update a clean runnable checkout to the exact pushed commit using fast-forward pull or a safe separate checkout when required. Record environment, repository path, commit, and GPU/runtime evidence in `artifacts/sprint-6/task-4.md`.
- [x] **Task 5 — Execute geometry extraction remotely.** Use the server checkout's repository source, a checkpoint under `checkpoints/`, the local materialized dataset, and an execution copy under `experiments/20260905/server-geometry-run-1/`; run the extraction notebook directly to completion on the available device and retain its executed notebook, logs, and complete geometry cache. Fix any source/notebook runtime failure through the same local commit → GitHub push → server pull loop before retrying. Record evidence in `artifacts/sprint-6/task-5.md`.
- [x] **Task 6 — Execute geometry analysis remotely.** Run the analysis notebook directly against the Task 5 cache, write outputs under the same experiment directory, and retain the executed notebook, metrics, neighbors, figures, diagnostic ZIP, logs, timings, and resource evidence. Fix runtime failures through the same Git workflow before retrying. Record evidence in `artifacts/sprint-6/task-6.md`.
- [x] **Task 8 — Bound and balance geometry analysis.** Bound normal-reference distance computation across query and reference dimensions without materializing a broadcasted three-dimensional matrix, and allocate deterministic pair budgets separately to same-class and different-class pairs so ordered data cannot suppress separation metrics. Add focused numerical, determinism, and memory-shape coverage. Record evidence in `artifacts/sprint-6/task-8.md`.
- [x] **Task 9 — Restore independent manifest validation.** Preserve legacy manifests that lack fleet metadata while independently requiring the supported format and a valid `splits` mapping, raising `GeometryCompatibilityError` for malformed inputs and retaining per-batch fleet-range guards. Record evidence in `artifacts/sprint-6/task-9.md`.
- [x] **Task 10 — Remove residual Kaggle source bundle.** Delete the packaging-only `data/generated/kaggle-20260903-01/src.zip`, extracted `src/` mirror, and Kaggle metadata without deleting the retained materialized production dataset. Record evidence in `artifacts/sprint-6/task-10.md`.
- [x] **Task 11 — Correct execution evidence.** Reconcile the generated figure count as 12 and correct rank, normal-reference-distance, same-class-similarity, and `S_pop` provenance statements in Tasks 6–7 evidence and result summaries. Record evidence in `artifacts/sprint-6/task-11.md`.
- [x] **Task 12 — Push, pull, and rerun corrected notebooks.** Commit and push Tasks 8–11, pull the exact commit on the server, rerun extraction and analysis directly, refresh the bounded local result set, and report the corrected commit, timings, artifacts, and metrics. Record evidence in `artifacts/sprint-6/task-12.md`.
- [x] **Task 13 — Review and close the server gate.** Re-run task-level evidence and sprint-wide review over deletion scope, commit/push/pull coherence, executed notebook status, output integrity, and result interpretation; correct all actionable findings before marking the sprint complete.
- [x] **Task 14 — Establish inference run provenance.** Inspect the executed notebook `experiments/20260905/server-geometry-run-1/infer-v1-representation.executed.ipynb` without re-executing it: code-cell execution counts, error outputs, kernel/timing metadata, configured `SRC_DIR`/dataset/checkpoint/output paths, checkpoint identity (step/hash) and dataset manifest identity; determine which source commit produced it relative to corrected commit `a326e48` and whether its outputs are stale relative to the Sprint 6 rerun. Record evidence in `artifacts/sprint-6/task-14.md`.
- [x] **Task 15 — Extract and validate inference results.** Read every executed cell output, log and metric: per-split record counts, `S_pred`/`S_pop` distributions and thresholds, detection/separation behavior, timestep localization where present, figures/tables, warnings and failures; validate internal consistency (counts, shapes, NaN/finiteness, score provenance) and flag anything meaningless such as random-weight fallback or missing-checkpoint defaults. Record evidence in `artifacts/sprint-6/task-15.md`.
- [x] **Task 16 — Cross-check against geometry run and close.** Compare inference scores and reference-bank behavior with the reviewed Sprint 6 geometry cache/analysis, reconcile `S_pop` provenance, state what the inference run does and does not establish about anomaly separability, and list exact rerun conditions if the executed notebook is stale or invalid. Record evidence in `artifacts/sprint-6/task-16.md`.

## Acceptance Criteria

- No `exports/kaggle-*` directory or ZIP remains, and active workflows no longer create or require `src.zip`.
- Active notebooks expose explicit editable server/local paths and use repository `src/`, `checkpoints/`, and `experiments/`; no Kaggle mount discovery or packaging path remains.
- The GitHub remote contains the exact reviewed source commit, and the server executes that exact commit without resetting or overwriting unrelated local work.
- The extraction notebook completes through final artifact publication using the real server dataset/checkpoint and produces a valid geometry cache.
- The analysis notebook completes against that cache and produces valid metrics, neighbors, figures, and a downloadable diagnostic bundle under `experiments/20260905/server-geometry-run-1/`.
- The hard-task worker returns concrete command status, commit identity, timings, device information, output paths, and scientific summary; failures are fixed and rerun rather than hidden.
- Credentials and checkpoint binaries are never committed or written into evidence artifacts.
- Normal-reference distance work is bounded in both query and reference dimensions, and capped class-pair diagnostics deterministically preserve both pair classes when both exist.

## Review Gates

- [x] Tasks 1–7 implementation and remote execution completed by retained `hard-task` worker.
- [x] Tasks 1–7 `evidence-reviewer` passed — PASS (`Sprint6ServerEvidence`; high confidence, zero actionable findings, `advance_to_sprint_gate=true`).
- [x] First sprint-wide `reviewer` completed — FAIL (`Sprint6FinalReview`; six actionable bounded-memory, sampling, schema, deletion, and evidence findings; `sprint_complete=false`).
- [x] Tasks 8–12 corrections and corrected remote rerun assigned to and completed by the retained `hard-task` worker.
- [x] Corrected Tasks 8–12 evidence review passed — PASS (`Sprint6CorrectionEvidence`; confidence 0.99, zero actionable findings, `advance_to_sprint_review=true`).
- [x] Tasks 14–16 inference result review completed by retained `hard-task` worker (read-only; no rerun, no source changes).
- [x] Tasks 14–16 `evidence-reviewer` passed — PASS (`Sprint6InferenceEvidence`; confidence 1.0, zero actionable findings, advance flag true). Evidence-only gate per user instruction; no `reviewer` stage run.

## Notes / Constraints

- Sprint 5's Kaggle package workflow is superseded and must not be regenerated.
- Use PowerShell OpenSSH from the workstation as requested. Prefer existing authenticated key/session; never persist credentials in commands, files, logs, Git history, or evidence.
- Treat local and remote uncommitted changes as user work. Never reset, clean, force-push, or overwrite them.
- The server run may execute real notebooks and inference because the user explicitly requested a direct trial on the authenticated server.
