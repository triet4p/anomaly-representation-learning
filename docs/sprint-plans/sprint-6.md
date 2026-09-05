# Sprint 6 — Server-Native Geometry Execution

**Goal:** Replace the Kaggle export workflow with a Git-synchronized server workflow that executes the geometry notebooks against repository source, `checkpoints/`, and `experiments/` on `trietlm@192.168.30.244` and returns real run artifacts for review.

**Status:** In progress — retained hard-task worker assigned

---

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [~] **Task 1 — Remove Kaggle export workflow.** Delete all `exports/kaggle-*` directories and ZIP files, remove source-ZIP packaging and active documentation/test assumptions, and preserve unrelated exports or user files. Record evidence in `artifacts/sprint-6/task-1.md`.
- [ ] **Task 2 — Convert active notebooks to server paths.** Make canonical train, inference, geometry extraction, and geometry analysis notebooks use explicit editable repository `src/`, local materialized-data, `checkpoints/`, cache, and `experiments/` paths with fail-fast validation and no Kaggle mount/source-archive logic. Keep notebook cells clean until the server run copies execute. Record evidence in `artifacts/sprint-6/task-2.md`.
- [ ] **Task 3 — Push a coherent source commit.** Stage only the intended Sprint 5/6 source, notebook, test, plan, and Kaggle-removal changes; preserve unrelated user work; commit them on the current branch; and push the exact commit to the configured GitHub remote. Record branch, commit, remote, and evidence in `artifacts/sprint-6/task-3.md` without recording credentials.
- [ ] **Task 4 — Pull the commit on the server.** From Windows PowerShell, SSH to `trietlm@192.168.30.244`, locate the existing repository without destructive search/reset, preserve remote user changes, and update a clean runnable checkout to the exact pushed commit using fast-forward pull or a safe separate checkout when required. Record environment, repository path, commit, and GPU/runtime evidence in `artifacts/sprint-6/task-4.md`.
- [ ] **Task 5 — Execute geometry extraction remotely.** Use the server checkout's repository source, a checkpoint under `checkpoints/`, the local materialized dataset, and an execution copy under `experiments/20260905/server-geometry-run-1/`; run the extraction notebook directly to completion on the available device and retain its executed notebook, logs, and complete geometry cache. Fix any source/notebook runtime failure through the same local commit → GitHub push → server pull loop before retrying. Record evidence in `artifacts/sprint-6/task-5.md`.
- [ ] **Task 6 — Execute geometry analysis remotely.** Run the analysis notebook directly against the Task 5 cache, write outputs under the same experiment directory, and retain the executed notebook, metrics, neighbors, figures, diagnostic ZIP, logs, timings, and resource evidence. Fix runtime failures through the same Git workflow before retrying. Record evidence in `artifacts/sprint-6/task-6.md`.
- [ ] **Task 7 — Collect and interpret server results.** Copy the bounded executed notebooks, manifests, metrics, logs, and diagnostic report needed for local review into the repository experiment directory without committing large generated data; summarize representation-health, neighborhood, separation, score, and resource results with exact artifact paths. Record evidence in `artifacts/sprint-6/task-7.md`.
- [ ] **Task 8 — Review and close the server gate.** Run an evidence review over deletion scope, commit/push/pull coherence, executed notebook status, output integrity, and result interpretation; correct all actionable findings through the retained worker before marking the sprint complete. Record the gate in this plan.

## Acceptance Criteria

- No `exports/kaggle-*` directory or ZIP remains, and active workflows no longer create or require `src.zip`.
- Active notebooks expose explicit editable server/local paths and use repository `src/`, `checkpoints/`, and `experiments/`; no Kaggle mount discovery or packaging path remains.
- The GitHub remote contains the exact reviewed source commit, and the server executes that exact commit without resetting or overwriting unrelated local work.
- The extraction notebook completes through final artifact publication using the real server dataset/checkpoint and produces a valid geometry cache.
- The analysis notebook completes against that cache and produces valid metrics, neighbors, figures, and a downloadable diagnostic bundle under `experiments/20260905/server-geometry-run-1/`.
- The hard-task worker returns concrete command status, commit identity, timings, device information, output paths, and scientific summary; failures are fixed and rerun rather than hidden.
- Credentials and checkpoint binaries are never committed or written into evidence artifacts.

## Review Gates

- [~] Tasks 1–7 implementation and remote execution assigned to retained `hard-task` worker.
- [ ] Tasks 1–7 evidence review passes with zero actionable findings.
- [ ] Sprint-wide server execution gate passes and plan status is updated.

## Notes / Constraints

- Sprint 5's Kaggle package workflow is superseded and must not be regenerated.
- Use PowerShell OpenSSH from the workstation as requested. Prefer existing authenticated key/session; never persist credentials in commands, files, logs, Git history, or evidence.
- Treat local and remote uncommitted changes as user work. Never reset, clean, force-push, or overwrite them.
- The server run may execute real notebooks and inference because the user explicitly requested a direct trial on the authenticated server.
