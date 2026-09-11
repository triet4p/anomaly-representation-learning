# Sprint 8 — V1 Results Summary

**Goal:** Consolidate every real V1 result (training checkpoint provenance, executed inference, geometry extraction/analysis) into one reviewed `README.md` with local figure assets, and rename the run folder to a clean numeric layout.

**Status:** Complete — rename, assets, README, and evidence-only gate passed

---

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] **Task 1 — Rename run folder and assemble assets.** Rename `experiments/20260905/server-geometry-run-1/` to `experiments/20260905/01/` via `git mv` for tracked files (plain `mv` for untracked ones), create `01/assets/`, and place all 12 geometry figures plus any inference figures there; fetch missing images read-only from `trietlm@192.168.30.244` via PowerShell OpenSSH key auth when absent locally. Keep large raw outputs (embeddings, neighbors CSV, diagnostics bundle) and the prior-run `-prev` copy server-side. Record evidence in `artifacts/sprint-8/task-1.md`.
- [x] **Task 2 — Write the V1 summary README.** Write `experiments/20260905/01/README.md` summarizing training provenance (checkpoint step/hash/config/reference bank, dataset identity), executed inference results (counts, precision/recall/F1, AUROC, per-family detection, score means, bank caveat, staleness verdict), and geometry results (extraction manifest, health metrics, neighborhoods, separation limits, resource/timing), with every number cited to its exact artifact path and every figure embedded from `assets/`. State limitations explicitly. Record evidence in `artifacts/sprint-8/task-2.md`.
- [x] **Task 3 — Review and close.** Run an `evidence-reviewer` gate over the rename scope, asset completeness, README accuracy against source artifacts, and link integrity; correct all actionable findings before marking the sprint complete. No `reviewer` stage is used.

## Acceptance Criteria

- Credentials, checkpoint binaries, and large generated data are never committed.

## Review Gates

- [x] Tasks 1–2 assigned to and completed by retained `hard-task` worker (uncommitted working-tree changes, per plan).
- [x] Tasks 1–2 `evidence-reviewer` passed — PASS (`Sprint8SummaryEvidence`; confidence 1.0, zero actionable findings, advance true). Evidence-only gate; no `reviewer` stage run. One INFO note: the 11 fetched PNGs, README, and untracked inference notebook remain unstaged in the working tree, ready for the user to commit.
- [x] Sprint-wide status updated and plan closed.

## Notes / Constraints

- Sprint 6 evidence files keep their historical `server-geometry-run-1/` paths; the README records the rename mapping.
- Treat unrelated uncommitted changes as user work; never reset, clean, force-push, or overwrite.
- Prefer key auth over SSH; never persist credentials in commands, files, logs, or evidence.
