# Sprint 3 — CLI Version Reporting

**Goal:** Add a small, side-effect-free version-reporting feature to both supported synthetic-data CLI entry points.

**Status:** Complete

---

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] **Task 1 — Add CLI `--version` reporting.** Make `python -m synth.cli --version` and the installed `synth-generate --version` entry point print the same project version and exit successfully without loading generation configuration, creating output, or running dataset generation. Reuse the project's authoritative package-version metadata rather than introducing a second version constant. Preserve existing generation behavior. Add or update focused behavioral coverage, run the specific version and regression smoke commands, update this task and sprint status, and write `artifacts/task_1_summary.md` with the changed scope and exact verification evidence.

## Review Gates

- [x] Task 1 or its explicitly tagged batch has a passing `evidence-reviewer` review — PASS (`Sprint3Evidence`; no findings).
- [x] The complete sprint has a passing `reviewer` review with no unresolved actionable findings — PASS (`Sprint3FinalReview`; confidence 0.96; no findings).

## Notes / Blockers

- This intentionally small sprint is an end-to-end smoke test of the `omp-subagent-flows` orchestration skill.
- No blockers at planning time.
