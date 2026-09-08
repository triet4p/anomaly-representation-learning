# Sprint 13 — Measurable Early-Warning Benchmark

## Sprint Goal

Create and freeze a synthetic telemetry benchmark on which causal early-warning ranking, positive lead time, false-alert behavior, and later calibrated-risk evaluation are structurally measurable before any representation or risk-model optimization resumes.

**Status:** Active — REMEDIATION R2B done (C2 renewed-accepted; Tasks 12–14 renewed: 11/13 STRUCT-PASS with 2 abrupt-category misses, E1–E5 computable on 9 non-sealed roles; D1 re-review pending; verdict superseded; Sprint 14 BLOCKED).

## Decision Context

Sprint 12 isolated two independent problems:

1. The learned representation-to-score pipeline does not retain or exploit observable anomaly signal as well as the diagnostic handcrafted baseline.
2. The temporal benchmark is not identifiable for early-warning evaluation: failures are too dense, clean control windows are scarce, clean pre-failure baselines cover only a small minority of failures, and weak/abrupt categories are absent.

Sprint 13 owns only the second problem. Its question is: **“Is the intended early-warning problem measurable under a frozen, independent benchmark contract?”** It must not optimize the representation, geometry, anomaly scorer, survival model, or alert threshold.

The benchmark is not considered generally invalid: it remains useful for simulator causality, static anomaly probes, and failure-adjacent diagnostics. This sprint specifically repairs its fitness for event-level early warning and risk evaluation.

## Non-Negotiable Scientific Constraints

- Preserve all Sprint 11 and Sprint 12 evidence and frozen artifacts unchanged.
- Define estimands, partitions, cohort floors, failure physics, and success/stop rules before generating or inspecting new benchmark outcomes.
- Use multiple independent histories. Never select or reject seeds using anomaly-model, representation, warning, or calibration performance.
- Separate generator validation, development, model-fit, calibration/operating-point selection, confirmation, and sealed evaluation roles by entire history wherever feasible.
- Keep sealed histories unavailable to model selection, feature selection, threshold selection, or component attribution.
- Include progressive, weak-precursor, and abrupt/no-precursor failure cohorts. Do not silently remove hard or non-predictable events from the overall ledger.
- Model inputs may use only information observable at scoring time. Hidden simulator state, labels, masks, failure times, and future outcomes are post-hoc diagnostics or targets only.
- Reset causal histories at maintenance/recommissioning. Exclude maintenance overlap and invalid/censored horizons symmetrically across roles.
- A file ending at failure onset may count as detection, but it must not count as positive early-warning lead time.
- Event metrics must never fall back to file AUROC when event controls are insufficient. Report the event result as `UNAVAILABLE` and stop.
- Calibration and probability-quality evaluation remain out of scope. They become eligible only after a later useful event-ranking gate passes.
- Follow the repository's Git-synchronized remote workflow for generated histories and bounded executions. Bulk datasets stay uncommitted on the execution host.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done.

### Batch A — Measurement contract

- [x] **Task 1 — Reconcile accepted benchmark failure evidence.** Inventory the accepted Sprint 12 facts that constrain redesign: MTBF around 11–13 days, only 2–4 valid negative control windows per history against the frozen floor of 6, 12/85 failures with clean baselines, 73/85 without them, progressive-only coverage, maintenance/quarantine saturation, and median warning lead time of 0.0 days. Verify the exact source artifacts, hashes, runtime roots, and meanings before proposing new settings. Evidence: `artifacts/sprint-13/task-1.md` (done; review pending).
- [x] **Task 2 — Define causal event estimands and window semantics.** Freeze the positive event unit, same-history/robot negative controls, prediction horizons, exclusion intervals, deterministic non-overlap/matching, censoring, maintenance resets, first-alert lead time, persistence, event recall, standard tie-aware event AUROC/G-rank, and false-alert episodes per robot-day. Explicitly distinguish failure-adjacent detection from advance warning. Evidence: `artifacts/sprint-13/task-2.md` (done; review pending).
- [x] **Task 3 — Derive statistical cohort floors and safety margins.** Determine per-history and aggregate minimum positive events, negative controls, clean baseline coverage, category coverage, healthy calibration rows, and independent-history counts from the intended uncertainty and false-alert target. Include a safety margin above the mathematical minimum rather than designing to barely meet six negative controls. Freeze `PASS`, `FAIL`, and `UNAVAILABLE` rules. Evidence: `artifacts/sprint-13/task-3.md` (done; review pending).

### Batch B — Factory and partition design

- [~] **Task 4 — Specify distinct failure-mode cohorts.** Define progressive, weak-precursor, and abrupt/no-precursor mechanisms with auditable physical parameters, occurrence rates, severity support, precursor visibility expectations, and post-hoc category labels. Preserve abrupt events in overall metrics while preventing them from being mislabeled as predictable. Evidence: `artifacts/sprint-13/task-4.md` (REOPENED under R1 - pending re-verification).
- [x] **Task 5 — Set history duration and failure density.** Choose history length, per-robot exposure, failure rates, maintenance cadence, degradation duration, quarantine policy, and robot count so clean pre-failure baselines and matched negative windows can exist without erasing realistic operations. Use structural calculations or simulations independent of anomaly-model performance. Evidence: `artifacts/sprint-13/task-5.md` (done; review pending).
- [~] **Task 6 — Assign independent data roles and holdouts.** Predeclare seeds and entire-history assignments for generator validation, development/diagnostics, fit, calibration/operating-point selection, confirmation, and sealed evaluation. Reserve complete mechanisms plus robot/program cold-start slices. State exactly which structural metadata may be inspected before sealing. Evidence: `artifacts/sprint-13/task-6.md` (REOPENED under R1 - pending re-verification).
- [~] **Task 7 — Freeze Benchmark Protocol v4 before generation.** Consolidate Tasks 2–6 into a versioned protocol containing generator settings, cohort roles, metric implementations, cohort floors, permitted inspections, advancement rules, and stop conditions. Record approval and cryptographic provenance before materializing outcomes. Any semantic correction requires an explicit protocol amendment before rerun. Evidence: `experiments/sprint13-protocol-v4.md` and `artifacts/sprint-13/task-7.md` (REOPENED under R1 - pending re-verification).

### Batch C — Benchmark implementation and materialization

- [~] **Task 8 — Implement versioned factory configuration changes.** Add only the generator/configuration controls required by Protocol v4 for duration, density, cohort mixture, maintenance cadence, and deterministic history identities. Reuse existing public generator contracts; do not add representation-aware seed selection or model-dependent generation. Evidence: `artifacts/sprint-13/task-8.md` (REOPENED under R1 - pending re-verification).
- [~] **Task 9 — Implement event and role metadata contracts.** Persist the fields needed to reproduce category membership, causal horizons, censoring, maintenance resets, quarantine, control-window eligibility, role assignment, and sealed provenance without leaking hidden state into model inputs. Evidence: `artifacts/sprint-13/task-9.md` (REOPENED under R1 - pending re-verification).
- [~] **Task 10 — Prove generator and metric-contract behavior.** Add focused behavioral checks for deterministic reloads, physical bounds, all three failure cohorts, maintenance segmentation, horizon censoring, control-window selection, positive-lead semantics, tie handling, and role isolation. Exercise the real public entry point with a tiny history. Evidence: `artifacts/sprint-13/task-10.md` (REOPENED under R1 - pending re-verification).
- [~] **Task 11 — Materialize independent Protocol v4 histories.** Generate all predeclared histories through the public entry point on the canonical remote checkout after commit/push/pull verification. Do not retry or replace histories based on model performance. Persist bounded manifests, hashes, counts, timing, device, and execution commit; keep bulk data remote. Evidence: `artifacts/sprint-13/task-11.md` (v4 run retired; v4.1 R2A done, C2 renewed-accepted).

### Batch D — Structural measurability validation

- [~] **Task 12 — Verify factory causality and partition integrity.** Confirm route/robot/program causality, chronological ordering, deterministic reloads, maintenance/recommissioning behavior, quarantine, failure-category assignment, role isolation, cold-start slices, and absence of future-state leakage. Evidence: `artifacts/sprint-13/task-12.md` (v4 retired; v4.1 R2B done, 13/13 pass, D1 re-review pending).
- [~] **Task 13 — Audit clean-window and baseline coverage.** Measure per-history/per-robot positive events, negative controls, clean pre-failure baselines, maintenance-free exposure, healthy calibration rows, category counts, and exclusion reasons. Report distributions, not only pooled totals. No model scores may influence acceptance. Evidence: `artifacts/sprint-13/task-13.md` (v4 retired; v4.1 R2B done, 11/13 STRUCT-PASS, D1 re-review pending).
- [~] **Task 14 — Demonstrate score-independent metric computability.** Run the frozen event-window and metric pipeline with deterministic synthetic score fixtures and trivial constant/observable-time baselines solely to prove event AUROC/G-rank, positive lead time, persistence, and false-alert episodes are defined. Do not treat baseline performance as model selection or scientific success. Evidence: `artifacts/sprint-13/task-14.md` (v4 retired; v4.1 R2B done on 9 non-sealed roles, D1 re-review pending).

### Batch E — Review and closeout

- [~] **Task 16 — Pass batch evidence gates.** Review Batches A–D with evidence reviewers. Any actionable defect returns to the same implementation owner, followed by a fresh review at the affected gate. Preserve failed and superseded review records. Evidence: `artifacts/sprint-13/review-*.md` plus `artifacts/sprint-13/review-index.md` (REOPENED under R1 - pending re-verification).

- [~] **Task 17 — Pass differential review and finalize Sprint 13.** Run a sprint-wide differential deep review over accepted evidence. Finalize only with zero actionable findings and an explicit benchmark measurability verdict. Update `docs/PLAN.md` and this plan without implying representation or calibrated-risk success. Evidence: `artifacts/sprint-13/task-17.md` and `artifacts/sprint-13/deep-review-final.md`. (Deep review 1: FAIL; R1 remediation in progress — see `artifacts/sprint-13/deep-review-1.md`.)

## Acceptance Criteria

Sprint 13 returns `MEASURABLE` only when all of the following hold under the pre-generated frozen contract:

1. Every required history and failure category satisfies its predeclared positive-event, negative-control, clean-baseline, healthy-calibration, and maintenance-free exposure floors with safety margin.
2. Progressive, weak-precursor, and abrupt/no-precursor events are all present, auditable, and retained in the overall ledger.
3. Event-level ranking is computable per the frozen unit of independence without substituting file-level AUROC.
4. Positive lead time is distinguishable from an alert emitted at failure onset.
5. Event recall, first-alert lead time, persistence, censoring, and false-alert episodes per robot-day are reproducible under maintenance resets.
6. Generator, development, fit, calibration, confirmation, and sealed histories have explicit non-overlapping roles and reproducible provenance.
7. No history or seed was selected using representation, anomaly, warning, or calibration outcomes.
8. Sealed evaluation roots are hashed and inaccessible to later model/component selection.
9. Evidence and differential review gates pass with zero actionable findings.

## Scientific Stop Rules

- If raw simulator cohorts violate physical or deterministic contracts, fix the generator and issue a protocol amendment before rematerialization.
- If structural cohort floors fail, report `NOT_MEASURABLE`; do not compensate by pooling away the independent-history unit or relaxing floors after observing counts.
- If event metrics remain undefined, report `UNAVAILABLE`; do not substitute descriptive file metrics.
- If abrupt failures are inherently unpredictable, retain them and report category-conditional plus overall results; do not silently exclude them.
- Sprint 14 may use only accepted development/confirmation roles. Sealed Sprint 13 evaluation histories remain closed until a future predeclared independent-evaluation gate explicitly permits access.

## Out of Scope

- Representation objective, encoder, geometry, or scorer redesign.
- Representation-aware generator tuning or seed selection.
- Flexible survival/risk-model fitting, probability calibration, Brier score, ECE, or reliability claims.
- Production threshold selection or deployment conclusions.
- Opening Sprint 12 sealed-static data or reinterpreting its blocked-final tasks.

## Notes / Blockers

- Sprint 12 is the accepted prerequisite and remains unchanged.
- Sprint 14 is queued but its benchmark-dependent execution is blocked until Task 15 returns `MEASURABLE` and Task 17 closes with zero actionable findings.
- A future risk-calibration sprint remains conditional on useful event-level ranking, not merely benchmark measurability.
