# Sprint 14 — Benchmark Measurability Recovery

## Sprint Goal

Produce a causally valid, independently replicated early-warning benchmark that passes the concrete structural, observable-signal, and generalization gates in [`../BENCHMARK_MEASURABILITY_EXIT_GATES.md`](../BENCHMARK_MEASURABILITY_EXIT_GATES.md) before representation attribution resumes.

**Status:** COMPLETE — final verdict `NOT_MEASURABLE` (Task 19; closeout review PASS, 0 findings). EG0 PASS, EG1 PASS, EG2 FAIL (0/4, 3/3 cycles), EG3 PASS, EG4–EG7 NOT_RUN; no deep review, no `deep-review-final.md`. Ledger tail `e065-task-19-final` (65 entries, chain-ok). Sprint 15 BLOCKED/INELIGIBLE with no attribution handoff. No code/data/protocol/matrix/report changes.

## Decision Context

Sprint 13 completed correctly but returned `NOT_MEASURABLE`: under corrected v4.1.1 semantics, 0/13 histories achieved `STRUCT-PASS`; the four sealed histories had only 15/22/17/23 negative controls against the frozen floor of 25, and every history missed negative-control and/or abrupt-category floors. The result was neither a model failure nor an unavailable computation. It showed that the benchmark itself could not support the intended independent event-level claims.

The previous Sprint 14 attribution plan is preserved as [Sprint 15](sprint-15.md). Sprint 15 remains blocked until this sprint returns `MEASURABLE` and its final deep review has zero actionable findings.

Sprint 14 therefore owns benchmark recovery only. It must diagnose why structural units disappear, change the minimum necessary generator/scheduling/maintenance/quarantine controls prospectively, and prove that the corrected benchmark generalizes across fresh confirmation and sealed histories. It must not tune representation or risk models.

## Normative Exit Contract

[`docs/BENCHMARK_MEASURABILITY_EXIT_GATES.md`](../BENCHMARK_MEASURABILITY_EXIT_GATES.md) is normative for:

- verdict definitions and precedence;
- role counts and whole-history isolation;
- hard per-history structural floors and design safety margins;
- causal, deterministic, and leakage checks;
- deterministic E1–E5 fixture behavior;
- the fixed observable-signal sanity probe;
- confirmation and sealed generalization;
- the three-cycle limit and retirement rules.

A task artifact may add diagnostics but may not weaken, reinterpret, or silently replace that contract. Any semantic change requires an explicit edit to the exit-gate document, a protocol version bump, user approval, and a fresh evidence review before new outcomes are generated.

## Non-Negotiable Scientific Constraints

- Preserve Sprint 11–13 evidence, protocols, roots, seals, and failed/superseded review records unchanged.
- Treat v4 and v4.1/v4.1.1 materializations as diagnosis-only historical evidence; never promote or selectively reuse them as Sprint 14 confirmation or sealed roots.
- Use whole histories as independent units. No pooled total, median, or favorable subset rescues a failed per-history gate.
- Freeze seeds, roles, physics, schedules, floors, probe features, fitting/calibration rules, thresholds, and verdict logic before materialization.
- Never retry, replace, omit, or relabel a history because its counts or probe outcomes are unfavorable.
- Generator changes may be driven only by design-role structural diagnostics and the prospectively frozen observable probe. Learned representation, anomaly, warning, and risk scores are prohibited inputs to benchmark selection.
- Confirmation histories are one-shot. Failure retires the complete confirmation set and requires a new protocol version plus fresh design and confirmation roots.
- Sealed histories receive structural audit and sealing only. No fixture scores, observable probe, learned model, or threshold may touch them in Sprint 14.
- Abrupt failures stay in category-conditional and overall ledgers. They need structural representation but no minimum predictability claim.
- At most three diagnostic cycles are permitted. Failure after cycle three closes the sprint as `NOT_MEASURABLE`.
- Operational PASS and scientific `MEASURABLE` are distinct. Successful code execution cannot substitute for failed exits.
- Run every Sprint 14 workflow locally from the repository root through the locked `uv` environment. Do not use SSH or the remote server; generated roots remain local and uncommitted.

## Diagnostic-to-Review Workflow

Every cycle uses this exact state machine:

```text
accepted prior evidence
→ design-only diagnosis
→ one prospectively frozen protocol amendment
→ evidence review of amendment
→ implementation and public-entry proof
→ fresh design materialization
→ structural/metric audit
→ evidence review
→ candidate freeze
→ fresh confirmation materialization
→ observable + structural generalization audit
→ evidence review
→ fresh sealed materialization
→ structural-only audit and seals
→ evidence review
→ provisional MEASURABLE? yes → sprint-wide differential deep review → final MEASURABLE
                         no  → final NOT_MEASURABLE | UNAVAILABLE without deep review
```

Workflow corrections:

1. An evidence review with an actionable finding fails the gate.
2. The retained worker corrects the same task or batch; code and evidence are updated while failed review records remain preserved.
3. A fresh evidence reviewer must review the corrected evidence. Worker self-acceptance is prohibited.
4. A Design failure returns to diagnosis under a new protocol version.
5. A Confirmation failure retires all complete Design/Fit/Calibration/Confirmation roots for that candidate, returns to diagnosis, and consumes a cycle.
6. A Sealed structural failure mechanically returns `NOT_MEASURABLE`; sealed outcomes must not be used to tune another candidate inside this sprint.
7. Call a sprint-wide differential `deep-reviewer` if and only if the mechanically derived provisional verdict is `MEASURABLE`; it must return zero actionable findings before the verdict becomes final. A provisional `NOT_MEASURABLE` or `UNAVAILABLE` closes through evidence-reviewed blocker and ledger artifacts without deep review.

OMP scheduling follows one retained implementation worker plus at most one reviewer. Reviewers are orchestrator gates, not worker tasks, and no more than two subagents may be active.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done.

### Batch A — Authority, contract, and prospective protocol

- [x] **Task 1 — Reconcile the authoritative Sprint 13 failure ledger.** Verify v4.1.1 roots, active versus retired seals, corrected temporal/reset predicates, per-history negative and cohort misses, and exact `NOT_MEASURABLE` precedence. Produce a machine-readable deficit table naming which configuration or eligibility mechanism can affect each miss without consulting model scores. Evidence: `artifacts/sprint-14/task-1.md` and `artifacts/sprint-14/deficit-ledger.json`.
- [x] **Task 2 — Freeze Benchmark Recovery Protocol v1.** Predeclare the first diagnostic-cycle number, complete role topology, seeds, generator settings, causal physics, cohort/subtype holdouts, metric semantics, hard floors, design promotion margins, fixed observable probe, amendment rules, and canonical runtime. Embed the exit-gate document digest and explicitly map every EG0–EG7 clause to protocol sections. Evidence: `experiments/sprint14-benchmark-protocol-v1.md` and `artifacts/sprint-14/task-2.md`.
- [x] **Task 3 — Establish the cycle and retirement ledger.** Create an append-only ledger for protocol versions, cycle numbers, root identities, roles, seeds, commits, manifests, reviews, gate outcomes, retirement reasons, and forbidden reuse. Reject duplicate seeds, overlapping roles, missing digests, and roots not declared by the active protocol. Evidence: `artifacts/sprint-14/task-3.md` and `artifacts/sprint-14/cycle-ledger.json`.

**Evidence Gate A:** An `evidence-reviewer` verifies Tasks 1–3 against EG0 and the accepted Sprint 13 chain. Any actionable finding returns to the same worker. No implementation or generation may start before a fresh Gate A PASS with zero actionable findings. Record: `artifacts/sprint-14/review-batch-a.md`.

### Batch B — Bounded design diagnostic cycle

- [x] **Task 4 — Diagnose structural-unit loss on design evidence.** Attribute missing positives, controls, clean baselines, robot-days, and abrupt coverage to generation density, route/schedule occupancy, failure timing, maintenance, quarantine, censoring, reset crossings, temporal-view membership, or greedy control-window conflicts. Report per-history and per-robot distributions; do not infer from pooled totals. Evidence: `artifacts/sprint-14/task-4-cycle-<N>.md`. (Cycle 2 diagnosis: `artifacts/sprint-14/task-4-cycle-2.md`; Cycle 3 diagnosis: `artifacts/sprint-14/task-4-cycle-3.md`.)
- [x] **Task 5 — Freeze one bounded DGP amendment.** Change only the generator or scheduling controls causally supported by Task 4. State predicted effects for every hard floor and design margin, physical invariants that must remain unchanged, rejected alternatives, and failure conditions before implementation. No amendment may change metric definitions or floors merely to fit observed counts. Evidence: `experiments/sprint14-benchmark-protocol-v<N>.md` and `artifacts/sprint-14/task-5-cycle-<N>.md`.

**Evidence Gate B1 — Amendment approval:** An `evidence-reviewer` checks Tasks 4–5, the predicted floor effects, physical invariants, exit-gate mapping, and prospective protocol freeze. Task 6 cannot start until Gate B1 passes with zero actionable findings. Corrections remain in the same cycle only if no outcome has been generated; otherwise they require a new cycle. Record: `artifacts/sprint-14/review-batch-b1-cycle-<N>.md`.
- [x] **Task 6 — Implement versioned benchmark controls.** Add the smallest public configuration, generation, metadata, or audit-contract changes required by the approved amendment. Preserve existing public generator contracts where possible; keep labels and hidden simulator state out of model-visible data. Evidence: `artifacts/sprint-14/task-6-cycle-<N>.md`.
- [ ] **Task 7 — Prove causal and deterministic behavior through the public entry point.** Exercise a tiny real history and verify fresh-process deterministic reload, chronological ordering, route/robot exclusivity, maintenance/reset segmentation, cohort physics, role isolation, future-state exclusion, metric fixtures, and manifest/seal integrity required by EG1 and EG3. Evidence: `artifacts/sprint-14/task-7-cycle-<N>.md`. (Cycle 3 full proof FAILED at share band — `artifacts/sprint-14/task-7-cycle-3.md`; v5.1 re-eval passed on preserved roots, pending fresh review; stopped per instruction.)
- [x] **Task 8 — Materialize the declared Design histories.** Generate at least four predeclared Design histories on the canonical local workstation at the verified commit. Record first-attempt status, runtime, counts, roots, manifests, hashes, and device; never replace an unfavorable history. Evidence: `artifacts/sprint-14/task-8-cycle-<N>.md`.
- [x] **Task 9 — Audit Design promotion margins.** Apply the frozen audit to every Design history and report every hard floor, promotion target, concentration limit, lead-support predicate, Fit/Calibration support projection, exclusion reason, and EG1/EG3 result. Pass only if every Design history reaches every promotion target. Evidence: `artifacts/sprint-14/task-9-cycle-<N>.md` and `artifacts/sprint-14/design-audit-cycle-<N>.json`.

**Evidence Gate B2 — Design promotion:** An `evidence-reviewer` checks Tasks 6–9 and the Gate B1-approved protocol. A PASS promotes the candidate to Batch C. A scientific miss or actionable defect returns to Task 4 under a new protocol version and fresh Design roots. Each Gate B2 attempt increments `<N>`; at most three cycles are allowed. Preserve every failed and superseded record. Record: `artifacts/sprint-14/review-batch-b2-cycle-<N>.md`.

### Batch C — One-shot confirmation and observable generalization

- [ ] **Task 10 — Freeze the candidate DGP and downstream roster.** After Gate B2 PASS, freeze the exact DGP bytes and predeclare Fit, Calibration, four Confirmation, and four Sealed histories plus reserved robot, program, and `P`/`W` mechanism subtypes. Record that no further generator, probe, threshold, or audit change is permitted for this candidate. Evidence: `artifacts/sprint-14/task-10.md`. (NOT_RUN — Gate B2 denied promotion; see record.)

**Evidence Gate C1 — Candidate and roster freeze:** An `evidence-reviewer` checks Task 10 for byte identity with the Gate B2 candidate, non-overlapping roles, fresh predeclared seeds, holdout coverage, probe immutability, and forbidden sealed access. Task 11 cannot start until Gate C1 passes with zero actionable findings. Record: `artifacts/sprint-14/review-batch-c1.md`.
- [ ] **Task 11 — Materialize Fit, Calibration, and Confirmation roles.** Generate every declared non-sealed root once through the canonical public entry point. Persist manifests, hashes, role metadata, first-attempt status, support counts, and runtime provenance. Evidence: `artifacts/sprint-14/task-11.md`. (NOT_RUN — Gate B2 denied promotion; see record.)
- [ ] **Task 12 — Execute the fixed observable-signal sanity probe.** Fit the predeclared simple causal telemetry probe on Fit only, calibrate its single operating threshold on verified-healthy Calibration only, and evaluate it once on all four Confirmation histories. Report `P`, `W`, `A`, combined, per-history, holdout, lead-time, recall, and false-alert results exactly as required by EG4. Evidence: `artifacts/sprint-14/task-12.md` and `artifacts/sprint-14/observable-confirmation.json`. (NOT_RUN — Gate B2 denied promotion; see record. No `observable-confirmation.json` exists.)
- [ ] **Task 13 — Audit Confirmation structural generalization.** Apply every hard per-history floor, concentration constraint, causal/metric check, Design-to-Confirmation stability ratio, and reserved holdout requirement in EG5. Report all four histories separately and issue `CONFIRMATION-PASS` or `CONFIRMATION-FAIL` mechanically. Evidence: `artifacts/sprint-14/task-13.md` and `artifacts/sprint-14/confirmation-audit.json`. (NOT_RUN — Gate B2 denied promotion; see record. No `confirmation-audit.json` exists.)

**Evidence Gate C2 — Confirmation acceptance:** An `evidence-reviewer` checks Tasks 11–13 against EG4 and EG5. Any failed hard condition retires the entire candidate Confirmation set; no favorable root is retained. A new candidate returns to Task 4 with a new protocol version and consumes the next diagnostic cycle. Record: `artifacts/sprint-14/review-batch-c2.md`.

### Batch D — Sealed structural generalization and verdict

- [ ] **Task 14 — Materialize the sealed histories.** After Gate C2 PASS, generate exactly four predeclared Sealed histories once through the canonical local public entry point at the verified commit. Do not run fixture, observable, representation, anomaly, warning, or risk scores. Evidence: `artifacts/sprint-14/task-14.md`. (NOT_RUN — Gate B2 denied promotion; see record.)
- [ ] **Task 15 — Audit and seal independent evaluation roots.** Apply structural EG1/EG6 checks to each sealed history, record all floors and concentration limits separately, write deterministic seals, verify every manifest digest round-trip, and preserve bulk roots locally and uncommitted. Evidence: `artifacts/sprint-14/task-15.md`, `artifacts/sprint-14/sealed-audit.json`, and `artifacts/sprint-14/seals.json`. (NOT_RUN — Gate B2 denied promotion; see record. No `sealed-audit.json` or `seals.json` exist.)
- [x] **Task 16 — Issue the provisional benchmark verdict mechanically.** Evaluate EG0–EG6 and return exactly provisional `MEASURABLE`, `NOT_MEASURABLE`, or `UNAVAILABLE`. Include a clause-by-clause gate matrix, no-pooling proof, cycle count, retired-root inventory, and explicit Sprint 15 eligibility condition. No fallback, partial pass, or post-hoc reinterpretation is permitted. Evidence: `artifacts/sprint-14/task-16.md` and `artifacts/sprint-14/exit-gate-matrix.json`. (Provisional `NOT_MEASURABLE` issued; pending Evidence Gate D.)

**Evidence Gate D:** An `evidence-reviewer` verifies Tasks 14–16, sealed-role isolation, hashes, gate precedence, and the verdict. A sealed structural miss is a final `NOT_MEASURABLE`, not a diagnostic input for another in-sprint cycle. Record: `artifacts/sprint-14/review-batch-d.md`.

### Batch E — Handoff and closeout

- [x] **Task 17 — Prepare the benchmark handoff.** If and only if Task 16 returns provisional `MEASURABLE` and Gate D passes, prepare the exact development/Fit/Calibration/Confirmation roles Sprint 15 may use, sealed roots it may not open, approved metric contract, known limitations, and canonical execution instructions. The handoff remains inactive until Task 19 passes EG7. A negative verdict instead publishes blockers without an attribution handoff. Evidence: `artifacts/sprint-14/task-17.md`. (Negative branch final: blockers published, decisive 0/4 abrupt misses restated, NO Sprint 15 handoff exists; closeout review PASS.)
- [x] **Task 18 — Consolidate the evidence and review ledger.** Index every task artifact, protocol version, root, seal, failed/superseded review, correction, reviewer identity, relay caveat, and latest standing. Confirm every batch has a fresh zero-actionable evidence review. Evidence: `artifacts/sprint-14/review-index.md` and `artifacts/sprint-14/task-18.md`. (Final: full index with digests, all standing batch reviews zero-actionable including closeout, superseded attempts preserved.)
- [x] **Task 19 — Conditionally deep-review and finalize Sprint 14.** If and only if Task 16 and Gate D establish provisional `MEASURABLE`, run a sprint-wide `deep-reviewer` over the accepted evidence-review reports and cross-task interactions. Correct every actionable finding through the retained worker, re-run affected evidence gates, and repeat deep review until zero findings; then apply EG7 and finalize `MEASURABLE`. For provisional `NOT_MEASURABLE` or `UNAVAILABLE`, do not call a deep reviewer: record the scientific stop and finalize after Tasks 17–18 preserve the evidence-reviewed blockers and ledger. Update `docs/PLAN.md` plus Sprint 15 status only after this conditional closeout. Evidence: `artifacts/sprint-14/task-19.md`, and `artifacts/sprint-14/deep-review-final.md` only for a provisional `MEASURABLE`. (Negative branch complete: final `NOT_MEASURABLE`, EG7 `NOT_RUN`, no deep review, Sprint 15 blocked, plans updated.)

## Acceptance Criteria

1. The final protocol and every root map to the normative exit-gate document by content digest.
2. At least four Design histories meet all promotion margins before Confirmation is opened.
3. Every one of four Confirmation histories independently meets every hard structural and generalization gate.
4. The fixed causal observable probe passes every EG4 threshold on the untouched Confirmation set without hidden-state inputs or post-outcome tuning.
5. Every one of four Sealed histories independently meets every EG6 structural gate and remains unscored.
6. Fit and Calibration support is sufficient and role-clean; reserved robot, program, and mechanism subtypes appear in Confirmation and Sealed roles.
7. No pooled result rescues a failed history; no history is retried, replaced, omitted, or selected by outcome.
8. No more than three diagnostic cycles are used, and every prior protocol/root/review is preserved with an explicit retirement reason.
9. Every batch has a latest evidence review with zero unresolved actionable findings.
10. If and only if EG0–EG6 yield provisional `MEASURABLE`, the latest sprint-wide differential deep review has zero unresolved actionable findings; negative or unavailable outcomes record `NOT_RUN` with the triggering evidence gate.
11. The final verdict is exactly `MEASURABLE`, `NOT_MEASURABLE`, or `UNAVAILABLE`; `MEASURABLE` mechanically matches EG0–EG7, while a negative or unavailable verdict mechanically matches its failed EG0–EG6 condition without invoking EG7 deep review.
12. Sprint 15 remains blocked unless the verdict is `MEASURABLE` and the conditional final deep review passes.

## Scientific Stop Rules

- If a causal, determinism, leakage, or provenance invariant fails, retire the affected roots before interpreting structural counts.
- If Design misses a promotion margin, start a new protocol cycle; do not weaken the margin after seeing outcomes.
- If Confirmation fails any hard floor, observable threshold, or generalization condition, retire all candidate Confirmation roots and return to a fresh cycle.
- If Sealed fails any hard structural condition, return `NOT_MEASURABLE`; never use sealed outcomes to design another candidate in this sprint.
- If metric fixtures remain undefined despite sufficient structural units, return `UNAVAILABLE`; never substitute file AUROC.
- If cycle three fails, close `NOT_MEASURABLE`; do not continue seed or configuration search.
- A favorable observable probe does not authorize representation, production-threshold, or calibrated-risk claims.

## Explicit Non-Goals

- Representation, objective, encoder, geometry, scorer, pooling, or aggregation redesign.
- Learned-model training, hyperparameter search, or checkpoint selection.
- Production anomaly thresholds or calibrated one-day/seven-day risk.
- Making abrupt failures artificially predictable.
- Using Sprint 13 sealed roots or Sprint 14 sealed roots for development.
- Real-data deployment conclusions.

## Downstream Boundary

Sprint 15 owns representation failure localization and component attribution only after an accepted Sprint 14 `MEASURABLE` verdict. Sprint 16, if later opened, owns targeted recovery of causally established bottlenecks. Calibrated-risk work remains additionally gated on useful event ranking and acceptable false-alert behavior.
