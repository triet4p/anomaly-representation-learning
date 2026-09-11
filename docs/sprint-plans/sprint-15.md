# Sprint 15 — Balanced Causal Benchmark Recovery and Measurability Certification

## Sprint Goal

Deliver a fresh, causally valid, quota-controlled evaluation benchmark with final verdict `MEASURABLE`, removing Sprint 14's reset-driven abrupt-support loss and passing every Design, observable-signal, Confirmation, Sealed, evidence-review, and deep-review gate.

**Status:** COMPLETE — final verdict `MEASURABLE` for Candidate 7 (Gate F-r12 PASS with 0 actionable findings; final differential deep review R3 PASS with 0 actionable findings and 0 unanswered questions; EG7 PASS; Task 22 DONE per `artifacts/sprint-15/task-22.md` on `deep-review-final-r3.md`). Historical correction rounds r1–r10 and all failed/deprecated reviews preserved as indexed evidence. No further rounds planned; mandatory post-final evidence review required before Sprint 16 starts.

## Decision Context

Sprint 14 is complete and remains immutable with final verdict `NOT_MEASURABLE`. Its final Cycle 3 repaired negative-control support but failed Design promotion in every history:

- evaluable abrupt counts were `11 / 11 / 7 / 7` against the Design target of `10`;
- abrupt shares were `12.79% / 12.79% / 9.33% / 8.24%` against the `15%` floor;
- raw abrupt arrivals were adequate, but reset-window intersections removed roughly half of their causal horizons;
- negative controls reached `43 / 38 / 46 / 39`, so further generic exposure expansion is not the primary fix;
- EG1 causality/determinism and EG3 metric fixtures passed.

Sprint 15 therefore does not add a fourth Sprint 14 cycle. It creates a new benchmark family with fresh protocols, seeds, roots, and evidence. The former Sprint 15 attribution plan has moved intact to [Sprint 16](sprint-16.md) and remains blocked until this sprint returns final `MEASURABLE` with a reviewed handoff.

## Benchmark Construction Decision

Sprint 15 uses a **balanced causal evaluation profile**, not a production-prevalence simulator:

1. Generate the shared-unit calendar, routes, operations, maintenance, recommissioning, and healthy background first.
2. Compute eligible causal event anchors from that frozen calendar using the unchanged seven-day horizon, reset, quarantine, censoring, spacing, and baseline predicates.
3. Allocate predeclared per-history `P`, `W`, and `A` episode quotas across eligible anchors with a dedicated RNG, robot/program caps, and subtype stratification before waveform generation.
4. Reject the active candidate if its frozen calendar cannot place the quotas without violating physical or causal constraints. Never resample or replace a root within that candidate; diagnose, freeze a new protocol version, and advance to the next deterministic seed namespace.
5. Keep event labels, cohorts, subtypes, failure times, simulator state, and allocation metadata out of model-visible rows.

This is transparent case-control benchmark sampling. It supports identifiable evaluation but must never be interpreted as a fleet prevalence estimate.

To make `MEASURABLE` a construction target rather than a lucky outcome, every candidate must also freeze bounded nuisance envelopes and explicit causal `P`/`W` telemetry signatures with a conservative analytic signal-to-noise margin before roots exist. A deterministic qualification harness must prove quota feasibility, physical bounds, and observable-feature separation on symbolic or disposable fixtures before Design materialization. Learned-model outputs remain forbidden from this process.

### Prospective per-history support margins

The v1 recovery protocol must reserve, after all eligibility exclusions:

- `P = 24`, split across `P1/P2`;
- `W = 24`, split across `W1/W2`;
- `A = 16`, split across `A1/A2`;
- at least `48` deterministic non-overlapping negative-control windows;
- at least `240` evaluable robot-days;
- at least `6` positive-contributing robots and `6` negative-contributing robots;
- at least `2` represented programs in each predictable cohort.

The resulting nominal mix is `37.5% P / 37.5% W / 25% A`, safely inside the unchanged `[15%, 60%]` per-cohort limits. Structural support is guaranteed by the allocator or the candidate is rejected before promotion; signal observability, cross-history replication, and sealed generalization still require the full reviewed gate chain.

## Non-Negotiable Scientific Constraints

- Preserve Sprint 14 plans, protocols, roots, reviews, ledgers, and final verdict byte-for-byte as historical evidence.
- Create `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md`; do not edit the Sprint 14 gate document whose digest is part of its final evidence.
- Preserve Sprint 14's event definitions, seven-day windows, hard floors, Design promotion targets, concentration caps, tie handling, no-pooling rule, and observable-probe thresholds unless a change is justified and frozen before any Sprint 15 outcome exists.
- The balanced profile may control evaluation-event support; it may not leak labels or simulator state to representation inputs or scores.
- Representation, anomaly, warning, and calibrated-risk outputs cannot select generator settings, quotas, histories, thresholds, or gates.
- Design, Fit, Calibration, Confirmation, and Sealed seeds are disjoint. No history is retried, replaced, omitted, or selected by outcome.
- Confirmation is one-shot candidate evidence, never new Design data. Sealed roots receive structural audits and seals only; no probe or learned score may touch them.
- All generation and audits run locally from the repository root through the locked `uv` environment. Bulk roots remain local and uncommitted.
- Every batch requires a fresh `evidence-reviewer` PASS with zero actionable findings. A sprint-wide `deep-reviewer` runs only after EG0–EG6 produce provisional `MEASURABLE`.
- Sprint 15 has no fixed candidate-iteration cap and no negative terminal branch. Each failed candidate is preserved and retired; completion occurs only at reviewed final `MEASURABLE`.

## Deterministic Candidate and Seed Namespace

Candidate iterations are numbered `N = 1, 2, 3, ...`. Candidate `N` uses base seed
`B_N = 1000 + 100 × (N - 1)` and the following immutable offsets:

| Role | Seeds |
|---|---|
| Design | `B_N + 0` through `B_N + 3` |
| Fit | `B_N + 4` through `B_N + 6` |
| Calibration | `B_N + 7` |
| Confirmation | `B_N + 8` through `B_N + 11` |
| Sealed | `B_N + 12` through `B_N + 15` |
| Disposable proof | `B_N + 16` and `B_N + 17` |

Task 4 binds these derived seeds to exact role IDs and output paths before candidate implementation. A seed may be retired but never reassigned. Candidate `N + 1` starts only after evidence-reviewed diagnosis of candidate `N`, a new prospective protocol version, and confirmation that every prior root remains immutable.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done.

### Batch A — Recovery contract and prospective freeze

- [x] **Task 1 — Freeze the Sprint 14 evidence boundary.** Inventory the accepted Sprint 14 final verdict, decisive abrupt-support mechanism, passing conditions, canonical digests, retired roots, and forbidden reuse. Prove no Sprint 14 artifact is changed. Evidence: `artifacts/sprint-15/task-1.md`.
- [x] **Task 2 — Freeze the v2 measurability exit gates.** Create `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` with unchanged structural and observable thresholds, success-only sprint completion, candidate-rejection precedence, deterministic candidate seed namespaces, role topology, evidence gates, and Sprint 16 handoff contract. Evidence: `artifacts/sprint-15/task-2.md`.
- [x] **Task 3 — Freeze the balanced causal sampling protocol.** Specify calendar-first anchor eligibility, exact per-history quotas, bounded nuisance envelopes, causal `P`/`W` signal margins, subtype balance, identity caps, candidate rejection, physical invariants, and the case-control interpretation boundary. Evidence: `experiments/sprint15-benchmark-protocol-v1.md` and `artifacts/sprint-15/task-3.md`.
- [x] **Task 4 — Bind candidate roles, seeds, runtime, and no-retry rules.** Resolve the deterministic seed formula for candidate `N`, bind every seed to one immutable role ID, freeze local runtime closure and collision-free output paths, and record forbidden within-candidate replacement, pooling, sealed access, and model influence. Evidence: `artifacts/sprint-15/task-4.md`.

**Evidence Gate A — Prospective contract review:** **PASS** (`artifacts/sprint-15/review-batch-a.md`, reviewer `Sprint15GateAR2`, 0 actionable findings; F-01 resolved, F-02 informational). Task 5 authorized; Design roots remain barred until Gate B passes. Record: `artifacts/sprint-15/review-batch-a.md`.

### Batch B — Balanced profile implementation and causal proof

- [x] **Task 5 — Add the versioned balanced evaluation profile.** Add the smallest public configuration surface for Sprint 15 calendar, quota, subtype, and support settings without changing existing production or Sprint 14 profiles. Evidence: `artifacts/sprint-15/task-5.md`.
- [x] **Task 6 — Implement the causal eligible-anchor allocator.** Allocate quota events only from frozen calendar anchors satisfying horizon, reset, quarantine, censoring, spacing, robot/program, and physical episode constraints; fail fast when infeasible. Evidence: `artifacts/sprint-15/task-6.md`.
- [x] **Task 7 — Extend manifests and seals for sampling provenance.** Persist profile, protocol, seed, role, quota, allocation, rejection, and root digests while keeping hidden event state outside model-visible file rows. Evidence: `artifacts/sprint-15/task-7.md`.
- [x] **Task 8 — Implement the frozen structural audit.** Report raw, allocated, rejected, and evaluable events; all hard floors, promotion margins, mix/cap constraints, lead support, controls, support projections, role isolation, and EG1/EG3 fixtures per history. Evidence: `artifacts/sprint-15/task-8.md`.
- [x] **Task 9 — Prove the public-entry-point contract.** On disposable proof roots, verify fresh-process determinism, chronological ordering, physical episode bounds, anchor eligibility, exact quotas, bounded nuisances, predeclared observable separation, subtype balance, no-overlap, no leakage, audit computability, and manifest/seal round trips. Evidence: `artifacts/sprint-15/task-9.md`.

**Evidence Gate B — Implementation and proof review:** **PASS** (`artifacts/sprint-15/review-batch-b.md`, reviewer `Sprint15GateBReview`, 0 actionable findings, 4 informational). Task 10 authorized; Task 12+ blocked until Gate C1. Record: `artifacts/sprint-15/review-batch-b.md`.

### Batch C — Design promotion and candidate freeze

- [x] **Task 10 — Materialize four current-candidate Design histories once.** Generate the four predeclared candidate `N` Design roots through the canonical local public entry point at the reviewed commit. Record first-attempt status, runtime, roots, manifests, hashes, and device. Evidence: `artifacts/sprint-15/task-10-cycle-<N>.md`. (Cycle 1: 4/4 first attempts exit 0.)
- [x] **Task 11 — Audit every Design promotion condition.** Apply the frozen audit separately to all four histories and issue `DESIGN-PASS` only when every history meets every target, concentration cap, support projection, EG1 invariant, and EG3 fixture. Evidence: `artifacts/sprint-15/task-11-cycle-<N>.md` and `artifacts/sprint-15/design-audit-cycle-<N>.json`. (Cycle 1: 3/4 PASS — H-DESIGN-16 FAIL on `w-dist-shape` (single 0.007 d W duration); unanimity not met, reported as candidate-level miss for Gate C1.)

**Evidence Gate C1 — Design promotion:** An `evidence-reviewer` recomputes every value from raw manifests. Four-of-four `DESIGN-PASS` is mandatory; no pooled or three-of-four rescue. Failure retires candidate `N`, returns to Tasks 2–4 for an evidence-supported prospective methodology revision, and advances to fresh candidate `N + 1`. Record: `artifacts/sprint-15/review-batch-c1-cycle-<N>.md`.

**Candidate 1 retired; Candidate 2 implementation (2026-09-10):** Gate C1 FAIL accepted
(`artifacts/sprint-15/review-batch-c1-cycle-1.md`, F-01 gate-blocking: H-DESIGN-16 `w-dist-shape`
breach, 3/4 `DESIGN-PASS`). Candidate-1 roots and cycle-1 evidence preserved immutable; Task 12
blocked. Tasks 2–4 cycle-2 prospective revision complete; Gate A cycle-2 **PASS**
(`artifacts/sprint-15/review-batch-a-cycle-2.md`, 0 actionable findings, F-01 informational).
Tasks 5–9 cycle-2 (candidate-2 implementation + proof) in progress below; Task 10 cycle-2 remains
barred pending Gate B cycle-2. No candidate-2 root exists yet outside disposable proof.

- [x] **Task 5 (cycle 2) — Implement protocol v2 §7a firing gate + `sprint15-v2` profile.** Smallest Appendix-A boundary only. Evidence: `artifacts/sprint-15/task-5-cycle-2.md`. (Done: flag-off byte-identity proven, gate efficacy + golden + preservation tests green.)
- [x] **Task 6 (cycle 2) — Revalidate allocator + exact quotas on candidate-2 proof.** Evidence: `artifacts/sprint-15/task-6-cycle-2.md`. (Done: exact 12/12/12/12/8/8 both roots.)
- [x] **Task 7 (cycle 2) — Revalidate provenance/manifests/seals + hidden-state isolation.** Evidence: `artifacts/sprint-15/task-7-cycle-2.md`. (Done: v2 tags, seal round trips, no leakage.)
- [x] **Task 8 (cycle 2) — Revalidate structural audit + fail-fast.** Evidence: `artifacts/sprint-15/task-8-cycle-2.md`. (Done: DESIGN-PASS ×2; candidate-1 re-audit identical.)
- [x] **Task 9 (cycle 2) — Disposable proof on H-PROOF-3/4.** Evidence: `artifacts/sprint-15/task-9-cycle-2.md`. (Done: PROOF-PASS, byte-identical determinism.)

**Evidence Gate B (cycle 2) — Implementation and proof review:** **PASS** (`artifacts/sprint-15/review-batch-b-cycle-2.md`, reviewer `Sprint15GateB2Review`, 0 actionable findings, F-01 informational). Task 10 cycle-2 authorized. Record: `artifacts/sprint-15/review-batch-b-cycle-2.md`.

- [x] **Task 10 (cycle 2) — Materialize H-DESIGN-17..20 (seeds 1100..1103) once each.** Canonical local public CLI under `data/generated/sprint15-v2/DESIGN/`, at Gate B cycle-2 reviewed bytes. No retry/overwrite/replace/omit/pool. Evidence: `artifacts/sprint-15/task-10-cycle-2.md`. (Done: 4/4 first attempts exit 0.)
- [x] **Task 11 (cycle 2) — Audit every Design promotion condition (4/4 mandatory).** Evidence: `artifacts/sprint-15/task-11-cycle-2.md` and `artifacts/sprint-15/design-audit-cycle-2.json`. (Done: 4/4 DESIGN-PASS, zero errors, 9/9 EG3 each.)

**Evidence Gate C1 (cycle 2) — Design promotion:** **PASS** (`artifacts/sprint-15/review-batch-c1-cycle-2.md`, reviewer `Sprint15GateC1R2`, 0 actionable findings, F-01 informational). Candidate 2 DESIGN-PROMOTED (unanimous 4/4). Record: `artifacts/sprint-15/review-batch-c1-cycle-2.md`.

- [x] **Task 12 (cycle 2) — Freeze the promoted candidate bytes.** Generator, gate, allocator, audit, dependency, profile, protocol, role, seed, and Design-root bytes at the promoted state. Evidence: `artifacts/sprint-15/task-12-cycle-2.md`. (Done: read-only inventory, zero bytes modified.)
- [x] **Task 13 (cycle 2) — Freeze the observable probe implementation.** Causal feature set, Fit-only fitting, Calibration-only threshold, aggregation, bootstrap, recall/lead/FAR/abrupt rules, frozen pre-outcome. Evidence: `experiments/sprint15-observable-probe-v2.md` and `artifacts/sprint-15/task-13-cycle-2.md`. (Done: new `probe15.py` + 11 tests green, pre-outcome fixtures exact.)

**Evidence Gate C2 (cycle 2) — Candidate and probe freeze:** **PASS** (`artifacts/sprint-15/review-batch-c2-cycle-2.md`, reviewer `Sprint15GateC2Review`, 0 actionable findings, F-01 informational). Candidate 2 + probe frozen. Record: `artifacts/sprint-15/review-batch-c2-cycle-2.md`.

- [x] **Task 14 (cycle 2) — Materialize Fit/Calibration/Confirmation once each.** H-FIT-13..15 (1104–1106), H-CAL-5 (1107), H-CONF-14..17 (1108–1111) via canonical CLI, first-attempt only. Evidence: `artifacts/sprint-15/task-14-cycle-2.md`. (Done: 4/8 roots exit 0; 4 Conf fail-fast infeasible on A-subtype quotas, zero roots — see record.)
- [ ] **Task 15 (cycle 2) — Execute the frozen probe once.** Fit-only fitting, Calibration-only threshold, all four Confirmation histories. Evidence: `artifacts/sprint-15/task-15-cycle-2.md` and `artifacts/sprint-15/observable-confirmation-cycle-2.json`. (NOT_RUN — zero Confirmation roots; probe frozen unexecuted; Fit/Cal support recorded.)
- [x] **Task 16 (cycle 2) — Audit Confirmation structural generalization.** Evidence: `artifacts/sprint-15/task-16-cycle-2.md` and `artifacts/sprint-15/confirmation-audit-cycle-2.json`. (Done: 4× NOT_MATERIALIZED basis + Fit/Cal support + no-retry evidence; EG5 NOT_RUN.)

**Evidence Gate D (cycle 2) — Confirmation acceptance:** **FAIL** (`artifacts/sprint-15/review-batch-d-cycle-2.md`, reviewer `Sprint15GateD2Review`, F-01 gate-blocking: 0/4 Confirmation materialized, abrupt-subtype quota-shortfalls 6–7 vs 8; EG4/EG5 NOT_RUN). **Candidate 2 retired in full**; Task 17 forbidden. Candidate-1/2 roots and evidence preserved immutable.

**Candidate 2 retired; Candidate 3 prospective revision (2026-09-10):** Gate D FAIL accepted.
Tasks 2–4 cycle-3 complete as prospective revision evidence —
`artifacts/sprint-15/task-2-cycle-3.md` (V2 continuity, gates unchanged),
`experiments/sprint15-benchmark-protocol-v3.md` + `artifacts/sprint-15/task-3-cycle-3.md`
(single-methodology-change §3a joint fair allocation + candidate-3 advance, frozen pre-outcome),
`artifacts/sprint-15/task-4-cycle-3.md` (`B_3 = 1200`, seeds 1200–1217, fresh roster/paths) —
awaiting fresh prospective evidence review. Task 5 cycle-3 (implementation) is blocked until that
review passes with zero actionable findings. No candidate-3 root exists.

**Gate A cycle-3 PASS; Candidate 3 implementation (2026-09-10):** prospective contract review
PASS (`artifacts/sprint-15/review-batch-a-cycle-3.md`, 0 actionable findings). Task 5 cycle-3
authorized; Task 10 cycle-3 remains barred pending Gate B cycle-3. No candidate-3 root exists yet
outside disposable proof.
- [x] **Task 5 (cycle 3) — Implement protocol v3 §3a joint allocation + `sprint15-v3` profile.** Smallest Appendix-A boundary only. Evidence: `artifacts/sprint-15/task-5-cycle-3.md`. (Done: starvation contrast calibrated, golden + preservation tests green.)
- [x] **Task 6 (cycle 3) — Revalidate allocator + exact quotas on candidate-3 proof.** Evidence: `artifacts/sprint-15/task-6-cycle-3.md`. (Done: exact 12/12/12/12/8/8 both roots.)
- [x] **Task 7 (cycle 3) — Revalidate provenance/manifests/seals + hidden-state isolation.** Evidence: `artifacts/sprint-15/task-7-cycle-3.md`. (Done: v3 tags, seal round trips, no leakage.)
- [x] **Task 8 (cycle 3) — Revalidate structural audit + fail-fast.** Evidence: `artifacts/sprint-15/task-8-cycle-3.md`. (Done: DESIGN-PASS ×2; candidate-2 re-audit identical.)
- [x] **Task 9 (cycle 3) — Disposable proof on H-PROOF-5/6.** Evidence: `artifacts/sprint-15/task-9-cycle-3.md`. (Done: PROOF-PASS, byte-identical determinism.)

**Evidence Gate B (cycle 3) — Implementation and proof review:** **PASS** (`artifacts/sprint-15/review-batch-b-cycle-3.md`, reviewer `Sprint15GateB3Review`, 0 actionable findings, F-01 informational). Task 10 cycle-3 authorized. Record: `artifacts/sprint-15/review-batch-b-cycle-3.md`.

- [x] **Task 10 (cycle 3) — Materialize H-DESIGN-21..24 (seeds 1200..1203) once each.** Canonical local public CLI (`sprint15-v3`) under `data/generated/sprint15-v3/DESIGN/`, at Gate B cycle-3 reviewed bytes. No retry/overwrite/replace/omit/pool. Evidence: `artifacts/sprint-15/task-10-cycle-3.md`. (Done: 4/4 first attempts exit 0.)
- [x] **Task 11 (cycle 3) — Audit every Design promotion condition (4/4 mandatory).** Evidence: `artifacts/sprint-15/task-11-cycle-3.md` and `artifacts/sprint-15/design-audit-cycle-3.json`. (Done: 4/4 DESIGN-PASS, zero errors, 9/9 EG3 each.)

**Evidence Gate C1 (cycle 3) — Design promotion:** **PASS** (`artifacts/sprint-15/review-batch-c1-cycle-3.md`, reviewer `Sprint15GateC13Review`, 0 actionable findings). Candidate 3 DESIGN-PROMOTED (unanimous 4/4). Record: `artifacts/sprint-15/review-batch-c1-cycle-3.md`.

- [x] **Task 12 (cycle 3) — Freeze the promoted candidate bytes.** Generator, §7a gate, §3a allocator, audit, dependencies/runtime, profile, protocol, role/seed, proof, and Design manifest/shard bytes. Evidence: `artifacts/sprint-15/task-12-cycle-3.md`. (Done: read-only inventory, zero bytes modified.)
- [x] **Task 13 (cycle 3) — Freeze the observable probe for Candidate 3.** Reuse reviewed probe behavior unchanged; bind Fit/Cal/Conf roles, metrics, bootstrap. Evidence: `experiments/sprint15-observable-probe-v3.md` and `artifacts/sprint-15/task-13-cycle-3.md`. (Done: probe behavior frozen, 11 tests green, pre-outcome fixtures exact.)

**Evidence Gate C2 (cycle 3) — Candidate and probe freeze:** **PASS** (`artifacts/sprint-15/review-batch-c2-cycle-3.md`, reviewer `Sprint15GateC23Review`, 0 actionable findings, F-01/F-02 informational). Candidate 3 + probe frozen. Record: `artifacts/sprint-15/review-batch-c2-cycle-3.md`.

- [x] **Task 14 (cycle 3) — Materialize Fit/Calibration/Confirmation once each.** H-FIT-16..18 (1204–1206), H-CAL-6 (1207), H-CONF-18..21 (1208–1211) via canonical CLI, first-attempt only. Evidence: `artifacts/sprint-15/task-14-cycle-3.md`. (Done: 6/8 roots exit 0; 1208/1210 fail-fast infeasible A/A1 7<8, zero roots — see record.)
- [x] **Task 15 (cycle 3) — Execute the frozen probe once.** Fit-only fitting, Calibration-only threshold, existing Confirmation histories. Evidence: `artifacts/sprint-15/task-15-cycle-3.md` and `artifacts/sprint-15/observable-confirmation-cycle-3.json`. (Done: executed once over H-CONF-19/21; coverage 2/4, EG4 unsatisfiable; every checkable threshold passes.)
- [x] **Task 16 (cycle 3) — Audit Confirmation structural generalization.** Evidence: `artifacts/sprint-15/task-16-cycle-3.md` and `artifacts/sprint-15/confirmation-audit-cycle-3.json`. (Done: DESIGN-PASS ×2 existing + absence basis; stability/holdouts pass; EG5 NOT_RUN.)

**Evidence Gate D (cycle 3) — Confirmation acceptance:** **FAIL** (`artifacts/sprint-15/review-batch-d-cycle-3.md`, reviewer `Sprint15GateD3Review`, F-01 gate-blocking: 2/4 Confirmation materialized, abrupt-subtype quota-shortfalls A/A1 7<8 on H-CONF-18/20 with zero roots; EG4/EG5 NOT_RUN). **Candidate 3 retired in full**; Task 17 forbidden. Candidate-1/2/3 roots and evidence preserved immutable.

**Candidate 3 retired; Candidate 4 prospective revision (2026-09-10):** Gate D FAIL accepted.
Tasks 2–4 cycle-4 complete as prospective revision evidence —
`artifacts/sprint-15/task-2-cycle-4.md` (V2 continuity, gates unchanged),
`experiments/sprint15-benchmark-protocol-v4.md` + `artifacts/sprint-15/task-3-cycle-4.md`
(single-methodology-change §3b exact deterministic CSP + candidate-4 advance, frozen pre-outcome),
`artifacts/sprint-15/task-4-cycle-4.md` (`B_4 = 1300`, seeds 1300–1317, fresh roster/paths).
Gate A cycle-4 re-review **PASS** accepted (`review-batch-a-cycle-4.md`, 0 actionable findings;
actionable F-01 floor-cap defect corrected: protocol v4 + task-3-cycle-4 amended to exact floor caps
robot ≤ 22 and program ≤ 14 with re-proof under stricter bounds (status 0, validated, repeat
identical on 1208/1210); re-freeze digest recorded in task-3-cycle-4. **Task 5 cycle-4 authorized.**
No candidate-4 verdict root exists (disposable proof roots under `proof-candidate-4/` only).
- [x] **Task 5 (cycle 4) — Implement protocol v4 §3b exact CSP allocator + `sprint15-v4` profile.** Smallest Appendix-A boundary only. Evidence: `artifacts/sprint-15/task-5-cycle-4.md`. (Done: exact path + v4 profile/roster/seal; 40/40 focused tests.)
- [x] **Task 6 (cycle 4) — Revalidate allocator + exact quotas on candidate-4 proof.** Evidence: `artifacts/sprint-15/task-6-cycle-4.md`. (Done: exact 12/12/12/12/8/8 both roots, solver status 0.)
- [x] **Task 7 (cycle 4) — Revalidate provenance/manifests/seals + hidden-state isolation.** Evidence: `artifacts/sprint-15/task-7-cycle-4.md`. (Done: v4 tags, deterministic solver subset, seal round trips, no leakage.)
- [x] **Task 8 (cycle 4) — Revalidate structural audit + fail-fast.** Evidence: `artifacts/sprint-15/task-8-cycle-4.md`. (Done: DESIGN-PASS ×2; candidate-2/3 re-audits identical.)
- [x] **Task 9 (cycle 4) — Disposable proof on H-PROOF-7/8.** Evidence: `artifacts/sprint-15/task-9-cycle-4.md`. (Done: PROOF-PASS, byte-identical determinism.)

**Evidence Gate B (cycle 4) — Implementation and proof review:** **PASS** (`artifacts/sprint-15/review-batch-b-cycle-4.md`, 0 actionable findings). Task 10 cycle-4 authorized; Tasks 12+ remain barred pending Gate C1 cycle-4.
- [x] **Task 10 (cycle 4) — Materialize H-DESIGN-25..28 (seeds 1300..1303) once each.** Canonical local public CLI (`sprint15-v4`) under `data/generated/sprint15-v4/DESIGN/`, at Gate B cycle-4 reviewed bytes. No retry/overwrite/replace/omit/pool. Evidence: `artifacts/sprint-15/task-10-cycle-4.md`. (Done: 4/4 first attempts exit 0, 316 s total.)
- [x] **Task 11 (cycle 4) — Audit every Design promotion condition (4/4 mandatory).** Evidence: `artifacts/sprint-15/task-11-cycle-4.md` and `artifacts/sprint-15/design-audit-cycle-4.json`. (Done: 4/4 DESIGN-PASS, zero errors, 9/9 EG3 each, witness re-verified.)

**Evidence Gate C1 (cycle 4) — Design promotion:** **PASS** (`artifacts/sprint-15/review-batch-c1-cycle-4.md`, 0 findings). Candidate 4 DESIGN-PROMOTED (unanimous 4/4). Record: `artifacts/sprint-15/review-batch-c1-cycle-4.md`.
- [x] **Task 12 (cycle 4) — Freeze candidate-4 code, physics, allocator, runtime, and Design bytes.** Evidence: `artifacts/sprint-15/task-12-cycle-4.md`. (Done: read-only inventory, all digests MATCH Gate B quotes.)
- [x] **Task 13 (cycle 4) — Freeze the observable probe for candidate 4.** Evidence: `experiments/sprint15-observable-probe-v4.md` and `artifacts/sprint-15/task-13-cycle-4.md`. (Done: probe bytes untouched, 11/11 fixtures green, pre-outcome.)

**Evidence Gate C2 — Candidate and probe freeze:** **PASS** (`artifacts/sprint-15/review-batch-c2-cycle-4.md`, 0 actionable findings). Task 14 cycle-4 authorized; Tasks 17+ (Sealed) remain barred pending Gate D.
- [x] **Task 14 (cycle 4) — Materialize Fit/Calibration/Confirmation roots once each.** Evidence: `artifacts/sprint-15/task-14-cycle-4.md`. (Done with miss: 7/8 first attempts exit 0; H-FIT-21 proven-infeasible, zero bytes, no retry.)
- [x] **Task 15 (cycle 4) — Run the frozen probe once over Confirmation.** Evidence: `artifacts/sprint-15/task-15-cycle-4.md` and `artifacts/sprint-15/observable-confirmation-cycle-4.json`. (Done: 2-Fit basis macros 0.78/LCB 0.76/directional 4; H-CONF-24 FAR 0.0536 breach preserved, no tuning.)
- [x] **Task 16 (cycle 4) — Audit all four Confirmation histories.** Evidence: `artifacts/sprint-15/task-16-cycle-4.md` and `artifacts/sprint-15/confirmation-audit-cycle-4.json`. (Done: DESIGN-PASS ×4; H-CONF-25 stability 1.28 breach recorded; REJECTED-CANDIDATE basis for Gate D.)

**Evidence Gate D — Independent observable and structural confirmation:** **FAIL** (`artifacts/sprint-15/review-batch-d-cycle-4.md`, 3 actionable findings F-01/F-02/F-03, 4 informational). Candidate 4 retired in full (`REJECTED-CANDIDATE`); Tasks 17+ cycle-4 forbidden; candidate-4 assets immutable.

**Candidate 4 retired; Candidate 5 prospective revision (2026-09-10):** Gate D FAIL accepted
(F-01 H-FIT-21 exact-CSP infeasible — 2/3 Fit roster; F-02 H-CONF-24 FAR 0.0536 > 0.05;
F-03 truncated 2-Fit probe basis; F-04 informational H-CONF-25 abrupt stability 1.2759).
- [x] **Task 2 (cycle 5) — Record V2 continuity, retirement, and diagnosis.** Evidence: `artifacts/sprint-15/task-2-cycle-5.md`. (Done: V2 digest match; 1306 counting-infeasible via ablation + 41-history slack census.)
- [x] **Task 3 (cycle 5) — Freeze the revised causal protocol (v5).** Evidence: `experiments/sprint15-benchmark-protocol-v5.md` and `artifacts/sprint-15/task-3-cycle-5.md`. (Done: single-change calendar-volume margin + C5 advance, pre-outcome.)
- [x] **Task 4 (cycle 5) — Bind candidate-5 roles, seeds, runtime, no-retry rules.** `B_5 = 1400`, seeds 1400–1417, roots under `data/generated/sprint15-v5/`. Evidence: `artifacts/sprint-15/task-4-cycle-5.md`. (Done: bands disjoint, paths absent, no C5 bytes.)

**Evidence Gate A (cycle 5) — Prospective contract review:** **PASS** (`artifacts/sprint-15/review-batch-a-cycle-5.md`, 0 actionable findings). Tasks 5–9 cycle-5 authorized; Candidate 5 Design remains barred until Gate B cycle-5.
- [x] **Task 5 (cycle 5) — Implement protocol v5 §1 volume profile + `sprint15-v5` bindings.** Smallest Appendix-A boundary only. Evidence: `artifacts/sprint-15/task-5-cycle-5.md`. (Done: volume profile + bindings; 54/54 focused tests; miss is methodological, not implementational.)
- [x] **Task 6 (cycle 5) — Revalidate allocator + exact quotas on candidate-5 proof.** Evidence: `artifacts/sprint-15/task-6-cycle-5.md`. (Done: 64-anchors optimal both seeds; controls 13/11 < 48.)
- [x] **Task 7 (cycle 5) — Revalidate provenance/manifests/seals + hidden-state isolation.** Evidence: `artifacts/sprint-15/task-7-cycle-5.md`. (Done vacuous: zero roots, absence verified.)
- [x] **Task 8 (cycle 5) — Revalidate structural audit + fail-fast.** Evidence: `artifacts/sprint-15/task-8-cycle-5.md`. (Done: C4 DESIGN 4/4 identical; fail-fast ×2 correct.)
- [x] **Task 9 (cycle 5) — Disposable proof on H-PROOF-9/10.** Evidence: `artifacts/sprint-15/task-9-cycle-5.md`. (Done with MISS: PROOF-FAIL, controls 13/11 < 48 both roots, REJECTED-CANDIDATE basis.)

**Evidence Gate B (cycle 5) — Implementation and proof review:** **FAIL** (`artifacts/sprint-15/review-batch-b-cycle-5.md`, 1 actionable finding: control-window collapse 13/11 < 48 under 1.5× volume at fixed span). Candidate 5 retired in full (`REJECTED-CANDIDATE`); seeds 1400–1417 burned; Design barred; candidate-5 assets absent-by-fail-fast; prior assets immutable.

- [x] **Task 2 (cycle 6) — Record V2 continuity, retirement, and tradeoff diagnosis.** Evidence: `artifacts/sprint-15/task-2-cycle-6.md`. (Done: V2 digest match; C5 tradeoff diagnosed; fixed-budget direction with projected 0/41 below need.)
- [x] **Task 3 (cycle 6) — Freeze the revised causal protocol (v6).** Evidence: `experiments/sprint15-benchmark-protocol-v6.md` and `artifacts/sprint-15/task-3-cycle-6.md`. (Done: volume revert + cohort reallocation + C6 advance, pre-outcome.)
- [x] **Task 4 (cycle 6) — Bind candidate-6 roles, seeds, runtime, no-retry rules.** `B_6 = 1500`, seeds 1500–1517, roots under `data/generated/sprint15-v6/`. Evidence: `artifacts/sprint-15/task-4-cycle-6.md`. (Done: bands disjoint, paths absent, no C6 bytes.)

**Evidence Gate A (cycle 6) — Prospective contract review:** **PASS** (`artifacts/sprint-15/review-batch-a-cycle-6.md`, 0 actionable findings). Tasks 5–9 cycle-6 authorized; Candidate 6 Design remains barred until Gate B cycle-6.
- [x] **Task 5 (cycle 6) — Implement protocol v6 §1/§1a reallocation profile + `sprint15-v6` bindings.** Smallest Appendix-A boundary only. Evidence: `artifacts/sprint-15/task-5-cycle-6.md`. (Done: volume revert + rate reallocation; 57/57 focused tests.)
- [x] **Task 6 (cycle 6) — Revalidate allocator + exact quotas on candidate-6 proof.** Evidence: `artifacts/sprint-15/task-6-cycle-6.md`. (Done: exact 12/12/12/12/8/8 both roots, controls 50/59, worst A-slack 3.)
- [x] **Task 7 (cycle 6) — Revalidate provenance/manifests/seals + hidden-state isolation.** Evidence: `artifacts/sprint-15/task-7-cycle-6.md`. (Done: v6 tags, deterministic solver subset, seal round trips, no leakage.)
- [x] **Task 8 (cycle 6) — Revalidate structural audit + fail-fast.** Evidence: `artifacts/sprint-15/task-8-cycle-6.md`. (Done: DESIGN-PASS ×2; C4 re-audit identical.)
- [x] **Task 9 (cycle 6) — Disposable proof on H-PROOF-11/12.** Evidence: `artifacts/sprint-15/task-9-cycle-6.md`. (Done: PROOF-PASS, byte-identical determinism.)

**Evidence Gate B (cycle 6) — Implementation and proof review:** **PASS** (`artifacts/sprint-15/review-batch-b-cycle-6.md`, 0 actionable findings). Task 10 cycle-6 authorized; Tasks 12+ remain barred pending Gate C1 cycle-6.
- [x] **Task 10 (cycle 6) — Materialize H-DESIGN-33..36 (seeds 1500..1503) once each.** Canonical local public CLI (`sprint15-v6`) under `data/generated/sprint15-v6/DESIGN/`, at Gate B cycle-6 reviewed bytes. No retry/overwrite/replace/omit/pool. Evidence: `artifacts/sprint-15/task-10-cycle-6.md`. (Done: 4/4 first attempts exit 0, 1055 s total.)
- [x] **Task 11 (cycle 6) — Audit every Design promotion condition (4/4 mandatory).** Evidence: `artifacts/sprint-15/task-11-cycle-6.md` and `artifacts/sprint-15/design-audit-cycle-6.json`. (Done: 4/4 DESIGN-PASS, zero errors, 9/9 EG3 each, witness re-verified.)

**Evidence Gate C1 (cycle 6) — Design promotion:** **PASS** (`artifacts/sprint-15/review-batch-c1-cycle-6.md`, 0 findings). Candidate 6 DESIGN-PROMOTED (unanimous 4/4). Record: `artifacts/sprint-15/review-batch-c1-cycle-6.md`.
- [x] **Task 12 (cycle 6) — Freeze candidate-6 code, physics, allocator, runtime, and Design bytes.** Evidence: `artifacts/sprint-15/task-12-cycle-6.md`. (Done: read-only inventory, all digests MATCH Gate B quotes, C4 preservation verified.)
- [x] **Task 13 (cycle 6) — Freeze the observable probe for candidate 6.** Evidence: `experiments/sprint15-observable-probe-v6.md` and `artifacts/sprint-15/task-13-cycle-6.md`. (Done: probe bytes untouched, 11/11 fixtures green, pre-outcome; authorized by Gate C1 §8 addendum.)

**Evidence Gate C2 (cycle 6) — Candidate and probe freeze:** **PASS** (`artifacts/sprint-15/review-batch-c2-cycle-6.md`, 0 actionable findings, 2 informational). Task 14 cycle-6 authorized; Tasks 15+ remain barred pending Task 14 completion and subsequent gates.
- [x] **Task 14 (cycle 6) — Materialize Fit/Calibration/Confirmation roots once each.** Evidence: `artifacts/sprint-15/task-14-cycle-6.md`. (Done with miss: 7/8 first attempts exit 0; H-CONF-31 W1 pool 8 < 12 proven-infeasible, zero bytes, no retry; ablation preserved.)
- [ ] **Task 15 (cycle 6) — Run the frozen probe once over Confirmation.** Will not run — candidate miss; Gate D pending.
- [ ] **Task 16 (cycle 6) — Audit all four Confirmation histories.** Will not run — candidate miss; Gate D pending.

**Evidence Gate D — Independent observable and structural confirmation:** **FAIL** (`artifacts/sprint-15/review-batch-d-cycle-6.md`, 1 actionable finding F-01: H-CONF-31 W1 pool 8 < 12; 5 informational). Candidate 6 retired in full (`REJECTED-CANDIDATE`); Tasks 15–16 cycle-6 NOT_RUN; Sealed/Tasks 17+ forbidden; candidate-6 assets immutable.

**Candidate 6 retired; Candidate 7 prospective revision (2026-09-10):** Gate D FAIL accepted (W1-subtype tail starvation under the v6 W trim; A side fixed). Tasks 2–4 cycle-7 below are the prospective revision evidence; no candidate-7 root exists yet.
- [x] **Task 2 (cycle 7) — Record V2 continuity, retirement, and subtype diagnosis.** Evidence: `artifacts/sprint-15/task-2-cycle-7.md`. (Done: V2 digest match; shared subtype-imbalance root cause + preflight rationale.)
- [x] **Task 3 (cycle 7) — Freeze the revised causal protocol (v7).** Evidence: `experiments/sprint15-benchmark-protocol-v7.md` and `artifacts/sprint-15/task-3-cycle-7.md`. (Done: rate revert + stratification + §1b preflight + C7 advance, pre-outcome.)
- [x] **Task 4 (cycle 7) — Bind candidate-7 roles, seeds, runtime, no-retry rules.** `B_7 = 1600`, seeds 1600–1617, roots under `data/generated/sprint15-v7/`. Evidence: `artifacts/sprint-15/task-4-cycle-7.md`. (Done: bands disjoint, paths absent, preflight roster frozen, no C7 bytes.)

**Evidence Gate A (cycle 7) — Prospective contract re-review:** **PASS** (`artifacts/sprint-15/review-batch-a-cycle-7-r2.md`, 0 actionable findings; corrected protocol digest `a1fc09db…`). Tasks 5–9 cycle-7 authorized; Candidate 7 Design remains barred until Gate B cycle-7.
- [x] **Task 5 (cycle 7) — Implement protocol v7 App. A (stratification + preflight + `sprint15-v7` bindings).** Smallest boundary only. Evidence: `artifacts/sprint-15/task-5-cycle-7.md`. (Done: flag + emission sites + preflight runner + bindings; 63/63 focused tests.)
- [x] **Task 6 (cycle 7) — Revalidate allocator + exact quotas on candidate-7 proof.** Evidence: `artifacts/sprint-15/task-6-cycle-7.md`. (Done: exact 12/12/12/12/8/8 both roots, controls 71/71.)
- [x] **Task 7 (cycle 7) — Revalidate provenance/manifests/seals + hidden-state isolation.** Evidence: `artifacts/sprint-15/task-7-cycle-7.md`. (Done: v7 tags, deterministic solver subset, seal round trips, no leakage.)
- [x] **Task 8 (cycle 7) — Revalidate structural audit + fail-fast.** Evidence: `artifacts/sprint-15/task-8-cycle-7.md`. (Done: DESIGN-PASS ×2; C6 re-audit identical.)
- [x] **Task 9 (cycle 7) — Replay 1306/1509 + proof H-PROOF-13/14 + roster preflight 1600–1617.** Evidence: `artifacts/sprint-15/task-9-cycle-7.md` and `artifacts/sprint-15/preflight-cycle-7.json`. (Done: replay 2/2 feasible, PROOF-PASS, preflight 18/18 PASS; no Design.)

**Evidence Gate B (cycle 7) — Implementation and proof review:** **PASS** (`artifacts/sprint-15/review-batch-b-cycle-7.md`, 0 actionable findings, 6 informational). Task 10 cycle-7 authorized; Tasks 12+ remain barred pending Gate C1 cycle-7.
- [x] **Task 10 (cycle 7) — Materialize H-DESIGN-37..40 (seeds 1600..1603) once each.** Canonical local public CLI (`sprint15-v7`) under `data/generated/sprint15-v7/DESIGN/`, at Gate B cycle-7 reviewed bytes. No retry/overwrite/replace/omit/pool. Evidence: `artifacts/sprint-15/task-10-cycle-7.md`. (Done: 4/4 first attempts exit 0, 458 s total.)
- [x] **Task 11 (cycle 7) — Audit every Design promotion condition (4/4 mandatory).** Evidence: `artifacts/sprint-15/task-11-cycle-7.md` and `artifacts/sprint-15/design-audit-cycle-7.json`. (Done: 4/4 DESIGN-PASS, zero errors, 9/9 EG3 each, witness re-verified, worst A-slack 4.)

**Evidence Gate C1 (cycle 7) — Design promotion:** **PASS** (`artifacts/sprint-15/review-batch-c1-cycle-7.md`, 0 findings). Candidate 7 DESIGN-PROMOTED (unanimous 4/4). Record: `artifacts/sprint-15/review-batch-c1-cycle-7.md`.
- [x] **Task 12 (cycle 7) — Freeze candidate-7 code, physics, stratification, preflight, allocator, runtime, and Design bytes.** Evidence: `artifacts/sprint-15/task-12-cycle-7.md`. (Done: read-only inventory, all digests MATCH Gate B quotes, C6 preservation verified.)
- [x] **Task 13 (cycle 7) — Freeze the observable probe for candidate 7.** Evidence: `experiments/sprint15-observable-probe-v7.md` and `artifacts/sprint-15/task-13-cycle-7.md`. (Done: probe bytes untouched, 11/11 fixtures green, pre-outcome; authorized by Gate C1 §8 addendum.)

**Evidence Gate C2 (cycle 7) — Candidate and probe freeze:** **PASS** (`artifacts/sprint-15/review-batch-c2-cycle-7.md`, 0 actionable findings, 2 informational). Task 14 cycle-7 authorized; Tasks 15+ remain barred pending Task 14 completion and subsequent gates.
- [x] **Task 14 (cycle 7) — Materialize Fit/Calibration/Confirmation roots once each.** Evidence: `artifacts/sprint-15/task-14-cycle-7.md`. (Done: 8/8 first attempts exit 0, 834 s total; preflight agreement FULL 8/8.)
- [x] **Task 15 (cycle 7) — Run the frozen probe once over Confirmation.** Evidence: `artifacts/sprint-15/task-15-cycle-7.md` and `artifacts/sprint-15/observable-confirmation-cycle-7.json`. (Done: full 3/3-Fit basis, macros 0.795/0.821/0.764, LCB 0.766, directional 4/4, FAR ≤ 0.0462, no tuning.)
- [x] **Task 16 (cycle 7) — Audit all four Confirmation histories.** Evidence: `artifacts/sprint-15/task-16-cycle-7.md` and `artifacts/sprint-15/confirmation-audit-cycle-7.json`. (Done: DESIGN-PASS ×4, 9/9 EG3 each; stability medians in band with per-history variance recorded for Gate D.)

**Evidence Gate D — Independent observable and structural confirmation:** **PASS** (`artifacts/sprint-15/review-batch-d-cycle-7.md`, 0 actionable findings, 4 informational; per-history variance classified under the median stability rule). Candidate 7 CONFIRMED (`CONFIRMED-CANDIDATE`); Tasks 17–19 explicitly authorized; Sealed quarantine bounds apply.
- [x] **Task 17 (cycle 7) — Materialize four Sealed histories once.** H-SEAL-37..40 (seeds 1612..1615) via canonical frozen `sprint15-v7` CLI; structural metadata only, zero probe/score contact. Evidence: `artifacts/sprint-15/task-17-cycle-7.md`. (Done: 4/4 first attempts exit 0, 423 s total.)
- [x] **Task 18 (cycle 7) — Audit and seal every Sealed history.** Evidence: `artifacts/sprint-15/task-18-cycle-7.md`, `artifacts/sprint-15/sealed-audit-cycle-7.json`, `seals-cycle-7.json`. (Done: 4/4 structural PASS, seals round-tripped, EG6 medians in band.)
- [x] **Task 19 (cycle 7) — Issue the candidate verdict.** Evidence: `artifacts/sprint-15/task-19-cycle-7.md` and `artifacts/sprint-15/exit-gate-matrix-cycle-7.json`. (Done: accepted `MEASURABLE`; EG0–EG6 PASS, Gate E PASS, Gate F-r12 PASS, deep-r3 PASS, EG7 PASS; see §17 final addendum.)

**Evidence Gate E — Sealed and candidate-verdict review:** **PASS** (`artifacts/sprint-15/review-batch-e-cycle-7.md`, 0 actionable findings, 5 informational). Candidate 7 accepted `MEASURABLE`; Tasks 20–22 DONE; reviewed handoff to Sprint 16 ACTIVE (post-final check required).
- [x] **Task 20 — Prepare the Sprint 16 benchmark handoff.** Reviewed handoff ACTIVE. Evidence: `artifacts/sprint-15/task-20.md`. (Done: exact C7 identifiers/uses, Sealed forbidden, frozen metric contract, limits; handoff active per §13 final addendum.)
- [x] **Task 21 — Consolidate the complete evidence ledger.** Evidence: `artifacts/sprint-15/task-21.md` and `artifacts/sprint-15/review-index.md`. (Done, repair round: 50 reviews + 95 narratives across cycles 1–7 (109 with Tasks 20–22 + corrections r1–r10 plus repair) + 18 JSONs; all Gate F attempts + deep reviews + failed final review indexed with cutoff; zero open findings except the final recheck.)

**Evidence Gate F — Batch F review (all attempts, preserved):** PASS (`review-batch-f.md`, EG7-PASS classification superseded BY live addenda/index/plans — file byte-identical), PASS (`review-batch-f-r2.md`, likewise superseded — file byte-identical), **FAIL** (`review-batch-f-r3.md`, D-01..D-04; corrected in r2 round), **FAIL** (`review-batch-f-r4.md`, R4-01 digest, R4-02 census, R4-03 symmetry, R4-04 sweep scope; corrected in r3 round), **FAIL** (`review-batch-f-r5.md`, R5-01 citation, R5-02 deep row, R5-03/04/06 currency, R5-05 pointers; corrected in r4 round), **FAIL** (`review-batch-f-r6.md`, R6-01 PLAN index-row staleness; corrected in r5 round), **PASS** (`review-batch-f-r7.md`, 0 actionable; repeated deep review authorized). **Deep review attempt 1:** findings CLOSED at Gate F-r7. **Deep review attempt 2:** **REJECT** (R2-01/R2-02; corrected in r6 round). **FAIL** (`review-batch-f-r8.md`, F-r8-01 supersession registry, F-r8-02 stale closure text; corrected in r7 round). **FAIL** (`review-batch-f-r9.md`, F-r9-01 registry exhaustiveness, F-r9-02 single state, F-r9-03 handoff sentence; corrected in r8 round). **PASS** (`review-batch-f-r10.md`, 0 actionable, 2 INFO precision notes; corrected in r9 round). **FAIL** (`review-batch-f-r11.md`, F-r11-01 handoff sentence, F-r11-02 cross-reference; corrected in r10 round). **PASS** (`review-batch-f-r12.md`, 0 actionable). **Deep review attempt 3:** **PASS** (0 actionable, 0 unanswered; Task 22 authorized). **EG7: PASS. Final verdict: `MEASURABLE`.** Task 22 DONE; Sprint 16 handoff ACTIVE (post-final check required); no finalization beyond the mandatory post-final evidence review.

**Evidence Gate E — Sealed and candidate-verdict review:** An `evidence-reviewer` verifies Tasks 17–19, sealed isolation, hashes, gate precedence, and candidate verdict. Any Sealed miss retires candidate `N`; it cannot select a replacement seed within that candidate. A rejected candidate advances prospectively to candidate `N + 1`. Record: `artifacts/sprint-15/review-batch-e-cycle-<N>.md`.

### Batch F — Handoff, evidence consolidation, and finalization (Tasks 20–22 completed; Task 22 DONE per `task-22.md` on deep-review-r3)
- [x] **Task 22 — Deep-review finalization as MEASURABLE.** Final differential deep review R3 PASS (0 actionable, 0 unanswered) authorized finalization; atomic transaction executed (F-r12 + deep-r3 indexed, counts 49/108, matrix mutated to EG7 PASS / final MEASURABLE / Sprint16 READY and repinned, final state propagated). Evidence: `artifacts/sprint-15/task-22.md` on `deep-review-final-r3.md` (never failed deep1). Sprint 15 COMPLETE as `MEASURABLE`, subject to the mandatory post-final evidence review.

## Acceptance Criteria

1. Sprint 14 remains immutable and retains final verdict `NOT_MEASURABLE`.
2. The new balanced profile is explicitly a case-control evaluation benchmark, not a fleet prevalence model.
3. All event quotas are placed prospectively from a frozen calendar and satisfy unchanged causal-window, reset, quarantine, censoring, physical, and no-overlap rules.
4. Every Design history independently meets `P >= 13`, `W >= 13`, `A >= 10`, total positives `>= 38`, negatives `>= 32`, robot-days `>= 188`, identity/program support, lead support, and all concentration caps.
5. Fit has at least 200 verified-healthy files and 6,000 valid patches; Calibration has at least 40 eligible verified-healthy rows.
6. The fixed observable probe passes every inherited threshold on all required Confirmation evidence: macro `P+W` AUROC `>= 0.65`, history-block lower confidence bound `> 0.50`, at least three histories above `0.55`, macro `P >= 0.70`, macro `W >= 0.60`, required recall/lead, and per-history false-alert episodes `<= 0.05` per robot-day.
7. All four Confirmation histories independently pass structural, concentration, stability, holdout, EG1, and EG3 gates.
8. All four Sealed histories independently pass structural, concentration, stability, holdout, integrity, and seal checks without any score touching them.
9. No root is retried, replaced, omitted, pooled for rescue, or selected by outcome.
10. Every batch has a fresh evidence review with zero actionable findings.
11. Sprint 15 has exactly one permitted terminal verdict: final `MEASURABLE` after a zero-actionable sprint-wide deep review.
12. Sprint 16 remains blocked unless Sprint 15 is final `MEASURABLE` and its reviewed handoff exists.

## Candidate Rejection and Continuation Rules

- An infeasible allocator, undefined metric, structural miss, observable-probe miss, Confirmation miss, Sealed miss, evidence-review finding, or deep-review rejection retires only the active candidate; it does not close Sprint 15.
- Every new candidate requires an evidence-reviewed diagnosis, a prospectively frozen protocol version, the next deterministic seed namespace, and collision-free roots.
- Within a candidate, never retry, replace, omit, pool, or regenerate a history. Candidate iteration is methodology revision, not seed search.
- Generator, scheduler, allocator, nuisance-envelope, and causal-signal methodology may change only prospectively when a reviewer accepts the justification before candidate outcomes exist. Global acceptance floors, metrics, observable-probe thresholds, no-pooling rules, and verdict semantics remain invariant across Sprint 15.
- Failed Confirmation and Sealed data may identify which frozen gate failed but may not be reused as training, Calibration, or Design data.
- A favorable observable probe cannot rescue a structural failure or authorize representation, anomaly, warning, or calibrated-risk claims.
- Sprint 15 remains `IN_PROGRESS` until one candidate passes the complete gate chain and deep review returns zero actionable findings.
- Finalization writes only `MEASURABLE`; candidate failures remain preserved, indexed evidence.

## Explicit Non-Goals

- Reopening Sprint 14 or adding Sprint 14 Cycle 4.
- Estimating natural fleet failure prevalence from the balanced benchmark.
- Tuning representation, anomaly, warning, survival, or calibrated-risk models.
- Changing the accepted seven-day causal horizon or removing reset/quarantine safeguards to increase counts.
- Selecting favorable seeds or regenerating failed histories.
- Scoring Sealed roots.
- Starting Sprint 16 attribution before final `MEASURABLE` and reviewed handoff.
