# Benchmark Measurability Exit Gates

## Purpose

This document is the normative acceptance contract for Sprint 14 benchmark recovery. It answers one question before representation attribution or risk modeling resumes:

> Does the benchmark contain enough independent, observable, causal early-warning evidence to estimate event ranking, positive lead time, recall, persistence, and false-alert behavior across unseen histories without pooling away the history unit or selecting favorable seeds?

Sprint 14 may change generator physics or scheduling only through versioned, prospectively reviewed protocol amendments. It may not tune a learned representation, anomaly scorer, risk model, production threshold, or sealed history.

## Verdicts

Sprint 14 returns exactly one verdict:

- **`MEASURABLE`**: every mandatory gate EG0–EG7 passes, every confirmation and sealed history passes its per-history structural floors, the fixed observable probe passes on confirmation histories, and the final evidence and deep-review gates have zero actionable findings.
- **`NOT_MEASURABLE`**: required roots exist and audits execute, but any mandatory structural, observability, replication, or generalization gate fails after the allowed diagnostic cycles.
- **`UNAVAILABLE`**: a required root cannot be generated or verified, provenance is incomplete, or the frozen metric implementation cannot produce a required estimand despite the structural units being present.

`NOT_MEASURABLE` and `UNAVAILABLE` both block Sprint 15. There is no partial advancement state.

## Non-Negotiable Boundaries

1. **Whole histories are the independence unit.** Per-history floors are mandatory; pooled totals never rescue a failed history.
2. **No favorable-seed selection.** Seeds and roles are frozen before materialization. Every declared history is retained, including failures.
3. **No learned-model influence.** Representation, anomaly, warning, and calibrated-risk outputs cannot select generator settings, histories, thresholds, or gates.
4. **One fixed observable probe is allowed only as a DGP sanity check.** Its features, fit/calibration roles, thresholds, and metrics are frozen before outcomes. It uses causal observable telemetry only; simulator health, future state, labels, cohort IDs, and failure times are forbidden as inputs.
5. **Confirmation is not design data.** A failed confirmation retires the entire confirmation set. Any correction requires a protocol version bump and fresh design and confirmation roots.
6. **Sealed evaluation remains sealed.** Sealed roots receive structural audit and seal verification only. No probe, model, threshold, or score is evaluated on them during Sprint 14.
7. **Every semantic change is prospective.** Changes to physics, scheduling, maintenance, quarantine, windows, floors, roles, probes, or verdict precedence require a new protocol version approved before materialization.
8. **At most three diagnostic cycles.** A cycle ends at an evidence review. Failure after cycle three closes Sprint 14 as `NOT_MEASURABLE`; it does not trigger an unbounded search.

## Frozen Role Topology

Each candidate protocol must predeclare non-overlapping whole-history roles:

| Role | Minimum histories | Permitted use |
|---|---:|---|
| Design diagnostic | 4 | Diagnose structural misses and estimate safety margins. May be replaced only by a new protocol version. |
| Fit | 3 | Fit the fixed observable probe and provide downstream healthy reference support. |
| Calibration | 1 | Select only the probe operating threshold and verify healthy support. |
| Confirmation | 4 | One-shot observability and generalization gate after the candidate DGP is frozen. Never reused as design data. |
| Sealed evaluation | 4 | Structural audit and sealing only in Sprint 14; reserved for future independent evaluation. |

The protocol must also reserve, outside Fit and Calibration:

- at least one complete robot identity;
- at least one complete program identity;
- at least one failure-mechanism subtype from each predictable cohort, progressive (`P`) and weak precursor (`W`).

The reserved robot, program, and mechanism subtypes must occur in Confirmation and Sealed roles. Abrupt (`A`) failures remain in all overall ledgers but are never required to be predictably rankable.

## Hard Structural Floors

The following floors apply **to every Design, Confirmation, and Sealed history independently** after all frozen temporal-view, maintenance, quarantine, censoring, and reset exclusions:

| Quantity | Hard floor | Design promotion target |
|---|---:|---:|
| Evaluable progressive events (`P`) | 10 | 13 |
| Evaluable weak-precursor events (`W`) | 10 | 13 |
| Evaluable abrupt events (`A`) | 8 | 10 |
| All evaluable positive events | 30 | 38 |
| Deterministic non-overlapping same-history/robot negative-control windows | 25 | 32 |
| Evaluable robot-days | 150 | 188 |
| Distinct robots contributing eligible positives | 6 | 6 |
| Distinct robots contributing negative controls | 6 | 6 |
| Programs represented in each `P` and `W` cohort | 2 | 2 |

Additional concentration limits apply per history:

- no robot contributes more than 35% of eligible positive events;
- no robot contributes more than 40% of negative controls;
- no program contributes more than 60% of either predictable cohort;
- each of `P`, `W`, and `A` contributes between 15% and 60% of all evaluable positives.

For positive-lead measurability, at least 80% of `P` events and 80% of `W` events in each history must have:

- at least three eligible file endpoints inside the causal seven-day pre-failure horizon;
- at least one eligible endpoint ending at least 24 hours before failure;
- no maintenance or recommissioning reset inside the evaluated window;
- a clean same-history/robot baseline window under the frozen eligibility predicate.

Downstream support floors apply to the frozen Fit and Calibration roles:

- Fit: at least 200 verified-healthy files and 6,000 valid patches in aggregate, with every fallback or sparse conditioning group reported;
- Calibration: at least 40 verified-healthy eligible rows;
- no sealed, confirmation, quarantined, censored, or maintenance-overlapping row may enter Fit or Calibration.

The hard floors are acceptance criteria. The higher design targets are mandatory promotion margins: a candidate DGP cannot advance from Design to Confirmation by merely touching the floor.

## Threshold Rationale

The numbers are fixed for identifiable reasons rather than chosen to make a generated sample pass:

- `P >= 10` and `W >= 10` ensure each predictable cohort can support the existing reportable median/IQR minimum of 10 recalled events under a full-recall fixture.
- `A >= 8`, 25 negative controls, and 150 robot-days preserve the accepted Sprint 13 statistical floors; benchmark recovery must solve the data support problem rather than weaken it.
- Design promotion targets are the hard floors multiplied by 1.25 and rounded up where needed. This restores the safety margin Sprint 13 intended but did not realize.
- Four Confirmation and four Sealed histories preserve the accepted minimum independent-history replication unit while keeping whole histories—not files—as the generalization unit.
- Robot/program concentration caps prevent a nominal event-count pass from being carried by one identity or one operating program.
- The observable-probe AUROC, recall, lead, and false-alert thresholds are minimum evidence that causal telemetry contains a non-chance, advance-warning signal. They do not represent a production target.
- Requiring both a history-block confidence bound above 0.5 and directional performance in at least three of four Confirmation histories prevents one pooled or favorable history from carrying EG4.
- The `[0.75, 1.25]` median stability band permits stochastic history variation while rejecting a candidate whose support collapses or inflates materially outside Design.

## Exit Gates

### EG0 — Prospective protocol and provenance freeze

Pass only when all of the following are recorded and independently reviewed before materialization:

- protocol version and content digest;
- exact generator configuration and causal physics;
- role names, history identities, and seeds;
- failure-cohort definitions and subtype holdouts;
- temporal views, seven-day horizon, censoring, maintenance/reset, quarantine, and control-window predicates;
- every hard floor, design target, concentration limit, probe metric, and verdict-precedence rule in this document;
- canonical commit, local workstation and locked `uv` runtime, local output roots, and seal format;
- diagnostic-cycle number, maximum cycle count, and retirement ledger.

Any missing field or post-generation semantic edit fails EG0 and retires the affected roots.

### EG1 — Causality, determinism, and role isolation

Pass only when a public-entry-point fixture and every materialized history show:

- byte-identical manifest and event ledger on two fresh-process reloads of the same root;
- strictly chronological file/event ordering;
- one robot performs at most one operation at a time;
- maintenance/recommissioning resets split eligibility and alert episodes exactly once;
- no future health, failure time, cohort label, anomaly label, or diagnostic-only metadata enters model-visible fields;
- role membership is mutually exclusive and roster-exact;
- all required `P`, `W`, and `A` physical bounds and manifestation rules pass;
- zero seal or manifest digest mismatch.

Every check is all-or-nothing: required pass count is 100%, with zero leakage or integrity exceptions.

### EG2 — Design structural margin

Run on at least four Design histories. Pass only when every Design history meets every design promotion target and concentration limit, and Fit/Calibration support floors are met by the planned role configuration.

A pooled pass, median pass, or three-of-four pass is a failure. A failed EG2 returns to diagnosis under a new protocol version and consumes one diagnostic cycle.

### EG3 — Metric-contract computability

Run deterministic score fixtures on Design and later Confirmation histories only. For every history, require:

- constant scores produce tie-aware event AUROC exactly `0.5`;
- strictly correct ordering produces event AUROC/G-rank exactly `1.0`;
- reversed ordering produces event AUROC/G-rank exactly `0.0`;
- onset-only alerts produce first-alert lead time exactly `0.0` days;
- the frozen advance-alert fixture produces positive lead time and a reportable median/IQR with at least 10 recalled `P` events and 10 recalled `W` events;
- the false-alert fixture reproduces its analytically declared episode count and episodes-per-robot-day rate;
- E1–E5 outputs are finite, schema-valid, deterministic, and no required metric is `UNAVAILABLE`;
- repeated execution produces byte-identical strict-JSON metric output.

Fixture success proves computability only. It cannot be cited as anomaly-model or early-warning performance.

### EG4 — Fixed observable-signal sanity gate

Freeze one simple probe before Design outcomes. It may use only causal telemetry features already justified by Sprint 12: channel level/RMS/variance/slope, differences and cross-channel relationships, spectral bands, phase/timing, transition counts, duration, usage, and time since maintenance. Standardization and reader fitting use Fit only; the operating threshold uses verified-healthy Calibration only.

On the four fresh Confirmation histories, pass only when all conditions hold:

- macro event AUROC across `P+W` is at least `0.65`;
- the history-block bootstrap 95% lower confidence bound for `P+W` event AUROC is greater than `0.50`;
- at least three of four histories have `P+W` event AUROC greater than `0.55`;
- macro `P` event AUROC is at least `0.70`;
- macro `W` event AUROC is at least `0.60`;
- at the frozen Calibration threshold, `P` event recall is at least `0.50` and median first-alert lead is at least `1.0` day;
- at the same threshold, `W` event recall is at least `0.25` and median first-alert lead is at least `0.5` day;
- false-alert episodes are at most `0.05` per evaluated robot-day on every Confirmation history;
- abrupt results are reported separately and overall, with no minimum predictive-performance requirement and no silent exclusion.

The probe is a benchmark observability check, not a production baseline. Failing EG4 retires all four Confirmation histories and returns to a new protocol version; it is forbidden to keep favorable histories or retune the probe.

### EG5 — Confirmation structural generalization

Materialize four fresh Confirmation histories once, after EG0–EG3 pass and the DGP is frozen. Pass only when:

- every Confirmation history independently meets every hard structural floor and concentration limit;
- every Confirmation history passes EG1 and EG3;
- for each of `P`, `W`, `A`, negative controls, and robot-days, the Confirmation median lies within `[0.75, 1.25]` times the frozen Design median;
- the reserved robot, program, and `P`/`W` mechanism subtypes are present and eligible;
- no history was retried, replaced, or omitted.

Any failure retires the complete Confirmation set. A correction requires a new protocol version, fresh Design histories, and fresh Confirmation histories, and consumes one diagnostic cycle.

### EG6 — Sealed structural generalization

After EG4 and EG5 pass, materialize exactly four fresh Sealed histories through the canonical local public entry point. Sprint 14 may inspect structural metadata only.

Pass only when:

- all four Sealed histories independently meet every hard structural floor and concentration limit;
- all four pass EG1 integrity and role checks;
- for each of `P`, `W`, `A`, negative controls, and robot-days, the Sealed median lies within `[0.75, 1.25]` times the frozen Confirmation median;
- reserved robot, program, and mechanism subtypes are present and eligible;
- manifests and roots are sealed, round-trip verified, and digest-recorded;
- no score, threshold, probe, learned representation, or model output touched a Sealed root;
- no history was retried, replaced, pooled for rescue, or regenerated.

One Sealed failure makes the benchmark `NOT_MEASURABLE`. Sealed failure cannot trigger seed replacement within the same protocol.

### EG7 — Evidence and positive sprint closeout

Evaluate EG7 if and only if EG0–EG6 mechanically yield provisional `MEASURABLE`. Pass only when:

- every diagnostic batch has a latest evidence-review report with zero unresolved actionable findings;
- every failed, superseded, retired, and corrected protocol/root/review remains preserved in the ledger;
- the latest sprint-wide differential deep review has zero unresolved actionable findings and zero unanswered questions;
- `docs/PLAN.md`, the Sprint 14 plan, manifests, seals, and evidence index agree on protocol version, role roster, digests, per-history results, and verdict;
- the verdict is derived mechanically from EG0–EG6 without discretionary override.

If EG0–EG6 instead yield provisional `NOT_MEASURABLE` or `UNAVAILABLE`, do not call a deep reviewer. Record EG7 as `NOT_RUN`, preserve the triggering evidence-reviewed blocker and complete ledger, finalize the negative or unavailable verdict, and keep Sprint 15 blocked.

## Diagnostic-to-Review Loop

Each cycle follows this exact order:

```text
Design-only diagnosis
→ one causal hypothesis with predicted count/coverage effects
→ versioned protocol amendment
→ independent pre-materialization evidence review
→ minimal generator/config implementation
→ fresh Design materialization
→ EG1–EG3 audit
→ independent post-audit evidence review
→ promote or start a new versioned cycle
```

Rules:

- The same implementation worker is retained across corrections unless the OMP replacement criteria are met.
- A reviewer finding or gate miss returns to the worker; the same review level is rerun after correction.
- Confirmation begins only after a clean Design evidence gate.
- A Confirmation failure starts a new versioned cycle with new Design and Confirmation roots; failed Confirmation data never becomes design data.
- The sprint stops after three failed diagnostic cycles or immediately on an unrepairable causal/provenance defect.

## Sprint 15 Handoff Contract

Sprint 15 attribution may start only after Sprint 14 returns `MEASURABLE` and EG7 passes. Its allowed inputs are:

- accepted Fit and Calibration roles;
- accepted Design roles for diagnostics;
- accepted Confirmation roles for bounded replication;
- sealed role identifiers and digests only, never their contents;
- the frozen event/window/metric implementation and this gate record.

Sprint 15 must not reinterpret these exit gates, reopen retired roots, or use Sealed histories for component selection. Any benchmark semantic change returns control to a new benchmark sprint rather than being hidden inside attribution.