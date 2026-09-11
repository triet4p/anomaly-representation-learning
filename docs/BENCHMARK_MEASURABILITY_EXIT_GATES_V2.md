# Benchmark Measurability Exit Gates V2 (Sprint 15, normative)

## Purpose

This document is the normative acceptance contract for Sprint 15 balanced causal benchmark recovery.
It answers one question before representation attribution resumes:

> Does a prospectively versioned, quota-controlled candidate benchmark contain enough independent,
> observable, causal early-warning evidence to estimate event ranking, positive lead time, recall,
> persistence, and false-alert behavior across unseen histories without pooling away the history unit
> or selecting favorable seeds?

Sprint 15 may change generator, scheduler, allocator, nuisance-envelope, and causal-signal methodology
only through versioned, prospectively reviewed protocol amendments. It may not tune a learned
representation, anomaly scorer, risk model, production threshold, or sealed history. Global acceptance
floors, metrics, observable-probe thresholds, no-pooling rules, and verdict semantics are invariant
across Sprint 15.

Sprint 14's gate document (`docs/BENCHMARK_MEASURABILITY_EXIT_GATES.md`,
SHA256 `44d085b86ab7cf80aca15162a50aea74fa35c5bf7e7773514822a9e02e71efb1`) is read-only history and
is never edited by this sprint. Threshold continuity with that document is proved in
`artifacts/sprint-15/task-2.md`.

## Success-only sprint completion

Sprint 15 has exactly one permitted terminal verdict: final **`MEASURABLE`**, issued only after
EG0–EG6 yield provisional `MEASURABLE` for one candidate, every evidence gate passes with zero
actionable findings, and a sprint-wide differential deep review returns zero actionable findings.

`NOT_MEASURABLE` and `UNAVAILABLE` are **candidate-rejection states**, not sprint completion states:

- **`REJECTED-CANDIDATE`**: required roots exist and audits execute, but any mandatory structural,
  observability, replication, or generalization gate fails for the active candidate.
- **`UNAVAILABLE-CANDIDATE`**: a required root cannot be generated or verified, provenance is
  incomplete, or the frozen metric implementation cannot produce a required estimand despite the
  structural units being present.

Either state retires only the active candidate in full, preserves it as indexed evidence, and returns
to Tasks 2–4 for an evidence-supported prospective methodology revision under the next deterministic
seed namespace. There is no fixed candidate-iteration cap and no negative terminal branch.

## Candidate rejection precedence

Within candidate `N`, verdict precedence is mechanical:

1. EG0 or EG1 failure → `REJECTED-CANDIDATE` (causal/provenance defects retire roots before counts
   are interpreted) or `UNAVAILABLE-CANDIDATE` (if roots cannot be generated/verified).
2. EG2 Design miss on any one history → `REJECTED-CANDIDATE`; Confirmation/Sealed roles never open.
3. EG3 incomputability despite sufficient units → `UNAVAILABLE-CANDIDATE`.
4. EG4 observable-probe miss → `REJECTED-CANDIDATE`; favorable structural support cannot rescue it,
   and failed Confirmation data never becomes Design data.
5. EG5 Confirmation miss on any one history → `REJECTED-CANDIDATE`; the complete Confirmation set
   retires.
6. EG6 Sealed miss on any one history → `REJECTED-CANDIDATE`; sealed outcomes never tune another
   candidate in this sprint.
7. A favorable observable probe never rescues a structural failure and never authorizes
   representation, anomaly, warning, or calibrated-risk claims.

No pooled total, median, or favorable subset ever rescues a failed per-history gate.

## Deterministic candidate and seed namespace

Candidate iterations are numbered `N = 1, 2, 3, …`. Candidate `N` uses base seed
`B_N = 1000 + 100 × (N − 1)` with these immutable offsets:

| Role | Seeds |
|---|---|
| Design | `B_N + 0` through `B_N + 3` |
| Fit | `B_N + 4` through `B_N + 6` |
| Calibration | `B_N + 7` |
| Confirmation | `B_N + 8` through `B_N + 11` |
| Sealed | `B_N + 12` through `B_N + 15` |
| Disposable proof | `B_N + 16` and `B_N + 17` |

A seed may be retired but never reassigned. Candidate `N + 1` starts only after evidence-reviewed
diagnosis of candidate `N`, a new prospective protocol version, and confirmation that every prior
root remains immutable. Task 4 binds derived seeds to exact role IDs and collision-free output paths
before candidate implementation. Sprint 15 seed bands are disjoint from every Sprint 14 band
(742–749, 780–797, proof 740/741/796/797, test 910–914/931–933).

## Non-negotiable boundaries

1. **Whole histories are the independence unit.** Per-history floors are mandatory; pooled totals never
   rescue a failed history.
2. **No favorable-seed selection.** Seeds and roles are frozen before materialization. Every declared
   history is retained, including failures. Within a candidate, never retry, replace, omit, pool, or
   regenerate a history. Candidate iteration is methodology revision, not seed search.
3. **No learned-model influence.** Representation, anomaly, warning, and calibrated-risk outputs cannot
   select generator settings, quotas, histories, thresholds, or gates.
4. **One fixed observable probe per candidate, as DGP sanity check only.** Its features,
   fit/calibration roles, thresholds, and metrics are frozen before outcomes. It uses causal observable
   telemetry only; simulator health, future state, labels, cohort IDs, and failure times are forbidden
   as inputs.
5. **Confirmation is one-shot candidate evidence, never new Design data.** A failed confirmation retires
   the entire confirmation set and the candidate.
6. **Sealed evaluation remains sealed.** Sealed roots receive structural audit and seal verification
   only. No fixture, probe, model, threshold, or score touches them.
7. **Every semantic change is prospective.** Changes to physics, scheduling, allocation, maintenance,
   quarantine, windows, floors, roles, probes, or verdict precedence require a new protocol version
   approved before materialization. Methodology may change only prospectively when a reviewer accepts
   the justification before candidate outcomes exist.
8. **Balanced case-control interpretation.** The balanced profile controls evaluation-event support; it
   may not leak labels or simulator state to representation inputs or scores, and must never be
   interpreted as a fleet prevalence estimate.
9. **Local execution.** All generation and audits run locally from the repository root through the
   locked `uv` environment. Bulk roots remain local and uncommitted.

## Frozen role topology

Each candidate protocol predeclares non-overlapping whole-history roles:

| Role | Histories | Permitted use |
|---|---:|---|
| Design diagnostic | 4 | Quota-allocation proof and structural promotion audit. Replaced only by a new protocol version. |
| Fit | 3 | Fit the fixed observable probe; downstream healthy reference support. |
| Calibration | 1 | Select only the probe operating threshold; verify healthy support. |
| Confirmation | 4 | One-shot observability and generalization gate after the candidate is frozen. Never reused as design data. |
| Sealed evaluation | 4 | Structural audit and sealing only; reserved for future independent evaluation. |

The protocol also reserves, outside Fit and Calibration:

- at least one complete robot identity;
- at least one complete program identity;
- at least one failure-mechanism subtype from each predictable cohort (`P` and `W`).

The reserved robot, program, and `P`/`W` mechanism subtypes must occur eligible in Confirmation and
Sealed roles. Abrupt (`A`) failures remain in all overall ledgers but are never required to be
predictably rankable.

## Prospective per-history support margins (balanced quotas)

The recovery protocol reserves, after all eligibility exclusions, per Design history:

- `P = 24`, split across `P1/P2`;
- `W = 24`, split across `W1/W2`;
- `A = 16`, split across `A1/A2`;
- at least `48` deterministic non-overlapping negative-control windows;
- at least `240` evaluable robot-days;
- at least `6` positive-contributing robots and `6` negative-contributing robots;
- at least `2` represented programs in each predictable cohort.

Nominal mix is `37.5% P / 37.5% W / 25% A`, inside the unchanged `[15%, 60%]` per-cohort limits.
Quotas are construction targets: the allocator guarantees them or the candidate is rejected before
promotion. Gate promotion is still judged against the hard floors and Design targets below.

## Hard structural floors (unchanged from Sprint 14)

These floors apply **to every Design, Confirmation, and Sealed history independently** after all frozen
temporal-view, maintenance, quarantine, censoring, and reset exclusions:

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

Concentration limits per history (unchanged):

- no robot contributes more than 35% of eligible positive events;
- no robot contributes more than 40% of negative controls;
- no program contributes more than 60% of either predictable cohort;
- each of `P`, `W`, and `A` contributes between 15% and 60% of all evaluable positives.

Positive-lead measurability (unchanged): at least 80% of `P` events and 80% of `W` events in each
history must have at least three eligible file endpoints inside the causal seven-day pre-failure
horizon, at least one eligible endpoint ending at least 24 hours before failure, no maintenance or
recommissioning reset inside the evaluated window, and a clean same-history/robot baseline window
under the frozen eligibility predicate.

Downstream support floors (unchanged):

- Fit: at least 200 verified-healthy files and 6,000 valid patches in aggregate, with every fallback
  or sparse conditioning group reported;
- Calibration: at least 40 verified-healthy eligible rows;
- no sealed, confirmation, quarantined, censored, or maintenance-overlapping row may enter Fit or
  Calibration.

Hard floors are acceptance criteria. Design targets are mandatory promotion margins: a candidate
cannot advance from Design to Confirmation by merely touching the floor.

## Threshold rationale (unchanged)

`P >= 10` / `W >= 10` support the reportable median/IQR minimum of 10 recalled events under a
full-recall fixture. `A >= 8`, 25 controls, and 150 robot-days preserve the Sprint 13 statistical
floors. Design targets are hard floors × 1.25 rounded up. Four Confirmation and four Sealed histories
preserve the minimum independent-history replication unit. Concentration caps block single-identity
passes. Probe thresholds below are minimum evidence of non-chance advance-warning signal, not
production targets. The history-block bound plus ≥3/4 directional rule blocks single-history rescue.
The `[0.75, 1.25]` stability band permits stochastic variation while rejecting support collapse.

## Exit gates

### EG0 — Prospective protocol and provenance freeze

Pass only when all of the following are recorded and independently reviewed before materialization:

- protocol version and content digest;
- exact generator configuration, causal physics, quota/allocation rules, nuisance envelopes, and
  causal `P`/`W` signal margins;
- role names, history identities, and seeds (derived from `B_N` per the namespace above);
- failure-cohort definitions, subtype stratification, and holdouts;
- temporal views, seven-day horizon, censoring, maintenance/reset, quarantine, control-window, and
  anchor-eligibility predicates;
- every hard floor, quota, design target, concentration limit, probe metric, and
  verdict-precedence rule in this document;
- canonical commit, local workstation and locked `uv` runtime, local output roots, and seal format;
- candidate number and retirement ledger.

Any missing field or post-generation semantic edit fails EG0 and retires the affected roots.

### EG1 — Causality, determinism, and role isolation

Pass only when a public-entry-point fixture and every materialized history show:

- byte-identical manifest and event ledger on two fresh-process reloads of the same root;
- strictly chronological file/event ordering;
- one robot performs at most one operation at a time;
- maintenance/recommissioning resets split eligibility and alert episodes exactly once;
- no future health, failure time, cohort label, anomaly label, or diagnostic-only metadata enters
  model-visible fields (event labels, cohorts, subtypes, failure times, simulator state, and
  allocation metadata stay out of model-visible rows);
- role membership is mutually exclusive and roster-exact;
- all required `P`, `W`, and `A` physical bounds and manifestation rules pass;
- zero seal or manifest digest mismatch.

All-or-nothing: required pass count is 100%, zero leakage or integrity exceptions.

### EG2 — Design structural margin

Run on the four candidate Design histories. Pass only when every Design history meets every design
promotion target and concentration limit, and Fit/Calibration support floors are met by the planned
role configuration. A pooled, median, or three-of-four pass is a failure. Failed EG2 retires
candidate `N`; a new candidate returns to Tasks 2–4.

### EG3 — Metric-contract computability

Run deterministic score fixtures on Design and later Confirmation histories only. For every history,
require: constant scores produce tie-aware event AUROC exactly `0.5`; strictly correct ordering
produces event AUROC/G-rank exactly `1.0`; reversed ordering produces exactly `0.0`; onset-only
alerts produce first-alert lead exactly `0.0` days; the frozen advance-alert fixture produces positive
lead and a reportable median/IQR with at least 10 recalled `P` and 10 recalled `W` events; the
false-alert fixture reproduces its analytic episode count and rate; E1–E5 outputs are finite,
schema-valid, deterministic, and no required metric is `UNAVAILABLE`; repeated execution is
byte-identical strict JSON. Fixture success proves computability only.

### EG4 — Fixed observable-signal sanity gate

One simple probe frozen before Design outcomes, using only causal telemetry features justified by
Sprint 12 (channel level/RMS/variance/slope, differences and cross-channel relationships, spectral
bands, phase/timing, transition counts, duration, usage, time since maintenance). Standardization and
fitting use Fit only; the operating threshold uses verified-healthy Calibration only.

On the four fresh Confirmation histories, pass only when all hold:

- macro event AUROC across `P+W` ≥ `0.65`;
- history-block bootstrap 95% lower confidence bound for `P+W` event AUROC > `0.50`;
- at least three of four histories have `P+W` event AUROC > `0.55`;
- macro `P` event AUROC ≥ `0.70`;
- macro `W` event AUROC ≥ `0.60`;
- at the frozen Calibration threshold, `P` event recall ≥ `0.50` with median first-alert lead ≥ `1.0` day;
- at the same threshold, `W` event recall ≥ `0.25` with median first-alert lead ≥ `0.5` day;
- false-alert episodes ≤ `0.05` per evaluated robot-day on every Confirmation history;
- abrupt results reported separately and overall, with no minimum predictive requirement and no
  silent exclusion.

Failing EG4 retires all four Confirmation histories and the candidate; keeping favorable histories or
retuning the probe is forbidden.

### EG5 — Confirmation structural generalization

Four fresh Confirmation histories, materialized once after EG0–EG3 pass and the candidate freeze.
Pass only when every Confirmation history independently meets every hard structural floor and
concentration limit; every Confirmation history passes EG1 and EG3; for each of `P`, `W`, `A`,
negative controls, and robot-days the Confirmation median lies within `[0.75, 1.25]` times the frozen
Design median; reserved robot, program, and `P`/`W` mechanism subtypes are present and eligible; and
no history was retried, replaced, or omitted. Any failure retires the complete Confirmation set and
the candidate.

### EG6 — Sealed structural generalization

After EG4 and EG5 pass, materialize exactly four fresh Sealed histories through the canonical local
public entry point; inspect structural metadata only. Pass only when all four Sealed histories
independently meet every hard structural floor and concentration limit; all four pass EG1 integrity
and role checks; for each of `P`, `W`, `A`, negative controls, and robot-days the Sealed median lies
within `[0.75, 1.25]` times the frozen Confirmation median; reserved subtypes are present and
eligible; manifests and roots are sealed, round-trip verified, and digest-recorded; no score,
threshold, probe, learned representation, or model output touched a Sealed root; and no history was
retried, replaced, pooled for rescue, or regenerated. One Sealed failure retires candidate `N`; it
cannot select a replacement seed within that candidate.

### EG7 — Evidence and sprint closeout (success path only)

Evaluate EG7 if and only if EG0–EG6 mechanically yield provisional `MEASURABLE` for a candidate. Pass
only when every batch has a latest evidence-review report with zero unresolved actionable findings;
every failed, superseded, retired, and corrected protocol/root/review is preserved in the ledger; the
latest sprint-wide differential deep review has zero unresolved actionable findings and zero
unanswered questions; `docs/PLAN.md`, the Sprint 15 plan, manifests, seals, and evidence index agree
on candidate number, protocol version, role roster, digests, per-history results, and verdict; and
the verdict is derived mechanically from EG0–EG6 without discretionary override. EG7 PASS finalizes
Sprint 15 as `MEASURABLE`. Any other standing keeps Sprint 15 open.

## Evidence gates

- **Gate A (Tasks 1–4):** prospective contract review — threshold continuity, success-only semantics,
  balanced-sampling physics, signal-margin construction, seed derivation, Sprint 14 immutability.
  Task 5 blocked until zero actionable findings. Record: `artifacts/sprint-15/review-batch-a.md`.
- **Gate B (Tasks 5–9):** implementation and causal-proof review, including failure-on-infeasibility
  and absence of result-dependent resampling. Design roots barred until zero actionable findings.
- **Gate C1 (Tasks 10–11):** Design promotion — every value recomputed from raw manifests;
  four-of-four `DESIGN-PASS` mandatory.
- **Gate C2 (Tasks 12–13):** candidate and probe freeze — byte identity, probe immutability, role
  isolation, no sealed access, exact downstream roster.
- **Gate D (Tasks 14–16):** Confirmation acceptance — every observable and structural condition
  without pooled rescue.
- **Gate E (Tasks 17–19):** Sealed and candidate-verdict review — sealed isolation, hashes, gate
  precedence, candidate verdict.

Every batch requires a fresh `evidence-reviewer` PASS with zero actionable findings. A sprint-wide
`deep-reviewer` runs only after EG0–EG6 produce provisional `MEASURABLE`.

## Sprint 16 handoff contract

Sprint 16 attribution starts only after Sprint 15 returns final `MEASURABLE` with a zero-actionable
closeout and explicit reviewed handoff. Allowed inputs:

- accepted Fit and Calibration roles;
- accepted Design roles for diagnostics;
- accepted Confirmation roles for bounded replication;
- sealed role identifiers and digests only, never their contents;
- the frozen event/window/metric implementation and this gate record.

Sprint 16 must not reinterpret these exit gates, reopen retired roots or candidates, or use Sealed
histories for component selection. Any benchmark semantic change returns control to a new benchmark
sprint rather than being hidden inside attribution. Production recovery and calibrated risk remain
future gated work.
