# Sprint 15 Benchmark Protocol v1 — Balanced Causal Sampling (FROZEN, prospective)

**Status:** Frozen prospectively by the Sprint 15 worker. NOT APPROVED — nothing may execute under v1
(no Task 7/9 proof, no Design materialization) before Evidence Gate A passes with zero actionable
findings. Any semantic change requires a new protocol version (`v2`, …) and a fresh evidence review
before outcomes exist.

**Normative contract:** `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` (this protocol never weakens,
reinterprets, or replaces it; on conflict the gate document governs).
**Protocol identifier:** `sprint15-benchmark-protocol-v1`.
**Candidate binding:** candidate `N = 1`, base seed `B_1 = 1000` (see §10 and Task 4).
**Sprint 14 lineage:** `NOT_MEASURABLE` record acknowledged in `artifacts/sprint-15/task-1.md`; zero
Sprint 14 roots, seeds, or gates reused (see §0).

## 0. Amendment record (genesis)

v1 is the genesis protocol of the Sprint 15 benchmark family. There is no prior Sprint 15 version to
amend. Sprint 14 Protocols v1–v5.2 and all Sprint 14 roots/seeds are retired history (see Task 1) and
contribute only the diagnosed failure mechanism (reset-driven abrupt-support loss: raw abrupt arrivals
adequate, ~half of abrupt causal horizons reset-killed) that this calendar-first quota design removes
by construction.

## 1. Calendar-first construction order (binding)

For each history, in this exact order:

1. Generate the shared-unit calendar: routes, operations, unit arrivals, preventive/corrective
   maintenance, recommissioning, and the healthy background signal — with **no failure episodes
   placed**.
2. Freeze the calendar bytes. All downstream eligibility is computed from this frozen calendar.
3. Compute the eligible causal event anchor set (§2) using the unchanged seven-day horizon, reset,
   quarantine, censoring, spacing, and baseline predicates.
4. Allocate the predeclared per-history quotas (§3) across eligible anchors with a dedicated RNG,
   robot/program caps, and subtype stratification — **before waveform generation**.
5. Generate waveforms with bounded nuisances (§4) and frozen causal `P`/`W` signatures (§5).
6. Emit manifests carrying profile, protocol, seed, role, quota, allocation, rejection, and root
   digests; hidden event state stays outside model-visible file rows (§9).

Waveform generation must never move, add, or remove an anchor. If the frozen calendar cannot place
the quotas without violating physical or causal constraints, §8 fires.

## 2. Anchor eligibility predicates (frozen)

An event anchor at failure time `T` on robot `R` is eligible only if all hold:

- **Horizon:** the seven-day window `[T − 7d, T]` lies inside recorded operation with at least the
  lead-support endpoints required by the gate document (≥3 eligible file endpoints in-window, ≥1
  ending ≥24 h before `T`).
- **Reset-split:** no maintenance or recommissioning reset intersects `[T − 7d, T]`; resets split
  eligibility and alert episodes exactly once.
- **Quarantine/censoring:** `T` is outside quarantine windows and uncensored; the anchor's window
  contains no quarantined, censored, or maintenance-overlapping file.
- **Spacing:** anchors on the same robot are separated by at least 14 days (`T`-to-`T`), so
  seven-day horizons never overlap and control windows stay disjoint.
- **Clean baseline:** a clean same-history/robot baseline window exists under the frozen eligibility
  predicate.
- **Physical feasibility:** the robot/program route assignment admits the cohort's manifestation
  physics (§7); one robot performs at most one operation at a time; file/event ordering is strictly
  chronological.

Control-window candidates use the same predicates minus the failure anchor, are non-overlapping with
each other, with any event horizon, and with reset spans, on the same history/robot.

## 3. Exact per-history quotas and allocation (frozen)

Per Design, Confirmation, and Sealed history:

| Cohort | Quota | Subtype split |
|---|---|---|
| `P` progressive | 24 | `P1 = 12`, `P2 = 12` |
| `W` weak precursor | 24 | `W1 = 12`, `W2 = 12` |
| `A` abrupt | 16 | `A1 = 8`, `A2 = 8` |
| Negative controls | ≥ 48 deterministic non-overlapping windows | — |
| Evaluable robot-days | ≥ 240 | — |
| Positive-contributing robots | ≥ 6 | robot cap: ≤ 35% of positives |
| Negative-contributing robots | ≥ 6 | robot cap: ≤ 40% of controls |
| Programs per predictable cohort | ≥ 2 | program cap: ≤ 60% of cohort |

Allocation rules:

- A **dedicated allocator RNG**, seeded deterministically from the history seed
  (`alloc_seed = history_seed × 31 + 7`, stream isolated from all generation streams), draws quota
  events from eligible anchors only.
- Robot/program caps are enforced **during** allocation: an anchor whose acceptance would breach a
  cap is skipped, not repaired afterward.
- Subtype stratification is exact: the per-history subtype counts above are placed, never
  approximated; subtype identity is drawn on the dedicated stream before waveform generation.
- Nominal mix `37.5% P / 37.5% W / 25% A` stays inside the `[15%, 60%]` per-cohort gate by construction.
- Fit/Calibration roles exclude reserved identities (§6); Design includes all subtypes with per-subtype
  tallies reported.

## 4. Bounded nuisance envelopes (frozen)

Realized nuisance variation must stay inside these envelopes; the Task 9 qualification harness proves
the bounds on symbolic/disposable fixtures before Design materialization:

| Nuisance axis | Frozen bound |
|---|---|
| Sensor-noise scale multiplier | `1.00 ± 0.05` of nominal |
| Timing/phase jitter per file | ≤ 2% of file duration |
| Benign usage-rate variation | ≤ 10% around program nominal |
| File-duration variation | within the frozen program duration table ± 10% |
| Cross-channel benign coupling drift | ≤ 0.05 absolute coherence change |

No nuisance axis may shift cohort-conditional means: nuisances are mean-preserving by construction.
Any realized violation fails the structural audit and, post-promotion, retires the candidate.

## 5. Causal `P`/`W` signatures and signal margin (frozen)

Minimum causal telemetry signatures (precursor severity coupling, same gain semantics as the frozen
manifestation table: `P1 κ = 1.0`, `P2 κ = 0.55`, `W1 κ = 0.30`, `W2 κ = 0.16`, severity cap 0.95):

- The weakest signature, `W2` at minimum severity, must exceed the worst-case nuisance ceiling (§4)
  by a **conservative analytic factor ≥ 3.0** in at least one frozen probe-feature family (level,
  slope, or band energy), computed symbolically from the frozen gains and envelope bounds — no
  learned model participates.
- `P1/P2/W1` margins follow monotonically from their larger gains under identical nuisances; the
  Task 9 harness checks all four subtype margins analytically.
- Suppression within 7 d before abrupt failures and the failure-file severity law are unchanged
  Sprint 14 semantics (descriptive, never applied retrospectively to precursor windows).

Learned-model outputs are forbidden from envelope, signature, quota, threshold, or gate selection.

## 6. Subtype, identity, and holdout constraints (frozen)

- Exact per-history subtype splits from §3; per-subtype structural tallies (evaluable positives,
  eligibility rate, concentration share) reported per history with no gate attached beyond the caps.
- Reserved outside Fit and Calibration: one complete robot identity, one complete program identity,
  one `P`-mechanism subtype, one `W`-mechanism subtype; all must occur eligible in Confirmation and
  Sealed. (`A1/A2` are quota-stratified everywhere; no `A` holdout is required since abrupt events
  carry no predictability claim.)
- Concentration caps (§V2 gates) apply per history to evaluable events.

## 7. Physical invariants (unchanged, frozen)

Strictly chronological files/events; one robot performs at most one operation at a time; upcoming
non-abrupt cohort drawn at history start and after every maintenance; episode opens at health ≥ onset
tagged with the upcoming cohort; deterministic fire at the cohort threshold; abrupt hazard always
active and aborts open episodes (`A` records carry onset None, duration 0); maintenance closes
interrupted episodes at recommissioning and redraws the upcoming cohort; manifestation severity
capped at 0.95. Trajectories never condition on future state.

## 8. Infeasibility rejection (binding, no-retry)

If the allocator cannot place every quota from the frozen calendar's eligible anchors without
violating §§2–7, it **fails fast with a machine-readable infeasibility record** (missing quota,
binding predicate, per-predicate rejection tallies) and the active candidate is **rejected before
promotion**. Never resample, replace, omit, pool, or regenerate a root within the candidate.
Diagnosis is evidence-reviewed; the fix is a new prospective protocol version under the next seed
namespace (`B_{N+1}`). Infeasibility of allocation, like any structural/observable miss, retires only
the candidate — never the sprint.

## 9. Case-control interpretation boundary (binding)

This is transparent case-control benchmark sampling for identifiable evaluation. Event labels,
cohorts, subtypes, failure times, simulator state, and allocation metadata are excluded from
model-visible rows. Realized cohort shares must never be interpreted as fleet prevalence estimates;
no prevalence, deployment-threshold, or calibrated-risk claim may be built on these quotas.

## 10. Candidate-1 roster, runtime, and roots (frozen)

`B_1 = 1000`. Profile `sprint15-v1`, protocol tag `sprint15-benchmark-protocol-v1`, canonical runtime
is the local repository root through the locked `uv` environment
(`uv run python -m synth.cli --chronological --profile sprint15-v1 --seed <S> --role <R>
--protocol sprint15-benchmark-protocol-v1 --output data/generated/sprint15-v1/<ROLE>/`).
Bulk roots are local and uncommitted; manifests and seals carry the v1 protocol tag.

| Role | Root identities | Seeds |
|---|---|---|
| Design | H-DESIGN-13, 14, 15, 16 | 1000, 1001, 1002, 1003 |
| Fit | H-FIT-10, 11, 12 | 1004, 1005, 1006 |
| Calibration | H-CAL-4 | 1007 |
| Confirmation | H-CONF-10, 11, 12, 13 | 1008, 1009, 1010, 1011 |
| Sealed | H-SEAL-13, 14, 15, 16 | 1012, 1013, 1014, 1015 |
| Disposable proof | H-PROOF-1, 2 | 1016, 1017 |

Root identities continue past every Sprint 13/14 family; all 18 seeds are fresh and disjoint from
every ledgered Sprint 14 seed (Task 4 proves disjointness). Every entry point refuses to start when
a target root exists and is non-empty; first-attempt status is recorded; no retry/replacement.

## 11. Verdict precedence (frozen)

`MEASURABLE-CANDIDATE` requires EG0–EG6 for candidate 1 with four-of-four `DESIGN-PASS`, full EG4
probe passage, EG5/EG6 structural passage, and zero-actionable evidence reviews. Any mandatory miss
returns `REJECTED-CANDIDATE`; ungeneratable/unverifiable roots or uncomputable metrics despite units
return `UNAVAILABLE-CANDIDATE`. Both retire candidate 1 only. Sprint finalization is success-only per
the V2 gate document.

## 12. EG0–EG7 clause crosswalk (V2 gate document)

| Gate clause | V2 gate section | Protocol section |
|---|---|---|
| EG0 version/digest/physics/roles/seeds | EG0 | Header (identifier, candidate binding); §§1, 3, 10 |
| EG0 cohorts/subtypes/holdouts | EG0 | §§3, 6 |
| EG0 temporal views/horizon/reset/quarantine/controls/anchors | EG0 | §2 |
| EG0 floors/quotas/targets/caps/probe metrics/precedence | EG0 | §§3–6, §11 |
| EG0 commit/runtime/roots/seals | EG0 | §10 |
| EG0 candidate number/retirement ledger | EG0 | Header; §8 |
| EG1 causality/determinism/isolation/leakage | EG1 | §§1–2, 7, 9–10 |
| EG2 design margin, no pooling | EG2 | §§3, 8 |
| EG3 fixtures | EG3 | §§4–5 (computability harness, Task 9) |
| EG4 probe thresholds | EG4 | §§4–5 (frozen probe inputs; implementation binds feature set) |
| EG5 confirmation + stability + holdouts | EG5 | §§3, 6, 8 |
| EG6 sealed + seals + no scores | EG6 | §§6, 8, 10 |
| EG7 success-path closeout | EG7 | §11 |
| Boundaries (history unit, no seed selection, no learned influence, one probe, confirmation/sealed discipline, prospective-only, case-control, local-only) | Boundaries | §§1, 5, 8–10 |
| Role minimums 4/3/1/4/4 + reserves | Role topology | §§3, 6, 10 |
| Floors + caps + 80% lead support + Fit/Cal support | Floors | §§2–4, 6 |
