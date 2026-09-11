# Sprint 15 Benchmark Protocol v2 — Balanced Causal Sampling (FROZEN, prospective)

**Status:** Frozen prospectively by the Sprint 15 worker. NOT APPROVED — nothing may execute under v2
(no Task 5 implementation, no Task 9 proof, no Design materialization) before a fresh evidence review
passes with zero actionable findings. Any further semantic change requires a new protocol version
(`v3`, …) and a fresh evidence review before outcomes exist.

**Normative contract:** `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` (this protocol never weakens,
reinterprets, or replaces it; on conflict the gate document governs; the gate document itself is
unchanged from candidate 1 — see Task 2 cycle-2 continuity).
**Protocol identifier:** `sprint15-benchmark-protocol-v2`.
**Candidate binding:** candidate `N = 2`, base seed `B_2 = 1100` (see §10 and Task 4 cycle-2).
**Lineage:** Sprint 14 `NOT_MEASURABLE` record acknowledged in `artifacts/sprint-15/task-1.md`; candidate 1
(`sprint15-benchmark-protocol-v1`, seeds 1000–1017) is RETIRED after Gate C1 FAIL (3/4 DESIGN-PASS;
H-DESIGN-16 `w-dist-shape` breach) and preserved immutable — zero candidate-1 roots, seeds, or gates
reused (see §0).

## 0. Amendment record (v1 → v2; single-DGP-change + candidate advance)

`experiments/sprint15-benchmark-protocol-v1.md`
(SHA256 `7aa547fdd535504d723c1a3ad50289acdbf179d42e3f41e8d3e74648d7515413`) is preserved byte-identical
as the retired candidate-1 protocol. The v1→v2 delta is exactly two items, nothing else:

1. **Minimum-degradation-duration firing gate (§7a, the only DGP change).** Non-abrupt (P/W) cohort
   failure firing is gated on a minimum degradation duration sourced verbatim from the frozen audit
   floors (`synth.balanced.audit_sprint15`: P `min ≥ 2.0 d`, W `min ≥ 6.0 d`): while
   `(event.end_time − open_degradation.start_time) < floor_cohort`, the cohort can fire neither by
   deterministic threshold crossing nor by stochastic base-rate draw on that operation. Abrupt (A)
   hazard, legacy no-cohort path, episode opening, wear/aging/noise dynamics, thresholds, gains,
   maintenance, and all RNG streams outside the gated draw are untouched. When the floor is unmet, no
   cohort RNG draw is consumed on that path (floor check precedes the draw); abrupt evaluation
   continues independently. Deterministic given seed.
2. **Candidate-2 advance (§10).** Profile `sprint15-v2`, protocol tag
   `sprint15-benchmark-protocol-v2`, fresh seed band 1100–1117 with fresh role identities continuing
   past candidate 1, collision-free roots under `data/generated/sprint15-v2/<ROLE>/` and disposable
   proof under `artifacts/sprint-15/proof-candidate-2/`.

Every other section below is carried forward byte-for-meaning identical from v1: calendar-first order,
anchor predicates, quotas/splits/caps, nuisance envelopes, signal margins, subtype/holdout rules,
physical invariants, infeasibility discipline, case-control boundary, verdict precedence, and the EG
crosswalk (plus one §7a row).

**Diagnosed mechanism (Gate C1 F-01/F-03):** candidate-1 history H-DESIGN-16 (seed 1003) carries W2
failure `fail-robot-04-0016` with `duration_d = 0.006944 d` (600 s, one operation) against the frozen
`6.0 d` W floor. Cause: `src/synth/health.py` lines 421–426 evaluate the cohort base-rate hazard from
the first operation after episode onset (`H ≥ 0.3`), so a left-tail draw (`p ≈ 8.8×10⁻⁶` at onset
health `H = 0.318 ≪ H* = 1.35`) fired a weak-cohort failure with no progressive wear. Quota, margin,
or seed changes cannot remove this left tail; only generator physics can, prospectively.

**Predicted effects:** sub-floor P/W left tail eliminated by construction (the only observed breach
class); P/W medians shift slightly upward within caps (current P ~7 d vs cap 10; W ~12.5–13 d vs cap
24); max bounds and A physics unchanged-emergent; expected count loss negligible (sub-floor events
were ~1 in ~150 W events). Verified on candidate-2 proof roots before any Design materialization
(Tasks 9/11 cycle 2).

**Rejected alternatives:** weakening the `6.0 d` audit floor (forbidden — inviolable EG1 gate, no
waiver per Gate C1); deterministic-threshold-only firing / removing base hazard (larger behavioral
change, alters intended DGP stochasticity and counts); seed search, retry, replacement, or pooling
(forbidden — unanimity rule); enforcing max-duration bounds in the generator (unnecessary — max
bounds pass emergently); reselecting quotas (support was never the problem: 140–145 evaluable
positives per history).

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

## 7a. Minimum-degradation-duration firing gate (v2 addition, binding)

For non-abrupt cohorts only, with floors sourced verbatim from the frozen audit
(`synth.balanced.audit_sprint15`: P `min ≥ 2.0 d`, W `min ≥ 6.0 d`):

- On any operation where `(event.end_time − open_degradation.start_time) < floor_cohort`
  (P: `2.0 d`; W: `6.0 d`), the upcoming cohort can fire **neither** by deterministic threshold
  crossing **nor** by stochastic base-rate draw. Abrupt-hazard evaluation on that operation continues
  independently and unchanged.
- Ordering (frozen): the floor check precedes any cohort RNG draw; when the floor is unmet, no
  cohort draw is consumed on that path. Deterministic given seed.
- Abrupt cohorts (`degradation_min_d = degradation_max_d = 0.0` by config contract) and the legacy
  no-cohort path are out of scope and byte-for-byte untouched.
- Maximum-duration bounds and medians remain emergent and audited, not generated.

This makes the generator compliant with its own inviolable EG1 floors; it weakens no gate and grants
no waiver.

## 8. Infeasibility rejection (binding, no-retry)

If the allocator cannot place every quota from the frozen calendar's eligible anchors without
violating §§2–7a, it **fails fast with a machine-readable infeasibility record** (missing quota,
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

## 10. Candidate-2 roster, runtime, and roots (frozen)

`B_2 = 1100`. Profile `sprint15-v2` (Task 5 cycle-2 adds it; does not exist yet), protocol tag
`sprint15-benchmark-protocol-v2`, canonical runtime is the local repository root through the locked
`uv` environment
(`uv run python -m synth.cli --chronological --profile sprint15-v2 --seed <S> --role <R>
--protocol sprint15-benchmark-protocol-v2 --output data/generated/sprint15-v2/<ROLE>/`).
Bulk roots are local and uncommitted; manifests and seals carry the v2 protocol tag.

| Role | Root identities | Seeds |
|---|---|---|
| Design | H-DESIGN-17, 18, 19, 20 | 1100, 1101, 1102, 1103 |
| Fit | H-FIT-13, 14, 15 | 1104, 1105, 1106 |
| Calibration | H-CAL-5 | 1107 |
| Confirmation | H-CONF-14, 15, 16, 17 | 1108, 1109, 1110, 1111 |
| Sealed | H-SEAL-17, 18, 19, 20 | 1112, 1113, 1114, 1115 |
| Disposable proof | H-PROOF-3, 4 | 1116, 1117 |

Root identities continue past candidate 1 (`H-DESIGN-16`, `H-FIT-12`, `H-CAL-4`, `H-CONF-13`,
`H-SEAL-16`, `H-PROOF-2` are the last used); all 18 seeds are fresh and disjoint from every Sprint 14
and candidate-1 seed (Task 4 cycle-2 proves disjointness and path absence). Every entry point refuses
to start when a target root exists and is non-empty; first-attempt status is recorded; no
retry/replacement. No candidate-2 root exists yet; none may be created before the fresh prospective
review passes and Task 5 cycle-2 implements the §7a change.

## 11. Verdict precedence (frozen)

`MEASURABLE-CANDIDATE` requires EG0–EG6 for candidate 2 with four-of-four `DESIGN-PASS`, full EG4
probe passage, EG5/EG6 structural passage, and zero-actionable evidence reviews. Any mandatory miss
returns `REJECTED-CANDIDATE`; ungeneratable/unverifiable roots or uncomputable metrics despite units
return `UNAVAILABLE-CANDIDATE`. Both retire candidate 2 only. Sprint finalization is success-only per
the V2 gate document.

## 12. EG0–EG7 clause crosswalk (V2 gate document)

| Gate clause | V2 gate section | Protocol section |
|---|---|---|
| EG0 version/digest/physics/roles/seeds | EG0 | Header (identifier, candidate binding); §§1, 3, 7a, 10 |
| EG0 cohorts/subtypes/holdouts | EG0 | §§3, 6 |
| EG0 temporal views/horizon/reset/quarantine/controls/anchors | EG0 | §2 |
| EG0 floors/quotas/targets/caps/probe metrics/precedence | EG0 | §§3–7a, §11 |
| EG0 commit/runtime/roots/seals | EG0 | §10 |
| EG0 candidate number/retirement ledger | EG0 | Header; §8 |
| EG1 causality/determinism/role isolation/leakage | EG1 | §§1–2, 7, 7a, 9–10 |
| EG1 physical duration/shape bounds | EG1 | §7a (generator compliance), §2 (audit) |
| EG2 design margin, no pooling | EG2 | §§3, 8 |
| EG3 fixtures | EG3 | §§4–5 (computability harness, Task 9) |
| EG4 probe thresholds | EG4 | §§4–5 (frozen probe inputs; implementation binds feature set) |
| EG5 confirmation + stability + holdouts | EG5 | §§3, 6, 8 |
| EG6 sealed + seals + no scores | EG6 | §§6, 8, 10 |
| EG7 success-path closeout | EG7 | §11 |
| Boundaries (history unit, no seed selection, no learned influence, one probe, confirmation/sealed discipline, prospective-only, case-control, local-only) | Boundaries | §§1, 5, 7a, 8–10 |
| Role minimums 4/3/1/4/4 + reserves | Role topology | §§3, 6, 10 |
| Floors + caps + 80% lead support + Fit/Cal support | Floors | §§2–4, 6 |

## Appendix A. Exact Task 5 cycle-2 implementation boundary (proposed, NOT YET AUTHORIZED)

- **Touch only** the cohort-firing evaluation in `src/synth/health.py` (the block at lines 409–426:
  deterministic threshold fire at 410–414 and stochastic base-rate fire at 421–426): insert the §7a
  floor precondition for non-abrupt cohorts; abrupt evaluation (415–420), legacy path (427–430), and
  all surrounding dynamics stay byte-identical.
- **Touch only** the profile constructor addition (`sprint15_v2_history_config`, mirroring
  `sprint15_v1_history_config` with v1 H3 settings carried forward), the `sprint15-v2` CLI choice +
  default tag, the seal-allowlist `+1` entry, and the quota/audit constants only if they reference
  the v1 tag (tag-parameterized, no threshold edits).
- **Must not touch:** V2 gate document, v1 protocol/roots/evidence, candidate-1 bytes, floors,
  quotas, caps, horizons, predicates, verdict precedence, or any Sprint 14 file.
- Implementation, focused tests, and proof roots belong to Task 5 cycle-2 **after** the fresh
  prospective review passes — not in this task.
