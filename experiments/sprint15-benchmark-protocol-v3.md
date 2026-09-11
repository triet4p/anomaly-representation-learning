# Sprint 15 Benchmark Protocol v3 — Balanced Causal Sampling (FROZEN, prospective)

**Status:** Frozen prospectively by the Sprint 15 worker. NOT APPROVED — nothing may execute under v3
(no Task 5 implementation, no Task 9 proof, no Design materialization) before a fresh evidence review
passes with zero actionable findings. Any further semantic change requires a new protocol version
(`v4`, …) and a fresh evidence review before outcomes exist.

**Normative contract:** `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` (this protocol never weakens,
reinterprets, or replaces it; on conflict the gate document governs; the gate document itself is
unchanged across all candidates — see Task 2 cycle-3 continuity).
**Protocol identifier:** `sprint15-benchmark-protocol-v3`.
**Candidate binding:** candidate `N = 3`, base seed `B_3 = 1200` (see §10 and Task 4 cycle-3).
**Lineage:** Sprint 14 `NOT_MEASURABLE` record acknowledged in `artifacts/sprint-15/task-1.md`;
candidate 1 (`sprint15-benchmark-protocol-v1`, seeds 1000–1017) RETIRED after Gate C1 FAIL (3/4
DESIGN-PASS; H-DESIGN-16 `w-dist-shape` breach); candidate 2 (`sprint15-benchmark-protocol-v2`,
seeds 1100–1117) RETIRED after Gate D FAIL (0/4 Confirmation materialized; P→W→A shared-spacing
starvation of abrupt subtypes). Both preserved immutable — zero retired roots, seeds, or gates
reused (see §0).

## 0. Amendment record (v2 → v3; allocator-scheduling revision + candidate advance)

`experiments/sprint15-benchmark-protocol-v2.md`
(SHA256 `337c99ce06be8632e452c6bc6072534ee3e0b60b4d9afe1baf854196dc98113e`) is preserved
byte-identical as the retired candidate-2 protocol. The v2→v3 delta is exactly two items, nothing
else:

1. **Joint fair constrained allocation (§3a, the only methodology change).** The order-biased
   sequential bucket fill (P, then W, then A against a shared spacing ledger) is replaced by a
   deterministic joint round-robin allocation over the six (cohort, subtype) buckets with symmetric
   global spacing/cap enforcement. Eligible-anchor predicates (§2), quotas/splits (§3), caps, spacing
   predicate, rejection taxonomy, fail-fast infeasibility, RNG isolation, and waveform/model
   blindness are all unchanged. Protocol v2 §7a (minimum-duration firing gate) is retained verbatim
   (§7a below); no DGP physics changes in v3.
2. **Candidate-3 advance (§10).** Profile `sprint15-v3`, protocol tag
   `sprint15-benchmark-protocol-v3`, fresh seed band 1200–1217 with fresh role identities continuing
   past candidate 2, collision-free roots under `data/generated/sprint15-v3/<ROLE>/` and disposable
   proof under `artifacts/sprint-15/proof-candidate-3/`.

Every other section below is carried forward byte-for-meaning identical from v2: calendar-first order,
anchor predicates, quotas/splits/caps, nuisance envelopes, signal margins, subtype/holdout rules,
physical invariants plus §7a, infeasibility discipline, case-control boundary, verdict precedence,
and the EG crosswalk (plus one §3a row).

**Diagnosed mechanism (Gate D F-01):** all four candidate-2 Confirmation histories failed quota
placement fail-fast before persistence on abrupt-subtype splits (`H-CONF-14` A/A2 6<8; `H-CONF-15`
A/A1 7<8; `H-CONF-16` A/A2 7<8; `H-CONF-17` A/A2 7<8). Cause, verified in frozen code
(`src/synth/balanced.py:347` fill loop over `("P", "W", "A")` sharing the `kept_moments` ledger at
:339/:377): P quota selections (24) and W quota selections (24) are placed first and each asserts a
symmetric ±14 d same-robot exclusion, saturating roughly 70% of robot-calendar time before any A
anchor is considered, so A1/A2 absorb only residual spacing. Passing histories show the tax directly
(8–15 spacing-skips per Design history, 2–5 per proof history) against evaluable A-subtype pools of
only 15–26 (Design A1 16–26, A2 15–18) versus the post-skip 8/8 requirement. Caps never bind
(`cap-skip: 0` everywhere); `reset-horizon` (21–42/history) sets pool size while spacing-skips decide
acceptance at the margin. Support was never the problem (140–146 evaluable positives per Design
history; Fit/Cal floors exceeded >25×).

**Predicted effects:** symmetric per-round competition removes order bias — no bucket can be starved
by another bucket's early bulk acceptance. Expected: A acceptance at the pool-limited rate (pools
adequate per Design/proof evidence: 15–26 evaluable per subtype vs 8 required); P/W acceptance
unchanged-or-better (equal turns replace asymmetric blocking); medians/counts remain emergent;
abrupt physics, DGP streams, and all predicates untouched. Verified on candidate-3 proof roots
before any Design materialization (Tasks 9/11 cycle 3).

**Rejected alternatives:** raising the abrupt rate (wrong cause — A pools are adequate at 33–41
raw/15–26 evaluable; the binding constraint is acceptance order, not arrivals); lowering A quotas or
subtype splits (weakens an inviolable gate); relaxing the 14 d spacing predicate (inviolable causal
predicate — same-robot horizons would overlap, contaminating events and controls); per-cohort
independent spacing ledgers (breaks the shared-horizon non-overlap guarantee: two cohorts' anchors
on one robot within 14 d would hold overlapping 7 d horizons); post-hoc swap/repair heuristics
(non-deterministic, result-dependent resampling in disguise); deterministic-threshold-only firing or
hazard removal (unrelated cause, larger behavioral change); seed search, retry, replacement, or
pooling (forbidden — unanimity rule).

## 1. Calendar-first construction order (binding)

For each history, in this exact order:

1. Generate the shared-unit calendar: routes, operations, unit arrivals, preventive/corrective
   maintenance, recommissioning, and the healthy background signal — with **no failure episodes
   placed**.
2. Freeze the calendar bytes. All downstream eligibility is computed from this frozen calendar.
3. Compute the eligible causal event anchor set (§2) using the unchanged seven-day horizon, reset,
   quarantine, censoring, spacing, and baseline predicates.
4. Allocate the predeclared per-history quotas (§3) across eligible anchors with the joint fair
   methodology (§3a): dedicated per-bucket RNG streams, robot/program caps, and exact subtype
   stratification — **before waveform generation**.
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

- Quota events are drawn from eligible anchors only, under the joint fair methodology (§3a) with
  dedicated RNG streams isolated from all generation streams.
- Robot/program caps are enforced **during** allocation: an anchor whose acceptance would breach a
  cap is skipped, not repaired afterward.
- Subtype stratification is exact: the per-history subtype counts above are placed, never
  approximated; subtype identity is drawn on the dedicated stream before waveform generation.
- Nominal mix `37.5% P / 37.5% W / 25% A` stays inside the `[15%, 60%]` per-cohort gate by construction.
- Fit/Calibration roles exclude reserved identities (§6); Design includes all subtypes with per-subtype
  tallies reported.

## 3a. Joint fair constrained allocation (v3 addition, binding)

Eligible sets (§2), quotas/splits (§3), caps, the symmetric 14 d same-robot spacing predicate, the
rejection taxonomy, fail-fast infeasibility, and waveform/model blindness are unchanged. Only the
acceptance *scheduling* across buckets changes, as follows:

- **Buckets.** Six (cohort, subtype) buckets with exact needs: `P1: 12`, `P2: 12`, `W1: 12`,
  `W2: 12`, `A1: 8`, `A2: 8`. Canonical visitation order, fixed for all histories:
  `[P1, P2, W1, W2, A1, A2]`.
- **RNG (dedicated, order-independent).** Master seed unchanged:
  `alloc_seed = history_seed × 31 + 7`. Each bucket shuffles its eligible list exactly once with an
  independent sub-RNG seeded deterministically from `(alloc_seed, cohort, subtype)` via
  `int.from_bytes(sha256(f"{alloc_seed}|{cohort}|{subtype}").digest()[:8], "big")` into
  `np.random.default_rng`. Pre-shuffle order is frozen as `(failure_time, failure_id)` sort, so each
  bucket's shuffled priority list depends only on its own pool — never on bucket visitation order or
  acceptance outcomes. Draw counts depend only on pool sizes. The acceptance scan itself consumes no
  RNG.
- **Rounds.** `r = 0, 1, 2, …`; each round visits every incomplete bucket once in canonical order.
  Per visit, scan that bucket's shuffled list from the first unscanned position and accept the first
  candidate satisfying all of: not already selected; robot cap unbreached (same formula);
  program cap unbreached for P/W (same formula); symmetric 14 d spacing against **all** accepted
  anchors across **all** buckets (same predicate); record the existing single rejection reason per
  skipped candidate. A bucket that completes a full scan of its remaining list with zero acceptances
  is terminally stalled (acceptance only hardens as exclusions/caps fill, so rescans cannot succeed).
- **Termination.** All buckets meet exact quotas → success. All incomplete buckets stalled →
  `InfeasibleCandidate` with the existing record fields (`reason`, per-bucket `missing`,
  `history_seed`, `alloc_seed`, `eligible_pool`, `rejection`) plus `stalled_buckets`.
- **Fairness invariant.** In each round every incomplete bucket receives exactly one acceptance
  opportunity before any bucket receives its next; no bucket's early bulk acceptance can starve
  another bucket. Spacing and caps remain global and symmetric — fairness governs opportunity order,
  never constraint strictness.
- **Determinism and blindness.** Fixed bucket order + fixed per-bucket shuffled lists +
  deterministic scan + zero RNG in acceptance ⇒ byte-identical allocation given `history_seed`.
  Inputs remain the same allowlisted metadata; no waveform, score, or model output participates.

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

## 7a. Minimum-degradation-duration firing gate (retained verbatim from v2, binding)

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

## 8. Infeasibility rejection (binding, no-retry)

If the allocator cannot place every quota from the frozen calendar's eligible anchors without
violating §§2–7a, it **fails fast with a machine-readable infeasibility record** (missing quota,
binding predicate, per-predicate rejection tallies, stalled buckets where applicable) and the active
candidate is **rejected before promotion**. Never resample, replace, omit, pool, or regenerate a
root within the candidate. Diagnosis is evidence-reviewed; the fix is a new prospective protocol
version under the next seed namespace (`B_{N+1}`). Infeasibility of allocation, like any
structural/observable miss, retires only the candidate — never the sprint.

## 9. Case-control interpretation boundary (binding)

This is transparent case-control benchmark sampling for identifiable evaluation. Event labels,
cohorts, subtypes, failure times, simulator state, and allocation metadata are excluded from
model-visible rows. Realized cohort shares must never be interpreted as fleet prevalence estimates;
no prevalence, deployment-threshold, or calibrated-risk claim may be built on these quotas.

## 10. Candidate-3 roster, runtime, and roots (frozen)

`B_3 = 1200`. Profile `sprint15-v3` (Task 5 cycle-3 adds it; does not exist yet), protocol tag
`sprint15-benchmark-protocol-v3`, canonical runtime is the local repository root through the locked
`uv` environment
(`uv run python -m synth.cli --chronological --profile sprint15-v3 --seed <S> --role <R>
--protocol sprint15-benchmark-protocol-v3 --output data/generated/sprint15-v3/<ROLE>/`).
Bulk roots are local and uncommitted; manifests and seals carry the v3 protocol tag.

| Role | Root identities | Seeds |
|---|---|---|
| Design | H-DESIGN-21, 22, 23, 24 | 1200, 1201, 1202, 1203 |
| Fit | H-FIT-16, 17, 18 | 1204, 1205, 1206 |
| Calibration | H-CAL-6 | 1207 |
| Confirmation | H-CONF-18, 19, 20, 21 | 1208, 1209, 1210, 1211 |
| Sealed | H-SEAL-21, 22, 23, 24 | 1212, 1213, 1214, 1215 |
| Disposable proof | H-PROOF-5, 6 | 1216, 1217 |

Root identities continue past candidate 2 (`H-DESIGN-20`, `H-FIT-15`, `H-CAL-5`, `H-CONF-17`
unmaterialized-but-reserved, `H-SEAL-20`, `H-PROOF-4` are the last used); all 18 seeds are fresh and
disjoint from every Sprint 14, candidate-1, and candidate-2 seed (Task 4 cycle-3 proves
disjointness and path absence). Every entry point refuses to start when a target root exists and is
non-empty; first-attempt status is recorded; no retry/replacement. No candidate-3 root exists yet;
none may be created before the fresh prospective review passes and Task 5 cycle-3 implements §3a.

## 11. Verdict precedence (frozen)

`MEASURABLE-CANDIDATE` requires EG0–EG6 for candidate 3 with four-of-four `DESIGN-PASS`, full EG4
probe passage, EG5/EG6 structural passage, and zero-actionable evidence reviews. Any mandatory miss
returns `REJECTED-CANDIDATE`; ungeneratable/unverifiable roots or uncomputable metrics despite units
return `UNAVAILABLE-CANDIDATE`. Both retire candidate 3 only. Sprint finalization is success-only per
the V2 gate document.

## 12. EG0–EG7 clause crosswalk (V2 gate document)

| Gate clause | V2 gate section | Protocol section |
|---|---|---|
| EG0 version/digest/physics/roles/seeds | EG0 | Header (identifier, candidate binding); §§1, 3, 3a, 7a, 10 |
| EG0 cohorts/subtypes/holdouts | EG0 | §§3, 3a, 6 |
| EG0 temporal views/horizon/reset/quarantine/controls/anchors | EG0 | §2 |
| EG0 floors/quotas/targets/caps/probe metrics/precedence | EG0 | §§3–7a, §11 |
| EG0 commit/runtime/roots/seals | EG0 | §10 |
| EG0 candidate number/retirement ledger | EG0 | Header; §8 |
| EG1 causality/determinism/role isolation/leakage | EG1 | §§1–2, 7, 7a, 9–10 |
| EG1 physical duration/shape bounds | EG1 | §7a (generator compliance), §2 (audit) |
| EG2 design margin, no pooling | EG2 | §§3, 3a, 8 |
| EG3 fixtures | EG3 | §§4–5 (computability harness, Task 9) |
| EG4 probe thresholds | EG4 | §§4–5 (frozen probe inputs; implementation binds feature set) |
| EG5 confirmation + stability + holdouts | EG5 | §§3, 3a, 6, 8 |
| EG6 sealed + seals + no scores | EG6 | §§3a, 6, 8, 10 |
| EG7 success-path closeout | EG7 | §11 |
| Boundaries (history unit, no seed selection, no learned influence, one probe, confirmation/sealed discipline, prospective-only, case-control, local-only) | Boundaries | §§1, 5, 7a, 8–10 |
| Role minimums 4/3/1/4/4 + reserves | Role topology | §§3, 6, 10 |
| Floors + caps + 80% lead support + Fit/Cal support | Floors | §§2–4, 6 |

## Appendix A. Exact Task 5 cycle-3 implementation boundary (proposed, NOT YET AUTHORIZED)

- **Touch only** the acceptance scheduling inside `allocate_quotas` in `src/synth/balanced.py` (the
  sequential-bucket greedy fill): replace it with the §3a joint round-robin (canonical bucket order,
  per-bucket SHA256-derived sub-RNG shuffles over `(failure_time, failure_id)`-sorted pools, global
  symmetric spacing/cap enforcement, terminal stall rule, extended infeasibility record). Keep
  `eligible_anchors`, all predicates, cap formulas, the spacing predicate, control-window selection,
  and the fail-fast call sites exactly as they are.
- **Touch only** the profile constructor addition (`sprint15_v3_history_config`, mirroring
  `sprint15_v2_history_config` with v2 H3 settings and `min_duration_gate=True` carried forward),
  the `sprint15-v3` CLI choice + default tag, the seal-allowlist `+1` entry, roster/tag constants,
  and quota/audit constants only if they reference the v2 tag (tag-parameterized, no threshold edits).
- **Must not touch:** V2 gate document, v1/v2 protocols/roots/evidence, candidate-1/2 bytes, floors,
  quotas, caps, horizons, predicates, verdict precedence, DGP physics (`health.py`), or any Sprint 14
  file.
- Implementation, focused tests, and proof roots belong to Task 5 cycle-3 **after** the fresh
  prospective review passes — not in this task.
