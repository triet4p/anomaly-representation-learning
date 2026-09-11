# Sprint 15 Benchmark Protocol v5 — Balanced Causal Sampling (FROZEN, prospective)

**Status:** Frozen prospectively by the Sprint 15 worker. NOT APPROVED — nothing may execute under v5
(no Task 5 implementation, no Task 9 proof, no Design materialization) before a fresh evidence review
passes with zero actionable findings. Any further semantic change requires a new protocol version
(`v6`, …) and a fresh evidence review before outcomes exist.

**Normative contract:** `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` (this protocol never weakens,
reinterprets, or replaces it; on conflict the gate document governs; the gate document itself is
unchanged across all candidates — see Task 2 cycle-5 continuity).
**Protocol identifier:** `sprint15-benchmark-protocol-v5`.
**Candidate binding:** candidate `N = 5`, base seed `B_5 = 1400` (see §10 and Task 4 cycle-5).
**Lineage:** Sprint 14 `NOT_MEASURABLE` record acknowledged in `artifacts/sprint-15/task-1.md`;
candidate 1 (`sprint15-benchmark-protocol-v1`, seeds 1000–1017) RETIRED after Gate C1 FAIL (3/4
DESIGN-PASS; H-DESIGN-16 `w-dist-shape` breach); candidate 2 (`sprint15-benchmark-protocol-v2`,
seeds 1100–1117) RETIRED after Gate D FAIL (0/4 Confirmation materialized; P→W→A shared-spacing
starvation of abrupt subtypes); candidate 3 (`sprint15-benchmark-protocol-v3`, seeds 1200–1217)
RETIRED after Gate D FAIL (2/4 Confirmation materialized; H-CONF-18/20 abrupt-subtype
quota-shortfalls A/A1 7<8 under greedy joint allocation despite globally feasible calendars);
candidate 4 (`sprint15-benchmark-protocol-v4`, seeds 1300–1317) RETIRED after Gate D FAIL (7/8
downstream materialized; H-FIT-21 exact-CSP proven-infeasible on a counting-thin abrupt pool;
H-CONF-24 FAR 0.0536 > 0.05 on a truncated 2-Fit probe basis).
All preserved immutable — zero retired roots, seeds, or gates reused (see §0).

## 0. Amendment record (v4 → v5; calendar-volume margin + candidate advance)

`experiments/sprint15-benchmark-protocol-v4.md`
(SHA256 `498d45b767a0b0f106adfc5340bd8646ad6f36d25d4c76ca7e2cc1403530cd62`) is preserved
byte-identical as the retired candidate-4 protocol. The v4→v5 delta is exactly two items, nothing
else:

1. **Calendar-volume margin (§1, the only methodology change).** Operation volume per history scales
   by exactly 1.5× at frozen per-unit DGP mechanics: `n_units 2304 → 3456` with the arrival cadence
   rescaled inversely to preserve the frozen 360-day span (arrival interval/jitter
   `11200.0 s → 7466.67 s`, same 299-day arrival coverage). Routes, fleet (nine robots), span
   (360 d), dev cutoff (180 d), quarantine (7 d), per-unit failure physics, rates, wear, thresholds,
   gains, maintenance, and all eligibility/allocation/audit predicates are unchanged. Expected effect
   is ~1.5× failures, files, eligible anchors, controls, and robot-days per history at identical
   per-event mechanisms — absolute feasibility slack against the fixed 64-anchor quotas, with no
   floor, cap, quota, threshold, or metric touched.
2. **Candidate-5 advance (§10).** Profile `sprint15-v5`, protocol tag
   `sprint15-benchmark-protocol-v5`, fresh seed band 1400–1417 with fresh role identities continuing
   past candidate 4, collision-free roots under `data/generated/sprint15-v5/<ROLE>/` and disposable
   proof under `artifacts/sprint-15/proof-candidate-5/`.

Every other section below is carried forward byte-for-meaning identical from v4: calendar-first order,
anchor predicates, quotas/splits/caps, the exact §3b allocation formulation, nuisance envelopes,
signal margins, subtype/holdout rules, physical invariants plus §7a, infeasibility discipline,
case-control boundary, verdict precedence, and the EG crosswalk.

**Diagnosed mechanism (Gate D F-01, cycle 4):** candidate-4 Fit history H-FIT-21 (seed 1306) failed
exact allocation fail-fast before persistence. Read-only re-derivation on the frozen pre-allocation
state proves counting-infeasibility, not a solver or ordering defect: the eligible A/A2 pool holds
7 anchors against need 8 (spacing-alone maximum 6). Drop-one-group ablations (no spacing edges / no
robot caps / no program caps) all remain proven-infeasible (HiGHS status 8); P/W buckets carry large
spacing-alone maxima (18–33 vs needs 12). Full evidence in `artifacts/sprint-15/task-2-cycle-5.md`:
across all 41 materialized C1–C4 histories, eligible A2 pools range 9–24 (mean 15.3) with the worst
feasible slack at 1 anchor, and 1306 sits below the feasible floor on both total pool (132, global
minimum) and A2 pool (7). The exact v4 method is already optimal — no allocation-side change can
place 8 anchors from 7. Quota, spacing, cap, rate, horizon, and threshold changes are all rejected
(see below); only deeper realized pools add margin, prospectively.

**Predicted effects:** eligible pools deepen ~1.5× at fixed needs (A2 mean ~15.3 → ~23.0; worst
observed class 7–9 → ~10–14 vs need 8), restoring multi-anchor slack on every subtype; P/W worst
slacks (2) widen to ~9+; controls, robot-days, and Fit/Calibration healthy support deepen
proportionally, so the full 3/3 Fit roster materializes and the probe runs on its predeclared
3-Fit basis. Poisson left-tail bound on the A2 pool: P(pool < 8) falls from ~1.5% per history to
~0.01% (~100× candidate-level risk reduction). Observable FAR margin is restored through the
complete Fit basis plus deeper Calibration support with the frozen 95th-percentile threshold and
EG4 ceilings unchanged (candidate-3 full-Fit precedent: FARs 0.0421/0.0414 with margin). Abrupt
per-history stability variance (F-04 class) is dampened by deeper subtype pools but not directly
targeted; C5 medians and bands will judge. Verified on candidate-5 proof roots before any Design
materialization (Tasks 9/11 cycle 5).

**Rejected alternatives:** lowering A quotas or subtype splits (weakens an inviolable V2
construction target); relaxing the 14 d spacing predicate (inviolable causal predicate — seven-day
horizons would overlap); relaxing robot/program concentration caps (V2 limits); raising abrupt or
any failure rate (wrong cause — mechanisms are adequate; changes DGP physics); shortening the
seven-day horizon or touching quarantine/censoring/reset safeguards (explicit sprint non-goals);
retuning the probe threshold or any EG4 metric (invariant thresholds); widening the fleet instead
of unit volume (alters the nine-robot topology and holdout structure the audit expects — larger
ripple for the same volume); seed search, retry, replacement, resampling, or pooling (forbidden —
unanimity rule).

## 1. Calendar-first construction order (binding)

For each history, in this exact order:

1. Generate the shared-unit calendar: routes, operations, unit arrivals, preventive/corrective
   maintenance, recommissioning, and the healthy background signal — with **no failure episodes
   placed**. Operation volume is exactly 1.5× the v4 level at identical mechanics (`n_units = 3456`;
   arrival interval and jitter rescaled to `7466.67 s` so arrival coverage stays inside the frozen
   360-day span; routes, fleet, span, dev cutoff, and per-unit DGP rates/physics identical to v4).
2. Freeze the calendar bytes. All downstream eligibility is computed from this frozen calendar.
3. Compute the eligible causal event anchor set (§2) using the unchanged seven-day horizon, reset,
   quarantine, censoring, spacing, and baseline predicates.
4. Allocate the predeclared per-history quotas (§3) across eligible anchors with the exact
   constraint-satisfaction methodology (§3b) — **before waveform generation**.
5. Generate waveforms with bounded nuisances (§4) and frozen causal `P`/`W` signatures (§5).
6. Emit manifests carrying profile, protocol, seed, role, quota, allocation, rejection, solver
   provenance, and root digests; hidden event state stays outside model-visible file rows (§9).

Waveform generation must never move, add, or remove an anchor. If the frozen calendar admits no
quota/cap/spacing-satisfying anchor set, §8 fires.

## 2. Anchor eligibility predicates (frozen)

An event anchor at failure time `T` on robot `R` is eligible only if all hold:

- **Horizon:** the seven-day window `[T − 7d, T]` lies inside recorded operation with at least the
  lead-support endpoints required by the gate document (≥3 eligible file endpoints in-window, ≥1
  ending ≥24 h before `T`).
- **Reset-split:** no maintenance or recommissioning reset intersects `[T − 7d, T]`; resets split
  eligibility and alert episodes exactly once.
- **Quarantine/censoring:** `T` is outside quarantine windows and uncensored; the anchor's window
  contains no quarantined, censored, or maintenance-overlapping file.
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

- Quota events are drawn from eligible anchors only, under the exact methodology (§3b) with
  solver provenance isolated from all generation streams.
- Robot/program caps are enforced **in** allocation as hard constraints: no accepted set may breach
  a cap; there is no post-hoc repair step.
- Subtype stratification is exact: the per-history subtype counts above are placed, never
  approximated; subtype identity is recorded before waveform generation.
- Nominal mix `37.5% P / 37.5% W / 25% A` stays inside the `[15%, 60%]` per-cohort gate by construction.
- Fit/Calibration roles exclude reserved identities (§6); Design includes all subtypes with per-subtype
  tallies reported.

## 3b. Exact deterministic global constraint satisfaction (v4 formulation retained verbatim, binding)

Eligible sets (§2), quotas/splits (§3), caps, the symmetric 14 d same-robot spacing predicate, the
rejection taxonomy for eligibility filtering, fail-fast infeasibility, and waveform/model blindness
are unchanged. The exact formulation below is carried forward byte-for-meaning from v4. No greedy
ordering, priority list, shuffle, round, restart, repair, or swap heuristic participates anywhere
in acceptance.

- **Decision variables.** One binary variable per eligible anchor, indexed in canonical
  construction order: eligible anchors sorted by `(failure_time, failure_id)`. Let the selected set
  be exactly the anchors whose variable equals 1 in the accepted witness.
- **Objective.** Zero vector (pure feasibility program). Selection among multiple feasible sets is
  determined solely by the frozen solver build over the frozen construction order (see
  determinism below); no rank weights, costs, or preferences enter.
- **Constraints (all hard, all frozen).**
  1. *Exact subtype quotas (6 equalities):* `P1 = 12`, `P2 = 12`, `W1 = 12`, `W2 = 12`, `A1 = 8`,
     `A2 = 8` over the eligible anchors of each (cohort, subtype).
  2. *Same-robot spacing conflict edges:* for every pair of eligible anchors on one robot with
     `|t_i − t_j| < spacing_s` (the unchanged predicate, `spacing_s` = 14 d), `x_i + x_j ≤ 1`.
  3. *Robot caps:* per robot, selected positives `≤ robot_cap` with
     `robot_cap = floor(0.35 × 64) = 22` — the exact V2-compliant bound
     (`22/64 = 34.375% ≤ 35%`), enforced as a hard global constraint. The
     greedy skip rules of retired candidates used `ceil` bounds and are
     superseded for v4/v5 (retired records stand unmodified).
  4. *Program caps (P/W only):* per program identity within each predictable cohort, selected
     positives `≤ floor(0.60 × 24) = 14` — the exact V2-compliant bound
     (`14/24 = 58.333% ≤ 60%`), enforced globally; anchors whose program is unknown (`None` in
     the frozen mapping) carry no program constraint, exactly as the frozen eligibility rule
     prescribed.
- **Solver (frozen).** `scipy.optimize.milp` (HiGHS backend) from the locked environment
  (`scipy==1.18.1` per `uv.lock`; exact wheel hashes recorded in Task 3 cycle-4, re-verified in
  Task 2 cycle-5). Default solver options; no time limit (per-history programs are small: on the
  order of 10² variables and 10³ constraints); no thread-count overrides — the locked build's default
  deterministic execution applies. No alternative solver, no network service, no floating-point
  relaxation of integrality.
- **Status mapping (binding).**
  - Solver reports proven optimal (`scipy.optimize.milp` `status == 0`, `success is True`,
    `"(HiGHS Status 7: Optimal)"`) **and** the independent witness validator (§3b, below) accepts
    the witness → allocation succeeds; downstream control-window selection and margin arithmetic
    (unchanged deterministic stages) proceed.
  - Solver reports proven infeasible (`status == 2`, `"(HiGHS Status 8: … Infeasible)"`) →
    `InfeasibleCandidate` with the existing record fields (`reason`, per-bucket `missing`,
    `history_seed`, `alloc_seed`, `eligible_pool`, `rejection` from the eligibility stage) plus
    solver provenance (`solver`, `scipy_version`, `solver_status`, `solver_message`, `solve_s`,
    `n_variables`, `n_constraints`).
  - Any other solver status (numerical error, iteration abort, unreturned witness) → surfaced as
    `UNAVAILABLE` (the candidate cannot be verified; it is never promoted on an unverified status).
  - The `stalled_buckets` record field is retired with §3a (no stalls exist in an exact method).
- **Witness validation (independent, binding).** Before any manifest persistence, a
  solver-independent checker re-verifies the witness using only the frozen predicates: all 64 ids
  unique and members of the eligible set; exact per-subtype counts; every same-robot pair satisfies
  `|Δ| ≥ spacing_s`; per-robot counts ≤ 22; per-program P/W counts ≤ 14. Validation failure is a
  hard error (no root is written).
- **Determinism and blindness.** Frozen construction order + frozen solver build + zero objective ⇒
  deterministic allocation given `history_seed`. Task 5 proves byte-identity empirically (two
  fresh-process allocations of one seed → identical selection). Inputs remain the same allowlisted
  metadata; allocation consumes no RNG stream and no waveform, score, or model output participates
  (`alloc_seed` is retained in provenance records for continuity).
- **Tie-breaking.** Canonical index order in matrix construction; no RNG, no priority heuristic.

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

If the exact allocator proves no quota/cap/spacing-satisfying anchor set exists on the frozen
calendar, it **fails fast with a machine-readable infeasibility record** (missing quota, binding
predicate, per-predicate rejection tallies from eligibility filtering, solver provenance) and the
active candidate is **rejected before promotion**. Never resample, replace, omit, pool, or
regenerate a root within the candidate. Diagnosis is evidence-reviewed; the fix is a new prospective
protocol version under the next seed namespace (`B_{N+1}`). Infeasibility of allocation, like any
structural/observable miss, retires only the candidate — never the sprint.

## 9. Case-control interpretation boundary (binding)

This is transparent case-control benchmark sampling for identifiable evaluation. Event labels,
cohorts, subtypes, failure times, simulator state, and allocation metadata are excluded from
model-visible rows. Realized cohort shares must never be interpreted as fleet prevalence estimates;
no prevalence, deployment-threshold, or calibrated-risk claim may be built on these quotas.

## 10. Candidate-5 roster, runtime, and roots (frozen)

`B_5 = 1400`. Profile `sprint15-v5` (Task 5 cycle-5 adds it; does not exist yet), protocol tag
`sprint15-benchmark-protocol-v5`, canonical runtime is the local repository root through the locked
`uv` environment
(`uv run python -m synth.cli --chronological --profile sprint15-v5 --seed <S> --role <R>
--protocol sprint15-benchmark-protocol-v5 --output data/generated/sprint15-v5/<ROLE>/`).
Bulk roots are local and uncommitted; manifests and seals carry the v5 protocol tag.

| Role | Root identities | Seeds |
|---|---|---|
| Design | H-DESIGN-29, 30, 31, 32 | 1400, 1401, 1402, 1403 |
| Fit | H-FIT-22, 23, 24 | 1404, 1405, 1406 |
| Calibration | H-CAL-8 | 1407 |
| Confirmation | H-CONF-26, 27, 28, 29 | 1408, 1409, 1410, 1411 |
| Sealed | H-SEAL-29, 30, 31, 32 | 1412, 1413, 1414, 1415 |
| Disposable proof | H-PROOF-9, 10 | 1416, 1417 |

Root identities continue past candidate 4 (`H-DESIGN-28`, `H-FIT-21` never materialized but burned,
`H-CAL-7`, `H-CONF-25`, `H-SEAL-28` never materialized but burned, `H-PROOF-8` are the last used;
reserved-but-unmaterialized candidate-4 Confirmation/Sealed identities are skipped, never reused).
All 18 seeds are fresh and disjoint from every Sprint 14 and candidate-1/2/3/4 seed (Task 4 cycle-5
proves disjointness and path absence). Every entry point refuses to start when a target root exists
and is non-empty; first-attempt status is recorded; no retry/replacement. No candidate-5 root exists
yet; none may be created before the fresh prospective review passes and Task 5 cycle-5 implements
the volume profile.

## 11. Verdict precedence (frozen)

`MEASURABLE-CANDIDATE` requires EG0–EG6 for candidate 5 with four-of-four `DESIGN-PASS`, full EG4
probe passage, EG5/EG6 structural passage, and zero-actionable evidence reviews. Any mandatory miss
returns `REJECTED-CANDIDATE`; ungeneratable/unverifiable roots or uncomputable metrics despite units
return `UNAVAILABLE-CANDIDATE`. Both retire candidate 5 only. Sprint finalization is success-only per
the V2 gate document.

## 12. EG0–EG7 clause crosswalk (V2 gate document)

| Gate clause | V2 gate section | Protocol section |
|---|---|---|
| EG0 version/digest/physics/roles/seeds | EG0 | Header (identifier, candidate binding); §§1, 3, 3b, 7a, 10 |
| EG0 cohorts/subtypes/holdouts | EG0 | §§3, 3b, 6 |
| EG0 temporal views/horizon/reset/quarantine/controls/anchors | EG0 | §2 |
| EG0 floors/quotas/targets/caps/probe metrics/precedence | EG0 | §§3–7a, §11 |
| EG0 commit/runtime/roots/seals | EG0 | §10 |
| EG0 candidate number/retirement ledger | EG0 | Header; §8 |
| EG1 causality/determinism/role isolation/leakage | EG1 | §§1–2, 7, 7a, 9–10 |
| EG1 physical duration/shape bounds | EG1 | §7a (generator compliance), §2 (audit) |
| EG2 design margin, no pooling | EG2 | §§3, 3b, 8 |
| EG3 fixtures | EG3 | §§4–5 (computability harness, Task 9) |
| EG4 probe thresholds | EG4 | §§4–5 (frozen probe inputs; implementation binds feature set) |
| EG5 confirmation + stability + holdouts | EG5 | §§1, 3, 3b, 6, 8 |
| EG6 sealed + seals + no scores | EG6 | §§3b, 6, 8, 10 |
| EG7 success-path closeout | EG7 | §11 |
| Boundaries (history unit, no seed selection, no learned influence, one probe, confirmation/sealed discipline, prospective-only, case-control, local-only) | Boundaries | §§1, 5, 7a, 8–10 |
| Role minimums 4/3/1/4/4 + reserves | Role topology | §§3, 6, 10 |
| Floors + caps + 80% lead support + Fit/Cal support | Floors | §§1–4, 6 |

## Appendix A. Exact Task 5 cycle-5 implementation boundary (proposed, NOT YET AUTHORIZED)

- **Touch only** the profile constructor addition (`sprint15_v5_history_config`, mirroring
  `sprint15_v4_history_config` with v4 settings carried forward) with exactly one parameter change:
  operation volume `n_units 2304 → 3456` and the arrival interval/jitter rescaled to
  `7466.67 s` (same 299-day arrival coverage inside the frozen 360-day span); the `sprint15-v5`
  CLI choice + default tag; the seal-allowlist `+1` entry; roster/tag constants (`S15_PROFILE_V5`,
  `S15_PROTOCOL_V5`, `S15_B5`, roster 1400–1417); and quota/audit constants only if they reference
  the v4 tag (tag-parameterized, no threshold edits). No allocator, predicate, cap, quota, nuisance,
  margin, holdout, physics, or probe line changes.
- **Must not touch:** V2 gate document, v1/v2/v3/v4 protocols/roots/evidence, candidate-1/2/3/4 bytes,
  floors, quotas, caps, spacing, horizons, predicates, thresholds, verdict precedence, DGP physics
  (`health.py`, `config.py`), probe (`probe15.py`), or any Sprint 14 file. (The volume parameter
  lives in the new `sprint15_v5_history_config` profile constructor in `chronicle.py`, which carries
  forward all v4 scheduler/route/fleet/factory settings except `n_units` and the rescaled cadence.)
- Implementation, focused tests, and proof roots belong to Task 5 cycle-5 **after** the fresh
  prospective review passes — not in this task.
