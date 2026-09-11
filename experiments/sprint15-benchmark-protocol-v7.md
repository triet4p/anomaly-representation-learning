# Sprint 15 Benchmark Protocol v7 — Balanced Causal Sampling (FROZEN, prospective)

**Status:** Frozen prospectively by the Sprint 15 worker. NOT APPROVED — nothing may execute under v7
(no Task 5 implementation, no Task 9 proof, no Design materialization) before a fresh evidence review
passes with zero actionable findings. Any further semantic change requires a new protocol version
(`v8`, …) and a fresh evidence review before outcomes exist.

**Normative contract:** `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` (this protocol never weakens,
reinterprets, or replaces it; on conflict the gate document governs; the gate document itself is
unchanged across all candidates — see Task 2 cycle-7 continuity).
**Protocol identifier:** `sprint15-benchmark-protocol-v7`.
**Candidate binding:** candidate `N = 7`, base seed `B_7 = 1600` (see §10 and Task 4 cycle-7).
**Lineage:** Sprint 14 `NOT_MEASURABLE` record acknowledged in `artifacts/sprint-15/task-1.md`;
candidate 1 (`sprint15-benchmark-protocol-v1`, seeds 1000–1017) RETIRED after Gate C1 FAIL (3/4
DESIGN-PASS; H-DESIGN-16 `w-dist-shape` breach); candidate 2 (`sprint15-benchmark-protocol-v2`,
seeds 1100–1117) RETIRED after Gate D FAIL (0/4 Confirmation materialized; P→W→A shared-spacing
starvation of abrupt subtypes); candidate 3 (`sprint15-benchmark-protocol-v3`, seeds 1200–1217)
RETIRED after Gate D FAIL (2/4 Confirmation materialized; H-CONF-18/20 abrupt-subtype
quota-shortfalls A/A1 7<8 under greedy joint allocation despite globally feasible calendars);
candidate 4 (`sprint15-benchmark-protocol-v4`, seeds 1300–1317) RETIRED after Gate D FAIL (7/8
downstream materialized; H-FIT-21 exact-CSP proven-infeasible on a counting-thin abrupt pool;
H-CONF-24 FAR 0.0536 > 0.05 on a truncated 2-Fit probe basis); candidate 5
(`sprint15-benchmark-protocol-v5`, seeds 1400–1417) RETIRED after Gate B FAIL (0/2 proof feasible;
exact 64-anchor quotas placed proven-optimal on both roots but control windows collapsed to 13/11
vs 48 under 1.5× volume at fixed span); candidate 6 (`sprint15-benchmark-protocol-v6`, seeds
1500–1517) RETIRED after Gate D FAIL (7/8 downstream materialized; full 3/3 Fit restored, but
H-CONF-31 exact-CSP proven-infeasible on W1 pool 8 < 12 — the v6 W trim stacked with adverse
subtype variance).
All preserved immutable — zero retired roots, seeds, or gates reused (see §0).

## 0. Amendment record (v6 → v7; rate revert + subtype stratification + rejection-only preflight + candidate advance)

`experiments/sprint15-benchmark-protocol-v6.md`
(SHA256 `7ed6e20816f4c15824b231b159b2a372b7b6dc0fbac312c3e1b0715af90492b9`) is preserved
byte-identical as the retired candidate-6 protocol. The v6→v7 delta is exactly four items, nothing
else:

1. **Rate revert (§1a).** Cohort outcome shares return to the stable v4 baseline P 0.45 / W 0.30 /
   A 0.25 with the v4 dominant hazard rates (P `base_rate` 5.0e-10; W `base_rate` 3.0e-9; A
   `abrupt_rate` 2.2e-5; `upcoming_p` 0.52; wear/thresholds/gains unchanged). No element of the v6
   reallocation is retained: at stratified splits the v4 cohort totals already carry slack ≥3 at
   every subtype, while the v6 W trim demonstrably created the 1509 W1 tail.
2. **Deterministic subtype stratification (§1a, the only DGP-semantics change).** Within-cohort
   subtype emission changes from uniform-random draws to deterministic round-robin alternation per
   (robot, cohort) episode/abrupt ordinal with a hash-derived starting offset (specification below).
   Emitted sequences balance to ≤1 per robot-cohort; subtype-blind eligibility filtering preserves
   the balance into eligible pools. Timing, density, and all hazard draws are invariant (proof
   obligations below).
3. **Rejection-only roster-wide structural preflight (§1b, workflow gate — no DGP semantics).**
   After roster/seed freeze and before any waveform/shard materialization, a deterministic no-write
   preflight runs the calendar → eligibility → exact-CSP-witness → subtype-pool → control/cap/window
   → support chain for all 18 fixed candidate seeds plus replay regression of retired failure seeds
   1306 and 1509 under the v7 method. It generates no waveforms, shards, manifests, or observable
   metrics. Any preflight miss retires candidate 7 immediately (§8) with no replacement, retry, or
   tuning; passing preflight merely authorizes the normal gates/materialization. Preflight is
   rejection-only evidence, never seed selection (seeds fixed pre-preflight) nor outcome tuning
   (no thresholds/metrics/scores involved).
4. **Candidate-7 advance (§10).** Profile `sprint15-v7`, protocol tag

Every other section below is carried forward byte-for-meaning identical from v6: calendar-first order
(v4 schedule), anchor predicates, quotas/splits/caps, the exact §3b allocation formulation, nuisance
envelopes, signal margins, subtype/holdout rules, physical invariants plus §7a, infeasibility
discipline, case-control boundary, verdict precedence, and the EG crosswalk.

**Diagnosed mechanism (Gate D F-01, cycle 6):** H-CONF-31 (seed 1509) failed exact allocation
fail-fast with W1 pool 8 < 12 (spacing-alone max 8; all drop-one-group ablations infeasible) while
W total stood at 34 and every other bucket carried slack. The three retired misses (C3 A/A1 7<8,
C4 A2 7<8, C6 W1 8<12) share one mechanism: adequate cohort totals with starved subtype splits
under uniform-random emission. Cohort-level fixes (v4 exact method, v5 volume, v6 mix shift) each
moved the failure to the next-thinnest split. Full evidence in
`artifacts/sprint-15/task-2-cycle-7.md`.

**Predicted effects:** holding every cohort total at v4 levels and splitting each evenly (minus one
unit of filtering-skew allowance), projected worst subtype pools over all 41 prior histories are
P 24, W 15, A 11 vs needs 12/12/8 — 0/41 below need (observed worsts were 14/14/9 with misses at
7/8). Control space stays at the v4 regime (50–66 observed) since total density is unchanged;
P/W spacing maxima absorb any residual skew; subtype ratios no longer depend on draw luck. Full
3/3 Fit roster materialization restores the predeclared probe basis with the frozen 95th-percentile
threshold (C3 precedent FARs 0.0421/0.0414); balanced subtype pools dampen count variance (F-04
class). Verified on candidate-7 proof roots before any Design materialization (Tasks 9/11
cycle 7).

**Rejected alternatives:** lowering subtype needs or quotas (V2 construction targets); spacing /
horizon / window changes (causal predicates); cap changes (V2 limits); any density change
(C5 proved density kills controls; cuts thin totals); retaining any v6 rate shift (created the
1509 tail); horizon/quarantine/reset/censoring changes (explicit sprint non-goals); threshold/metric
retuning (V2-invariant); fleet-topology change (larger ripple); post-hoc subtype rebalancing or
relabeling (breaks the subtype→gain severity coupling and waveform causality — labels must be
assigned at episode/failure time); seed search, retry, replacement, resampling, or pooling
(forbidden — unanimity rule).

## 1. Calendar-first construction order (binding)

For each history, in this exact order:

1. Generate the shared-unit calendar: routes, operations, unit arrivals, preventive/corrective
   maintenance, recommissioning, and the healthy background signal — with **no failure episodes
   placed**. Operation volume is the frozen v4 level (`n_units = 2304`; arrival interval and jitter
   `11200.0 s`; 360-day span; nine-robot fleet; dev cutoff 180 d).
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

## 1a. Cohort outcome shares and deterministic subtype stratification (binding)

Cohort outcome shares (documentary targets; realized mix emerges from the rates) are the stable v4
baseline P 0.45 / W 0.30 / A 0.25 (sum 1.0), realized by the v4 dominant hazard rates: P
`base_rate` 5.0e-10, W `base_rate` 3.0e-9, A `abrupt_rate` 2.2e-5, with `upcoming_p` 0.52, wear
rates, thresholds, gains, and severity law unchanged. Expected total failure density is the v4
level: the only level ever observed to yield ≥48 controls.

Subtype emission within each cohort is deterministic round-robin alternation, replacing the
uniform-random draws:

- **P/W (episode opening).** The existing dedicated hash sub-stream keyed by (health seed, robot,
  cohort, per-cohort episode ordinal) is retained, so hazard draws and failure timing are invariant
  by construction. The uniform pick over the two labels is replaced by
  `labels[(ordinal + digest[0]) % 2]` with 1-based ordinal and the same digest: the emitted
  sequence per (robot, cohort) alternates with a hash-derived starting label (no systematic
  first-label bias), balancing emitted P1/P2 and W1/W2 to ≤1 per robot-cohort.
- **A (failure time).** The legacy shared-RNG draws at the abrupt branch are retained verbatim as
  stream-preserving no-ops (executed in identical order and count — the severity draw remains
  functional), so failure timing and density are bit-identical; the assigned label comes from a
  dedicated alternating counter per (seed, robot) abrupt ordinal with hash-derived starting offset,
  balancing emitted A1/A2 to ≤1 per robot.
- **Invariants (binding on implementation).** Subtype-blind eligibility is unchanged, so eligible
  pools inherit the emission balance up to filtering skew. Subtype→gain severity coupling (§5) is
  unchanged — labels are assigned at episode/failure time, never relabeled post-hoc. Task 5
  cycle-7 proves before any root: (i) failure-timing/density invariance (flag-off/flag-on ledger
  comparison on focused synthetic seeds); (ii) the alternation bound (emitted per-(robot,cohort)

  splits differ by ≤1); (iii) flag-off byte-identity (v1–v6 profiles reproduce frozen bytes);
  (iv) the full focused suite green.

No floor, cap, quota, threshold, window, or metric is touched by this section; the C7 proof verifies
the projection before any Design materialization.

## 1b. Roster-wide structural preflight (rejection-only, no-write; binding)

After roster/seed freeze (§10) and before any waveform/shard materialization, a deterministic
no-write preflight executes the structural chain for every fixed candidate seed 1600–1617:
calendar construction → frozen-calendar bytes → eligible anchor set (§2) → exact-CSP witness (§3b,
validated) → per-subtype pool tallies → control-window selection → cap/window/support arithmetic.
It generates no waveforms, no shards, no manifests, no seals, and no observable metrics (no probe,
threshold, score, or model runs at any point in preflight).

- **Fixed roster, retained in full.** All 18 seeds are frozen before preflight and all 18 are
  retained regardless of outcome: preflight may not drop, replace, resample, or reorder any seed.
  Any preflight miss — quota/controls/caps/windows/support shortfall on any seed, witness breach,
  or replay-regression failure — retires candidate 7 immediately under §8 with no replacement,
  retry, tuning, or second run. Passing preflight merely authorizes the normal evidence gates and
  first-attempt materialization; it confers no design credit and selects nothing.
- **Replay regression (binding).** Preflight additionally replays retired failure seeds 1306 and
  1509 under the v7 method (calendar → eligibility → exact witness, no-write, no roots — retired
  seeds are never rematerialized as candidate roots): both must place exact quotas with valid
  witnesses and ≥48 controls. Failure retires candidate 7 under §8.
- **Rejection-only classification (binding).** Preflight is rejection-only evidence: it can only
  retire the candidate, never promote, rank, or select seeds, and it tunes no threshold, metric,
  quota, or model. It is not seed selection (the roster is fixed before preflight runs) and not
  outcome tuning (no observable metrics exist at preflight time). Preflight records (per-seed
  witness/pool/control/support tallies plus replay results) join the Task 9 evidence.
- **Already-passed components by digest continuity.** Preflight reuses the frozen allocator, audit
  predicates, control selection, and solver pins by exact digest continuity with the Gate A/C1/C2
  reviewed bytes — it reimplements nothing. Only the §1a emission change and the preflight runner
  itself are new code (Appendix A).

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
are unchanged. The exact formulation below is carried forward byte-for-meaning from v4/v5/v6. No greedy
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
     superseded for v4/v5/v6/v7 (retired records stand unmodified).
  4. *Program caps (P/W only):* per program identity within each predictable cohort, selected
     positives `≤ floor(0.60 × 24) = 14` — the exact V2-compliant bound
     (`14/24 = 58.333% ≤ 60%`), enforced globally; anchors whose program is unknown (`None` in
     the frozen mapping) carry no program constraint, exactly as the frozen eligibility rule
     prescribed.
- **Solver (frozen).** `scipy.optimize.milp` (HiGHS backend) from the locked environment
  (`scipy==1.18.1` per `uv.lock`; exact wheel hashes recorded in Task 3 cycle-4, re-verified in
  Tasks 2 cycle-5/6/7). Default solver options; no time limit (per-history programs are small: on the
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

## 10. Candidate-7 roster, runtime, and roots (frozen)

`B_7 = 1600`. Profile `sprint15-v7` (Task 5 cycle-7 adds it; does not exist yet), protocol tag
`sprint15-benchmark-protocol-v7`, canonical runtime is the local repository root through the locked
`uv` environment
(`uv run python -m synth.cli --chronological --profile sprint15-v7 --seed <S> --role <R>
--protocol sprint15-benchmark-protocol-v7 --output data/generated/sprint15-v7/<ROLE>/`).
Bulk roots are local and uncommitted; manifests and seals carry the v7 protocol tag.

| Role | Root identities | Seeds |
|---|---|---|
| Design | H-DESIGN-37, 38, 39, 40 | 1600, 1601, 1602, 1603 |
| Fit | H-FIT-28, 29, 30 | 1604, 1605, 1606 |
| Calibration | H-CAL-10 | 1607 |
| Confirmation | H-CONF-34, 35, 36, 37 | 1608, 1609, 1610, 1611 |
| Sealed | H-SEAL-37, 38, 39, 40 | 1612, 1613, 1614, 1615 |
| Disposable proof | H-PROOF-13, 14 | 1616, 1617 |

Root identities continue past candidate 6 (all C6 identities burned, H-CONF-31 with zero bytes
materialized; the last materialized identities remain candidate-4's (`H-DESIGN-28`, `H-CAL-7`,
`H-CONF-25`, `H-PROOF-8`) alongside candidate-6's (`H-DESIGN-36`, `H-FIT-27`, `H-CAL-9`,
`H-CONF-33`); burned-but-never-materialized `H-FIT-21`, `H-SEAL-25..28`, H-CONF-31 and
reserved-but-unmaterialized candidate-5/6 identities are skipped, never reused).
All 18 seeds are fresh and disjoint from every Sprint 14 and candidate-1/2/3/4/5/6 seed (Task 4
cycle-7 proves disjointness and path absence). Every entry point refuses to start when a target
root exists and is non-empty; first-attempt status is recorded; no retry/replacement. No
candidate-7 root exists yet; none may be created before the fresh prospective review passes and
Task 5 cycle-7 implements the stratification profile.

## 11. Verdict precedence (frozen)

`MEASURABLE-CANDIDATE` requires EG0–EG6 for candidate 7 with four-of-four `DESIGN-PASS`, full EG4
probe passage, EG5/EG6 structural passage, and zero-actionable evidence reviews. Any mandatory miss
returns `REJECTED-CANDIDATE`; ungeneratable/unverifiable roots or uncomputable metrics despite units
return `UNAVAILABLE-CANDIDATE`. Both retire candidate 7 only. Sprint finalization is success-only per
the V2 gate document.

## 12. EG0–EG7 clause crosswalk (V2 gate document)

| Gate clause | V2 gate section | Protocol section |
|---|---|---|
| EG0 version/digest/physics/roles/seeds | EG0 | Header (identifier, candidate binding); §§1, 1a, 3, 3b, 7a, 10 |
| EG0 cohorts/subtypes/holdouts | EG0 | §§1a, 3, 3b, 6 |
| EG0 temporal views/horizon/reset/quarantine/controls/anchors | EG0 | §2 |
| EG0 floors/quotas/targets/caps/probe metrics/precedence | EG0 | §§1a, 3–7a, §11 |
| EG0 commit/runtime/roots/seals | EG0 | §10 |
| EG0 candidate number/retirement ledger | EG0 | Header; §8 |
| EG1 causality/determinism/role isolation/leakage | EG1 | §§1–2, 1a, 7, 7a, 9–10 |
| EG1 physical duration/shape bounds | EG1 | §7a (generator compliance), §2 (audit) |
| EG2 design margin, no pooling | EG2 | §§3, 3b, 8 |
| EG3 fixtures | EG3 | §§4–5 (computability harness, Task 9) |
| EG4 probe thresholds | EG4 | §§4–5 (frozen probe inputs; implementation binds feature set) |
| EG5 confirmation + stability + holdouts | EG5 | §§1, 1a, 3, 3b, 6, 8 |
| EG6 sealed + seals + no scores | EG6 | §§3b, 6, 8, 10 |
| EG7 success-path closeout | EG7 | §11 |
| Rejection-only preflight gate (no-write, fixed roster, replay) | EG0/EG2 | §1b (authorizes gates; retires on miss under §8) |
| Boundaries (history unit, no seed selection, no learned influence, one probe, confirmation/sealed discipline, prospective-only, case-control, local-only) | Boundaries | §§1, 1a, 5, 7a, 8–10 |
| Role minimums 4/3/1/4/4 + reserves | Role topology | §§3, 6, 10 |
| Floors + caps + 80% lead support + Fit/Cal support | Floors | §§1a, 2–4, 6 |

## Appendix A. Exact Task 5 cycle-7 implementation boundary (proposed, NOT YET AUTHORIZED)

- **Touch only** the profile constructor addition (`sprint15_v7_history_config`, mirroring
  `sprint15_v4_history_config` values exactly: `n_units` 2304, cadence `11200.0 s`, v4 cohort
  shares/rates/upcoming/wear/thresholds) plus a new `HealthConfig` flag
  `stratified_subtype_emission: bool = False` set `True` by the v7 constructor only; the two
  subtype-emission sites in `health.py` (`_draw_pw_subtype` uniform pick → round-robin alternation
  `labels[(ordinal + digest[0]) % 2]` on the retained dedicated sub-stream; abrupt-branch label →
  dedicated per-(seed, robot) alternating counter with the legacy shared-RNG draws retained
  verbatim as stream-preserving no-ops); the `sprint15-v7` CLI choice + default tag; the
  roster 1600–1617); a deterministic no-write preflight runner executing §1b over the 18 fixed
  seeds plus 1306/1509 replay (calendar → eligibility → exact witness → pools → controls/caps/support;
  no waveform/shard/manifest/seal/metric code paths); and quota/audit constants only if they
  reference the v6 tag (tag-parameterized, no threshold edits). No allocator, predicate, cap, quota,
  nuisance, margin, holdout, hazard-rate, wear, threshold, gain, schedule, fleet, probe, or
  verdict-precedence line changes.
- **Must not touch (beyond the above):** V2 gate document, v1–v6 protocols/roots/evidence,
  candidate-1/2/3/4/5/6 bytes, floors, quotas, caps, spacing, horizons, predicates, thresholds,
  verdict precedence, hazard/wear/threshold/gain/schedule/fleet DGP structure (`health.py` except
  the two emission sites + flag read; `config.py` except the new defaulted flag), probe
  (`probe15.py`), or any Sprint 14 file.
- **Task 5 proof obligations (before any root):** (i) failure-timing/density invariance
  (flag-off/flag-on ledger comparison on focused synthetic seeds — identical timing, only labels
  differ); (ii) alternation bound (emitted per-(robot, cohort) subtype splits differ by ≤1);
  (iii) flag-off byte-identity (v1–v6 profiles reproduce frozen bytes); (iv) full focused suite
  green (existing semantics unweakened); (v) preflight execution (all 18 seeds feasible with valid
  witnesses and ≥48 controls, plus 1306/1509 replay feasible — miss anywhere retires the candidate
  before any waveform/shard work).
- Implementation, focused tests, and proof roots belong to Task 5 cycle-7 **after** the fresh
  prospective review passes — not in this task.
