# Sprint 15 Benchmark Protocol v4 — Balanced Causal Sampling (FROZEN, prospective)

**Status:** Frozen prospectively by the Sprint 15 worker. NOT APPROVED — nothing may execute under v4
(no Task 5 implementation, no Task 9 proof, no Design materialization) before a fresh evidence review
passes with zero actionable findings. Any further semantic change requires a new protocol version
(`v5`, …) and a fresh evidence review before outcomes exist.

**Normative contract:** `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` (this protocol never weakens,
reinterprets, or replaces it; on conflict the gate document governs; the gate document itself is
unchanged across all candidates — see Task 2 cycle-4 continuity).
**Protocol identifier:** `sprint15-benchmark-protocol-v4`.
**Candidate binding:** candidate `N = 4`, base seed `B_4 = 1300` (see §10 and Task 4 cycle-4).
**Lineage:** Sprint 14 `NOT_MEASURABLE` record acknowledged in `artifacts/sprint-15/task-1.md`;
candidate 1 (`sprint15-benchmark-protocol-v1`, seeds 1000–1017) RETIRED after Gate C1 FAIL (3/4
DESIGN-PASS; H-DESIGN-16 `w-dist-shape` breach); candidate 2 (`sprint15-benchmark-protocol-v2`,
seeds 1100–1117) RETIRED after Gate D FAIL (0/4 Confirmation materialized; P→W→A shared-spacing
starvation of abrupt subtypes); candidate 3 (`sprint15-benchmark-protocol-v3`, seeds 1200–1217)
RETIRED after Gate D FAIL (2/4 Confirmation materialized; H-CONF-18/20 abrupt-subtype
quota-shortfalls A/A1 7<8 under greedy joint allocation despite globally feasible calendars).
All preserved immutable — zero retired roots, seeds, or gates reused (see §0).

## 0. Amendment record (v3 → v4; exact allocation methodology + candidate advance)

`experiments/sprint15-benchmark-protocol-v3.md`
(SHA256 `1d76e630787f5f248eaa5cd771eb929984476c4220f89722d75765876675cedb`) is preserved
byte-identical as the retired candidate-3 protocol. The v3→v4 delta is exactly two items, nothing
else:

1. **Exact deterministic global constraint-satisfaction allocation (§3b, the only methodology
   change).** The heuristic §3a joint round-robin acceptance scheduling is replaced by an exact
   binary constraint program over the eligible anchors: feasibility is declared only on a validated
   solver witness, infeasibility only on a proven infeasible solver status. Quotas, spacing
   predicate, caps, predicates, fail-fast discipline, and waveform/model blindness are unchanged;
   protocol v2 §7a (minimum-duration firing gate) is retained verbatim (§7a below); no DGP physics
   changes in v4.
2. **Candidate-4 advance (§10).** Profile `sprint15-v4`, protocol tag
   `sprint15-benchmark-protocol-v4`, fresh seed band 1300–1317 with fresh role identities continuing
   past candidate 3, collision-free roots under `data/generated/sprint15-v4/<ROLE>/` and disposable
   proof under `artifacts/sprint-15/proof-candidate-4/`.

Every other section below is carried forward byte-for-meaning identical from v3: calendar-first order,
anchor predicates, quotas/splits/caps, nuisance envelopes, signal margins, subtype/holdout rules,
physical invariants plus §7a, infeasibility discipline, case-control boundary, verdict precedence,
and the EG crosswalk (with §3a rows superseded by §3b rows).

**Diagnosed mechanism (Gate D F-01, cycle 3):** two of four candidate-3 Confirmation histories failed
quota placement fail-fast before persistence on `A/A1` (`H-CONF-18` seed 1208 and `H-CONF-20` seed
1210, accepted 7 vs required 8). Independent exact binary MILP (HiGHS via `scipy.optimize.milp`)
proves both failed calendars admit globally feasible 64-anchor assignments satisfying exact quotas,
14-day pairwise spacing, and all caps — full witness evidence in `artifacts/sprint-15/task-3-cycle-4.md`.
infeasibility: round-robin Cascades — early accepted anchors assert spacing exclusions that later
visits cannot repair, and no backtracking exists. Quota, margin, rate, or seed changes cannot remove
an order-dependent acceptance defect; only an exact method can, prospectively.

**Predicted effects:** feasibility-declared if and only if a quota/cap/spacing-satisfying anchor set
exists on the frozen calendar — no order luck, for any cohort. Expected: abrupt-subtype placement at
the pool-limited rate on every history (pools adequate per Design/proof/failed-seed MILP evidence);
P/W medians/counts emergent as before; abrupt physics, DGP streams, and all predicates untouched.
Verified on candidate-4 proof roots before any Design materialization (Tasks 9/11 cycle 4).

**Rejected alternatives:** raising the abrupt rate (wrong cause — A pools are adequate at 33–41 raw
and MILP-feasible on the failed seeds); lowering A quotas or subtype splits (weakens an inviolable
gate); relaxing the 14 d spacing predicate (inviolable causal predicate); per-cohort independent
spacing ledgers (breaks the shared-horizon non-overlap guarantee); post-hoc swap/repair or
scarce-first greedy heuristics (still order-dependent: any greedy acceptance order admits a
starvation counterexample, as cycles 2 and 3 respectively demonstrate); larger round budgets or
restarts (result-dependent resampling in disguise); seed search, retry, replacement, or pooling
(forbidden — unanimity rule).

## 1. Calendar-first construction order (binding)

For each history, in this exact order:

1. Generate the shared-unit calendar: routes, operations, unit arrivals, preventive/corrective
   maintenance, recommissioning, and the healthy background signal — with **no failure episodes
   placed**.
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

## 3b. Exact deterministic global constraint satisfaction (v4 addition, binding)

Eligible sets (§2), quotas/splits (§3), caps, the symmetric 14 d same-robot spacing predicate, the
rejection taxonomy for eligibility filtering, fail-fast infeasibility, and waveform/model blindness
are unchanged. The heuristic §3a acceptance scheduling is retired in full and replaced by the
following exact formulation. No greedy ordering, priority list, shuffle, round, restart, repair, or
swap heuristic participates anywhere in acceptance.

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
     superseded for v4 (retired records stand unmodified).
  4. *Program caps (P/W only):* per program identity within each predictable cohort, selected
     positives `≤ floor(0.60 × 24) = 14` — the exact V2-compliant bound
     (`14/24 = 58.333% ≤ 60%`), enforced globally; anchors whose program is unknown (`None` in
     the frozen mapping) carry no program constraint, exactly as the frozen eligibility rule
     prescribed.
- **Solver (frozen).** `scipy.optimize.milp` (HiGHS backend) from the locked environment
  (`scipy==1.18.1` per `uv.lock`; exact wheel hashes recorded in Task 3 cycle-4). Default solver
  options; no time limit (per-history programs are small: on the order of 10² variables and 10³
  constraints); no thread-count overrides — the locked build's default deterministic execution
  applies. No alternative solver, no network service, no floating-point relaxation of integrality.
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

## 10. Candidate-4 roster, runtime, and roots (frozen)

`B_4 = 1300`. Profile `sprint15-v4` (Task 5 cycle-4 adds it; does not exist yet), protocol tag
`sprint15-benchmark-protocol-v4`, canonical runtime is the local repository root through the locked
`uv` environment
(`uv run python -m synth.cli --chronological --profile sprint15-v4 --seed <S> --role <R>
--protocol sprint15-benchmark-protocol-v4 --output data/generated/sprint15-v4/<ROLE>/`).
Bulk roots are local and uncommitted; manifests and seals carry the v4 protocol tag.

| Role | Root identities | Seeds |
|---|---|---|
| Design | H-DESIGN-25, 26, 27, 28 | 1300, 1301, 1302, 1303 |
| Fit | H-FIT-19, 20, 21 | 1304, 1305, 1306 |
| Calibration | H-CAL-7 | 1307 |
| Confirmation | H-CONF-22, 23, 24, 25 | 1308, 1309, 1310, 1311 |
| Sealed | H-SEAL-25, 26, 27, 28 | 1312, 1313, 1314, 1315 |
| Disposable proof | H-PROOF-7, 8 | 1316, 1317 |

Root identities continue past candidate 3 (`H-DESIGN-24`, `H-FIT-18`, `H-CAL-6`, `H-CONF-21`,
`H-SEAL-20` reserved-but-unmaterialized, `H-PROOF-6` are the last used; reserved-but-unmaterialized
candidate-3 Confirmation identities `H-CONF-18`/`H-CONF-20` are skipped, never reused). All 18 seeds
are fresh and disjoint from every Sprint 14 and candidate-1/2/3 seed (Task 4 cycle-4 proves
disjointness and path absence). Every entry point refuses to start when a target root exists and is
non-empty; first-attempt status is recorded; no retry/replacement. No candidate-4 root exists yet;
none may be created before the fresh prospective review passes and Task 5 cycle-4 implements §3b.

## 11. Verdict precedence (frozen)

`MEASURABLE-CANDIDATE` requires EG0–EG6 for candidate 4 with four-of-four `DESIGN-PASS`, full EG4
probe passage, EG5/EG6 structural passage, and zero-actionable evidence reviews. Any mandatory miss
returns `REJECTED-CANDIDATE`; ungeneratable/unverifiable roots or uncomputable metrics despite units
return `UNAVAILABLE-CANDIDATE`. Both retire candidate 4 only. Sprint finalization is success-only per
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
| EG5 confirmation + stability + holdouts | EG5 | §§3, 3b, 6, 8 |
| EG6 sealed + seals + no scores | EG6 | §§3b, 6, 8, 10 |
| EG7 success-path closeout | EG7 | §11 |
| Boundaries (history unit, no seed selection, no learned influence, one probe, confirmation/sealed discipline, prospective-only, case-control, local-only) | Boundaries | §§1, 5, 7a, 8–10 |
| Role minimums 4/3/1/4/4 + reserves | Role topology | §§3, 6, 10 |
| Floors + caps + 80% lead support + Fit/Cal support | Floors | §§2–4, 6 |

## Appendix A. Exact Task 5 cycle-4 implementation boundary (proposed, NOT YET AUTHORIZED)

- **Touch only** the acceptance scheduling inside `allocate_quotas` in `src/synth/balanced.py` (the
  §3a joint round-robin fill): replace it with the §3b exact program (canonical index construction,
  zero objective, quota equalities, spacing edges, cap inequalities, HiGHS via
  `scipy.optimize.milp`, status mapping, independent witness validation before persistence,
  extended solver-provenance record). Keep `eligible_anchors`, all predicates, cap formulas, the
  spacing predicate, control-window selection, margin arithmetic, and the fail-fast call sites
  exactly as they are.
- **Touch only** the profile constructor addition (`sprint15_v4_history_config`, mirroring
  `sprint15_v3_history_config` with v3 settings and `min_duration_gate=True` carried forward),
  the `sprint15-v4` CLI choice + default tag, the seal-allowlist `+1` entry, roster/tag constants
  (`S15_PROFILE_V4`, `S15_PROTOCOL_V4`, `S15_B4`, roster 1300–1317), and quota/audit constants only
  if they reference the v3 tag (tag-parameterized, no threshold edits). New test module additions
  for solver witness determinism and status mapping; no existing test semantics weakened.
- **Must not touch:** V2 gate document, v1/v2/v3 protocols/roots/evidence, candidate-1/2/3 bytes,
  floors, quotas, caps, horizons, predicates, verdict precedence, DGP physics (`health.py`), or any
  Sprint 14 file.
- Implementation, focused tests, and proof roots belong to Task 5 cycle-4 **after** the fresh
  prospective review passes — not in this task.
