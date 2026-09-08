# Sprint 13 Benchmark Protocol v4.1 (AMENDMENT — INTERNALLY FROZEN, AWAITING MAIN APPROVAL)

**Status:** DRAFT AMENDMENT frozen by worker before any v4.1-outcome generation.
NOT APPROVED — generation under v4.1 (rematerialization) MUST NOT start on
worker authority. Amends `experiments/sprint13-protocol-v4.md` (which stays
immutable); v4 materializations (seeds 300–308/400–403) retire untouched as
void-for-verdict historical records (bulk preserved, seals preserved as
tombstones, never reused).
**Reason:** sprint-wide deep review FAIL (`artifacts/sprint-13/deep-review-1.md`,
5 actionable findings). This amendment resolves findings 1–5 before any rerun.

## 1. Amendment record (v4 → v4.1)

| # | Change | Driver |
|---|---|---|
| 1 | Density/rates set analytically below (target MTBF 22 d retained); exact values frozen here, not tuned post-freeze | Finding 1 |
| 2 | Fresh roster: VAL 500, DEV 501–503, FIT 504–506, CAL 507, CONF 508, SEAL 600–603; old roster retired/void | Findings 1–2 |
| 3 | Causal cohort dynamics: upcoming-cohort draw at recommissioning; cohort wear (`w_P`/`w_W`); base hazard gated on open degradation episodes; abort handling; no retrospective window draws | Finding 3 |
| 4 | Task 14 fixture proof restricted to non-sealed roles (predeclared computability procedure); sealed roles are Task-13-structural-audit-only until a future score gate | Finding 2 |
| 5 | One shared eligible-operational-row predicate (censor exclusion + full-interval maintenance overlap) for E5 numerator/denominator and robot-day floors; holdout exclusions on FIT/CAL pools; per-file valid-patch counts persisted, ≥6000 aggregate enforced | Findings 4–5 |

Unchanged from v4 unless stated above: span/duration/topology, horizons,
exclusions, estimands E1–E5, floors, §9a verdict logic, quarantine policy.

## 2. Density, rates, and duration physics (analytic, score-independent)

Operating cadence (measured, structural): φ ≈ 1.6 ops/robot-day, mean op
duration d̄ = 600 s. Target per-robot failure rate Λ = 1/22 d⁻¹ ≈ 0.04545,
split Λ_P = 0.02045, Λ_W = 0.01364, Λ_A = 0.01136 d⁻¹.

- **Abrupt (closed form):** per-op fire prob p_A = Λ_A/φ ≈ 0.00710 →
  `abrupt_rate` = −ln(1−p_A)/d̄ ≈ **1.1e-5 s⁻¹** (rounded, verified emergent).
- **P/W failures are threshold-crossing with a small stochastic term:**
  base hazard `base·exp(min(cap, αH))` per op (α = 5.0) PLUS deterministic
  fire when H ≥ H* (H*_P = 2.0, H*_W = 1.35). Threshold crossing dominates:
  stochastic bases (P 5.0e-10, W 3.0e-9) contribute only rare early fires
  (leak ≈ 0.05%/episode), preserving schedule-driven spread without
  breaking minima.
- **Wear (causal climb):** pre-episode wear 3.0e-5 + aging 1e-7 gives onset
  (≈0.3) in ≈7 d; in-episode wear w_P = 2.0e-4 (r ≈ 0.20/d) and
  w_W = 5.0e-5 (r ≈ 0.057/d). Predicted climb durations: P ≈ 8.5 d,
  W ≈ 16–18 d. Cycle ≈ onset 7 d + duration + maintenance 2 d, plus the
  abrupt race → predicted MTBF ≈ 22–24 d.
- **Why distributional duration bounds:** per-op draws × bursty robot
  schedules make hard per-episode minima infeasible under any stochastic
  hazard (verified by experiment: dense op clusters produce fast tails);
  deterministic thresholds bound medians/maxima while minima follow climb
  physics with schedule spread. Frozen bounds (falsifiable — a common
  trajectory or stochastic-only mechanism violates the separated medians):
  P: min ≥ 2 d, median ∈ [5, 10] d, max ≤ 15 d;
  W: min ≥ 6 d, median ∈ [12, 24] d, max ≤ 28 d; A: exactly 0.
- **Acceptance of this design (throwaway verification, not tuning):**
  emergent MTBF ∈ [19, 25] d, shares within ±8 pp of 45/30/25, durations
  inside the distributional bounds. Seed-930 full-scale throwaway:
  MTBF 24.4, mix 46/27/27, P 2.97/5.97/12.66, W 8.19/14.70/21.25 — all hold
  (seed-929: MTBF 23.2, mix 35/40/24 with P-share 2 pp under band — ordinary
  seed variation around a centered design, disclosed not hidden).
  Draft iterations (initial gated-hazard point → concentrated firing →
  threshold-crossing) ran on throwaway seeds 920–927 (deleted) before this
  freeze; no protocol seed was observed in the process.

Frozen values: `base_rate_P=5.0e-10`, `base_rate_W=3.0e-9`,
`abrupt_rate_A=1.1e-5`, `wear_P=2.0e-4`, `wear_W=5.0e-5`,
`wear_base=3.0e-5`, `aging=1e-7`, `onset=0.3`, `alpha=5.0`,
`H*_P=2.0`, `H*_W=1.35`, κ_W=0.3, severity {1.0,2.0,4.0},
`upcoming_p=0.55`, maint 2 d corrective / 30 d + 1 d preventive.

## 3. Causal cohort dynamics (frozen mechanism)

- At history start and after every maintenance, each robot draws its
  **upcoming** non-abrupt cohort (P with prob `upcoming_p` = 0.55, else W) —
  dedicated stream, fixed order, pre-failure truth. Abrupt failures arrive
  via the independent constant hazard at any time.
- Pre-episode wear = base; on H ≥ onset an episode opens tagged with the
  upcoming cohort; wear switches to w_P/w_W from the open event onward.
- Base hazard fires ONLY while an episode is open (per-cohort base rate);
  additionally, reaching the cohort failure threshold (H*_P = 2.0,
  H*_W = 1.35) fires deterministically on that operation. The threshold
  carries durations; the small stochastic term carries schedule spread.
  The failure links that episode (onset = episode start — nothing placed).
- Abrupt hazard is always active; an A failure aborts any open episode
  (closed at recommissioning, never linked); A records carry onset None,
  duration 0, subtype A1/A2.
- Maintenance (corrective or preventive) closes interrupted episodes at
  recommissioning and redraws the upcoming cohort. Trajectories never
  connect across maintenance.

## 4. Manifestation (frozen)

Precursor severity = progressive_severity(G) × κ_cohort (κ_P = 1.0,
κ_W = 0.3), capped 0.95; precursor suppression within 7 d before abrupt
failures (isolated nuisance preserved). Failure-file severity =
min(0.95, 0.9 × s/2); s ∈ {1,2,4} is a label plus failure-magnitude scale
only — never applied retrospectively to precursor windows.

## 5. Roster and retirement

| Role | v4.1 histories (seeds) |
|---|---|
| Generator validation | H-VAL-DESIGN (500) |
| Development | H-DEV-1..3 (501–503) |
| Fit | H-FIT-1..3 (504–506) |
| Calibration/selection | H-CAL-1 (507) |
| Confirmation | H-CONF-1 (508) |
| Sealed evaluation | H-SEAL-1..4 (600–603) |

v4 roots (300–308/400–403, bulk + seals) are RETIRED/VOID for verdict
purposes: preserved untouched, never re-audited for acceptance, never
grandfathered into v4.1. Cold-start slices (`program-03`, `robot-08`) and
mechanism novelty (A2 out of FIT/CAL/CONF) carry over unchanged.

## 6. Roles, inspections, and the Task 14 boundary (predeclared)

Whole-history isolation and the Task 6 permitted/forbidden inspection lists
carry over. Clarified exception (this amendment IS the predeclaration):
**score/threshold fixture proof (Task 14 procedure) runs ONLY on allowed
non-sealed roles** (VAL/DEV/FIT/CAL/CONF) with deterministic fixtures and a
fixed arbitrary threshold; sealed roles are Task-13-structural-audit-only
until a future predeclared score gate (which does not exist in Sprint 13).
Any fixture run touching a sealed root voids it per the standing rule.

## 7. Metrics, floors, and shared predicates

Estimands E1–E5, horizons, windows, clearance, censoring, resets, and all
Task 3 floors (including FIT ≥200 files / ≥6000 valid patch rows and CAL
≥40) carry over. Frozen additions:
- **Eligible-operational-row predicate** (code: `eligible_operational_row`):
  row counts iff NOT censored AND NOT full-interval maintenance-overlapping
  (`win.start < row.end AND row.start < win.end`). Used identically for the
  E5 numerator (scored rows), the E5 denominator days, and robot-day floors.
  A (robot, day) counts iff ≥1 eligible row ends that day.
- **Holdout-eligible pools:** FIT/CAL pool counts exclude `program-03` and
  `robot-08` rows; raw pools reported alongside for transparency.
- **Valid patch rows:** per-file `n_valid_patches` persisted at
  materialization under the frozen patch contract (Patchifier emission
  count); aggregate ≥6000 over holdout-eligible FIT healthy files.

## 8. Sealing and verdict (unchanged logic)

Task 3 §9a `STRUCT-PASS`/`STRUCT-FAIL`/`UNAVAILABLE` → `MEASURABLE` /
`NOT_MEASURABLE` / `UNAVAILABLE`, applied to the v4.1 sealed roster only.
No G/C performance, no file-AUROC substitution, no seed search.

## 9. Execution plan (R2 and beyond — NOT R1)

After Main approval: materialize the §5 roster once via the public entry
point; re-run structural audit (Task 13 procedure with §7 predicates),
fixture proof on non-sealed roles, seal the four evaluation roots, publish
the verdict; renew gates C2/D1/D2 and the ledger with fresh reviews. R1
delivers code + contracts + focused test evidence only.
