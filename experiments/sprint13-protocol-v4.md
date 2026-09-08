# Sprint 13 Benchmark Protocol v4 (FROZEN — APPROVED)

**Status:** FROZEN before any v4-outcome generation. Any change needs an explicit
protocol amendment (v4.1+, reason, re-freeze hashes) plus approval before
rematerialization. Prior protocols (`sprint12-protocol-v1..v3.md`) stay immutable.

**Approval:** Main, 2026-09-08 — approved for Batch C implementation/
materialization contingent on preserving these semantics (SHA256
`09ef31426682c513239ad8c984dec9d061a52cb00de1c4d02da7ef42cdb8f079`).
Batch B gate PASS (`artifacts/sprint-13/review-batch-b.md`).
**Consolidates:** Tasks 2 (estimands), 3 (floors/verdicts), 4 (cohorts), 5 (density),
6 (roles/holdouts). Detail lives in `artifacts/sprint-13/task-{2,3,4,5,6}.md`;
this document is the authoritative freeze — conflicts resolve in its favor and
trigger a task-artifact correction note.

## 1. Pre-generation Batch A clarification (adopted here, pre-freeze)

Task 2 §5 over-generalized quarantine exclusion across roles. Frozen predicates
(faithful to accepted Protocol v3: positives never required non-quarantined;
control anchors did):

- **POS-ELIGIBLE(file, T):** temporal view, uncensored `future_targets`,
  holdout-pass, no maintenance overlap, same robot, end ∈ `[T−H, T]`.
  Quarantine is NOT a condition (in-window files are expected precursor files).
- **ANCHOR-ELIGIBLE(file):** POS conditions plus non-quarantined, plus the §4
  clearance below. Members inherit anchor eligibility.
- **HEALTHY-ELIGIBLE(file):** pre-cutoff, normal label, non-quarantined, no
  maintenance overlap (development/reference cohorts only).

## 2. Generator settings (Task 8 implementation target)

- Base: existing `SynthConfig` machinery, C=6 channels, generator v2 physics.
- History: span 180 d (≤ 183 bound), 8 robots, 8 programs, origin fixed by Task 8.
- Health/hazard: latent `H_r(t)`; `λ = abrupt_rate + base_rate·exp(alpha·H + beta·W)`
  per cohort: P (`base_rate>0`, wear `w_P`, 45%), W (`base_rate>0`, wear `w_W`,
  amplitude ×0.3, 30%), A (`base_rate=0`, `abrupt_rate>0`, subtypes A1/A2, 25%).
  Overall per-robot MTBF 22 d. Degradation draws: P U[7,14] d, W U[14,28] d, A 0.
  Severity s ∈ {1.0, 2.0, 4.0}.
- Maintenance: preventive every 30 d (1 d) + corrective 2 d per failure;
  per-robot downtime < 15%; resets per Task 2 §7 (no cross-maintenance lookback).
- Quarantine 7 d pre-failure; file rate ≈ 1.6/robot-day via span-preserving
  `--units` scale. No representation-aware or performance-driven adjustments.

## 3. History roles and holdouts (frozen roster)

H-VAL-DESIGN (300); H-DEV-1..3 (301–303); H-FIT-1..3 (304–306); H-CAL-1 (307);
H-CONF-1 (308); H-SEAL-1..4 (400–403). Whole-history isolation; no
performance-based seed/history selection or replacement. Holdouts from all
fit/selection: `program-03`, `robot-08` (descriptive-only in eval), abrupt
subtype A2 (descriptive-only; A1-only fitting). FIT keeps hierarchy identity
(≥ 32/group where reachable, else fallback-documented, never merged).
Chronological cutoff day 120/180; 80/20 pre-cutoff healthy split.

## 4. Event metrics (frozen implementations)

H = 7 d primary (1 d descriptive companion). Positives: one max-file-score per
`(h, r, T)` over POS-ELIGIBLE files in `[T−H, T]`; no eligible file →
unevaluable (counted, never imputed). Negatives: ANCHOR-ELIGIBLE end-times in
end-time order, greedy 7 d non-overlap, kept iff no failure start in
`[e−H, e+H)` and no maintenance overlap; score = window max; pooled across
robots within one history only. E1 tie-aware AUROC (sklearn convention;
constant ≡ 0.5); E2 first-alert lead `T − e_first` (median/IQR; `lead = 0` is
detection, positive warning requires `lead > 0` strictly); E3 recall with
Clopper–Pearson 95%; E4 persistence median/IQR; E5 false episodes (≤ 2 d
grouping) per evaluated robot-day (zero → rule-of-three `3/D` bound).
Thresholds selected on H-CAL-1 only at FPR ≤ 0.10 tie-safe; sealed data never
enters fitting/selection. Category labels (`mechanism_id`, cohort, subtype,
onset, T, severity, duration) are generator-written post-hoc metadata, never
model inputs. Abrupt events stay in the overall ledger with conditional
reporting; re-labeling forbidden.

## 5. Cohort floors (all per-history unless noted)

Positives ≥ 25 evaluable; negatives ≥ 25 matched; evaluated robot-days ≥ 150;
category presence ≥ 8/category/history; clean baselines (Task 4 §6 rule)
aggregate ≥ 25/category; CAL ≥ 40/selection history; FIT ≥ 200 files / 6000 rows
aggregate; sealed histories ≥ 4. Absolute computability triggers: ≥ 2
evaluable positives, ≥ 6 negatives. Design points (Task 5): 40/40 events.
Lead medians only with ≥ 10 recalled events (else SPARSE raw values).

## 6. Permitted inspections and sealing

Pre-seal inspectable (exhaustive): seed/config hash, manifest counts, exclusion
tallies by reason, timestamp-derived event/window/control counts, maintenance
calendars, generator cohort tallies, reload identity. Forbidden pre-seal:
scores, thresholds, threshold-derived counts, model comparisons, performance
orderings, density/seed re-tuning. Violated histories are void. Sealed roots:
`seal.json` (manifest SHA256 + config hash + role + seed), bulk data remote and
uncommitted, no fit/selection/attribution access; structural counts readable
only by the Task 13 audit; scores open only at a future predeclared gate.

## 7. Advancement rules and stop conditions

- Task 10 proves determinism, bounds (`7 ≤ d_P ≤ 14`, `14 ≤ d_W ≤ 28`,
  `d_A = 0`), A-NULL precursor check, censoring, control selection, lead
  semantics, ties, and role isolation on a tiny history + fixtures.
- Task 11 generates exactly the §3 roster via the public entry point; no
  performance retries.
- Task 13 audits §§5–6 from manifests without scores; ≥ 2 missed cohort floors
  rejects a history.
- Task 14 proves E1–E5 computability with deterministic fixtures (not model
  selection or success).
- Task 15 verdict (§9a rules): `MEASURABLE` iff structural PASS on all sealed
  histories and E1–E5 computable without file-AUROC fallback; `NOT_MEASURABLE`
  on structural floor failure; `UNAVAILABLE` on unmet triggers/undefined
  metrics. Non-pass blocks Sprint 14 benchmark-dependent work and calibrated
  risk; no metric fallback, no post-hoc seed search. Downstream model gate
  (G ≥ 0.70 and ≥ constant + 0.10, unanimous) applies only to future model
  evaluation, never to this verdict.
- Post-freeze semantic corrections require an amendment (new version, reason,
  re-freeze) before any rerun; already-materialized histories under a superseded
  version are retired untouched, never patched in place.
