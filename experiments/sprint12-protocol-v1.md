# Sprint 12 Experiment Protocol v1 (frozen)

**Status:** FROZEN — any change requires a version bump (v2) plus explicit user
approval; v1 text below is immutable. Frozen before any new outcome is generated
or inspected (Batch C gate). Historical Sprint 11 data stays diagnostic only.

## 1. Histories

Three independent server-scale histories, seeds frozen (never performance-selected):

| history | role | seed | server root (uncommitted, server-side) |
|---|---|---|---|
| H-DEV | development: fitting, calibration, selection | 100 | `data/generated/sprint12-dev/` |
| H-SEAL-A | sealed final evaluation | 200 | `data/generated/sprint12-seal-a/` |
| H-SEAL-B | sealed final evaluation (2nd uncertainty unit) | 201 | `data/generated/sprint12-seal-b/` |

Scale: server profile at 4× units — `n_units=1200`, `arrival_interval_s=5400`,
`arrival_jitter_s=5400` (total arrival span preserved: 300×21600 = 1200×5400;
higher line density, same 90-day calendar). Expected ≈2400 files/history.
Generation: public entry point `python -m synth.cli --chronological --profile
server --units 1200 --seed <seed> --output <root>` (`--units` added in Batch C;
span-preserving by construction).

## 2. Cohort roles (per history; disjoint by construction + exclusion rules)

- FIT: manifest `dev_train` minus reserved-family files minus program-03 files.
  Healthy fitting only (assert all `NORMAL`).
- CAL: manifest `dev_val` minus the same exclusions. Threshold calibration and
  development selection ONLY.
- SEL: selection reads CAL metrics only; no separate split, no sealed access.
- EVAL (sealed histories only): full `test_static` + `test_temporal` views —
  all families (incl. reserved), all programs (incl. program-03).
- H-DEV test views: untouched by training/selection; diagnostic use only with
  explicit record.
- Excluded-from-dev sets (reserved families §3, program-03 §4) remain IN sealed
  EVAL. A checker (Task 6 verifier, reused downstream) proves zero excluded
  files in any FIT/CAL/SEL cohort before any downstream run.

## 3. Reserved mechanisms (entire families, never in FIT/CAL/SEL)

`wrong_transition` (temporal-structural) and `cross_channel_inconsistency`
(relational). Rationale: span distinct mechanism classes; development
corruptions (Task 7) operate on the remaining six families' territory plus
mechanism-based synthesis. Both reserved families must appear in sealed EVAL
(verified counts; absence of either in a sealed history is a recorded deviation,
absence in both is a blocker).

## 4. Held-out cold-start axis

program-03 is excluded from ALL FIT/CAL/SEL cohorts in every history and scored
only in sealed EVAL (genuine program cold-start). robot-03 stays in dev
(known-robot coverage retained). Sealed program-03 abnormal count verified and
recorded (floor: ≥10 abnormal program-03 files per sealed history; shortfall is
a recorded deviation with reduced cold-start power, not silent pooling).

## 5. Quarantine and maintenance

- Seven-day precursor quarantine preserved exactly (generator-enforced; verified
  per history: no FIT/CAL file within quarantine_s of a failure).
- Non-saturated maintenance per robot per history: maintenance-time fraction
  < 25% of robot calendar span.
- Unsaturated contrast: ≥1 completed maintenance→recommission cycle followed by
  ≥7 days of healthy operation before the next failure episode.
- Shortfalls are recorded per robot (deviation), never repaired by weakening
  quarantine (add healthy exposure instead — out of scope for v1).

## 6. Cohort floors (from the 5% file-FPR operating target)

- CAL: ≥40 healthy files/history (tail-quantile resolution 1/41 ≈ 0.024; CIs
  reported, never hidden).
- EVAL per sealed history: ≥400 normal files (FPR SE ≤ 1.1% at 5%), ≥100
  abnormal files (recall SE ≤ 5% at 50%), ≥8 failure episodes (event-recall
  granularity 1/8).
- FIT: ≥32 healthy files per (robot, program) training group; smaller groups
  are fallback-documented (hierarchy handles them; fallback fractions recorded),
  never silently merged across robots.
- Task 6 verifies every floor from manifests; any miss is a RECORDED DEVIATION
  with power consequences, not a silent narrowing. Two or more missed EVAL
  floors in a sealed history = blocker (history rejected, no replacement seeds
  selected by performance).

## 7. Development selection rules (Task 9)

Single predeclared metric: dev paired corrupt-vs-clean ranking rate on H-DEV
CAL cohort. Caps: CAL file FPR ≤ 0.10, background stability, nuisance flag
≤ 0.10. Select exactly ONE configuration; no sealed access; no seed re-rolls;
failed gate → bounded documented development-only corrections, never a long run
on a failed gate.

## 8. Computational limits

Generation ≤ 12 h total; single Task-9 training ≤ 4 GPU-h; Batch D total
≤ 24 GPU-h; any eval run ≤ 1 h. Actuals recorded per run.

## 9. Uncertainty units and forbidden statistics

Unit hierarchy: history → robot → episode. Report medians/IQRs across the two
sealed histories plus per-robot/per-episode slices. NEVER bootstrap correlated
patches as independent samples; NEVER pool sealed histories for selection.

## 10. Numeric scientific advancement gates (frozen)

- G-learn (Task 9, H-DEV only): ranking rate ≥ 0.80; severity Spearman ≥ 0.50;
  background ≤ 0.10× corrupt gap; nuisance flag rate ≤ 0.10.
- G-static (Task 10, sealed): on EACH sealed history — learned AUROC ≥ 0.70 AND
  learned − handcrafted ≥ +0.05; file FPR ≤ 0.10 at the frozen threshold;
  localization hit ≥ 0.50. All four sub-conditions on both histories, no
  cherry-picking.
- G-rank (Task 12): 7-day event AUROC ≥ 0.70 on sealed AND ≥ +0.10 over the
  constant-risk baseline at comparable false-alert rate.
- G-risk (Task 13): ECE ≤ 0.15 AND Brier skill > 0 vs the operating-history
  baseline; else no calibration claim.
- 1-day recall: REPORTED ONLY (precursor visibility unknown until Task 11).

## 11. Seal procedure

Sealed roots carry `seal.json` (manifest + shard checksums + frozen date).
Training/selection code paths take dev roots only; any sealed-root read outside
an approved evaluation run is a protocol violation. Bulk data stays
server-side, uncommitted; only bounded manifests/counts/checksums return.

## 12. Deviation policy

Scientific failures stay explicit. Reachable work finishes; blockers name the
missing prerequisite with exact attempts. No threshold/seed/model substitution
for a failed gate without a protocol version bump and explicit user approval.
