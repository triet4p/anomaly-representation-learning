# Sprint 12 Experiment Protocol v2 — amendment (frozen)

**Status:** FROZEN by explicit user selection ("Multiple 90-day histories
(recommended)"). v1 (`experiments/sprint12-protocol-v1.md`) stays immutable;
this amendment supersedes ONLY the parts below. Any further change needs v3 +
explicit user approval. Frozen before any v2 outcome is generated.

## Supersession rationale (measured, not assumed)

v1 required 4× line density (1200 units, 5400 s cadence) with unchanged health
rates. Materialized H-DEV/H-SEAL-A/H-SEAL-B (seeds 100/200/201) verified 16/18
check families PASS but structurally failed: failures scaled ~3× (60–64 failed
episodes/history vs 21 at base), the 7-day quarantine blanketed 1440/1440
pre-cutoff files → dev_train = dev_val = 0 on all three histories, and
robot-02 maintenance saturated (0.30–0.32 ≥ 0.25). Inverse wear/hazard scaling
to compensate is REJECTED (validated base physics must be preserved). The v1
roots stand as rejected evidence (retired, untouched, VOID — never deleted).
Scale-back is achieved by returning to base density, NOT by weakening
quarantine or retuning health: healthy exposure is enlarged by NUMBER OF
HISTORIES instead of density.

## 1. Histories (v2)

Base physics preserved exactly: 300 units, 90-day calendar, 3 robots,
unchanged health/wear/hazard/quarantine (server profile defaults; NO `--units`
override). Eight independent histories, seeds frozen (never
performance-selected; the four sealed were the user-selected preview):

| history | role | seed | server root (uncommitted) |
|---|---|---|---|
| H-DEV-1 | development | 100 | `data/generated/sprint12v2-dev-1/` |
| H-DEV-2 | development | 101 | `data/generated/sprint12v2-dev-2/` |
| H-DEV-3 | development | 102 | `data/generated/sprint12v2-dev-3/` |
| H-DEV-4 | development | 103 | `data/generated/sprint12v2-dev-4/` |
| H-SEAL-1 | sealed eval | 200 | `data/generated/sprint12v2-seal-1/` |
| H-SEAL-2 | sealed eval | 201 | `data/generated/sprint12v2-seal-2/` |
| H-SEAL-3 | sealed eval | 202 | `data/generated/sprint12v2-seal-3/` |
| H-SEAL-4 | sealed eval | 203 | `data/generated/sprint12v2-seal-4/` |

Generation: public entry point WITHOUT scale override —
`python -m synth.cli --chronological --profile server --seed <seed> --output
<root>`. Expected ≈600 files/history (base yield ≈2/unit).

## 2. Development cohorts: pooled across dev histories (v2)

Per-history manifest `dev_train`/`dev_val` (≈44/≈11 at base) are too small
alone. Effective cohorts pool VERIFIED-HEALTHY UNQUARANTINED rows across the
four dev histories AFTER the §3/§4 exclusions:

- FIT = ⋃ effective dev_train (target ≈176; floor: total ≥120; per pooled
  (robot, program) group ≥32 else fallback-documented; group×history identity
  matrix preserved — never merged silently, never cross-robot).
- CAL = ⋃ effective dev_val (target aggregate ≥40; floor: aggregate ≥40 with
  ≥3 dev histories contributing ≥5 files each — no single-history calibration).
- SEL reads CAL metrics only. H-DEV-1..4 test views untouched (diagnostic only
  with explicit record).
- The verifier ESTABLISHES actual aggregate and per-history counts (no assumed
  ≈44/≈11); short of floors = recorded deviation, not silent narrowing.

## 3. Sealed histories: separate uncertainty units (v2)

Each sealed history must meet realistic per-history floors derived from the
base profile (sprint11-server: 145 normals, 219 abnormals, 21 episodes,
18–37/static-family, 54 program-03 abnormals, maintenance ≤ 0.111):

- EVAL normals ≥ 100 (FPR SE ≈ 2.1% at 5%), abnormals ≥ 150 (recall SE ≈ 4%
  at 50%), failure episodes ≥ 15, EVERY static family ≥ 10 files,
  program-03 abnormals ≥ 10, maintenance fraction < 25% per robot,
  maintenance contrast (v1 rule).
- The impossible v1 per-history normal≥400 floor is RETIRED (it assumed 4×
  density that quarantine forbids).
- Sealed histories are NEVER pooled: report per-history metrics plus
  median/IQR across the four; pooling sealed files for selection or as one
  uncertainty unit is a protocol violation.

## 4. Aggregate power ledger (predeclared; files across histories are NOT one unit)

Computed from ACTUAL verified counts (formulas frozen, inputs measured):

- CAL resolution: 1/(n_CAL + 1) on the aggregate CAL n.
- FIT capacity: aggregate n plus pooled-group matrix (min/median group n).
- EVAL precision per sealed history: binomial SE at 5% FPR on actual normals;
  at 50% recall on actual abnormals; event granularity 1/(failed episodes).
- History-level uncertainty: median/IQR across 4 sealed histories (n=4 units).
- The ledger reports power; it does NOT license treating pooled dev files as
  independent uncertainty units for sealed claims.

## 5. Preserved v1 invariants (unchanged)

Reserved whole mechanisms (`wrong_transition`, `cross_channel_inconsistency`;
never in FIT/CAL/SEL); held-out program-03 (never in FIT/CAL/SEL);
maintenance < 25%; quarantine 7 d exact; G-learn/G-static/G-rank/G-risk gates
and 1-day-report-only rule (G-static "both histories" now reads "ALL FOUR
sealed histories"); no performance seed replacement; sealed no-access except
approved evaluation; bulk data server-side uncommitted; deviation/blocker
policy (§12 v1).

## 6. v1 roots disposition

`sprint12-dev/`, `sprint12-seal-a/`, `sprint12-seal-b/` (4× density, VOID:
empty dev cohorts) are RETIRED — left untouched on the server as audit
evidence, never read by downstream work. Their stray FAIL-marked `seal.json`
files are void. Deletion is NOT authorized (audit preservation wins over disk
tidiness; ~41 MB each).
