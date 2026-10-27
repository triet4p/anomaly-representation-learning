# Sprint 24 low-burden, stable baseline support audit (Task 7-M5) — metadata only

- Task / attempt / agent: `S24-7-M5` / `S24-7-M5-A01` / `S24M5GoldA01` (owner-selected Gold via the standing default-answer instruction, quota `unknown`, full route).
- Criterion: `docs/sprint-plans/sprint-24.md:171-173` — audit existing accepted Development metadata for weekly/monthly baseline regions and comparable elevated-burden regions; freeze a bounded, justified definition/grid before candidate-count summaries; report complete candidate/support/rejection accounting with pair identities, exposure/context gaps, overlap versus independent support, and honest feasible/limited/unsupported conclusions with missing-data requirements.
- Status: **implemented and executed on the accepted Development metadata.** Zero forwards, zero model loads, zero waveform reads, zero embedding/score/cache reads. This report is a feasibility/support audit, **not** a detector, **not** a movement measurement, and it defines **no new numerical success gate and no operational threshold**.
- Implementation: `src/representation/sprint24_low_burden_baseline.py`; CLI `experiments/sprint24_task7m5_low_burden_baseline.py`; focused tests `tests/representation/test_sprint24_low_burden_baseline.py`.
- Executed outputs: `artifacts/sprint-24/evidence/s24-7-m5-a01/` (`audit-freeze.json`, `period-candidates.json`/`.csv`, `pair-regions.json`/`.csv`, `cells.json`, `study-manifest-digest.json`, `run.json`).
- Frozen audit digest (`S24-LOW-BURDEN-BASELINE-v1`): `b4e276858add37f00c13419d42c56f5c7d8304a56bc160f4792cb531bf2357a8` (computed from the audit identity, grid, rules and vocabularies only, before any candidate was counted; the run fails closed on any audit/study/manifest digest mismatch).

## 0. Scope and what this audit is not

In scope: whether the accepted Development metadata can support low-burden, temporally stable baseline comparison regions — and comparable elevated-burden contrast regions — at weekly/monthly granularity, for a future collective-movement comparison. Out of scope: embeddings, scores, checkpoints, waveform shards, model loading, forwards, retraining, a new corpus, Confirmation access, Phase C execution, any movement endpoint, and any revision of M4's `INCONCLUSIVE` result.

Five binding boundaries:

1. **Freeze before counting.** The audit identity, grid, burden/stability rules, disposition vocabulary and output specification are frozen in code and written to `audit-freeze.json` before the first candidate is classified. Support statistics never alter the grid; no candidate is ranked, preferred or selected; no threshold was adjusted after counting to obtain support. No M5 candidate count was executed before this freeze.
2. **Labels annotate, never construct.** Candidate *sets* are the frozen label-blind study manifests (`S24-BIZ-WM-v1` period/pair identities built from timestamps and deployment-observable context only). Annotations only *classify* periods/pairs afterwards. A small nonzero abnormal fraction is eligible when every other rule holds; absence of annotation is `UNANNOTATED`, never health, never a control.
3. **Prior knowledge is exploratory, not confirmation.** The M1/M2/M3/M4 results that motivated this audit (zero clean controls, burden ranges, coarse monthly support) are exploratory prior knowledge recorded in the run provenance note. No embedding movement output was inspected to choose audit candidates or thresholds, and no prior result is presented as unseen confirmation.
4. **Overlap is disclosed, not counted as independent.** Adjacent pairs share a period, so overlapping counts reuse the same files. Nonoverlapping subsets are reported as distinct-period support; the report never calls nonoverlap statistical independence.
5. **Thresholds are research assumptions.** Every burden/stability number below is a bounded audit assumption declared before counting, with a descriptive ladder published beside it. None is an owner-approved operational cutoff or success gate, and none may be relaxed retrospectively.

## 1. Frozen audit grid

The grid reuses the accepted M3 study cells exactly: four Development histories × week/month horizons × calendar (primary) / rolling (sensitivity) modes. The daily horizon is deferred by owner prioritization and never enumerated. Reference custody, membership, exposure and comparability are inherited unchanged from `S24-BIZ-WM-v1` (30 files/period, 200 reference files, 0.8 exposure ratio, 0.25 mix-drift reference).

| Research assumption | Primary value | Justification (not an owner threshold) | Descriptive ladder |
|---|---|---|---|
| Low-burden ceiling (abnormal member fraction) | `≤ 0.10` | at most one file in ten is abnormal: faults are a small minority of a normal-trending region | 0.05 / 0.10 / 0.15 |
| Elevated floor | `≥ 0.20` | twice the low ceiling, so baseline and elevated regions are separated by a contrast gap | 0.15 / 0.20 / 0.25 |
| Stability range (max fraction range inside a run) | `≤ 0.05` | half the low ceiling; a larger swing is not demonstrably stable at these counts | — |
| Minimum stable run | `≥ 2` consecutive low periods | a singleton low period is a point, not a trend; reported, never a segment | — |

Period dispositions (closed): `LOW-BASELINE-CANDIDATE`, `BORDERLINE-BURDEN` (inside the 0.10–0.20 gap, neither region), `ELEVATED-CANDIDATE`, `REJECTED-INSUFFICIENT-SUPPORT`, `REJECTED-PARTIAL-PERIOD`, `REJECTED-UNANNOTATED`. Run dispositions: `STABLE-BASELINE-SEGMENT`, `LOW-RUN-UNSTABLE`, `LOW-RUN-SINGLETON`. Pair regions: `BASELINE-STABLE`, `ELEVATED-CONSISTENT`, `TRANSITION` (low↔elevated), `MIXED-BORDERLINE`, exposure-limited variants, `REJECTED-NONCOMPARABLE`, `REJECTED-UNANNOTATED`. Mix flags: `MIX-OK` (drift ≤ 0.25), `MIX-CONFOUNDED`, `MIX-UNKNOWN`. Cell verdicts: `FEASIBLE` (stable segment + mix-ok stable baseline pair + comparable elevated/transition pair), `LIMITED` (partial support), `UNSUPPORTED` (no stable segment).

## 2. Construction lineage and missing dimensions

- Candidate sets: `candidate_period_keys` / `candidate_pair_keys` enumerate label-blind manifest identities in time order; the AST guard `assert_candidate_construction_label_blind` fails closed on any annotation reference in that path. Classification, stability, region and disposition functions read annotations only afterwards.
- Each `BASELINE-STABLE` region requires a comparable matched pair fully inside a stable segment; `in_stable_segment` marks exactly those pairs. Baseline pairs also carry the mix flag; only `MIX-OK` pairs count toward `FEASIBLE`.
- Cross-region (baseline versus elevated) robot/program mix comparability is an explicit **missing dimension**: the persisted study manifests carry mix drift per pair only, not per arbitrary nonadjacent region combination, so `cross_region_mix_status` is `UNKNOWN-MISSING-DIMENSION` everywhere rather than an assumed match. Exposure compatibility of elevated periods against each segment (period exposure-days ratio ≥ 0.8) is reported instead.
- Within-file burden (per-file severity/area) has no accepted annotation field — the manifest carries only the file-level `file_label` — so the audit distinguishes file prevalence only and reports within-file burden as unavailable. Degradation onsets and reset edges are carried as post-hoc/observable covariates, never as burden.

## 3. Inputs, custody and field coverage

The four accepted Development manifests were hash-verified before use; they match the digests bound in the executed M2/M3 runs:

| History | Manifest SHA-256 | Eligible | Analysis span |
|---|---|---:|---|
| `S18I-ITER-0005-DEVELOPMENT-01` | `ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480847e37b48fe1e08` | 2,076 | 224.00 d |
| `S18I-ITER-0005-DEVELOPMENT-02` | `cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310` | 2,327 | 223.91 d |
| `S18I-ITER-0005-DEVELOPMENT-03` | `b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e` | 2,053 | 223.97 d |
| `S18I-ITER-0005-DEVELOPMENT-04` | `3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af` | 2,298 | 223.88 d |

All 328 label-blind periods and 312 pairs (week/month × calendar/rolling) were audited. Observable-field coverage is complete and every row carries a file label, so no `UNANNOTATED` period exists on the accepted metadata — but every annotated period carries abnormal members and almost all carry covered events, reproducing the M2/M3 negative-control background at a graded burden level.

## 4. Results

### 4.1 Period candidacy — every candidate accounted

Over all 328 audited periods (both horizons, both modes): 68 `LOW-BASELINE-CANDIDATE`, 161 `BORDERLINE-BURDEN`, 21 `ELEVATED-CANDIDATE`, 50 `REJECTED-INSUFFICIENT-SUPPORT`, 28 `REJECTED-PARTIAL-PERIOD`, 0 `REJECTED-UNANNOTATED`. The borderline gap is the modal outcome: most periods sit between the two regions, neither low nor elevated under the frozen grid.

Calendar-mode detail by history × horizon (low / borderline / elevated / insufficient-support / partial):

| Cell | Low | Borderline | Elevated | Insufficient | Partial |
|---|---|---:|---:|---:|---:|
| DEV-01 week | 6 | 15 | 3 | 7 | 2 |
| DEV-01 month | 1 | 6 | 0 | 0 | 2 |
| DEV-02 week | 6 | 15 | 4 | 6 | 2 |
| DEV-02 month | 0 | 7 | 0 | 0 | 2 |
| DEV-03 week | 9 | 13 | 1 | 8 | 2 |
| DEV-03 month | 1 | 6 | 0 | 0 | 2 |
| DEV-04 week | 8 | 17 | 1 | 5 | 2 |
| DEV-04 month | 2 | 5 | 0 | 0 | 2 |
**Low nonzero burden is eligible and observed.** All 68 low candidates carry nonzero abnormal fractions (minimum 0.022; maximum at the ceiling 0.10); the zero-fault case never occurs on this corpus. A demonstrative low-nonzero pair of consecutive weeks (DEV-04 `W2024-12-16`/`W2024-12-23`, fractions 0.0526/0.0488, range 0.0039) is eligible when the other rules hold and counts are independent of embeddings.

### 4.2 Stability — weekly segments exist, monthly support does not

Calendar mode:

| Cell | Runs | Segment detail |
|---|---|---|
| DEV-01 week | 1 segment (2 periods, range 0.0218) + 4 singletons | `W2024-09-23`/`W2024-09-30` (0.0968/0.0750) |
| DEV-02 week | 1 segment (2 periods, range 0.0189) + 4 singletons | `W2024-11-11`/`W2024-11-18` (0.1000/0.0811) |
| DEV-03 week | 1 segment (3 periods, range 0.0264) + 6 singletons | `W2024-07-22`/`-29`/`-08-05` (0.0667/0.0882/0.0930) |
| DEV-04 week | 2 segments (ranges 0.0112, 0.0039) + 4 singletons | `W2024-07-15`/`-22`, `W2024-12-16`/`-23` |
| DEV-0x month | no segment in any history | only singletons or empty runs |

The rolling sensitivity reproduces the same pattern (one segment per history at week, none at month), so the horizon asymmetry is not a boundary-phase artefact.

### 4.3 Pair regions — baseline pairs are thin and mix-confounded or exposure-limited

Over all 312 pairs: 171 `MIXED-BORDERLINE`, 98 `REJECTED-NONCOMPARABLE`, 12 `REJECTED-UNANNOTATED` (comparison-state accounting, attribution-class convention), 9 `TRANSITION`, 8 `BASELINE-EXPOSURE-LIMITED`, 7 `TRANSITION-EXPOSURE-LIMITED`, 4 `BASELINE-STABLE`, 2 `ELEVATED-CONSISTENT`, 1 `ELEVATED-EXPOSURE-LIMITED`.

Calendar week by history: DEV-01 holds 0 in-segment stable baseline pairs (its in-segment pair is exposure-mismatched, ratio 0.775) and 1 `ELEVATED-CONSISTENT` contrast pair (`W2024-09-09`, fraction 0.267, nonoverlapping 1); DEV-02 holds 1 stable pair but mix-confounded (drift 0.742 > 0.25) and no contrast pair; DEV-03 holds 0 stable pairs (both in-segment pairs exposure-mismatched) and 1 `TRANSITION` contrast pair; DEV-04 holds 1 stable pair, mix-confounded (drift 0.457), plus 1 `TRANSITION` contrast pair (`W2024-12-16`, matched, mix-confounded 0.583, exposure 0.905). Every stable weekly baseline pair is therefore either exposure-limited (capped-`n` sensitivity only) or mix-confounded — none is `MIX-OK`. Calendar month holds no `BASELINE-STABLE`, `ELEVATED-CONSISTENT` or `TRANSITION` pair in any history; monthly comparable pairs are `MIXED-BORDERLINE` throughout (fractions 0.106–0.156, 33–44 events each).

Overlap accounting: weekly in-segment baseline pairs carry 1 overlapping / 1 nonoverlapping pair on DEV-02 and DEV-04 (the single pair is trivially nonoverlapping); contrast pairs carry 1/1 everywhere one exists. Nonoverlap here means distinct periods, never independence.

### 4.4 Feasibility findings

1. **Weekly baseline support is `LIMITED`, not `FEASIBLE`.** All four histories hold at least one stable low-burden segment under the frozen grid, and low-nonzero burden is demonstrably eligible — but no history combines a mix-ok stable baseline pair with a comparable elevated/transition pair. The stable pairs that exist are mix-confounded or exposure-limited; DEV-02 has no contrast pair at all.
2. **Monthly baseline support is `UNSUPPORTED`.** No stable segment exists in any history at month granularity: the only low months are isolated singletons, and every monthly comparable pair is borderline with 33–44 covered events. Relaxing the two-period stability minimum or the ceiling to admit singletons would be a threshold change after counting, and is not made.
3. **The elevated pole is the binding constraint.** Only 21 elevated periods exist across the whole grid and monthly elevated support is zero; the audit's contrast gap (0.10–0.20) absorbed the modal period. An elevated-contrast comparison needs either more histories or a prospectively re-justified grid — not a silent lowering.
4. **Mix confounding dominates the surviving baseline pairs.** All three calendar-week in-segment `BASELINE-STABLE` pairs carry drift 0.457–0.742 against the 0.25 reference; the rolling sensitivity is identical (0 mix-ok pairs). Any future movement comparison on these regions must carry the mix residual explicitly.
5. **This is a feasibility answer, not a specificity answer.** No calibrated claim, no control pair, no movement measurement and no deployment statement follow from these counts. Zero annotated-clean periods persist on the accepted span; low-burden here means few faults, never verified health.

## 5. Missing-data requirements (targeted, not mandatory positive support)

- **More histories** (or denser weekly exposure): the `FEASIBLE` bar needs three of four histories with a mix-ok stable baseline pair plus a contrast pair; currently zero histories clear it. The gap is mix-ok exposure-matched baseline pairs and monthly elevated periods, counted exactly above.
- **Cross-region mix comparability**: baseline-versus-elevated robot/program mix drift for nonadjacent regions is not persisted in the study manifests; a future design must record it prospectively rather than assume it.
- **Within-file burden annotation**: no accepted field distinguishes a barely-abnormal file from a severely-abnormal one; only file prevalence is auditable.
- **A verified healthy/quiet source** if any specificity claim is ever wanted: the graded burden association is the only available contrast on this corpus. M4 remains `INCONCLUSIVE`.

## 6. Limitations

- Metadata support only. Every count describes file/event/context availability; none measures embedding movement or detection quality.
- Thresholds are frozen research assumptions (0.10 / 0.20 / 0.05 / 2-period minimum) with published ladders; they are not owner-approved operational cutoffs and cannot be re-tuned after counting.
- Pair overlap is structural (adjacent pairs share a period); nonoverlapping counts are distinct-period support, never independence evidence.
- Four histories at `n = 4`; all horizon verdicts are descriptive.
- The daily horizon is deferred; no daily claim follows.
- Prior corpus knowledge motivated the grid; no embedding output selected it. Every candidate/support count above was computed after the audit freeze recorded in `audit-freeze.json`.

## 7. Reproduction

```bash
python experiments/sprint24_task7m5_low_burden_baseline.py \
  --data-root <dir containing DEVELOPMENT/<history>/manifest.json> \
  --edges experiments/sprint24_task7m1_dev-edges.json \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-01=ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480847e37b48fe1e08 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-02=cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-03=b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-04=3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af \
  --expected-audit-sha256 b4e276858add37f00c13419d42c56f5c7d8304a56bc160f4792cb531bf2357a8 \
  --expected-study-sha256 6f96c2750c15552bcdf25678a51ed7b24a1f7aecdb5914bf5985d3012d1daefd \
  --out artifacts/sprint-24/evidence/s24-7-m5-a01
```

The CLI fails closed on an audit, study or manifest digest mismatch, writes `audit-freeze.json` before analysis, and records output digests in `run.json`. No disposition is revised after outcomes. Focused tests: `python -m pytest tests/representation/test_sprint24_low_burden_baseline.py -q`.

## 8. What is not authorized

This report does not authorize a new movement study, a new window or reference lock, a threshold change, retraining, a new corpus, Confirmation access, Phase C execution or any protocol change. Any follow-on comparison study needs a separately approved prospective amendment and execution release; if no grouping yields feasible baseline-plus-contrast support, Phase C stays held with this reason recorded.
