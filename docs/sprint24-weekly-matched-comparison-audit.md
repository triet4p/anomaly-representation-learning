# Sprint 24 weekly matched-comparison feasibility audit (Task 7-M6) — metadata only

- Task / attempt / agent: `S24-7-M6` / `S24-7-M6-A01` / `S24M6GoldA01` (owner-selected Gold via the standing default-answer instruction, quota `unknown`, full route).
- Criterion: `docs/sprint-plans/sprint-24.md:176-178` — audit existing accepted Development metadata for weekly matched baseline-versus-elevated and baseline-versus-baseline comparisons; freeze a bounded, justified matching design before feasibility counts; report complete candidate/support/rejection accounting with exact file/region identities, common-support/context/exposure balance before/after, attrition and burden contrast; publish an honest feasible/limited/unsupported conclusion with targeted missing-data requirements.
- Status: **implemented and executed on the accepted Development metadata.** Zero forwards, zero model loads, zero waveform reads, zero embedding/score/cache reads. This report is a feasibility/support audit, **not** a detector, **not** a movement measurement, and it defines **no new numerical success gate and no operational threshold**.
- Implementation: `src/representation/sprint24_weekly_matched_comparison.py`; CLI `experiments/sprint24_task7m6_weekly_matched_comparison.py`; focused tests `tests/representation/test_sprint24_weekly_matched_comparison.py`.
- Executed outputs: `artifacts/sprint-24/evidence/s24-7-m6-a01/` (`matching-freeze.json`, `comparisons.json`/`.csv`, `members.json`, `exclusions.json`/`.csv`, `controls.json`, `summary.json`, `study-manifest-digest.json`, `run.json`).
- Frozen matching digest (`S24-WEEKLY-MATCHED-v1`): `1eff95d18fb521b8b9722107b487ced620567357b10154f069a3fd4f412d8ac3` (computed from the matching identity, grid, rules and vocabularies only, before any feasibility count; the run fails closed on any matching/baseline-audit/study/manifest digest mismatch).

## 0. Scope and what this audit is not

In scope: whether the accepted Development metadata can support deterministic, label-blind weekly matched comparisons between low-burden stable baseline regions and elevated-burden contrast regions (plus baseline-versus-baseline controls), with measured attrition and retained burden contrast. Out of scope: embeddings, scores, checkpoints, waveform shards, model loading, forwards, retraining, a new corpus, Confirmation access, Phase C execution, any movement endpoint, and any revision of M4's `INCONCLUSIVE` or M5's `LIMITED`/`UNSUPPORTED` results.

Six binding boundaries:

1. **Freeze before counting.** The matching identity, grid, stratum definition, balancing rule, disposition vocabulary and output specification are frozen in code and written to `matching-freeze.json` before the first comparison is balanced. Support statistics never alter the design; no region is ranked, preferred or selected by outcome; no threshold was adjusted after counting to obtain support. Exploratory real-metadata probes (see §7) preceded the freeze and are disclosed as exploratory, never as prospectively frozen confirmation.
2. **Labels annotate regions, never select retained files.** Candidate *sets* are the frozen label-blind M5 period identities (stable-segment and exposure-compatible elevated periods built from timestamps and deployment-observable context only). Post-hoc burden quantifies retained contrast afterwards. Within-stratum retention is hash-ordered over observable `(robot_id, program_id)` cells under the inherited null seed; no label enters the ordering or the cap.
3. **Post-balancing contrast is measured, never assumed.** Label-blind selection can shift file fractions: every comparison reports burden before *and* after balancing, and the feasibility verdict uses only the retained (after) contrast.
4. **Prior knowledge is exploratory, not confirmation.** The M1/M2/M3/M4/M5 results that motivated this audit (zero clean controls, burden ranges, the M5 weekly `LIMITED` verdict) are exploratory prior knowledge recorded in the run provenance note. No embedding movement output was inspected to choose regions, and no prior result is presented as unseen confirmation.
5. **Disjoint support is disclosed, not independence.** `disjoint` means two regions share no file and no period; the report never calls nonoverlap statistical independence. Adjacent chronology (pairs sharing a period) is distinguished from nonadjacent research pairs (baseline segment versus a distant elevated period sharing neither files nor periods).
6. **Thresholds are research assumptions.** Every burden/support number below is a bounded audit assumption declared before counting, frozen under `S24-WEEKLY-MATCHED-v1`. None is an owner-approved operational cutoff or success gate, and none may be relaxed retrospectively. No `FEASIBLE` outcome is required.

## 1. Frozen matching design

The design reuses the M5 definitions unchanged (low-burden ceiling 0.10, elevated floor 0.20, stability range 0.05 over runs of at least two consecutive low periods; inherited floors 30/200/0.8 and the 0.25 mix reference) and adds one deterministic balancing layer:

- Horizon/mode: weekly calendar only (primary). Month stays deferred after the M5 `UNSUPPORTED` verdict (no stable monthly segment exists); day was already deferred by owner prioritization. Rolling is not enumerated: the M6 question is adjacent-week common support, and the M5 rolling sensitivity already reproduced the weekly pattern.
- Stratum: the observable `(robot_id, program_id)` file cell. Stratum membership is deployment-observable; no label is read to include or exclude a file.
- Baseline region per history: the union of that history's exact M5 primary-calendar stable-segment member files. Rule for multi-segment histories (frozen): the latest segment is the PRIMARY baseline; earlier segments are disjoint reserve controls. Histories without a stable segment contribute no baseline region and no comparison.
- Contrast region per history: the union of that history's exact M5 exposure-compatible elevated periods (`ELEVATED-CANDIDATE` periods whose period-exposure-days ratio against at least one segment member meets the inherited 0.8 floor). Elevated periods are post-hoc region *types*; the elevated pole is the binding constraint (21 elevated periods grid-wide, zero at month).
- Common support: strata present in both arms of the same history. Files outside common support are excluded label-blind.
- Balancing rule (`capped-n` per stratum): within each common stratum, keep the first `min(n_baseline, n_contrast)` members of each arm ordered by `sha256(f'{null_seed}:{history_id}:{region_identity}:{file_id}')` under the inherited null seed (2407), replicating the M3 capped-`n` ordering idea per stratum instead of per period. Balanced arms have equal per-stratum and equal total sizes; mix drift is recomputed over the same frozen stratum universe before and after.
- Controls: a reserve baseline-versus-baseline comparison exists only where a history holds two disjoint stable segments. Every other control candidate shares a period with (or is) the baseline region and is rejected as non-independent support. Singleton low periods are points, never segments, and are rejected as controls.

Feasibility rules (frozen research assumptions, not owner-approved gates; no positive result required):

- `FEASIBLE`: at least one history holds a balanced baseline-versus-elevated comparison with common support on at least two strata, at least 10 retained files per arm, retained burden contrast (elevated minus baseline) of at least 0.10, and an available disjoint baseline control.
- `LIMITED`: at least one history holds a balanced comparison meeting the support minima (two strata, ten files per arm) but no history meets the full `FEASIBLE` bar (contrast lost under balancing or no disjoint control).
- `UNSUPPORTED`: no history meets even the support minima.

## 2. Construction lineage and missing dimensions

- Candidate *sets* are the label-blind study manifest rows; the region assembler reuses exact M5 period identities and observable member lists. The AST guard `assert_matching_construction_label_blind` fails closed on any annotation reference in the selection/matching functions.
- Cross-region (baseline-versus-elevated) robot/program mix drift between nonadjacent regions is **computed, not missing**: per-file `(robot_id, program_id)` cells are projected from the accepted manifests and the total-variation drift is reported before and after balancing over the frozen stratum universe. (M5 recorded this as a missing dimension because the persisted study manifests carry mix drift per adjacent pair only; M6 resolves it from the same accepted metadata without new collection.)
- Within-file burden (per-file severity/area) has no accepted annotation field — the manifest carries only the file-level `file_label` — so the audit classifies file prevalence only and reports within-file burden as unavailable.
- Exposure balance uses period-exposure-days sums per arm; the region-level ratio is descriptive (arms span different numbers of weeks), so exposure *compatibility* (the inherited 0.8 floor at period granularity, already enforced by the M5 elevated selection) is the binding constraint, not the raw region ratio.
- Degradation onsets and reset edges are carried as post-hoc/observable covariates, never as burden.

## 3. Inputs, custody and field coverage

The four accepted Development manifests were hash-verified before use; they match the digests bound in the executed M2/M3/M5 runs:

| History | Manifest SHA-256 | Eligible | Analysis span |
|---|---|---:|---|
| `S18I-ITER-0005-DEVELOPMENT-01` | `ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480747e37b48fe1e08` | 2,076 | 224.00 d |
| `S18I-ITER-0005-DEVELOPMENT-02` | `cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310` | 2,327 | 223.91 d |
| `S18I-ITER-0005-DEVELOPMENT-03` | `b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e` | 2,053 | 223.97 d |
| `S18I-ITER-0005-DEVELOPMENT-04` | `3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af` | 2,298 | 223.88 d |

Observable-field coverage is complete: every audited file carries `robot_id`/`program_id`, so every arm member maps to a stratum and no file is `UNSTRATIFIED`. Every annotated period carries abnormal members and almost all carry covered events, reproducing the M2/M3/M5 negative-control background at a graded burden level.

## 4. Results

### 4.1 Primary comparisons — every history accounted

Four PRIMARY baseline-versus-elevated comparisons were balanced (one per history; DEV-04 uses its latest segment as PRIMARY per the frozen rule):

| History | Baseline periods | Contrast periods | Common strata | Before (n / burden / gap) | After (n/arm / burden / gap) | Attrition base/cont | Mix drift before→after | Verdict |
|---|---|---:|---|---|---|---|---|
| DEV-01 | W2024-09-23, W2024-09-30 | W2024-09-02, W2024-09-09, W2024-10-21 | 7 | 71 / 0.0845 vs 127 / 0.2598 / 0.1753 | 71 / 0.0845 vs 0.3239 / 0.2394 | 0.000 / 0.441 | 0.136→0.000 | LIMITED (no disjoint control) |
| DEV-02 | W2024-11-11, W2024-11-18 | W2024-06-10, W2024-09-23, W2024-10-21 | 7 | 77 / 0.0909 vs 123 / 0.2195 / 0.1286 | 75 / 0.0800 vs 0.2000 / 0.1200 | 0.026 / 0.390 | 0.179→0.000 | LIMITED (no disjoint control) |
| DEV-03 | W2024-07-22, -29, -08-05 | W2024-11-25 | 6 | 122 / 0.0820 vs 34 / 0.2059 / 0.1239 | 34 / 0.1471 vs 0.2059 / 0.0588 | 0.721 / 0.000 | 0.289→0.000 | LIMITED (contrast lost under balancing) |
| DEV-04 | W2024-12-16, W2024-12-23 | W2024-12-09 | 5 | 79 / 0.0506 vs 42 / 0.2143 / 0.1637 | 37 / 0.0270 vs 0.2162 / 0.1892 | 0.532 / 0.119 | 0.360→0.000 | FEASIBLE (with reserve control) |

Deterministic retention identities for every comparison are persisted in `members.json`; every exclusion is audit-trailed per stratum in `exclusions.json`/`.csv` with `in_common_support` and a closed reason.

### 4.2 Sampling sensitivity — one elevated period at a time

Eight SINGLE sensitivity comparisons (every compatible elevated period balanced alone against its history's primary baseline) hold the same verdicts as their PRIMARY unions: all DEV-01/DEV-02 singles are LIMITED with retained gaps 0.094–0.191; DEV-03's single is LIMITED with the same collapsed gap (0.059). No single-period choice rescues or destroys a verdict beyond what the union already shows: the sampling sensitivity is stable, not a selection lever.

### 4.3 Baseline-versus-baseline controls

Only DEV-04 holds a disjoint reserve control (July segment `W2024-07-15`/`W2024-07-22` versus December primary `W2024-12-16`/`W2024-12-23`): 5 common strata, 59 retained files per arm, retained burden 0.1017 (reserve) versus 0.0678 (primary) for a gap of 0.0339 — a near-zero control gap as a baseline-versus-baseline pairing should show. Its verdict is LIMITED (support minima met, burden-contrast minimum not met, which is the expected control behavior, not a failure). DEV-01/DEV-02/DEV-03 have `UNAVAILABLE-SINGLE-SEGMENT` status: each holds exactly one stable segment, so any control candidate shares periods with the baseline region. The 4–6 singleton low periods per history are rejected as controls (points, never demonstrably stable trends).

### 4.4 Rejections and non-support

- One elevated period is rejected grid-wide: DEV-02 `W2024-10-07` (fraction 0.20) is exposure-incompatible with the DEV-02 segment under the inherited 0.8 floor and never enters a contrast region.
- Zero files are unstratified; zero comparisons lack common support (minimum 5 common strata); zero histories lack a supported balanced comparison.
- Chronology: all 13 comparisons are `NONADJACENT-RESEARCH-PAIR` (baseline segment versus a distant elevated period, or two distant segments, sharing neither files nor periods) except the structural `NONADJACENT-SHARED-SUPPORT` cases where a SINGLE sensitivity shares the baseline region with its PRIMARY union. No adjacent-week pair surviving exposure matching exists inside any stable segment (the in-segment pairs are exposure-mismatched per M5), so no `ADJACENT-CHRONOLOGY` matched comparison is available.

### 4.5 Feasibility findings

1. **Weekly matched support is `LIMITED`, not `FEASIBLE`.** One history (DEV-04) meets the full per-comparison bar, but the horizon rule requires three of four: only one history holds a disjoint baseline control, and DEV-03's retained contrast collapses under label-blind balancing (0.124 → 0.059).
2. **Balancing removes mix confounding exactly.** Mix drift falls to 0.000 on all 13 comparisons (balanced arms share identical stratum proportions by construction). Context balance is achieved; the residual limits are contrast retention and control availability, not mix.
3. **Label-blind selection shifts burden and must be measured.** DEV-03's baseline burden nearly doubles under balancing (0.082 → 0.147) because the retained strata carry more abnormal files — the mechanism the audit's measure-after-balancing rule exists to catch. Contrast is preserved on DEV-01/DEV-02/DEV-04 but lost on DEV-03; no assumption substitutes for the measurement.
4. **The control inventory is the binding constraint.** Three of four histories hold exactly one stable segment, so no disjoint baseline control exists for them. A future comparison can use at most one history's control (DEV-04) without new data.
5. **This is a feasibility answer, not a specificity answer.** No calibrated claim, no control-pair significance, no movement measurement and no deployment statement follow from these counts. Zero annotated-clean periods persist on the accepted span; low-burden here means few faults, never verified health. M4 remains `INCONCLUSIVE` and M5's weekly `LIMITED` / monthly `UNSUPPORTED` stand unchanged.

## 5. Missing-data requirements (targeted, not mandatory positive support)

- **More histories with two disjoint stable segments** (or denser weekly exposure): the horizon bar needs three of four histories with a mix-balanced, contrast-preserving comparison *and* a disjoint control; currently one history clears it. The gap is control availability, not balancing machinery.
- **Elevated periods that survive balancing with contrast intact:** DEV-03 shows that common-support restriction can halve the gap; more elevated periods (or exposure-compatible ones nearer the segment) would let the contrast survive restriction without re-tuning any rule.
- **A verified healthy/quiet source** if any specificity claim is ever wanted: the graded burden association is the only available contrast on this corpus.
- **Within-file burden annotation**: no accepted field distinguishes a barely-abnormal file from a severely-abnormal one; only file prevalence is auditable.

## 6. Limitations

- Metadata support only. Every count describes file/stratum availability and measured burden; none measures embedding movement or detection quality.
- Thresholds are frozen research assumptions (0.10 / 0.20 / 0.05, two-period minimum, two strata / ten files per arm / 0.10 retained gap, three-of-four horizon bar); they are not owner-approved operational cutoffs and cannot be re-tuned after counting.
- Disjoint rows mean no shared file and no shared period, never statistical independence. Adjacent SINGLE-versus-PRIMARY comparisons share their baseline region by construction.
- Four histories at `n = 4`; all horizon verdicts are descriptive.
- The daily horizon is deferred and the monthly horizon stays `UNSUPPORTED` per M5; no claim follows for either.
- Exploratory probes (correct constants, wrong expected persistence) motivated the design but every feasibility count above was computed after the matching freeze recorded in `matching-freeze.json`. The module/CLI digest `1eff95d18fb52…` is unchanged since the freeze probe.

## 7. Exploratory-probe disclosure (chronology gap)

No pre-probe design freeze exists for M6: real-metadata balancing probes were run during design to understand stratum structure (common cells, capped-`n` behavior, DEV-04 control overlap, per-period sensitivity) *before* the matching module was written. Those probes are exploratory, not prospectively frozen confirmation — they selected no retained file, tuned no threshold, and their constants (null seed 2407, inherited floors, M5 burden/stability values) are carried unchanged into the frozen design rather than fitted from probe outcomes. The frozen `matching_sha256` (`1eff95d1…`) was computed from the written module *before* the final CLI counts; the CLI fails closed on any drift of the matching, baseline-audit, study or manifest digests between freeze and analysis. No threshold changed after any outcome was inspected.

## 8. Reproduction

```bash
python experiments/sprint24_task7m6_weekly_matched_comparison.py \
  --data-root <dir containing DEVELOPMENT/<history>/manifest.json> \
  --edges experiments/sprint24_task7m1_dev-edges.json \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-01=ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480747e37b48fe1e08 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-02=cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-03=b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-04=3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af \
  --expected-matching-sha256 1eff95d18fb521b8b9722107b487ced620567357b10154f069a3fd4f412d8ac3 \
  --expected-baseline-audit-sha256 b4e276858add37f00c13419d42c56f5c7d8304a56bc160f4792cb531bf2357a8 \
  --expected-study-sha256 6f96c2750c15552bcdf25678a51ed7b24a1f7aecdb5914bf5985d3012d1daefd \
  --out artifacts/sprint-24/evidence/s24-7-m6-a01
```

The CLI fails closed on a matching, baseline-audit, study or manifest digest mismatch, writes `matching-freeze.json` before analysis, and records output digests in `run.json`. No disposition is revised after outcomes. Focused tests: `python -m pytest tests/representation/test_sprint24_weekly_matched_comparison.py -q`.

## 9. What is not authorized

This report does not authorize a new movement study, a new window or reference lock, a threshold change, retraining, a new corpus, Confirmation access, Phase C execution or any protocol change. Any follow-on comparison study needs a separately approved prospective amendment and execution release; with weekly support `LIMITED` and a single-history control inventory, Phase C stays held with this reason recorded.
