# Sprint 24 context/time control-support audit (Task 7-M1) — metadata only

- Task / attempt / agent: `S24-7-M1` / `S24-7-M1-A01` / `S24M1GoldA01` (owner-selected Gold, quota `unknown`, full route).
- Criterion: `docs/sprint-plans/sprint-24.md:148-150` — on the four immutable Development histories, freeze a bounded candidate grid of observable context groupings and temporal windows **before** post-hoc label summaries, then audit window density, causal reference depth, healthy-control density, ordinary-transition/reset coverage, missingness and overlap/effective independent support; publish every candidate and zero-support case; no embeddings/scores/model loading/forwards and no calibrated-specificity claim.
- Inputs (unchanged): accepted decision `docs/sprint24-geometry-time-decision-g3.md` §4, locked protocol `docs/sprint24-geometry-time-diagnostic-v1.md`, G2 worker evidence and derived `NO-CALIBRATION` report, G2/G3 review and checkpoint records.
- Status: **implemented and executed on the accepted Development manifests.** Zero forwards, zero model loads, zero waveform reads. This report is a feasibility/support audit, **not** a detector, **not** calibrated specificity, and it defines **no new numerical success gate**.
- Implementation: `src/representation/sprint24_context_audit.py`; CLI `experiments/sprint24_task7m1_context_audit.py`; frozen decile-edge input `experiments/sprint24_task7m1_dev-edges.json`; focused tests `tests/representation/test_sprint24_context_audit.py`.
- Executed outputs: `artifacts/sprint-24/evidence/s24-7-m1-a01/` (`grid-freeze.json`, `support-table.json`, `support-table.csv`, `summary.json`, `field-audit.json`, `label-audit.json`, `run.json`).
- Frozen grid digest (`S24-CTX-TIME-AUDIT-v1`): `8c6d70ce641c105922a7aa416d4eeeeda9a8093ecdf1189630c9b7533d0f1dc3` (computed from the grid, support policy and output spec only; written to `grid-freeze.json` before any support statistic).

## 0. Scope and what this audit is not

In scope: whether the accepted Development metadata can support temporal windows, causal pre-evaluation references and post-hoc healthy/ordinary-transition controls for a future geometry design. Out of scope: embeddings, scores, checkpoints, waveform shards, model loading, forwards, retraining, a new corpus, Confirmation access, Phase C execution, any protocol change, and any outcome-tuned window or threshold choice.

Three binding boundaries:

1. **Grid before summaries.** The candidate grid, the support-minimum policy and the complete output specification are frozen in code and written to `grid-freeze.json` before the first window is built. Support statistics never alter the grid, and no candidate is ranked, preferred or selected.
2. **Labels after membership.** Construction reads only `file_id`, `start_time`, `end_time`, `robot_id`, `program_id`, `last_reset_time`, `is_quarantined`, `is_censored`, `n_valid_patches` plus the manifest `maintenance_windows` table and the frozen inherited eligibility predicate. `project_observable` strips every annotation field before construction can see a row, `project_labels` is a separate post-hoc projection, and `assert_construction_label_blind` fails closed if construction code references an annotation, failure or episode name. Labels annotate support; they never choose files inside a window, never define a window and never enter a score.
3. **Dependent support is disclosed, not counted as independent.** Overlapping windows are reported with reuse factors and effective independent windows. The three accepted encoder seeds (`181801/181802/181803`) are metadata-independent: this audit performs **zero** seed computations and they add **no** additional data.

## 1. Frozen candidate grid

Groupings are the four deployment-observable context columns already allowed by protocol §3.4. Widths/strides are bounded around the executed design (fleet-wide 14-day / 7-day) and around the narrower/personal-context scales that evidence suggested might thin support. Regime/type sequences and `operating_context_json` ambient fields are excluded (protocol §3.5 — they are not v1 window/reference inputs).

| Grouping | Membership rule | Contexts per history |
|---|---|---|
| `fleet` | all eligible rows | 1 |
| `robot` | `robot_id` only | 9 |
| `program` | `program_id` only | 8 |
| `robot_x_program` | `robot_id` and `program_id` cell | 9 observed cells |

| Width (days) | Admitted strides (days, `stride <= width`) | Cells |
|---|---|---|
| 3 | 1 | 1 |
| 7 | 1, 7 | 2 |
| 14 | 1, 7, 14 | 3 |
| 28 | 1, 7, 14, 28 | 4 |

Four groupings × ten temporal cells = **40 frozen candidates**; on four histories with 27 context instances each the grid yields **1,080 candidate-instance rows** (`4 × (1×10 + 9×10 + 8×10 + 9×10)`). Window anchors are history-local trailing windows `(T − width, T]` with `T = c1 + k·stride` while `T ≤ tmax`; `c1` is the frozen Task 2/Task 7 40 % decile edge supplied by `experiments/sprint24_task7m1_dev-edges.json`.

## 2. Frozen support-minimum policy

Both floors are **inherited unchanged** from the locked protocol; this audit classifies support and does not tune them. A sensitivity ladder is published beside each floor so a zero at the primary floor cannot be confused with a zero at a smaller count.

| Quantity | Value | Basis |
|---|---|---|
| Minimum files per window | 30 | protocol §4 inherited floor |
| Minimum fixed causal reference pool (pre-`c1`, per context) | 200 | protocol §5 inherited floor |
| Descriptive support ladder | 10 / 30 / 100 | published counts, not gates |
| Descriptive reference ladder | 30 / 100 / 200 | published counts, not gates |
| Descriptive "fresh reset" age | 7 days | descriptive reference value only |
| Descriptive "material mix drift" | TV ≥ 0.25 | descriptive reference value only |

Closed disposition vocabulary (support classification only): `NO-WINDOW-SUPPORT` (no window at the inherited floor), `WINDOW-SUPPORT-NO-REFERENCE-FLOOR` (windows supported, fixed causal reference below 200), `WINDOW-SUPPORT-REFERENCE-FLOOR` (both met). Post-hoc control presence is a separate boolean, so a zero-control outcome cannot masquerade as an unsupported design.

## 3. Construction and eligibility lineage

- Analysis span `(c1, tmax]` on each history's own clock, never concatenated across histories; `tmax` is the history's maximum file `end_time`.
- Eligible membership: frozen inherited predicate `is_eligible_for_training` (not quarantined, not censored, no reserve robot/program identity, no manifest maintenance-window overlap). The audit re-derives the *reason* for each exclusion and asserts it against the inherited predicate on every row of all four histories.
- Maintenance windows and `last_reset_time` remain observable inputs: reset edges are maintenance-window **end** times (recommissioning) for the context's robots inside the evaluation span.
- Causal reference: the fixed pre-`c1` eligible pool per context (inherited protocol §5) **and** a per-window causal prior count (`end_time ≤ window_start`). Both are reported; neither uses labels.
- Windows still reach before `c1` when `width > stride` (the anchor rule is inherited); **3,527 windows** across the grid contain pre-`c1` members. This inherited reference-window overlap is disclosed per row as `n_windows_with_pre_c1_members` and is never re-labelled as independent support.

## 4. Inputs, custody and field coverage

All four accepted Development manifests were copied as metadata only and hash-verified before use; they match the digests bound in the executed G2 runner:

| History | Manifest SHA-256 | Rows | Eligible | Eval span (days) |
|---|---|---|---:|---:|---:|
| `S18I-ITER-0005-DEVELOPMENT-01` | `ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480747e37b48fe1e08` | 6,513 | 2,076 | 224.00 |
| `S18I-ITER-0005-DEVELOPMENT-02` | `cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310` | 6,460 | 2,327 | 223.91 |
| `S18I-ITER-0005-DEVELOPMENT-03` | `b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e` | 6,483 | 2,053 | 223.97 |
| `S18I-ITER-0005-DEVELOPMENT-04` | `3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af` | 6,505 | 2,298 | 223.88 |

Observable-field coverage is complete: every row carries all nine observable fields, `last_reset_time` is never missing, and no row has zero valid patches. Whole-manifest inherited exclusions (the lineage this audit preserves, `field-audit.json`): quarantined 3,671 / 3,284 / 3,608 / 3,307; censored 154 / 199 / 215 / 156; reserve identity 611 / 649 / 607 / 743; maintenance overlap 1 / 1 / 0 / 1. Support-table rows report the same reasons restricted to the evaluation span and to each context (e.g. fleet DEV-01: 2,183 quarantined, 154 censored, 337 reserve identity, 1 maintenance overlap).

Post-hoc annotation denominators (`label-audit.json`, evaluation-only): failure events 246 / 229 / 254 / 221 (in-span 147 / 139 / 154 / 131); degradation episodes 213 / 208 / 214 / 208 (in-span 128 / 127 / 128 / 124); file labels are 77.1–78.6 % normal.

## 5. Results

### 5.1 The grouping axis is degenerate on accepted Development metadata

Eligible Development data contains only **7 eligible robot × program pairs** (9 pairs in the raw manifest). Robot and program are a near-bijection: `robot-01 ↔ program-01` (shared with reserve `robot-08`), `robot-02 ↔ program-02`, `robot-03 ↔ program-04`, `robot-04 ↔ program-05`, `robot-05 ↔ program-06`, `robot-06 ↔ program-07`, `robot-07 ↔ program-08`; reserve `robot-09 ↔ program-03` is excluded by identity. After the reserve exclusions, **program and `robot_x_program` contexts reproduce the robot contexts exactly**, so their support tables are identical by construction (all three grouped rows in `summary.json` report the same 1,857 supported windows and the same per-cell counts).

Consequence: the 40-cell grid contains **two genuinely distinct designs** (fleet and per-robot), with `program` and `robot_x_program` as replicate encodings. This is a property of the corpus, not of the audit, and it must not be read as three independent confirmations.

### 5.2 Causal reference depth

| Grouping | Context instances | Populated (≥1 eligible row) | Prefix min / median / max | ≥30 | ≥100 | ≥200 (inherited floor) |
|---|---:|---:|---|---:|---:|---:|
| `fleet` | 4 | 4 | 850 / 927 / 974 | 4 | 4 | **4** |
| `robot` | 36 | 28 | 0 / 127 / 168 | 28 | 27 | **0** |
| `program` | 32 | 28 | 0 / 130 / 168 | 28 | 27 | **0** |
| `robot_x_program` | 36 | 28 | 0 / 127 / 168 | 28 | 27 | **0** |

No per-robot, per-program or per-cell context can supply the inherited 200-file fixed reference pool; the deepest grouped context reaches 168 files. On the per-window causal prior depth, 100 % of fleet windows have ≥200 prior observations, while only 44.9 % of per-robot windows do (`16,514 / 36,792`; the `program` and `robot_x_program` encodings carry the same memberships) and `28,360 / 36,792` cross ≥100. Reading: a *past-only* reference exists everywhere, but the **fixed pre-`c1` reference convention** is fleet-only under these floors.

### 5.3 Window density and count support

Fleet supports every temporal cell on every history (`40 / 40` candidate instances). Grouped contexts do not:

| Grouped cell | Instances | Instances with ≥1 supported window | Supported windows (total) | Max files in a window |
|---|---:|---:|---:|---:|
| `3 d / 1 d` | 36 | 0 | 0 | 13 |
| `7 d / 1 d` | 36 | 0 | 0 | 21 |
| `7 d / 7 d` | 36 | 0 | 0 | 20 |
| `14 d / 1 d` | 36 | 5 | 19 | 34 |
| `14 d / 7 d` | 36 | 3 | 3 | 33 |
| `14 d / 14 d` | 36 | 1 | 1 | 31 |
| `28 d / 1 d` | 36 | 28 | 1,487 | 51 |
| `28 d / 7 d` | 36 | 28 | 200 | 50 |
| `28 d / 14 d` | 36 | 27 | 100 | 49 |
| `28 d / 28 d` | 36 | 23 | 47 | 49 |

The same counts hold for `program` (32 instances) and `robot_x_program` (36 instances) by the degeneracy above. Dispositions over all 1,080 rows: `NO-WINDOW-SUPPORT` 695, `WINDOW-SUPPORT-NO-REFERENCE-FLOOR` 345, `WINDOW-SUPPORT-REFERENCE-FLOOR` 40 (fleet only). Zero-member windows are counted, not dropped (e.g. fleet 3-day windows: 4 / 6 / 10 / 8 empty windows across the four histories).

Fleet density (per history across cells): 14-day windows hold a median of 76–88 files (maximum 105–143), 28-day windows a median of 149–185 files (maximum 170–244); only 24 of the 123 supported fleet 14-day/7-day windows reach ≥100 files, and 6 of those are on DEV-01. 3-day fleet windows are mostly below the inherited floor (8–22 supported windows per history out of 223) with median size 15–19 files.

### 5.4 Post-hoc healthy-control support — the headline feasibility result

**Zero healthy-control windows exist anywhere in the frozen grid.** Across `8,588` supported window-instances (`3,017` fleet + `1,857` × 3 grouped) and `1,080` candidate instances:

- `candidate_instances_with_healthy_control = 0`;
- `candidate_instances_with_reference_floor_and_healthy_control = 0`;
- `any_posthoc_healthy_control = false` and `any_healthy_control_with_inherited_reference_floor = false`.

A healthy window is a supported window with **zero abnormal member files and zero covered failure events** (post-hoc definition inherited from protocol §8). Only one distinct context comes close: `DEVELOPMENT-04 / robot:robot-02 / robot|28d|1d` (duplicated exactly by its `program:program-02` and `robot_x_program:robot-02|program-02` encodings) has `9 / 35` supported windows with zero abnormal member files — but all `9` also cover failure events (`27 / 35` windows cover ≥1 of that robot's 26 manifest failure events), so **none is event-free** and `n_healthy_supported_windows = 0`. Its abnormal member fraction is 0.051 and its fixed prefix is 103 files (below the inherited 200 floor).

Reading: widening fleet windows, narrowing to a single robot, and re-encoding the same membership as program/cell does not produce a post-hoc healthy, event-free window on Development. This reproduces and generalizes the G2 `NO-CALIBRATION` finding (0/369 supported fleet 14-day windows) across every candidate in the frozen grid.

### 5.5 Ordinary-transition / reset coverage

Reset edges are observable (maintenance-window ends); they are not label-derived.

| Fleet cell (per history) | Reset edges in span | Supported windows spanning ≥1 reset | Median reset age p50 / p95 (days) | Windows with median age ≤ 7 days |
|---|---:|---:|---|---:|
| `14 d / 7 d` DEV-01 | 182 | 30 / 30 | 11.60 / 16.37 | 2 |
| `14 d / 7 d` DEV-02 | 176 | 31 / 31 | 11.29 / 16.59 | 3 |
| `14 d / 7 d` DEV-03 | 191 | 31 / 31 | 11.64 / 16.10 | 2 |
| `14 d / 7 d` DEV-04 | 166 | 31 / 31 | 10.95 / 16.50 | 1 |

Ordinary-transition coverage is structurally strong for fleet at the executed shape: every supported fleet 14-day/7-day window (123 / 123) spans at least one recommissioning edge, and the daily-stride variant spans a reset in 212–223 of its windows. The *fresh*-reset subgroup (median age ≤ 7 days) is thin (`8 / 123` supported windows across all four histories). Grouped contexts cannot be assessed this way at the executed shape: only 3 of their 14-day/7-day windows are supported at all (2 of those span a reset edge), and across all grouped 14-day cells only 23 windows reach the inherited floor per grouped encoding (the `program` and `robot_x_program` encodings repeat the same windows). Mix-drift (total variation of robot/program composition vs the causal prefix) is reported per row; its fleet p50 range across cells is 0.114–0.375, i.e. most windows also carry non-trivial ordinary operational mix change — the same confounding direction G2 measured.

### 5.6 Missingness and calendar gaps

No observable field is missing and no row has zero valid patches. Calendar gaps inside supported fleet windows remain material: the p95 of the window's largest internal gap ranges from 12.8 h (3-day windows) to 162.5 h (28-day windows; e.g. DEV-03/04 28-day p95 159.2 h / 162.5 h ≈ 6.6–6.8 days). Fleet 14-day/7-day gap p95 is 39.8–75.1 h, and the median of the per-window maximum gap is 32.5 h for DEV-01. Gaps are expected from quarantine exclusions and maintenance downtime; they are published so a future design cannot silently treat a nominally 14-day window as continuously observed.

### 5.7 Overlap and effective independent support

| Fleet cell | Windows | Reuse factor | Mean adjacent overlap | Effective independent windows | Supported windows |
|---|---:|---:|---:|---:|---:|
| `14 d / 7 d` DEV-01 | 31 | 1.97 | 0.50 | 15.4 | 30 |
| `28 d / 7 d` DEV-01 | 31 | 3.78 | 0.75 | 8.2 | 31 |
| `28 d / 14 d` DEV-01 | 15 | 1.92 | 0.51 | 7.8 | 15 |
| `28 d / 28 d` DEV-01 | 7 | 1.00 | 0.00 | 7.0 | 7 |
| `28 d / 1 d` DEV-01 | 223 | 26.00 | 0.96 | 8.6 | 223 |

Daily-stride windows are almost entirely redundant: 223 windows at `28d|1d` carry only ~8.6 independent window-equivalents of distinct data, and the same distinct files are re-counted up to 26 times. The executed `14d|7d` design carries at most ~15.4–15.9 independent window-equivalents per history — roughly half its 30–31 supported windows. Across the grid, window-level event coverage reuses each covered failure event ~1.9–2.0 times at 14-day/7-day (e.g. 277 assignments for 144 unique events on fleet DEV-01). Any downstream count must use the unique event/window figures and the digest-bound membership, never the raw row multiplicities.

### 5.8 Post-hoc event and degradation coverage (evaluation only)

Fleet 14-day/7-day (the executed design, exactly reconciled with G2's retained `controls.jsonl`): `30/30`, `31/31`, `31/31`, `31/31` supported windows cover ≥1 failure event; unique covered failure events `144 / 140 / 152 / 131` of `246 / 229 / 254 / 221`; unique degradation onsets `122 / 126 / 132 / 122` of `213 / 208 / 214 / 208`; abnormal member fraction `0.135 / 0.142 / 0.126 / 0.122`. Wider fleet windows cover more events by construction (`28d|1d`: up to `173 / 254`) — a longer window trivially overlaps more event times, which is why coverage counts are **support**, not detection performance.

## 6. Feasibility findings

1. **A metadata-only healthy-control design does not exist on accepted Development under this bounded grid.** Zero of 8,588 supported window-instances are post-hoc healthy and event-free; zero of 1,080 candidate instances carry a healthy control. The single near-miss context (DEV-04 robot-02, 28-day) is disqualified by covered events, not by abnormal files.
2. **Only the fleet grouping can supply the inherited fixed causal reference pool** (850–974 files vs the 200 floor). Grouped contexts top out at 168 files, below the inherited floor even though 27 of the 28 populated contexts have ≥100 prefix files.
3. **Finer temporal resolution does not create healthy controls; it only creates unsupported windows.** 3- and 7-day grouped windows cannot reach 30 files at all (max 13–21), and 14-day grouped windows reach support in only 1–5 of 28 populated contexts.
4. **The grouping axis is not four independent designs.** Robot↔program is a bijection on eligible data, so `program` and `robot_x_program` duplicate `robot`; any future claim of cross-grouping replication would be false.
5. **Daily-stride candidates are dependent sampling, not evidence volume.** Up to 26× reuse with 0.96 adjacent overlap means row counts must never be treated as independent samples; across the 14-day and 28-day fleet cells the effective-independent figures are 7.0–16.5 window-equivalents per history.
6. **Ordinary-transition coverage is available but confounded and thin at the fresh-reset edge.** Every fleet 14-day/7-day window spans a reset edge, yet only 8 of 123 are freshly recommissioned (≤7 days), and mix drift p50 is 0.114–0.375.
7. **This is a feasibility answer, not a specificity answer.** No operating point, recall, delay, false-alarm rate or calibrated claim follows from these counts, and no candidate is selected or preferred. A zero-support / zero-control outcome is a valid result: it says a future geometry design cannot be built on these groupings without a separate prospective amendment.

## 7. Limitations

- Metadata support only. Every count describes observable file/event/context availability; none measures embedding movement or detection quality, and none is calibrated specificity.
- Post-hoc definitions (healthy = zero abnormal members and zero covered failure events) are inherited from protocol §8 and are not re-tuned here. Changing them would be a protocol amendment, not an audit option.
- Event coverage uses the inherited all-manifest failure-event denominator per context robot scope; degradation onsets are reported separately and are not a second event denominator.
- Windows that reach before `c1` are counted and disclosed (3,527 windows with pre-`c1` members) because the anchor rule is inherited, not re-designed.
- The 200-file reference floor is an inherited convention calibrated for fleet statistics. Grouped contexts are reported against it **and** against the 30/100 ladders so a single floor choice does not hide their actual depth.
- The four Development histories are the only audited units; `n = 4` histories means all cross-history aggregation is descriptive, and history-level uncertainty is not estimated here.
- Encoder seeds are irrelevant to this audit: zero seed computations were performed and the three accepted seeds add no independent metadata support.
- The accepted manifests were staged locally for execution (metadata only, hash-verified). No waveform shard, embedding cache, checkpoint or score table was read; the audited bytes match the digests bound in the executed G2 run.

## 8. Reproduction

```bash
python experiments/sprint24_task7m1_context_audit.py \
  --data-root <dir containing DEVELOPMENT/<history>/manifest.json> \
  --edges experiments/sprint24_task7m1_dev-edges.json \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-01=ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480747e37b48fe1e08 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-02=cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-03=b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-04=3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af \
  --expected-grid-sha256 8c6d70ce641c105922a7aa416d4eeeeda9a8093ecdf1189630c9b7533d0f1dc3 \
  --out artifacts/sprint-24/evidence/s24-7-m1-a01
```

The CLI fails closed on a manifest or grid digest mismatch, writes `grid-freeze.json` before analysis, and records output digests in `run.json`. Focused tests: `python -m pytest tests/representation/test_sprint24_context_audit.py -q`.

## 9. What is not authorized

This report does not authorize a new geometry design, a new window or reference lock, retraining, a new corpus, fresh Confirmation access, Phase C execution or any protocol change. Any follow-on geometry study needs a separately approved prospective amendment and execution release; if no grouping yields healthy-control support, Phase C stays held with this reason recorded.
