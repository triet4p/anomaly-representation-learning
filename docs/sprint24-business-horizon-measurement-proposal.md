# Sprint 24 business-horizon movement measurement proposal (prospective design, metadata feasibility)

- Task / attempt / agent: `S24-7-M2` / `S24-7-M2-A01` / `S24M2GoldA01` (owner-selected Gold via the standing default-answer instruction, quota `unknown`, full route).
- Criterion: `docs/sprint-plans/sprint-24.md:153-156` — design prospective day-over-day, week-over-week and month-over-month collective movement endpoints with abnormal-period/ordinary-transition controls, using G3 and M1 as established evidence; specify calendar/rolling boundaries, timezone and week start, partial periods, exposure/context comparability, reference custody, support/abstention, post-hoc attribution and what would distinguish abnormal from normal variation; separate deployment-observable comparisons from retrospective labels and proposed choices from business facts.
- Status: **proposal + metadata-only feasibility.** This document is **not** an execution release, not a detector design approval, and not a Phase C release. It proposes defaults and records what the accepted metadata can and cannot support.
- Preserved unchanged: locked protocol `docs/sprint24-geometry-time-diagnostic-v1.md`, G1/G2 records and the accepted G3 decision `docs/sprint24-geometry-time-decision-g3.md`, the M1 audit report and evidence, Task 7 `NOT_TRIGGERED`, Confirmation custody, Sprint 18 `UNRESOLVED`, and every existing numerical floor.
- Cost of this task: zero forwards, zero model loads, zero waveform reads, zero embeddings or scores read; ~7 s per full metadata pass. The four Development histories remain the only data.

## 0. What this document is and is not

The owner's business requirement is a comparison of **collective cluster/distribution movement** at three operational horizons: today vs yesterday, this week vs last week, this month vs last month. That is an episode/period-level endpoint. It is explicitly **not** a per-file classification endpoint, and it is not a retrospective re-scoring of the frozen G1/G2 windows.

In scope here:

- a complete prospective measurement design for the three horizons;
- a metadata-only feasibility and unknowns matrix computed from the accepted Development manifests;
- reproducible worked examples for period membership and matched exposure;
- the owner decisions and prerequisites that must be settled before a separately authorized execution amendment.

Out of scope and not performed: any embedding, score, model, checkpoint, waveform or forward; any new corpus; any retraining; Confirmation access; Phase C execution; any plan edit; any numerical gate change.

## 1. Established evidence carried in (no re-derivation)

Two accepted results bound what a new design may claim:

- **G3/G2 (`NO-CALIBRATION`, owner-accepted).** Healthy-control support was `0 / 369` SUPPORTED windows under the frozen fleet 14-day / 7-day design, so no calibrated operating point existed. All `369 / 369` window-seeds crossed their own split-half `q95` (100 %) — preserved as descriptive movement, never as detections. Movement tracked operational mix drift (`r(MMD², TV drift) ≈ +0.78`) more strongly than abnormal load (`r(MMD², n_abnormal) ≈ −0.40`).
- **M1 metadata audit.** `0 / 1,080` candidate instances and `0 / 8,588` supported window-instances contained a post-hoc healthy, event-free control. Fleet was the only grouping reaching the inherited 200-file reference pool (850–974); grouped contexts topped out at 168. Robot↔program is effectively a bijection on eligible Development data, so `program` and `robot×program` are replicate encodings, not independent designs.

Consequence for this proposal: a business-horizon design **must not** assume that annotated healthy exposure exists on Development, and must abstain rather than relax a floor. The metadata pass in §11 reproduces the same negative control conclusion at day, week and month granularity.

## 2. Established facts, proposed defaults, and owner decisions

Every element below is labelled in one of three tiers. Confusing the tiers is the main design risk; the code freeze in §17 binds the second tier as declared assumptions.

| Element | Tier | Value used here | Source or consequence |
|---|---|---|---|
| Development clock origin | Established | `t = 0` per history; `span_s = 38,880,000` (450 d); `cutoff_time = 19,440,000` (225 d); `quarantine_s = 604,800` (7 d) | manifest `calendar` table, all four histories |
| Clock is relative, not wall-clock | Established | no epoch, no timezone, no calendar dates in the metadata | manifest inspection; period boundaries are therefore a **projection** |
| Eligibility predicate | Established | `is_eligible_for_training` (not quarantined, not censored, no reserve identity, no maintenance overlap) | inherited, unchanged |
| Analysis span | Established | `(c1, tmax]` per history, `c1` = 40 % decile edge, never concatenated | inherited protocol §4 |
| Window support floor | Established | `>= 30` eligible files | inherited protocol §4, unchanged |
| Reference floor | Established | `>= 200` eligible pre-`c1` files | inherited protocol §5, unchanged |
| Timezone | **Proposed** | `UTC` | owner decision D1 |
| Projection anchor | **Proposed (for this pass only)** | `t = 0` ↦ `2024-01-01T00:00:00Z` | owner decision D2; this is an arithmetic projection, not a business fact |
| Week start | **Proposed** | ISO-8601 Monday | owner decision D3 |
| Month definition | Established requirement | real calendar months, actual lengths 28/29/30/31 | owner requirement; 28-day substitution forbidden |
| Rolling month anchor | **Proposed** | same day-of-month 15 | owner decision D4 |
| Primary boundary convention | **Proposed** | calendar-aligned | owner decision D4 |
| Exposure comparability floor | **Proposed** | ratio `>= 0.8` | owner decision D10 |
| Sampling-null resamples / seed | **Proposed** | 200 resamples, seed inherited convention `2407` | owner decision D8 |
| Cluster count for composition endpoint | **Proposed** | `K = 8` (sensitivity 4/16) | owner decision D9 |
| Minimum pairs for a contrast | **Proposed** | 2 | owner decision D8 |
| Business timezone / week start / shift calendar | **Unknown** | — | no metadata field or accepted record establishes these |

## 3. Time base and calendar projection

The accepted manifests carry monotone relative timestamps in seconds (median file duration 600 s) with a per-history origin. They contain no wall-clock epoch, no timezone and no calendar date. Business periods are therefore defined by an explicit projection:

$$ t_{\text{abs}} = \text{anchor} + t_{\text{rel}}, \qquad \text{local} = \text{tz}(t_{\text{abs}}) $$

- `anchor` is a declared epoch offset; `tz` is a declared IANA zone.
- Day and week boundaries are local midnights; month boundaries are local first-of-month midnights.
- Every boundary is recomputed from the declared calendar, so a month is never approximated by a nominal length.
- The projection is a **deployment-observable** construct (it reads only `start_time`/`end_time` and the declared calendar). It uses no annotation.

Because the anchor is unverifiable from the metadata, §11 reports a projection-sensitivity pass. That pass changes which files sit in which period and how many pairs are comparable; it does **not** change the qualitative conclusion.

## 4. Horizon definitions

Each horizon compares two strictly adjacent periods of the same type. A missing period is an abstention, never a skip to the next available period (a "this week vs two weeks ago" comparison is not the requested week-over-week measurement).

| Horizon | Current period | Compared with | Adjacent rule |
|---|---|---|---|
| `day` | calendar day `[00:00, 24:00)` local | the immediately preceding calendar day | no substitution of day−2 |
| `week` | calendar week starting on the declared week-start day, 7 days | the immediately preceding week | no substitution of week−2 |
| `month` | calendar month `[1st 00:00, 1st of next month 00:00)` local | the immediately preceding calendar month | actual lengths; no nominal month |

Month lengths vary and are preserved. On the accepted span the observed full months are 30 and 31 days (for example `M2024-06` = 30 d vs `M2024-07` = 31 d); leap February (29 d) and non-leap February (28 d) are exercised in the illustrative calendar cases of §12 because the analysis span does not reach them.

Declared rolling alternatives (sensitivity only, never primary):

- `day`/`week` rolling = tumbling blocks anchored at `c1` rather than at local calendar boundaries. This measures boundary-phase sensitivity.
- `month` rolling = same-day-of-month windows (`D = 15` proposed): consecutive anchors are 15 Jan → 15 Feb (31 d) → 15 Mar (29 d in 2024) → … so the window length is the real inter-anchor gap (28–31 days). A fixed 28- or 30-day month is never admitted.

## 5. Period membership, exposure and partial periods

Membership is label-blind and follows the inherited trailing-window convention:

$$ \text{file} \to p \iff p_{\text{start}} < \text{end\_time} \le p_{\text{end}} $$

- The assignment key is the file's **`end_time`**, consistent with the frozen `(T − W, T]` window rule.
- A file ending exactly on a boundary belongs to the period that ends there. No floating tolerance is used.
- A file may **straddle** a boundary: it is assigned by `end_time` while its observation time inside the period is clipped. On the calendar pass there are 36 such eligible files at the day horizon, 7 at the week horizon and 2 at the month horizon.
- Period membership only considers eligible files with `end_time > c1`; pre-`c1` rows are reference custody and never enter an evaluation period.

**Exposure** is reported in member file-days:

$$ E_p = \frac{1}{86{,}400}\sum_{i \in p} \max\bigl(0,\ \min(e_i, p_{\text{end}}) - \max(s_i, p_{\text{start}})\bigr) $$

File-days rather than calendar-days is the honest unit here: files are short (median 600 s) and sparse (median 6 eligible files per day), so calendar coverage is not observation coverage. Across the calendar pass the summed exposure is 35.13 (day), 35.23 (week) and 35.24 (month) file-days over the four histories, consistent with G3's reported 35.3 eligible-file-days.

**Partial periods** are classified, never dropped silently:

| State | Meaning | Comparison effect |
|---|---|---|
| `FULL` | `period.start >= c1` and `period.end <= tmax` | eligible |
| `PARTIAL-PRE` | period begins before `c1` (reference-era part is not an evaluation period) | abstain |
| `PARTIAL-POST` | period is truncated by the history's `tmax` | abstain |
| `PARTIAL-BOTH` | both edges cut | abstain |

On the calendar pass each history contributes exactly 2 partial periods per horizon (8 across four histories). The first business period therefore always abstains; the final truncated one does too. This is deliberate: comparing a partial period would confound movement with exposure.

Period-pair comparability uses a fixed precedence:

1. `INSUFFICIENT-REFERENCE` — the history's fixed pre-`c1` reference pool is below 200 files.
2. `PARTIAL-PERIOD` — either period is not `FULL`.
3. `INSUFFICIENT-PERIOD-SUPPORT` — either period is below 30 eligible files.
4. `COMPARABLE-UNMATCHED-EXPOSURE` — both periods pass support, but `min(E)/max(E) < 0.8`.
5. `COMPARABLE-MATCHED` — all of the above pass.

`COMPARABLE-UNMATCHED-EXPOSURE` pairs are published and excluded from the primary aggregate; a capped-`n` deterministic sensitivity (both periods reduced to the smaller member count by a hash-seeded, label-blind selection) is the declared fallback, not a silent replacement.

## 6. Collective movement endpoints (specification only; not executed here)

All endpoints operate on the frozen accepted file embeddings in the original 128-d space, one vector per file, no projection. No endpoint is computed by this task; the catalogue is frozen so that an execution amendment cannot quietly change the estimand.

| ID | Endpoint | Definition | Role |
|---|---|---|---|
| `E1` | DIST-MOVE | unbiased squared MMD with a fixed Gaussian kernel; bandwidth = median pairwise distance on the reference pool, computed once per history × seed | primary distribution movement |
| `E2` | CLUSTER-MOVE | total-variation distance between the two periods' cluster-occupancy vectors; `K` prototypes fitted once on the reference pool | collective cluster/composition movement |
| `E3` | LOC-MOVE | squared mean shift standardized by reference per-dimension SD | location component |
| `E4` | DISP-MOVE | `|log(Tr(Σ_cur) / Tr(Σ_prev))|` | dispersion component |

Estimator rules:

- `E1` uses the unbiased U-statistic so unequal period sizes do not create an `O(1/n)` bias; the biased plug-in value may be reported alongside for continuity with G2.
- Kernel bandwidth, cluster prototypes and per-dimension standardization come from the **reference pool only** and are never re-fitted per period or per pair.
- Exposure-weighted variants (`w_i = ` the file's clipped overlap fraction inside its period) are published as sensitivity; equal-file-weight is primary.
- Every row reports `n`, exposure and estimator flags; a period below floor or with non-finite inputs is `UNSUPPORTED`, never dropped from the denominator.

## 7. Normal variation: what is estimable and what is not

Distinguishing abnormal movement from normal variation requires a reference distribution of "no-anomaly" movement. Four candidate references exist; the metadata decides which are available.

| ID | Reference | Definition | Availability |
|---|---|---|---|
| `N1` | sampling null | paired label permutation inside the pooled pair, preserving `n`; 200 resamples, seed derived from `2407` + pair identity | computable; **sampling noise only** |
| `N2` | half-split band | seeded half-splits of each period; movement half-vs-half | computable; conservative within-period noise |
| `N3` | annotated-clean control | pair where both periods have complete annotation coverage, zero abnormal member files and zero covered events | **0 pairs on the accepted span** |
| `N3b` | quiet-period control | pair where both periods have complete annotation coverage and the pair covers no annotated failure event | **0 pairs on the accepted span** |

Two facts must be stated plainly rather than engineered around:

- G2 already showed that the sampling null is not a normal-variation reference: `369 / 369` frozen windows crossed their own sampling `q95`. Crossing `N1` is therefore expected and is not evidence of anomaly specificity.
- On the accepted Development span, **every** supported week and month carries abnormal member files and covered failure events, and crosses at least one recommissioning edge. Weekly comparable pairs cover a median of 9 events (range 1–13); monthly pairs cover a median of 39 (range 33–44). Reset adjacency is equally ubiquitous (median 10 edges per weekly pair, 48 per monthly pair), so it is not discriminative either. Perfectly annotated-clean pairs do not exist at these horizons.

Consequence: the only abnormal-versus-normal contrast available on this corpus is a **graded event-load association**, not a case-control comparison:

- `C1` EVENT-LOAD-ASSOCIATION: within history, Spearman correlation of `E1`/`E2` with the number of covered failure events and with the abnormal member fraction; reported per history and pooled descriptively.
- `C2` MIX-ADJUSTED-RESIDUAL: the `E1` residual after regressing on observable robot-mix drift, then associated with event load — the direct answer to the G3 `+0.78` drift confound.
- `C3` QUIET-PERIOD-CONTRAST and `C4` TEMPORAL-SPECIFICITY **abstain** (`UNSUPPORTED`), because no quiet period and no event-free neighbour exists at week or month granularity on this corpus. Abstention is the declared outcome, not an implementation failure.

Absence of annotation is `UNANNOTATED` or `UNANNOTATED-PARTIAL`, never `ANNOTATED-CLEAN` and never "quiet". The projection keeps absence explicit (`None`), and a pair whose attribution class is `UNANNOTATED` is excluded from both control counters. A stripped-annotation probe on real metadata confirms this: the affected history's periods all become `UNANNOTATED` and the horizon control state does not become available (§12).

## 8. Post-hoc attribution

Annotations are joined only after period and pair membership, support and comparability exist. They are assessment only.

- `ANNOTATED-CLEAN`: complete label coverage of every member, zero abnormal members, zero covered failure events.
- `ANNOTATED-ABNORMAL`: complete coverage and at least one abnormal member or covered event.
- `UNANNOTATED-PARTIAL` / `UNANNOTATED`: some or all members lack a label.
- Pair attribution: `EVENT-CURRENT`, `EVENT-PREVIOUS`, `NO-EVENT-ANNOTATED` (quiet), `UNANNOTATED`.

Event coverage uses the inherited denominator: a failure event is covered iff `period_start < failure_time <= period_end`. Degradation onsets are reported separately and are never a second event denominator. An event is never a membership or support input.

## 9. Support, abstention and dispositions

- Floors are inherited unchanged: 30 eligible files per period, 200 reference files per history. A descriptive ladder (`5 / 10 / 15 / 20 / 30 / 100`) is published beside them so a zero at 30 cannot be mistaken for a zero at a smaller count.
- Per-period states: `SUPPORTED`, `INSUFFICIENT-PERIOD-SUPPORT`; plus the coverage and exposure states above.
- Per-horizon states: `FEASIBLE-DESCRIPTIVE` (every contributing history has at least one comparable pair), `PARTIALLY-FEASIBLE` (some histories do), `INFEASIBLE` (none).
- Control states: `ANNOTATED-CLEAN-CONTROL-AVAILABLE`, `QUIET-PERIOD-CONTROL-AVAILABLE`, `NO-QUIET-PERIOD-CONTROL`, `NO-SUPPORTED-PERIODS`.
- No floor is relaxed to manufacture comparability, and no horizon is pooled into a coarser one to hide an unsupported daily comparison. If the daily horizon is unsupported, it is reported as `INFEASIBLE` and daily comparisons are not made.

## 10. Reference custody and representation reuse

- Reference pool: eligible files with `end_time <= c1`, fixed per history, digest recorded. Observed sizes are 850 / 974 / 935 / 919 (all `REFERENCE-OK`).
- Kernel bandwidth, cluster prototypes and per-dimension SD come from that pool only; no period-matched resampling, no label filtering, no evaluation contact.
- No cross-history pooling; no Confirmation access or reuse; the three accepted encoder seeds are metadata-identical and add no independent data.
- A future execution amendment must freeze the label-blind period manifest (history, period key, member file IDs, digest) **before** any embedding outcome is inspected.
- Representation reuse envelope: the fixed reference pools plus the analysis spans contain 8,754 eligible files in total (3,678 reference + 5,076 span). With three seeds the maximum embedding rows are `8,754 × 3 = 26,262`; if accepted reference embeddings are reused, the new span rows are `5,076 × 3 = 15,228`. Both bounds are below G2's 41,000 cap. If the accepted caches already cover those files, no forward is needed; any missing coverage requires an explicitly bounded inference authorization in the amendment, not an implicit one.

## 11. Per-horizon feasibility and unknowns (accepted Development metadata)

Primary calendar-aligned convention, four histories, inherited floors:

| Horizon | Periods (full) | Supported periods | Pairs | Comparable matched | Comparable unmatched | Median / max members | Clean pairs | Quiet pairs | State | Control state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `day` | 900 (892) | 0 | 896 | 0 | 0 | 6 / 17 | 0 | 0 | `INFEASIBLE` | `NO-SUPPORTED-PERIODS` |
| `week` | 132 (124) | 99 | 128 | 36 | 42 | 40 / 74 | 0 | 0 | `FEASIBLE-DESCRIPTIVE` | `NO-QUIET-PERIOD-CONTROL` |
| `month` | 36 (28) | 28 | 32 | 14 | 10 | 171.5 / 237 | 0 | 0 | `FEASIBLE-DESCRIPTIVE` | `NO-QUIET-PERIOD-CONTROL` |

Per-history detail:

| History | Day full periods at ≥10 / ≥15 / ≥20 | Week comparable (matched / unmatched) | Month comparable (matched / unmatched) | Reference |
|---|---|---|---:|---:|
| DEV-01 | 21 / 2 / 0 | 12 / 7 | 3 / 3 | 850 |
| DEV-02 | 37 / 2 / 0 | 5 / 15 | 5 / 1 | 974 |
| DEV-03 | 21 / 1 / 0 | 6 / 12 | 3 / 3 | 935 |
| DEV-04 | 44 / 2 / 0 | 13 / 8 | 3 / 3 | 919 |

Rolling convention (sensitivity):

| Horizon | Periods (full) | Supported periods | Pairs | Matched | Unmatched | Median / max members | Full-period lengths |
|---|---:|---:|---:|---:|---:|---:|---|
| `day` | 896 (892) | 0 | 892 | 0 | 0 | 5.5 / 16 | 1 d |
| `week` | 128 (124) | 100 | 124 | 34 | 46 | 40.75 / 74 | 7 d |
| `month` | 32 (24) | 32 | 28 | 18 | 2 | 185 / 231 | 30–31 d |

Projection sensitivity (alternate anchor `2024-03-07T00:00:00Z`, zone `Etc/GMT-7` i.e. +07:00, Sunday week start), deltas versus primary:

| Cell | Δ matched | Δ unmatched | Δ pairs | Δ supported periods |
|---|---:|---:|---:|---:|
| `week|calendar` | +3 | 0 | 0 | +2 |
| `month|calendar` | +4 | −8 | −4 | +1 |
| `month|rolling` | −9 | +13 | +4 | 0 |
| `day|*` | 0 | 0 | 0 | 0 |

The projection changes which pairs are comparable (up to ±9 pairs for rolling months) and is therefore an owner decision, not an implementation detail. It does not change any qualitative state: the daily horizon stays infeasible and no clean or quiet control appears under any variant.

Feasibility and unknowns matrix:

| Horizon | What the metadata establishes | What is unknown | Executable after which decisions |
|---|---|---|---|
| `day` | 223 full days per history; median 6 eligible files, p90 9–11, maximum 17; zero days reach the 30-file floor; 80 calendar periods have no eligible member | whether a lower day floor is acceptable, and what distribution-statistic validity it buys; whether intra-day shifts are the real business unit | D6 (floor) and, if lowered, a documented tradeoff; otherwise `INFEASIBLE` and no daily claim |
| `week` | 31 full weeks per history; 99 of 124 full periods supported; 36 matched + 42 unmatched comparable pairs; per-history median mix drift 0.435–0.504 — weekly comparison is strongly composition-confounded | whether weekly movement can be separated from robot-mix change without per-robot strata (median per-robot weekly counts are 3–9 files, maximum 18, far below floor) | D4, D7, D10 |
| `month` | 7 full months per history; all supported; 14 matched + 10 unmatched comparable pairs; real lengths 30–31 d; per-history median mix drift 0.126–0.216, roughly a third of the weekly figure | whether a 6-pair-per-history association is informative; anchor/timing sensitivity (±4 matched pairs) | D2, D4, D7, D10 |
| all | zero annotated-clean and zero quiet pairs at every horizon; every week/month covers events and crossings | any source of verified healthy exposure, or an owner-accepted substitute normal-variation reference | D7, D8 |

## 12. Reproducible worked examples

All values below come from `artifacts/sprint-24/evidence/s24-7-m2-a01/worked-examples.json` (history `S18I-ITER-0005-DEVELOPMENT-01`, calendar projection) and are reproduced by the command in §17.

Boundary and straddling examples:

| Case | Observed value | Units |
|---|---|---|
| First day period | `D2024-05-29`, `PARTIAL-PRE`, 2 members, 0.0139 exposure | rel s / days |
| Last day period | `D2025-01-08`, `PARTIAL-POST`, 0 members, 0 exposure | rel s / days |
| Largest day period | `D2024-07-01`, 16 members, `INSUFFICIENT-PERIOD-SUPPORT` against the 30 floor | files |
| Straddling file | `S-op-003178-6f4ec73932`, starts `2024-06-30T23:56:03.51Z`, ends `2024-07-01T00:06:03.51Z`, assigned to `D2024-07-01` / `W2024-07-01` / `M2024-07` | ISO projection |
| Exposure inside the assigned day | 363.51 s = 0.00421 file-days of the 600 s file | days |
| Assignment inequality | `15724800.0 < 15725163.51 <= 15784800.0` holds for the day period | rel s |
| Adjacent unequal months | `M2024-06` = 30 d with 164 members, `M2024-07` = 31 d with 164 members | days / files |
| Matched weekly pair | `W2024-07-29` (44 files, 0.3056 file-days) vs `W2024-08-05` (48 files, 0.3333 file-days), ratio 0.9167, mix drift 0.4545, 11 reset edges | days / dimensionless |
| Matched monthly pair | `M2024-10` (31 d, 190 files, 1.3194 file-days) vs `M2024-11` (30 d, 177 files, 1.2292 file-days), ratio 0.9316 | days / dimensionless |
| Day comparison outcome | 222 pairs `INSUFFICIENT-PERIOD-SUPPORT`, 2 `PARTIAL-PERIOD`, 0 comparable | pairs |

Illustrative synthetic calendar cases (`illustrative-examples.json`, explicitly **not** corpus evidence):

- an `end_time` exactly on a day boundary is assigned to the earlier period (right-inclusive rule);
- calendar month lengths `2024-01` 31, `2024-02` 29, `2024-03` 31, `2025-02` 28, `2024-06` 30, `2024-07` 31;
- rolling-month lengths 31, 29, 31, 30, 31 for the 15th-anchored windows, i.e. never a nominal 28;
- the same instant falls in different week keys under Monday and Sunday week starts.

Annotation-absence probe (real metadata, one history's labels stripped): all of its periods report `UNANNOTATED` and no control state becomes available; `0` clean and `0` quiet pairs are counted. Label-permutation probe: 0 structural mismatches across 1,068 period rows and 1,056 pair rows while 693 / 836 rows changed post-hoc.

## 13. What would distinguish abnormal movement from normal variation

The design can only support association-level answers on this corpus. Within that limit, the predeclared discriminators are:

1. **Component separation.** Movement must not be attributable to composition alone: `E1` must exceed what the observable mix-drift change predicts (`C2` residual). A pair whose entire movement is composition is ordinary variation by definition.
2. **Graded event load.** Higher covered-event counts or abnormal member fractions should associate with higher `E1`/`E2` within a history (`C1`). G3's own association was negative (`r ≈ −0.40` with abnormal count), so a negative result is a legitimate and expected outcome.
3. **Sampling-noise separation.** `E1` exceeds `N1`/`N2` — necessary for the movement to be real, and explicitly not sufficient for specificity (G2: 100 % crossing).
4. **Temporal concentration and recovery.** Movement should be elevated in the event-containing period rather than in the adjacent periods — predeclared as `C4`, currently `UNSUPPORTED` because no event-free neighbour exists at these horizons.
5. **Rank agreement.** Event-loaded pairs should rank above low-event pairs within the same history, reported as a descriptive rank share rather than a tuned threshold.
6. **Replication across histories and seeds.** The same sign and ordering in all four histories and three seeds, reported with history-cluster uncertainty at `n = 4`.

What would **not** distinguish abnormal movement from normal variation, and must not be presented as if it did:

- crossing a sampling-null quantile alone (G2 evidence);
- any threshold chosen after inspecting outcomes, or a q95 chosen to make a target count appear;
- pooling day into week or week into month to hide an infeasible horizon;
- treating unannotated periods as healthy, or reset adjacency as a control;
- per-file classification accuracy, precision/recall, or file-level PR-AUC (the endpoint is period-level);
- a single history or a single seed presented as replication.

## 14. Owner decisions required

Each decision is required before a separate execution amendment. The value in §11 is what was used for feasibility; none is a business fact.

| ID | Decision | Proposed default | Consequence if unchanged |
|---|---|---|---|
| D1 | Business timezone (IANA) | `UTC` | day/week boundaries may sit at the wrong local time; membership and pair counts shift |
| D2 | Projection anchor mapping the relative clock to a calendar | `2024-01-01T00:00:00Z` (probe only) | month boundaries and week phase are arbitrary; ±4 matched monthly pairs observed under an alternate anchor |
| D3 | Week start | Monday (ISO) | Sunday-start shifts week membership; ±3 matched weekly pairs observed |
| D4 | Primary boundary convention | calendar-aligned, rolling as sensitivity | a rolling primary would change which pairs are compared and must be predeclared |
| D5 | Partial-period policy | exclude partial periods from primary comparisons | including them would confound movement with exposure |
| D6 | Daily-horizon support floor | keep inherited 30 → `INFEASIBLE` | daily business comparisons cannot be made; observed maxima are 16–17 files/day. A lower floor is a new documented tradeoff, not a silent relaxation |
| D7 | Normal-variation reference | none available → event-load association only | no abnormal-vs-quiet contrast, no calibrated specificity; a verified healthy source would be a new data decision |
| D8 | Operating-point policy | descriptive only; no calibrated detector | no recall/delay/false-alarm claims. Any operating point needs owner-approved calibration support that this corpus does not contain |
| D9 | Cluster count for `E2` | `K = 8` primary, 4/16 sensitivity | composition endpoint depends on `K`; it must be fixed before outcomes |
| D10 | Exposure-mismatch policy | exclude from primary, capped-`n` sensitivity | 42 of 78 weekly pairs are exposure-mismatched; including them without the sensitivity would bias `E1` |

Two further decisions are implicitly required and should be recorded with the above: whether the business unit is the whole fleet or a permitted sub-context (per-robot strata are unsupported at these horizons: 3–9 files per robot-week), and whether a deployment source with real timestamps and incident labels may be used for the healthy control that Development cannot supply.

## 15. Prerequisites before a separately authorized execution amendment

1. Owner decisions D1–D10 (plus the two implicit decisions) recorded with their values.
2. A new prospective protocol identity and a frozen label-blind period manifest written **before** any embedding outcome is inspected.
3. Verified coverage of the frozen accepted embeddings for all 8,754 eligible reference-plus-span files per seed (5,076 span rows plus the fixed reference pools), or an explicitly bounded missing-feature inference authorization with its own forward cap.
4. Confirmation custody unchanged: evaluation-only secondary context, no adaptation, no tuning, no checkpoint selection.
5. No retraining, no new corpus, no Phase C execution; Task 7 `NOT_TRIGGERED` stands.
6. The G1/G2 historical contract is not rewritten; the business horizons are a new design, not a correction.

## 16. Limitations

- Metadata feasibility only: nothing here measures embedding movement, and no support count is a performance estimate.
- Four histories at `n = 4`: every cross-history aggregate is descriptive; history-level uncertainty is not estimated.
- The clock is relative: the calendar projection, timezone and week start are declared assumptions. All calendar dates in this document are projections, not corpus facts.
- The corpus is sparse in observation time (~35 file-days per four histories), so file-day exposure rather than calendar-day exposure is the honest denominator.
- Per-robot and robot×program strata cannot support distribution statistics at these horizons, so composition adjustment is limited to fleet-level mix covariates.
- Weekly comparison is strongly confounded by robot-mix drift (per-history median TV 0.435–0.504); any weekly movement claim must carry the `C2` residual analysis.
- No annotated-clean and no quiet period exists at any horizon on the accepted span; abstention is the correct outcome, and a negative feasibility is a valid answer.
- Cross-horizon statements (for example "the month moved less than the week") are not supported: the periods have different exposure and the pairs are not independent.

## 17. Reproduction, evidence and freeze

Command (metadata only; digest-verified inputs):

```bash
python experiments/sprint24_task7m2_business_horizon.py \
  --data-root <dir containing DEVELOPMENT/<history>/manifest.json> \
  --edges experiments/sprint24_task7m1_dev-edges.json \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-01=ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480747e37b48fe1e08 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-02=cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-03=b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-04=3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af \
  --out artifacts/sprint-24/evidence/s24-7-m2-a01
```

Focused tests:

```bash
python -m pytest tests/representation/test_sprint24_business_horizon.py -q
```

Frozen proposal digest (`S24-BIZ-HORIZON-v1`): written to `proposal-freeze.json` before any support statistic; the CLI fails closed on a mismatch and records every output digest in `run.json`.

Evidence files (`artifacts/sprint-24/evidence/s24-7-m2-a01/`): `proposal-freeze.json`, `period-table.json` / `.csv`, `pair-table.json` / `.csv`, `feasibility-matrix.json`, `worked-examples.json`, `illustrative-examples.json`, `sensitivity.json`, `run.json`, and `label-permutation-smoke.md`.

This proposal authorizes nothing. Any movement computation needs a separately approved prospectively frozen amendment and an explicit execution release.
