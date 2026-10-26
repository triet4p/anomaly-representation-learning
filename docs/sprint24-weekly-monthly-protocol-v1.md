# Sprint 24 weekly/monthly collective-movement study protocol v1 (pre-execution)

- Protocol ID: `S24-WM-PROTOCOL-v1`; machine-readable study identity `S24-BIZ-WM-v1`.
- Task / attempt / agent: `S24-7-M3` / `S24-7-M3-A01` / `S24M3GoldA01` (role Gold under the owner's standing default-answer instruction; quota `unknown`; full route with an independent evidence review and a separate Bronze exact-snapshot checkpoint).
- Criterion: `docs/sprint-plans/sprint-24.md:159-162` — freeze the bounded weekly/monthly prospective execution amendment on existing accepted Development data, with exact period/membership/reference identities, calendar assumptions, exposure/context controls and abstentions, post-hoc abnormal-period association, small-history uncertainty, sensitivity/multiplicity handling, a prospective decision rule for "promising enough to propose scaling", complete method/seed/history/cost coverage, representation-coverage unknowns, unambiguous PROMISING / NOT-PROMISING / INCONCLUSIVE dispositions, and no embedding/score/model/Confirmation/remote/new-corpus/retraining/Phase-C work.
- Base commit: `7da595277ae9b1258b6d8dfff0bbf81272b45c3a` (accepted Task 7-M2 checkpoint).
- Status: **frozen prospective design candidate, pre-execution.** This document authorizes nothing. It is not an execution release, not a scaling release, not a detector design, and not a Phase C release.
- Upstream contract resolved here: `docs/sprint24-business-horizon-measurement-proposal.md` (`S24-BIZ-HORIZON-v1`, digest `c8c7b8d4a396a98163bdf3e13b86538454cff8a45905eb7382a65c72b911dc57`).
- Preserved unchanged: `docs/sprint24-geometry-time-diagnostic-v1.md`, `docs/sprint24-geometry-time-decision-g3.md`, the M1 audit report and evidence, Task 7 `NOT_TRIGGERED`, Confirmation custody, Sprint 18 `UNRESOLVED`, and every inherited floor (30 eligible files per period, 200 reference files).

## 0. What this document is, and what it is not

The owner resolved a business measurement question: compare **collective cluster/distribution movement** week over week and month over month on the existing accepted Development data, and decide prospectively whether such measurement is promising enough to justify proposing a larger, denser study. This document is the executable study contract for that question.

It is **not**:

- a detector specification: it defines no anomaly threshold, no operating point and no alarm rule;
- an accuracy study: it makes no per-file classification, PR-AUC, recall, delay or false-alarm claim;
- a generalization study: everything here is exposed-benchmark exploratory research on data the encoder already saw;
- an execution release: no embedding, score, cache, checkpoint or model is opened to produce this document, and no forward is authorized;
- a scaling release: a PROMISING disposition only *proposes* a separately authorized scaling study.

Daily resolution remains part of the business requirement but is **deferred** by owner prioritization. It is not evaluated here, no daily floor is lowered, and no daily statement follows from this study. The owner-reported deployment volume (approximately 500–1,000 files per day) is business context; it is not the observed density of the current synthetic corpus (median 6 eligible files per day) and must never be used to reinterpret current-data infeasibility as deployment infeasibility.

Three interpretation rules are binding on every reader and on the execution task:

1. **Movement is not an anomaly.** A distribution shift is a description of the data, not evidence of a fault. Every disposition in this protocol is an association statement about periods.
2. **Absence of annotation is not health.** A period with missing or partial labels is `UNANNOTATED`/`UNANNOTATED-PARTIAL`; it is never `ANNOTATED-CLEAN`, never "quiet", and never a control.
3. **The calendar is a projection.** The accepted Development clock is relative seconds from a per-history origin with no wall-clock epoch. Every calendar date in this protocol is an arithmetic projection from a declared anchor, not a corpus fact and not a deployment timestamp.

## 1. Established evidence carried in (no re-derivation, no re-audit)

| Source | Accepted fact used here | Consequence for this study |
|---|---|---|
| G3/G2 (`NO-CALIBRATION`, owner-accepted) | Healthy-control support was `0 / 369` SUPPORTED windows under the frozen fleet 14-day/7-day design; all `369 / 369` window-seeds crossed their own split-half `q95` (100 %) | Sampling-null crossing is necessary only, never sufficient; no calibrated operating point exists |
| G3 association summary | `r(MMD², TV drift) ≈ +0.78` versus `r(MMD², n_abnormal) ≈ −0.40` | Composition drift must be adjusted for, and a negative burden association is a legitimate expected outcome |
| M1 audit | `0 / 1,080` candidate instances and `0 / 8,588` supported window-instances contained a post-hoc healthy, event-free control; grouped contexts topped out at 168 reference files against the inherited 200 floor | Fleet is the only supportable business unit; no grouped-context substitute exists |
| M2 feasibility | Week `FEASIBLE-DESCRIPTIVE` (99 of 124 full periods supported; 36 matched + 42 unmatched comparable pairs), month `FEASIBLE-DESCRIPTIVE` (28 of 28 full periods; 14 matched + 10 unmatched), day `INFEASIBLE` | Week and month are executable in principle; day is out of scope |
| M2 controls | Zero annotated-clean pairs and zero quiet-period pairs at every horizon; every supported week/month covers events and reset edges | No specificity, quiet-period contrast or temporal-specificity claim is possible |
| Task 7 | `NOT_TRIGGERED`; file-level PR-AUC results are file-level context only | No file-level metric enters this study's endpoint set |

## 2. Decision resolution: retained, research default, deferred, or owner-required

Each row resolves one element of the upstream proposal. Tiers are used exactly as labeled. **No row records an owner decision that was not given.** The calendar-anchor rows are declared research defaults precisely because the metadata contains no wall-clock epoch and no deployment timezone was established; they are not reported as business facts.

| ID | Topic | Tier | Value bound by this protocol | What would change it |
|---|---|---|---|---|
| D1 | Business timezone | RESEARCH-DEFAULT | `UTC` | An owner-confirmed deployment zone, via a new protocol version |
| D2 | Projection anchor | RESEARCH-DEFAULT | `t = 0` ↦ `2024-01-01T00:00:00Z` (arithmetic projection only) | An owner-confirmed real epoch, via a new protocol version |
| D3 | Week start | RESEARCH-DEFAULT | ISO-8601 Monday (Sunday probe declared as sensitivity) | An owner-confirmed shift calendar |
| D4 | Boundary convention | RETAINED | Calendar-aligned primary; rolling declared as sensitivity | A predeclared amendment making rolling primary |
| D5 | Partial periods | RETAINED | Excluded from primary comparisons; published as `PARTIAL-*` | An exposure-sharing estimator that makes truncation harmless |
| D6 | Daily horizon | DEFERRED | Floor unchanged at 30; daily not evaluated; no daily claim | An owner release plus a documented density/floor tradeoff |
| D7 | Normal-variation reference | RETAINED | None available; event-load association only; no specificity | A verified healthy/quiet source, which is a new data decision |
| D8 | Operating point | RETAINED | Descriptive only; no calibrated detector or threshold | Owner-approved calibration support that this corpus lacks |
| D9 | Cluster count for `E2` | RESEARCH-DEFAULT | `K = 8` primary; 4 and 16 as declared sensitivity | A predeclared amendment before outcomes |
| D10 | Exposure-mismatch policy | RESEARCH-DEFAULT | Ratio floor `0.8`; mismatched pairs excluded from the primary aggregate and published with the capped-`n` sensitivity | An exposure-weighted primary estimator, predeclared |
| I1 | Business unit | RETAINED | Fleet only; per-robot and robot×program strata are unsupported (3–9 files per robot-week) | A data density that supports grouped-context statistics |
| I2 | Deployment healthy source | DEFERRED | Not used; requires a separate data decision and release | Owner decision plus a new data authorization |
| M3-1 | Horizon set | RETAINED | Week and month primary; daily deferred without removal from the requirement | Owner re-prioritization |
| M3-2 | Decision rule | RESEARCH-DEFAULT | The frozen triage in §12 | Nothing after outcomes: a revision invalidates the study |
| M3-3 | Endpoint hierarchy | RETAINED | `E1` primary, `E2` secondary, `E3`/`E4` descriptive; reference-pool-only fitting | Nothing per period or per pair |
| M3-4 | Multiplicity | RESEARCH-DEFAULT | Holm over the 4 declared primary contrasts at `alpha = 0.05`, used for auxiliary reported inference only | An owner-approved analysis plan change before outcomes |
| M3-5 | Uncertainty | RETAINED | History-cluster bootstrap, `B = 2000`, seed `2407`, descriptive at `n = 4` | Nothing in this study |
| M3-6 | Inference cost and cap | OWNER-REQUIRED | Zero-new-forward cache reuse required; any incremental row needs explicit approval | Owner approval of incremental rows |
| M3-7 | Scaling release | OWNER-REQUIRED | Separate release; a PROMISING disposition only proposes it | Owner decision |

The three numeric research defaults that most affect the outcome (`E1` magnitude floors, the minimum estimable pairs, and the sensitivity-retention requirement) are **methodological choices of this protocol**, not owner-approved performance targets. The inherited floors (30 files/period, 200 reference files) are unchanged; nothing in this protocol relaxes them.

## 3. Time base and calendar projection

Member time is the accepted per-history relative clock. The projection is

$$ t_{\text{abs}} = \text{anchor} + t_{\text{rel}}, \qquad \text{local} = \mathrm{tz}\,(t_{\text{abs}}) $$

with `anchor = 2024-01-01T00:00:00Z`, `UTC`, Monday week start and a 15th-of-month anchor for the rolling-month sensitivity. Day and week boundaries are local midnights; month boundaries are local first-of-month midnights, so a month is never approximated by a nominal length. The observed span contains no leap February; leap and non-leap behaviour is exercised only in the illustrative calendar examples of the upstream proposal.

Because the anchor is unverifiable from the metadata, the execution records the declared projection and reports a projection sensitivity pass. The primary consequence measured on the accepted span is:

| Cell | Matched pairs (primary → alternate) | Unmatched pairs | Pairs total | Supported periods |
|---|---|---|---|---|
| `week|calendar` | 36 → 39 | 42 → 42 | 128 → 128 | 99 → 101 |
| `week|rolling` | 34 → 34 | 46 → 46 | 124 → 124 | 100 → 100 |
| `month|calendar` | 14 → 18 | 10 → 2 | 32 → 28 | 28 → 29 |
| `month|rolling` | 18 → 9 | 2 → 15 | 28 → 32 | 32 → 32 |

The alternate probe (`2024-03-07T00:00:00Z`, `Etc/GMT-7`, Sunday start) moves up to 9 matched monthly pairs. A sensitivity variant is a *robustness check on the primary disposition*; it can never be promoted to primary, and it can never rescue a failed primary cell.

## 4. Cohort, membership and the frozen label-blind manifests

Cohort: the four accepted Development histories, each on its own clock, never concatenated. Analysis span `(c1, tmax]` per history; `c1` is the inherited 40 % decile edge.

Membership rule (inherited trailing-window convention): a file belongs to the period whose interval satisfies `period_start < end_time <= period_end`, evaluated exactly with no floating tolerance. A file ending exactly on a boundary belongs to the period that ends there. Straddling files are assigned by `end_time` while their observation time inside the period is clipped for exposure.

Deterministic identities, frozen before any outcome:

| Identity | Definition |
|---|---|
| Period | `{history_id}|{horizon}|{mode}|{period_key}` |
| Pair | `{history_id}|{horizon}|{mode}|PAIR|{current_period_key}` |
| Reference | `{history_id}|REFERENCE|end_le_c1` |
| Period membership digest | `sha256` over the sorted member file IDs |
| Pair membership digest | `sha256` over the concatenated previous and current period digests |
| Row digest | `sha256` over the canonical label-blind row |

Every identity digest excludes all post-hoc annotation fields by construction. The manifest builder fails closed on three conditions: a reference file ending after `c1`, a period extending outside `(c1, tmax]` while labeled `FULL`, and a non-adjacent or non-touching pair.

Frozen manifest content on the accepted span (study digest `6f96c2750c15552bcdf25678a51ed7b24a1f7aecdb5914bf5985d3012d1daefd`):

| Cell | Periods (FULL) | Supported periods | Pairs | Matched | Unmatched | Insufficient support | Partial |
|---|---:|---:|---:|---:|---:|---:|---:|
| `week|calendar` | 132 (124) | 99 | 128 | 36 | 42 | 42 | 8 |
| `week|rolling` | 128 (124) | 100 | 124 | 34 | 46 | 40 | 4 |
| `month|calendar` | 36 (28) | 28 | 32 | 14 | 10 | 0 | 8 |
| `month|rolling` | 32 (24) | 32 | 28 | 18 | 2 | 0 | 8 |

- Membership manifests: 328 period rows, 312 pair rows. Digests: periods `e86157b056d292081a1f0007a7e8ae57ce121448fd817c56d7a2dbabde9c2f3d`, pairs `6b848cf6972f25a03078df94a7ae64518801db2272d0d6cac67626304726cf0e`.
- Weekly membership median 39 files, maximum 74; monthly median 173.5, maximum 237.
- Overlap-weighted exposure per convention: week 35.227 file-days (calendar) and 35.223 (rolling); month 35.244 and 35.246. These are observation-time totals, not calendar coverage.
- Straddling eligible files: 7 per weekly convention and 2 / 1 at month. Each is visible in the manifest through its period assignment and clipped exposure.

## 5. Reference custody and causal ordering

- The reference pool is `eligible files with end_time <= c1`, fixed per history before any evaluation period exists. Observed sizes and digests: DEV-01 850 (`dd9c9f98…`), DEV-02 974 (`cd05f55f…`), DEV-03 935 (`ee8f0ec2…`), DEV-04 919 (`be814983…`), all `REFERENCE-OK`. The latest reference row ends at `12,895,130.9 s`, earlier than every history's `c1`.
- Kernel bandwidth, cluster prototypes and per-dimension standardization are fitted on that pool only, once per history and representation seed. Nothing is re-fitted per period, per pair, per convention or per sensitivity variant.
- Ordering is enforced structurally: period and pair membership are constructed and digested first, annotations are joined only afterwards by identity, and the post-hoc burden block is a separate file with its own digest.
- No cross-history pooling for any fitted quantity; no Confirmation contact of any kind; no reuse of the evaluation periods to choose a bandwidth, `K`, threshold or covariate set.

## 6. Endpoint hierarchy and estimator rules

| ID | Endpoint | Role | Definition |
|---|---|---|---|
| `E1` | DIST-MOVE | PRIMARY (decision-relevant) | Unbiased squared MMD between the two periods' file-embedding samples in the original 128-d space; Gaussian kernel with bandwidth = median pairwise distance on the reference pool |
| `E2` | CLUSTER-MOVE | SECONDARY (descriptive) | Total-variation distance between the two periods' cluster-occupancy vectors; prototypes fitted once on the reference pool, `K = 8` (4/16 sensitivity) |
| `E3` | LOC-MOVE | DESCRIPTIVE-ONLY | Squared mean shift standardized by reference per-dimension SD |
| `E4` | DISP-MOVE | DESCRIPTIVE-ONLY | `abs(log(Tr(Sigma_current) / Tr(Sigma_previous)))` |

Only `E1` feeds the decision rule. `E2` is the composition endpoint and is reported alongside; `E3`/`E4` decompose the movement and can never create or block a disposition. That hierarchy is deliberate: `E2` is strongly composition-driven, so promoting it to decision-bearing would confuse "movement" with "mix change", and the `C2` residual criterion already handles composition explicitly.

Estimator rules: the unbiased U-statistic is used so unequal period sizes do not create an `O(1/n)` bias; the biased plug-in value is reported beside it for continuity with G2; equal-file weighting is primary with exposure weighting as a declared sensitivity; every pair row reports `n`, exposure and estimator flags; a period below floor or with non-finite inputs is `UNSUPPORTED` and stays in the denominator.

## 7. Post-hoc burden and the burden relation

Burden is an annotation-only quantity joined after membership:

| Field | Definition | Role |
|---|---|---|
| Covered failure events | failure events with `period_start < failure_time <= period_end` summed over both periods of the pair | primary burden variable |
| Abnormal member fraction | abnormal-labeled members divided by members, both periods | secondary burden variable |
| Covered degradation onsets | episode onsets inside the periods | reported separately, never a second event denominator |
| Reset edges crossed | maintenance-window ends inside the pair span | observable covariate, never burden |

The burden relation is therefore **graded association**, not a case-control contrast: every supported week and month on this corpus carries events. A pair whose attribution class is `UNANNOTATED` is excluded from the burden association and from both control counters; it is never imputed as zero burden and never counted as quiet.

## 8. Exposure and context confound handling

- Exposure is measured in overlap-weighted file-days and compared per pair; the declared comparability ladder is `INSUFFICIENT-REFERENCE` → `PARTIAL-PERIOD` → `INSUFFICIENT-PERIOD-SUPPORT` → `COMPARABLE-UNMATCHED-EXPOSURE` → `COMPARABLE-MATCHED`, applied in that order.
- Only `COMPARABLE-MATCHED` pairs enter the primary aggregate. `COMPARABLE-UNMATCHED-EXPOSURE` pairs are published and enter the declared capped-`n` sensitivity, which reduces both periods to the smaller member count by a deterministic hash order under the inherited seed `2407`.
- Composition is handled by the `C2` residual: within each history and horizon, the movement values are rank-transformed, regressed on the rank-transformed pair-level robot-mix total-variation distance, and the residual is associated with burden. The observed weekly composition drift is large — median pair-level drift `0.4026–0.5479` per history, maximum `0.7419` — while monthly drift is much smaller (`0.1369–0.2081`). This is why the residual association is a co-required criterion for a promising disposition and not a post-hoc explanation.
- Context strata below fleet are unsupported: per-robot weekly counts are 3–9 files, far below any usable distributional support. Any per-robot claim would be fabricated precision.
- A label-blind median split of the mix-drift covariate within history × horizon is reported as a descriptive stratification. The split threshold is computed from observable covariates only and never from movement or burden.

## 9. Method, seed, history and cost matrix — complete coverage

The declared cell space is `horizon × mode × endpoint × representation seed × history`:

$$ 2 \times 2 \times 4 \times 3 \times 4 = 192 \text{ cells} $$

| Axis | Values | Role |
|---|---|---|
| Horizon | `week`, `month` | co-primary business horizons; `day` deferred |
| Mode | `calendar` (primary), `rolling` (sensitivity) | rolling months keep real inter-anchor gaps (28–31 d) and never a nominal month |
| Endpoint | `E1` primary, `E2` secondary, `E3`/`E4` descriptive | only `E1` is decision-relevant |
| Representation seed | `181801`, `181802`, `181803` | accepted B0 file embeddings; identity-selected; **not independent data** |
| History | DEVELOPMENT-01…04 | never pooled for fitting; `n = 4` |

Every one of the 192 cells is published with a status from a closed vocabulary, and no cell is silently dropped. On the accepted span all 192 are accounted for: 24 `ESTIMABLE` (the decision-relevant `E1`/calendar cells: 4 histories × 3 seeds × 2 horizons), 72 `DESCRIPTIVE-SUPPORTED` (the calendar `E2`–`E4` cells, which can never create or block a disposition) and 96 `SENSITIVITY-SUPPORTED` (every rolling-mode cell, computed and reported but never decision-bearing). No cell is `UNSUPPORTED-*` on this span; the vocabulary exists so that a future run with a missing reference pool or an empty comparable set reports the state instead of omitting the cell.

Cost accounting for the full matrix, per the accepted custody envelope:

| Quantity | Value |
|---|---:|
| Causal reference rows per seed | 3,678 |
| Analysis-span rows per seed | 5,076 |
| Rows per seed | 8,754 |
| Rows across three seeds | 26,262 |
| Span-only rows across three seeds (if reference caches are reused) | 15,228 |
| New forwards authorized by this protocol | 0 |

## 10. Supported, unsupported and abstention reporting

- Period states: `SUPPORTED` / `INSUFFICIENT-PERIOD-SUPPORT`, plus `FULL` / `PARTIAL-PRE` / `PARTIAL-POST` / `PARTIAL-BOTH` coverage.
- Pair states: `COMPARABLE-MATCHED`, `COMPARABLE-UNMATCHED-EXPOSURE`, `INSUFFICIENT-PERIOD-SUPPORT`, `PARTIAL-PERIOD`, `INSUFFICIENT-REFERENCE`.
- Contrast states, derived from the metadata on this span:

| Contrast | State | Reason |
|---|---|---|
| `C1` event-load association | `AVAILABLE` | primary burden association |
| `C2` mix-adjusted residual | `AVAILABLE` | co-required confound residual |
| `C3` quiet-period contrast | `UNSUPPORTED-CONTROL-ABSENT` | zero quiet-period pairs among the 202 comparable pairs |
| `C4` temporal specificity | `UNSUPPORTED-CONTROL-ABSENT` | no event-free neighbouring period exists at either horizon |
| `N1` sampling null | `DESCRIPTIVE-ONLY` | G2 crossed its own null in 369/369 windows, so crossing is necessary and never sufficient |
| `N2` half-split band | `DESCRIPTIVE-ONLY` | within-period noise reference, never a decision criterion |
| `N3` annotated-clean control | `UNSUPPORTED-CONTROL-ABSENT` | zero annotated-clean comparable pairs |
| `N3b` quiet-period control | `UNSUPPORTED-CONTROL-ABSENT` | identical availability to `C3` |

- Among the 202 comparable pairs, 0 are annotated-clean, 0 are quiet and 0 are unannotated. Absence of a control is a published abstention, not an implementation gap.
- No floor is relaxed to manufacture comparability, and no horizon is pooled into a coarser one to hide an unsupported comparison.

## 11. Uncertainty, sensitivity and multiplicity

- **Uncertainty unit.** The history is the resampling unit. The paired uncertainty statement is a history-cluster bootstrap over the four histories, `B = 2000`, seed `2407`, reported as descriptive. With `n = 4` clusters the interval is coarse by construction and is never read as a precise standard error.
- **Pair dependence.** Adjacent pairs share a period, so pair-level resampling is not an independent bootstrap. Pairs inside a history are treated as dependent; the within-history statistic is computed once per history rather than aggregated from independent pairs.
- **Seed replication.** The three accepted encoder seeds change the representation but share the same files, so they are run replicates and not independent data. Seed agreement is a stability requirement for a promising disposition, never a second test and never an argument for `n = 12`.
- **Declared sensitivity variants**, all computed from the frozen manifests and frozen embeddings:
  1. rolling boundary convention (tumbling blocks for week, same-day-of-month windows for month);
  2. capped-`n` exposure-matched subsampling applied to exposure-mismatched pairs;
  3. alternate projection anchor with `Etc/GMT-7` and a Sunday week start;
  4. cluster count `K ∈ {4, 16}` for `E2`;
  5. equal-file versus exposure-weighted movement weighting;
  6. biased plug-in MMD versus the unbiased U-statistic.

- **Multiplicity.** Any auxiliary inferential statement is Holm-adjusted over the declared family of four primary contrasts (two horizons × two association endpoints) at `alpha = 0.05`. The disposition does **not** depend on a p-value. The exact rank reference is published beside every result so this cannot be over-read: with no ties the two-sided permutation tail of a perfect rank association is `2/n!`, i.e. `0.333` at three pairs, `0.0833` at four, `0.0167` at five and `0.00278` at six. At the month maximum of five comparable matched pairs per history even a perfect association cannot reach the Holm threshold of `0.0125`, so no month per-history significance claim is possible; week histories with six or more pairs can reach it.

## 12. The decision rule: PROMISING / NOT-PROMISING / INCONCLUSIVE

The rule is a **prospective triage for whether to propose a separately authorized scaling study**. It is not a performance gate, not a detector operating point, and not a significance test. Its inputs are declared outcome summaries; the frozen implementation is `representation.sprint24_weekly_monthly_study.evaluate_decision`.

### 12.1 Inputs per horizon

For each history: the number of comparable matched pairs, the number of distinct burden values among them, the within-history rank association (`rho_burden`, `C1`) and the mix-adjusted residual association (`rho_residual`, `C2`). For the horizon: the same association computed under each declared sensitivity variant. The association is Spearman's rank correlation with average ranks for ties, computed within the history and never pooled across histories.

### 12.2 Estimability gate

A history is *estimable* when it has at least three comparable matched pairs and at least two distinct burden values. Three is chosen as the smallest count at which a rank association has a defined, non-degenerate permutation distribution; it is deliberately not raised to the count that would permit a per-history significant test, because the accepted design supplies at most five comparable matched month pairs per history and a higher floor would predetermine the month disposition regardless of the evidence — an impossible gate rather than a rule. At least three of the four histories must be estimable for any decisive disposition.

### 12.3 Promise criteria (all must hold)

| Criterion | Requirement |
|---|---|
| Magnitude | median `rho_burden ≥ 0.5` **and** median `rho_residual ≥ 0.3` |
| Consistency | the burden association is positive in at least three of the estimable histories |
| Sensitivity | every declared variant keeps the primary sign and retains at least half of the median magnitude |
| Specificity | not applicable and not claimed: no disposition may assert calibrated specificity, detector behaviour or per-file accuracy |

The magnitude floors are research choices justified as follows: a burden rank association below 0.5 explains under a quarter of the rank variance, and the 0.3 residual floor requires a third of the signal to survive composition adjustment — necessary because the observed weekly composition drift reaches 0.74 and G3 measured `r(MMD², TV drift) ≈ +0.78`.

### 12.4 Non-promise criteria (all must hold, with the same evidence burden)

| Criterion | Requirement |
|---|---|
| Magnitude | median `rho_burden ≤ 0` **and** median `rho_residual ≤ 0` |
| Consistency | the burden association is non-positive in at least three of the estimable histories |
| Sensitivity | every declared variant is non-positive |

A negative answer therefore requires the same estimable support, the same cross-history consistency and the same sensitivity stability as a positive one.

### 12.5 Dispositions and precedence

| Disposition | Meaning | Precedence |
|---|---|---|
| `INCONCLUSIVE` | the estimability gate fails, or the promise and non-promise criteria conflict, or a declared sensitivity variant is unknown or sign-flipped | first: an inestimable or conflicting analysis cannot yield a decisive answer |
| `PROMISING` | every promise criterion holds | proposes a separately authorized scaling study; claims no specificity and releases nothing |
| `NOT-PROMISING` | every non-promise criterion holds | the association-level question is answered negatively for this corpus and design |

Every result also reports: the per-criterion status trace, the estimable histories, the medians, the sensitivity values, an explicit `specificity_claim: false` with its reason, the blocked-claims list, the rank-resolution block (including whether the horizon is coarse), and a `scaling_status` that is `PROPOSAL-ONLY-SEPARATE-AUTHORIZATION-REQUIRED` only for a PROMISING outcome and `NOT-RELEASED` otherwise.

### 12.6 What this rule explicitly cannot do

- It cannot claim calibrated specificity, recall, detection delay, false-alarm rate or per-file accuracy: the corpus has zero verified controls at both horizons.
- It cannot convert a sampling-null crossing into evidence: `N1`/`N2` are descriptive only.
- It cannot be revised after outcomes: a post-outcome revision invalidates the study rather than re-deciding it.
- It cannot promote a sensitivity variant to primary or rescue a failed primary cell with a variant that happens to pass.
- It cannot pool histories, seeds or horizons to reach a threshold.

## 13. Worked decision examples (illustrative only)

`artifacts/sprint-24/evidence/s24-7-m3-a01/worked-decision-examples.json` contains eight worked cases exercising every boundary. **Every number in that file is invented to exercise the rule; none is a measurement, a simulation of the corpus, or evidence about it.**

| Case | Input shape | Expected disposition | What it demonstrates |
|---|---|---|---|
| `promise-boundary` | strong burden and residual association, all variants retained | `PROMISING` | the promise boundary, including a variant at exactly half the median magnitude |
| `nonpromise-boundary` | median burden and residual ≤ 0, three of four non-positive, all variants non-positive | `NOT-PROMISING` | the non-promise boundary with full evidence burden |
| `inconclusive-insufficient-pairs` | strongly negative but only one estimable history | `INCONCLUSIVE` | underpowered evidence never yields a negative answer |
| `inconclusive-composition-confound` | raw association clears its floor, residual does not | `INCONCLUSIVE` | the G3 drift confound blocks a promise |
| `inconclusive-sensitivity-flip` | promise-shaped primary, rolling variant reverses sign | `INCONCLUSIVE` | boundary-phase dependence blocks a promise |
| `inconclusive-unverified-sensitivity` | one variant unavailable | `INCONCLUSIVE` | unknown is never treated as a pass |
| `promise-coarse-month-resolution` | month with four to five pairs, strong and stable association | `PROMISING` with `coarse = true` | a coarse horizon can propose scaling but carries no per-history resolution |
| `promise-without-specificity` | maximum-strength synthetic association | `PROMISING`, `specificity_claim = false` | even the best case claims no specificity and releases nothing |

## 14. Cost, cap and representation-coverage prerequisites

- **No inference in this task.** The metadata pass opens no embedding, score, cache, checkpoint, waveform or model and performs no forward. Measured cost of a full metadata pass: about 3–4 s of analysis on this workstation (including the declared projection-sensitivity variant).
- **Reuse-first.** The accepted G2 run already captured the same 8,754 files per seed (3,678 causal reference + 5,076 analysis span) under the same three accepted seeds. Reuse is permitted only when every cache-identity input matches exactly, including device; a mismatch fails closed and forces recomputation.
- **Declared envelope.** 8,754 rows per seed and 26,262 rows across three seeds. The CLI refuses any plan whose per-seed requirement exceeds that envelope.
- **Inherited accounting consequence.** The geometry-time study recorded 26,431 cumulative forwards against a strict 41,000 cap with no reset, leaving 14,569. A fresh span-only capture would need `5,076 × 3 = 15,228` rows and therefore exceeds the remaining headroom by **659 rows**. Zero-new-forward cache reuse is consequently a release prerequisite unless the owner explicitly approves incremental headroom. This protocol authorizes no forward.
- **Representation coverage is unknown, not assumed.** Whether the accepted caches actually cover all 8,754 files is unknown at protocol time; verifying cache identity coverage is a preflight step of the future execution task, and no coverage claim appears in this document.
- **No retraining.** Zero fits, zero trainable parameters, no corpus generation, no architecture search.

## 15. Release prerequisites; scaling is conditional and separate

A later, separately authorized execution task must satisfy all of the following before computing a single movement value:

1. Owner-recorded decisions for the rows still marked `OWNER-REQUIRED` (incremental inference budget, scaling release) and explicit acknowledgement that D1–D3 and D10 are research defaults, not deployment facts.
2. Verification that the frozen study digest and all three label-blind manifest digests recorded here still match the source tree, and that the protocol document digest recorded in `run.json` matches this file.
3. A cache-coverage preflight proving every required row is reusable under an identical cache identity, or an explicit bounded approval for the missing rows within an owner-approved cap.
4. Unchanged Confirmation custody and no retraining, new corpus or Phase C execution.
5. A written execution record that accounts for every declared cell and every declared contrast, including the unsupported states.

A PROMISING disposition then produces only a **proposal**: a scaling study with more histories, denser periods and a verified healthy-control source is the only path to a specificity or operating-point claim. A NOT-PROMISING disposition closes the association-level question for this corpus and design. An INCONCLUSIVE disposition records what additional data or control support would be needed. None of these outcomes authorizes corpus generation, scaled inference, retraining, Confirmation reuse or Phase C execution.

## 16. Limitations

- Metadata only: nothing here measures embedding movement, and no support count is a performance estimate.
- The calendar anchor, timezone, week start and month anchor day are declared research assumptions. Every calendar date in this document is a projection.
- The corpus supplies zero annotated-clean and zero quiet-period pairs at both horizons, so no calibrated specificity exists and none may be claimed by any disposition.
- `n = 4` histories; 3–5 comparable matched monthly pairs per history. Monthly rank associations are coarse (`0.083` smallest attainable two-sided tail at four pairs, `0.0167` at five), so a monthly PROMISING only proposes more density or histories.
- Weekly comparison is strongly composition-confounded (median pair-level drift up to `0.5479` per history), so the residual criterion is load-bearing.
- Six decimal-free comparisons across horizons are not independent (different exposure, overlapping periods); no cross-horizon magnitude statement is supported.
- Per-robot and robot×program strata cannot support distribution statistics at these horizons, so context adjustment is fleet-level only.
- The Development corpus is exposed-benchmark exploratory research; the encoder, normalization, bank and thresholds previously saw Fit/Calibration-healthy rows spanning the full timeline.
- Sprint 18 remains `UNRESOLVED`; original Confirmation remains evaluation-only secondary context.

## 17. Reproduction and evidence inventory

Command (metadata only; digest-verified inputs, accepted Development manifests and the frozen M1 decile edges):

```bash
python experiments/sprint24_task7m3_weekly_monthly.py \
  --data-root <dir containing DEVELOPMENT/<history>/manifest.json> \
  --edges experiments/sprint24_task7m1_dev-edges.json \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-01=ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480747e37b48fe1e08 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-02=cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-03=b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-04=3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af \
  --out artifacts/sprint-24/evidence/s24-7-m3-a01
```

Focused behavioral checks:

```bash
python -m pytest tests/representation/test_sprint24_weekly_monthly_study.py -q
```

Frozen study identity (`S24-BIZ-WM-v1`): `6f96c2750c15552bcdf25678a51ed7b24a1f7aecdb5914bf5985d3012d1daefd`. The CLI writes `study-freeze.json` before any membership or support value and fails closed on a digest mismatch.

Evidence files under `artifacts/sprint-24/evidence/s24-7-m3-a01/`:

- `study-freeze.json` — study identity, decision tiers, endpoint hierarchy, decision policy, cost policy and freeze digest;
- `study-manifest-references.json` — causal reference pools with digests and custody validity;
- `study-manifest-periods.json` / `.csv`, `study-manifest-pairs.json` / `.csv` — the frozen label-blind membership manifests with deterministic identities;
- `posthoc-burden.json` — the annotation-only burden join keyed by those identities;
- `coverage-matrix.json`, `coverage-cells.csv`, `estimability.json`, `estimability.csv` — the 192 declared cells, their statuses, per-history support and the metadata estimability pre-check;
- `projection-sensitivity.json` — the declared alternate-anchor probe restricted to the study's horizons;
- `worked-decision-examples.json` — the illustrative decision-rule cases;
- `cost.json` — envelope, reuse requirement and the inherited-headroom consequence;
- `run.json` — inputs, digests, validity, environment, timing and the explicit statement that no disposition was computed.

This protocol authorizes nothing. Movement computation requires the release prerequisites of §15; scaling requires a separate owner release.
