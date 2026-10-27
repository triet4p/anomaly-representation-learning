# Sprint 24 supplemental controlled-data execution protocol v2 (Task 7-M8)

- Protocol ID: `S24-CONTROLLED-DATA-v2`.
- Task / attempt / agent: `S24-7-M8` / `S24-7-M8-A01` / `S24M8GoldA01` (role Gold under the owner's standing default-answer instruction; quota `unknown`; full route with independent evidence `S24-7-M8-C1` and a separate Bronze exact-snapshot checkpoint).
- Criterion: `docs/sprint-plans/sprint-24.md:189-191` — implement controlled-data generator readiness; executable planner entry point, deterministic full planned-metadata readiness plus tiny real waveform smoke; verify burden, duration/severity/channel provenance, roster/exposure comparability, monthly/weekly nesting, reference causality, disjointness, history/seed/lineage custody, shortfall refusal and non-overwrite behavior.
- Base commit: `df28d9f9caec2bd5aa6543a1a28e180580b4fc3a` (accepted Task 7-M7 checkpoint; v1 design digest `a5762fc572eaa861190b486d372724fad0fd4c48f8d5ea599a2db8c6803b1493` preserved as historical, never edited).
- Status: **execution readiness candidate, pre-corpus.** This document authorizes no official corpus generation, no corpus execution, no inference, no retraining, no business-density scale, no Confirmation contact and no Phase C release.

## 0. What this document is, and what it is not

This protocol is the bounded owner-authorized amendment of the frozen M7 design (`S24-CONTROLLED-DATA-v1`) into an executable readiness: the same scientific contract (estimands, thresholds, null controls, lineage custody, refusal policy, interpretation limits) with only the calendar/band layout, planned counts and cost re-frozen for executability — a single shared 30-week span, 7 monthly offsets, a second elevated month, and remapped arm weeks. It binds the burden-schedule planner, the multi-history calendar builder, the duration/roster controls, the generation path, and the machine-checkable execution freeze that must be pinned before any synthesis.

It is **not**:

- an official corpus, a manifest, a waveform store, an embedding cache or a measured result (only a 96-file clearly-marked throwaway smoke subset exists, never an official corpus);
- a detector specification, an accuracy study, a generalization study or a deployment qualification;
- an execution, inference, budget-consuming, scaling or Phase C release — including the business 500–1000 files/day density, which is later scale context, never this protocol's target;
- a revision of any frozen threshold, floor, seed, estimand, identity, disposition or custody rule: every inherited value is re-declared unchanged (§3).

Two interpretation rules from the M3 protocol remain binding unchanged: movement is not an anomaly, and absence of annotation is not health.

No DEV-04 or old embedding/movement output selected any planned value in this protocol. M4/M5/M6 are exploratory motivation only (§1); every planned identity, band, seed and quota below was frozen before any generation.

## 1. Prior evidence carried in (no re-derivation, no re-audit)

| Source | Accepted fact used here | Consequence for this design |
|---|---|---|
| M4 results | Weekly burden association leans positive but fails both promise floors and capped-`n` retention; monthly leans negative but rolling flips sign; both horizons `INCONCLUSIVE`; no quiet-period controls at either horizon; median movement of the same order as the within-period half-split band | The old corpus cannot decide the association question; a new corpus with a designed low-burden pole and disjoint controls is the only non-threshold-tuning path to an answer |
| M5 audit | All four histories hold weekly stable low-burden segments under the frozen grid, but no history combines a mix-ok baseline pair with a comparable contrast pair; monthly has no stable segment anywhere; low nonzero burden eligible (all 68 low candidates nonzero); within-file burden unavailable; cross-region mix comparability missing | This protocol plans two disjoint baseline segments plus contrast blocks per history, persists mix per planned comparison, requires per-file severity/area provenance, and verifies nesting file-level on the real calendar |
| M6 audit | Deterministic label-blind balancing zeroes mix drift everywhere, but DEV-03's retained contrast collapses 0.124 → 0.059; only one history (DEV-04) holds a disjoint reserve control; no adjacent-week matched comparison survives exposure matching | This protocol plans nonadjacent research pairs (baseline segment versus a distant elevated block sharing neither files nor periods), buffer weeks between arms, and contrast blocks large enough to survive balancing attrition; the second elevated month doubles monthly contrast |
| G3/M4 context | `r(MMD², TV drift) ≈ +0.78`; weekly composition drift reaches 0.74; three encoder seeds are run replicates, never independent data; cumulative forwards 52,789 of a former 54,000 cap, now under owner budget authority for this branch | Planned arms must match observable robot/program cells; the residual `C2` endpoint stays co-required; seeds stay run replicates; readiness executes 0 forwards and inference needs a separate execution release |

## 2. Research questions this corpus is designed to answer — and not answer

A synthetic result on this corpus would establish, at most, that under controlled burden schedules with matched operating context, disjoint support and frozen custody, the E1 movement endpoint separates planned low-burden stable baselines from planned elevated blocks with the declared estimands — and that baseline-to-baseline control pairs sit inside the within-period noise band.

It would **not** establish: calibrated specificity or quiet-period contrast on real data; a detector operating point, recall, delay, false-alarm rate or per-file accuracy; generalization beyond the planned generator plan; deployment readiness; or any business performance threshold. The corpus is synthetic by construction: its burden schedule is imposed, its nuisance envelope is bounded, and its provenance is recorded — none of that transfers to deployment without a separate real-data study.

## 3. Inherited decisions: retained, research default, or owner-required

Tiers are used exactly as in the M3 protocol. **No row records an owner decision that was not given**, except the calendar/band amendment rows marked OWNER-AMENDED below.

| ID | Topic | Tier | Value bound by this protocol | What would change it |
|---|---|---|---|---|
| D1 | Business timezone | RETAINED | `UTC` (from `S24-BIZ-WM-v1`) | An owner-confirmed deployment zone, via a new protocol version |
| D2 | Projection anchor | RETAINED | `t = 0` ↦ `2024-01-01T00:00:00Z` (arithmetic projection) | An owner-confirmed real epoch, via a new protocol version |
| D3 | Week start | RETAINED | ISO-8601 Monday | An owner-confirmed shift calendar |
| D4 | Boundary convention | RETAINED | Calendar-aligned primary; no rolling sensitivity for the new corpus (M3 rolling answers already exist on the old corpus) | A predeclared amendment before outcomes |
| D5 | Partial periods / warmup | RETAINED | First week after `c1` (W00) and first partial month excluded from every arm | An exposure-sharing estimator, predeclared |
| D6 | Daily horizon | RETAINED-DEFERRED | Floor unchanged at 30; daily not planned; no daily claim | An owner release plus a density/floor tradeoff |
| D7 | Healthy/quiet source | RETAINED | None available; event-load association only; no specificity | A verified healthy/quiet source, a new data decision |
| D8 | Operating point | RETAINED | Descriptive only; no calibrated detector or threshold | Owner-approved calibration support this corpus lacks |
| D9 | Cluster count for `E2` | RESEARCH-DEFAULT | `K = 8` primary; 4 and 16 as sensitivity | A predeclared amendment before outcomes |
| D10 | Exposure-mismatch policy | RETAINED | Ratio floor `0.8`; mismatched pairs excluded from the primary aggregate with capped-`n` sensitivity | An exposure-weighted primary estimator, predeclared |
| N1 | Low-burden band | RESEARCH-DEFAULT | Abnormal fraction in `[0.02, 0.10]` inclusive for planned baseline scheduling; strictly positive (no zero-fault weeks) | A predeclared amendment before generation |
| N2 | Elevated-burden band | RESEARCH-DEFAULT | Abnormal fraction in `[0.20, 0.35]` inclusive for planned elevated scheduling | A predeclared amendment before generation |
| N3 | Stability band | RESEARCH-DEFAULT | Max fraction range within a run `≤ 0.05` over `≥ 2` consecutive low periods (M5 values) | A predeclared amendment before generation |
| N4 | Common-support quota | RESEARCH-DEFAULT | `≥ 4` shared observable `(robot_id, program_id)` cells per comparison; `≥ 12` retained files per arm; retained gap `≥ 0.10` (above the M6 2-strata / 10-files bar) | A predeclared amendment before generation |
| N5 | Research scale | RESEARCH-DEFAULT | 6 independent histories: a 3-history primary triple plus a 3-history replication triple (lower bound justified in §7.7, not a power claim) | A predeclared amendment before generation |
| N6 | Fresh data seeds | RESEARCH-DEFAULT | Proposed seeds `42401–42406`, one per planned history (verified disjoint from the certified old block at the pre-generation freeze) | A predeclared amendment before generation |
| N7 | Frozen encoder custody | RETAINED | Encoder frozen as-is; accepted B0 seeds `181801/181802/181803` as run replicates; bandwidth/prototypes/standardization fitted once per NEW history on its own causal reference pool | A separately approved refit protocol |
| O1 | Generator implementation | OWNER-AMENDED | v2 calendar/band layout authorized under the owner amendment (`docs/sprint-plans/sprint-24.md:192`); M7 v1 bytes/digest preserved as historical; no other threshold/estimand/lineage change | A further owner release |
| O2 | Corpus generation | OWNER-REQUIRED | Separate authorization after the pre-generation freeze (§9) | Owner generation release |
| O3 | Inference / forwards | OWNER-REQUIRED | Separate authorization; §7 gate is `READY-NO-FORWARDS-EXECUTED` in readiness (0 forwards execute here) | Owner execution release |
| O4 | Scaling / business density | OWNER-REQUIRED | Separate release; this protocol targets research density only | Owner decision |

Rows N1–N7 remain **proposed numerical research defaults**: declared before any generation, not owner-approved operational thresholds or business performance gates, and nothing after outcomes may revise them without a new protocol version.

## 4. Calendar, timezone, boundaries, warmup and reference custody

- Time base: per-history relative clock in seconds; the projection `t_abs = anchor + t_rel`, `local = tz(t_abs)` with `anchor = 2024-01-01T00:00:00Z`, `UTC`, ISO Monday start inherited unchanged from `S24-BIZ-WM-v1`.
- Membership: a file belongs to the period whose interval satisfies `period_start < end_time <= period_end`, exactly, with no float tolerance; straddling files are assigned by `end_time` with observation time clipped for exposure.
- Month boundary: local first-of-month midnight; a month is never a nominal length. The builder verifies on the real `c1` calendar that every planned weekly arm segment nests file-level inside a same-band month (`validate_week_month_nesting`).
- Warmup: the first planned week after `c1` (W00) and the first partial planned month (M01) belong to no arm and enter no estimand. Analysis span is `(c1, tmax]` per history, never concatenated.
- Reference custody: eligible files with `end_time <= c1`, fixed per history before any evaluation period exists. Bandwidth, cluster prototypes and per-dimension standardization are fitted on that pool once per NEW history and representation seed; nothing is re-fitted per period, per pair or per sensitivity variant; old-history banks are never reused. Reference state below 200 files refuses the history (`REFUSED-SHORTFALL`).

## 5. Planned layout: one 30-week span serving both horizons

Each planned history spans 30 weekly offsets after `c1`: W00 warmup plus 29 analysis weeks. The same planned files form 7 monthly offsets (M01–M07) by `end_time`:

Default week-resolution band plan (offsets from `c1`):

- `W00`: `WARMUP` (no arm).
- `W05–W07`: `BASELINE-A`; `W18–W20`: `BASELINE-B` (two disjoint 3-week stable segments; the control pair is A×B).
- `W13–W14`: `ELEVATED-A`; `W26–W27`: `ELEVATED-B` (two disjoint 2-week contrast blocks).
- All other offsets: `BUFFER` (temporal separation; entry into no estimand). Minimum baseline-to-elevated offset gap is 4 weeks.

Default month-resolution band plan: `M01 WARMUP-PARTIAL`, `M02 LOW`, `M03 LOW`, `M04 ELEVATED`, `M05 LOW`, `M06 LOW`, `M07 ELEVATED` — two disjoint LOW runs (`M02–M03`, `M05–M06`, each satisfying the `≥ 2`-period minimum) plus two elevated contrast months, remapped so every arm week nests inside a same-band month on the real calendar.

Planned comparable pairs per history per horizon (5: 4 TRANSITION + 1 CONTROL):

- Weekly: BASELINE-A×ELEVATED-A, BASELINE-B×ELEVATED-A, BASELINE-A×ELEVATED-B, BASELINE-B×ELEVATED-B, plus control BASELINE-A×BASELINE-B.
- Monthly: RUN-A×ELEVATED-M4, RUN-B×ELEVATED-M4, RUN-A×ELEVATED-M7, RUN-B×ELEVATED-M7, plus control RUN-A×RUN-B.

Every planned pair is nonadjacent by construction: the two arms share neither files nor periods. Adjacent chronology is never a primary comparison (M6 found no surviving adjacent-week support on the old corpus).

Per planned history this yields 10 comparable pairs (5 weekly + 5 monthly); grid-wide 60 pairs over 6 histories. Pair counts are support accounting, not independent samples: pairs inside a history are dependent, and the within-history statistic is computed once per history.

## 6. Baseline, contrast, context, exposure, severity and duration provenance

- Baseline-to-baseline controls: the two disjoint planned baseline segments per horizon (weekly A×B, monthly RUN-A×RUN-B) are first-class planned pairs with the expectation that their E1 movement lies inside the within-period half-split band with a near-zero burden gap. A history without two qualifying control segments refuses (`REFUSED-NO-WEEKLY-CONTROL` / `REFUSED-NO-MONTHLY-CONTROL`).
- Baseline-to-elevated transitions: each baseline segment crosses each elevated block, so contrast cannot depend on one lucky pairing; the retained (post-balancing) burden gap must clear `≥ 0.10`, measured after balancing, never assumed.
- Common context and exposure: strata are the observable `(robot_id, program_id)` file cells; deterministic capped-`n` balancing keeps the first `min(n_baseline, n_contrast)` members per common stratum per arm ordered by `sha256('2407:{history}:{region}:{file_id}')` (M6 rule, null seed 2407). Mix drift is recomputed over the same frozen stratum universe before and after balancing. No label enters member selection; no artificial marker, injected tag, or scheduled artifact may distinguish arms.
- Nuisance realism versus controlled contrast: nuisance channels (sensor noise, unit offsets, shift/diurnal context, regime structure) are drawn from the same bounded envelope in every arm; the *only* designed difference between arms is the imposed burden schedule (file fraction plus per-file severity/area). The contrast envelope is bounded (elevated fraction `≤ 0.35`) so the task is a burden association, not an out-of-envelope detection.
- Within-file severity/duration provenance: every planned abnormal file carries per-file `severity ∈ [0.35, 0.85]`, an affected-channel mask, and `[start, end)` step provenance — the field the old corpus lacks. File prevalence alone is no longer the burden measure: each planned arm reports the fraction abnormal, the mean severity, the severity histogram, and the mean abnormal-step fraction. Planned abnormal outcomes are generated through the real injector that is recorded; when the inherited strength gate legitimately rejects the primary family on the exact planned seed, the planned deterministic fallback order is tried on the SAME seed and the realized injector (never a copied request) feeds provenance and trace annotation.
- Dependence, overlap and restricted estimands: comparisons that share an arm are reported as `NONADJACENT-SHARED-SUPPORT` (the same class M6 disclosed for its sensitivity pairs). Common-support restriction changes the population: the primary estimand is the *restricted-common-support* association, reported alongside the full-population band gap so balancing attrition is visible and never hidden. If balancing removes the contrast, the verdict is LIMITED and the contrast is not rescued.

## 7. Endpoints, estimands, uncertainty, multiplicity and decision interpretation

- Endpoint hierarchy (unchanged from M3): `E1` primary; `E2` secondary; `E3`/`E4` descriptive decomposition; `N1`/`N2` descriptive-only; `C1` primary burden association; `C2` co-required mix-adjusted residual. Only `E1` feeds the disposition; `E2` can never create or block one.
- Estimands: the unit is the history. For each history: the within-history Spearman rank association between planned-pair E1 and planned burden (`C1`), and its mix-adjusted residual (`C2`), over the 5 planned comparable pairs per horizon. Encoder seeds are run replicates summarized by median, never independent data. No pooling across histories, seeds or horizons.
- Frozen encoder and reference custody: as-is (D7/N7); any refit needs a separately approved protocol. The frozen encoder is used as-is or not at all.
- Movement endpoints: `E1` unbiased squared MMD with a Gaussian kernel whose bandwidth is the median pairwise distance on the NEW causal reference pool; `E2` total-variation cluster occupancy with `K = 8` (4/16 sensitivity), prototypes on the new pool. Null controls: `N1` reference split-half null (crossing is necessary, never sufficient — G2 crossed 369/369); `N2` within-period half-split band; support abstentions (`INSUFFICIENT-REFERENCE`, `PARTIAL-PERIOD`, `INSUFFICIENT-PERIOD-SUPPORT`) stay in the denominator.
- Selection and multiplicity: every declared planned pair enters the association; no pair is selected, ranked or dropped by outcome. Auxiliary inferential statements are Holm-adjusted over the 4 declared primary contrasts (2 horizons × C1/C2) at `alpha = 0.05`; the disposition does not depend on a p-value. The exact rank reference is published beside every result: with 5 planned pairs the smallest two-sided permutation tail is `2/5! ≈ 0.0167`.
- Uncertainty: history-cluster bootstrap, `B = 2000`, seed 2407, descriptive at `n = 6`; pairs inside a history are dependent so no pair-level bootstrap is valid.
- Decision interpretation: the prospective triage is `PROMISING` / `NOT-PROMISING` / `INCONCLUSIVE` with the M3 promise/non-promise criteria carried over (median `rho_burden ≥ 0.5` and median `rho_residual ≥ 0.3` with consistency and sensitivity retention for PROMISING; the mirrored non-positive criteria for NOT-PROMISING; `INCONCLUSIVE` otherwise). A PROMISING disposition additionally requires that every promise criterion hold in BOTH the primary triple and the replication triple — replication agreement, never pooled significance.
- Scale justification (research lower bound, §N5): 6 histories because each horizon disposition needs at least 3 histories with complete weekly/monthly support (mirroring the M5/M6 three-of-four bar) and the replication triple must independently agree. This is a support/replication lower bound; no statistical power is fabricated and every cross-history statement stays descriptive at `n = 6`.
- Cost formulas, assumptions and the forward gate: planned counts are `files/history = 220 + 29 × 40 = 1380`; total files `6 × 1380 = 8280`; estimated forwards `8280 × 3 = 24840` (+ 96 bounded smoke envelope: 24936 estimated including smoke). Readiness executes 0 forwards, loads 0 models, reads 0 waveforms/embeddings; the gate is `READY-NO-FORWARDS-EXECUTED` and official inference needs the separate execution release under standing owner budget authority. Storage assumes mean 400 steps/file (inside the RegimeConfig 200–800 range, stated not measured).
- The owner-reported 500–1000 files/day density is later business scale context. This protocol plans ≈ 5.7 files/day research density and must never be read as business-density validation, a scaling release, or evidence that the business density is supportable.

## 8. Identities, history roles, seeds, lineage and custody

- Planned identities are structural only: `{history}|week|calendar|W{k:02d}`, `{history}|month|calendar|M{k:02d}`, `{history}|REFERENCE|end_le_c1`, and nonadjacent research pairs `{history}|{horizon}|calendar|PAIR|{prev}_TO_{current}`. The `PREV_TO_CUR` pair key replaces the M3 single-key form, which would be ambiguous for nonadjacent arms.
- History roles: `S24C-ITER-0001-CONTROLLED-01…03` form the primary triple; `…04…06` the replication triple. No history may reuse, extend, mutate or re-read the immutable old corpus (`S18I-ITER-0005-DEVELOPMENT-01…04`); a colliding identity or the shared `S18I-ITER-0005` prefix refuses (`REFUSED-LINEAGE`), as does any non-frozen identity (`REFUSED-IDENTITY`).
- Seeds: proposed data seeds `42401–42406`, one per planned history. Any seed inside the certified old block (`32064–32079`) or assigned twice refuses (`REFUSED-SEED-COLLISION`). The label-blind member-selection null seed 2407 and the bootstrap seed 2407 are inherited unchanged. Slot seeds derive deterministically per `(history_seed, stream, week_key, slot)` and are verified unique and outside the certified old block at plan time.
- Lineage and custody: the old corpus (waveforms, manifests, caches, support digests) stays immutable and is never an input to planning, generation or inference. The new corpus carries its own `S24C-ITER-0001` namespace, generator lineage tag, config hash, per-file seeds and per-arm burden-schedule provenance. No label, file, mask, anomaly meta, episode or diagnostic may enter encoder inputs (the `MODEL_INPUT_FIELD_NAMES` / `DIAGNOSTIC_ONLY_FIELD_NAMES` separation is inherited unchanged and verified per generated file).

## 9. Pre-generation freeze, structural checks and release prerequisites

Before any official supplemental-corpus generation, the following freeze must be recorded and pass, in order; any failure refuses the release without retries, reseeding, threshold/band rescue or silent rescaling:

1. Design digest: the frozen `S24-CONTROLLED-DATA-v1` digest matches the M7 document and the machine-readable `protocol_spec()` (`a5762fc5…6803b1493`, verified as the execution base).
2. Execution digest: the v2 `execution_freeze()` digest pins the remapped calendar, burden quotas, roster controls and cost; it is written to disk BEFORE any synthesis.
3. Real-`c1` nesting: on the actual planned calendar, every weekly arm segment nests file-level inside a same-band month (verified at `c1 = 0`, 30 weeks / 7 months enumerated on the real `PeriodGrid`).
4. Density verification: the planned 40-file weekly density is reproducible under the burden-schedule planner with per-week quotas exact (verified: 40/40 every analysis week, all histories).
5. Seed/lineage non-collision: proposed data seeds collide with no certified range; no planned identity collides with the old corpus; slot seeds are unique and outside the certified block.
6. Role/seed/support matrix: all 60 planned pairs enumerated with their structural status; roster support verified per pair (common cells `≥ 4`, retained files/arm `≥ 12`); shortfalls refuse with closed reasons.
7. Forward authorization: official inference needs the separate execution release; readiness executes 0 forwards (estimated 24,840 + 96 smoke envelope under standing budget authority).

Separate future releases, each needing explicit owner authorization: corpus generation (after this freeze), and inference (execution release). Digest persistence rule: the execution digest must be persisted BEFORE any future outcome-producing data is created; readiness verifies only planned identities, calendar, quotas, cost and consistency plus the bounded throwaway smoke.

## 10. Interpretation limits and what a synthetic result would not establish

- A synthetic PROMISING result would mean the movement endpoint tracks the imposed burden schedule under controlled, matched, disjoint conditions in both triples — a warrant to *propose* a separately authorized real-data study, not evidence of real-world sensitivity, specificity, calibration, delay, false-alarm behavior, per-file accuracy, generalization, or deployability.
- A synthetic NOT-PROMISING or INCONCLUSIVE result would mean the endpoint fails to track even an imposed, bounded burden contrast — a reason to stop or redesign, never a claim about real data.
- No synthetic result can claim calibrated specificity, a detector operating point, a business performance threshold, or deployment readiness. The owner approves the protocol design, never a business performance threshold.

## 11. Exact generator implementation and readiness evidence

Honest accounting: the M7 gap table named the burden-schedule planner, the multi-history calendar builder and planner-level duration/roster control as MISSING. This task closes exactly those rows — reusing the accepted `SessionGenerator` / `AnomalyFamily` / `FileSample` machinery, never a second generator:

| Need | Status | Evidence |
|---|---|---|
| Per-file waveform synthesis with severity/area provenance (`AnomalyMeta` + mask) | AVAILABLE, reused | `synth.generator.SessionGenerator`, `synth.anomalies` families, `FileSample` schema — called directly by the v2 generation path |
| Burden-schedule-driven multi-history planner (fraction bands, stability runs, disjoint arms, buffer weeks, common-support robot/program quotas) | IMPLEMENTED in readiness | `plan_history()` with deterministic quota selection (`slot_order_key`), per-band quota validation, shared-support roster verification; module `src/representation/sprint24_controlled_data_execution.py` |
| Emergent-episode schedules as a substitute for imposed bands | EXPLICITLY NOT AVAILABLE (unchanged) | Health-episode timing still cannot guarantee the bands; the planner imposes the schedule before synthesis |
| Within-file duration control at planned granularity | IMPLEMENTED via realized provenance | Planned severity band `[0.35, 0.85]` + deterministic rotation over all 9 hard families; realized `[start, end)`/channel provenance asserted mask-exact per file (`trace_annotation`) |
| Robot/program roster control with matched common support | IMPLEMENTED via prediction + verification | `predict_normal_roster` / `predict_anomaly_roster` replicate the real derivation exactly; every planned pair verified for common cells and retained depth |
| Lineage-separated multi-history factory calendars with non-colliding seeds | IMPLEMENTED | `build_history_calendar()` on the real `PeriodGrid`; per-history `c1 = 0` shared projection; slot-seed uniqueness and old-block exclusion verified |
| Encoder/frozen-custody path for a new corpus | AVAILABLE pattern (not executed) | Admission/cache-identity pattern from `sprint24_weekly_monthly_execution`; readiness executes 0 forwards and builds no bank |
| `missing_event` effect-size control (R2 root fix) | REPAIRED in implementation | Donor fill now level/scale-matched to the local original plus severity-blended (previously mean-only; severity never entered amplitude, so some bases exceeded the inherited strength gate at every severity). Wrong-regime donor pattern preserved |
| `freq_phase_mismatch` signature preservation (R2 root fix) | REPAIRED in implementation | Pointwise envelope renormalization (which divided out the phase/frequency signature) replaced by per-channel RMS match; blend weight restored to full severity |
| 4 semantically impossible `missing_event` slots (R2 owner amendment) | REASSIGNED prospectively | Edge-only 2-regime sessions have no interior event to miss; reassigned to `subtle_drift` at same seed/severity/week/arm/quota with original request retained (`requested_family_original` + `reassignment_reason`); single-shot `REFUSED-GENERATION` still applies to any other rejection |

No fake executable generator or stub is provided: the machine-readable v2 module freezes the execution plan and its consistency checks, and the readiness CLI proves the path with a bounded throwaway smoke.

## 12. Plan acceptance coverage

| Acceptance item (`sprint-plans/sprint-24.md:189-191`) | Where this protocol satisfies it |
|---|---|
| Executable public generator/planner entry point | `experiments/sprint24_task7m8_controlled_data_readiness.py` (`dryplan` + `readiness` through one generation path) |
| Deterministic full planned-metadata readiness run | `execution_freeze()` pinned before synthesis; 6 histories × 1380 slots = 8280 planned entries; determinism verified by repeated runs |
| Tiny real waveform-generation smoke through the same path | 96-file throwaway smoke (`SMOKE-THROWAWAY`, never official corpus) via `generate_planned_file` on planned slots; realized annotations verified mask-exact; no separate small-mode semantics |
| Baseline/elevated burden verification | Per-week band quotas validated at plan time; smoke subset spans BASELINE-A and ELEVATED-A with observed contrast |
| Within-file duration/severity/channel provenance | `trace_annotation` recomputed from the realized mask/meta per file; duration steps, severity, channels asserted exact |
| Exact robot/program/exposure comparability | Predicted rosters verified against realized per file; common-cell support verified per planned pair |
| Monthly/weekly actual nesting | File-level nesting verified on the real calendar (`validate_week_month_nesting`); freeze pins the enumerated 30-week/7-month grids |
| Reference causality | All reference end_times `<= c1`; reference quota 220/history against the 200 floor |
| Disjointness | Arm week-sets disjoint with buffer separation; pairs nonadjacent (no shared files or periods) |
| History/seed/lineage custody | v1 lineage/seed gates reused; slot seeds unique and outside the certified block; old corpus never an input |
| Shortfall refusal and non-overwrite behavior | `REFUSED-*` fail-fast with no rescue; existing output directories and existing waveforms refuse |
| No mocked success, placeholders, silent retries/reseeding, label leakage or threshold rescue | Real synthesis only; single-shot generation with fail-fast `REFUSED-GENERATION` (no retry/reseed/fallback); encoder-input audit per file |
| Actual supported/unsupported criteria and machine-checkable execution freeze | `execution-freeze.json` + `planned-slots.json` + `cost.json` + `run.json`; readiness vs official corpus distinguished |
| Specific behavioral contract tests | `tests/representation/test_sprint24_controlled_data_execution.py` (12 passed) plus v1/synth regression scope |

## 13. Reproduction (readiness CLI)

```bash
python experiments/sprint24_task7m8_controlled_data_readiness.py \
  --mode readiness \
  --expected-base-protocol-sha256 a5762fc572eaa861190b486d372724fad0fd4c48f8d5ea599a2db8c6803b1493 \
  --out artifacts/sprint-24/evidence/s24-7-m8-a01
```

The CLI pins `execution-freeze.json` before any synthesis, then runs the full planned-metadata validation (6 histories, 8280 slots, 60 pairs, cost) and the 96-file throwaway smoke through the same generation path. `--mode dryplan` runs metadata only (0 waveforms). Output digests land in `run.json`; the smoke manifest carries per-file realized annotations, waveform digests and the throwaway marker. Focused tests: `python -m pytest tests/representation/test_sprint24_controlled_data_execution.py -q` (10 passed). Readiness executes 0 forwards, 0 model loads, 0 waveform reads, 0 embedding/score/cache reads; it writes 96 throwaway waveforms explicitly marked not-official-corpus.

## 14. What is not authorized

This protocol authorizes no official corpus generation or execution, no inference, no forward budget consumption, no retraining, no new window or reference lock, no threshold change, no Confirmation access, no Phase C execution and no protocol change beyond the stated v2 calendar/band amendment. Any follow-on generation or inference study needs a separately approved prospective amendment, a readiness review with PASS, a Bronze exact-snapshot checkpoint, and (for inference) the explicit execution release. Until then Phase C stays held with this reason recorded.
