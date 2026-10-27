# Sprint 24 prospective supplemental controlled-data protocol v1 (design only, Task 7-M7)

- Protocol ID: `S24-CONTROLLED-DATA-v1`.
- Task / attempt / agent: `S24-7-M7` / `S24-7-M7-A01` / `S24M7GoldA01` (role Gold under the owner's standing default-answer instruction; quota `unknown`; full route with an independent evidence review `S24-7-M7-C1` and a separate Bronze exact-snapshot checkpoint).
- Criterion: `docs/sprint-plans/sprint-24.md:182-184` — design the prospective controlled-data protocol for weekly/monthly movement; acceptance items in §12.
- Base commit: `def6281dc36d6d7aa5c540dc0e7bf9f3349deddc` (accepted Task 7-M6 checkpoint).
- Status: **frozen prospective design candidate, pre-implementation.** This document authorizes no generator implementation, no corpus generation, no corpus execution, no inference, no budget, no retraining, no Confirmation contact and no Phase C release.
- Machine-readable design: `src/representation/sprint24_controlled_data_protocol.py`; public entrypoint `experiments/sprint24_task7m7_controlled_data_protocol.py`; focused tests `tests/representation/test_sprint24_controlled_data_protocol.py` (9 passed).
- Preserved unchanged: M4 `INCONCLUSIVE` (both horizons), M5 weekly `LIMITED` / monthly `UNSUPPORTED`, M6 exploratory weekly `LIMITED` (only DEV-04 feasible). No prior result is revised, re-decided or re-audited by this protocol.

## 0. What this document is, and what it is not

This protocol specifies a bounded supplemental synthetic research corpus — new histories, fresh data seeds, a burden-schedule-driven generator plan — so that future weekly/monthly collective movement comparisons can have low-nonzero-burden stable baseline-to-baseline controls and baseline-to-elevated-burden transitions with matched observable operating context and exposure, exact within-file severity/duration provenance, and disjoint temporal support. It answers the M5/M6 missing-data requirements (§1) with a concrete plan rather than a retrospective threshold change.

It is **not**:

- an implementation of any generator planner, scheduler, health process or anomaly injector: those are named prerequisites with explicit gaps (§11);
- a generated corpus, a manifest, a waveform store, an embedding cache or a measured result;
- a detector specification, an accuracy study, a generalization study or a deployment qualification;
- an execution, inference, budget, scaling or Phase C release — including the business 500–1000 files/day density, which is later scale context (§7.6), never this protocol's target;
- a revision of any frozen threshold, floor, seed, identity, disposition or custody rule: every inherited value is re-declared unchanged (§3).

Two interpretation rules from the M3 protocol are binding here unchanged: movement is not an anomaly (a distribution shift describes data; it never evidences a fault), and absence of annotation is not health.

No DEV-04 or old embedding/movement output selected any planned value in this protocol. M4/M5/M6 are exploratory motivation only (§1); every planned identity, band, seed and quota below was frozen before any generation — and no generation exists.

## 1. Prior evidence carried in (no re-derivation, no re-audit)

| Source | Accepted fact used here | Consequence for this design |
|---|---|---|
| M4 results | Weekly burden association leans positive but sits below both promise floors and fails capped-`n` retention; monthly leans negative but rolling flips sign; both horizons `INCONCLUSIVE`; no quiet-period controls at either horizon; median movement of the same order as the within-period half-split band | The old corpus cannot decide the association question; a new corpus with a designed low-burden pole and disjoint controls is the only non-threshold-tuning path to an answer |
| M5 audit | All four histories hold weekly stable low-burden segments under the frozen grid, but no history combines a mix-ok baseline pair with a comparable contrast pair; monthly has no stable segment anywhere; low nonzero burden eligible (all 68 low candidates nonzero); within-file burden unavailable; cross-region mix comparability missing | This protocol plans two disjoint baseline segments plus contrast blocks per history, persists mix per planned comparison, and requires per-file severity/area provenance |
| M6 audit | Deterministic label-blind balancing zeroes mix drift everywhere, but DEV-03's retained contrast collapses 0.124 → 0.059; only one history (DEV-04) holds a disjoint reserve control; no adjacent-week matched comparison survives exposure matching | This protocol plans nonadjacent research pairs (baseline segment versus a distant elevated block sharing neither files nor periods), adds buffer weeks between arms, and plans contrast blocks large enough to survive balancing attrition |
| G3/M4 context | `r(MMD², TV drift) ≈ +0.78`; weekly composition drift reaches 0.74; three encoder seeds are run replicates, never independent data; cumulative forwards 52,789 of a strict 54,000 cap (1,211 left) | Planned arms must match observable robot/program cells; the residual `C2` endpoint stays co-required; seeds stay run replicates; inference needs an explicit new budget (§7) |

## 2. Research questions this corpus is designed to answer — and not answer

A synthetic result on this corpus would establish, at most, that under controlled burden schedules with matched operating context, disjoint support and frozen custody, the E1 movement endpoint separates planned low-burden stable baselines from planned elevated blocks with the declared estimands — and that baseline-to-baseline control pairs sit inside the within-period noise band.

It would **not** establish: calibrated specificity or quiet-period contrast on real data; a detector operating point, recall, delay, false-alarm rate or per-file accuracy; generalization beyond the planned generator plan; deployment readiness; or any business performance threshold. The corpus is synthetic by construction: its burden schedule is imposed, its nuisance envelope is bounded, and its provenance is recorded — none of that transfers to deployment without a separate real-data study.

## 3. Inherited decisions: retained, research default, or owner-required

Tiers are used exactly as in the M3 protocol. **No row records an owner decision that was not given.**

| ID | Topic | Tier | Value bound by this protocol | What would change it |
|---|---|---|---|---|
| D1 | Business timezone | RETAINED | `UTC` (from `S24-BIZ-WM-v1`) | An owner-confirmed deployment zone, via a new protocol version |
| D2 | Projection anchor | RETAINED | `t = 0` ↦ `2024-01-01T00:00:00Z` (arithmetic projection) | An owner-confirmed real epoch, via a new protocol version |
| D3 | Week start | RETAINED | ISO-8601 Monday | An owner-confirmed shift calendar |
| D4 | Boundary convention | RETAINED | Calendar-aligned primary; no rolling sensitivity for the new corpus (M3 rolling answers already exist on the old corpus) | A predeclared amendment before outcomes |
| D5 | Partial periods / warmup | RETAINED | First week after `c1` and first partial month excluded from every arm | An exposure-sharing estimator, predeclared |
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
| O1 | Generator implementation | OWNER-REQUIRED | Separate authorization; §11 gaps must close first | Owner release of a generator readiness review |
| O2 | Corpus generation | OWNER-REQUIRED | Separate authorization after the pre-generation freeze (§9) | Owner generation release |
| O3 | Inference / forwards | OWNER-REQUIRED | Separate authorization with an explicit new forward budget; §7 gate is currently `INFERENCE-NOT-AUTHORIZED` | Owner budget + inference release |
| O4 | Scaling / business density | OWNER-REQUIRED | Separate release; this protocol targets research density only | Owner decision |

Rows N1–N7 are **proposed numerical research defaults** of this protocol: they are declared before any generation, they are not owner-approved operational thresholds or business performance gates, and nothing after outcomes may revise them without a new protocol version.

## 4. Calendar, timezone, boundaries, warmup and reference custody

- Time base: per-history relative clock in seconds; the projection `t_abs = anchor + t_rel`, `local = tz(t_abs)` with `anchor = 2024-01-01T00:00:00Z`, `UTC`, ISO Monday start is inherited unchanged from `S24-BIZ-WM-v1`.
- Membership: a file belongs to the period whose interval satisfies `period_start < end_time <= period_end`, exactly, with no float tolerance; straddling files are assigned by `end_time` with observation time clipped for exposure.
- Month boundary: local first-of-month midnight; a month is never a nominal length. Planned monthly bands aggregate the same planned files as the weekly bands; the pre-generation freeze (§9) verifies on real `c1` values that every planned weekly baseline segment nests inside a LOW month and every planned weekly elevated block nests inside an ELEVATED month.
- Warmup: the first planned week after `c1` (W00) and the first partial planned month belong to no arm and enter no estimand. Analysis span is `(c1, tmax]` per history, never concatenated.
- Reference custody: eligible files with `end_time <= c1`, fixed per history before any evaluation period exists. Bandwidth, cluster prototypes and per-dimension standardization are fitted on that pool once per NEW history and representation seed; nothing is re-fitted per period, per pair or per sensitivity variant; old-history banks are never reused. Reference state below 200 files refuses the history (`REFUSED-SHORTFALL`).

## 5. Planned layout: one 26-week span serving both horizons

Each planned history spans 26 weekly offsets after `c1`: W00 warmup plus 25 analysis weeks. The same planned files form 6 monthly offsets (M01–M06) by `end_time`:

Default week-resolution band plan (offsets from `c1`):

- `W00`: `WARMUP` (no arm).
- `W01–W03`: `BASELINE-A`; `W08–W10`: `BASELINE-B` (two disjoint 3-week stable segments; the control pair is A×B).
- `W05–W06`: `ELEVATED-A`; `W12–W13`: `ELEVATED-B` (two disjoint 2-week contrast blocks).
- All other offsets: `BUFFER` (temporal separation; entry into no estimand).

Default month-resolution band plan: `M01 WARMUP-PARTIAL`, `M02 LOW`, `M03 LOW`, `M04 ELEVATED`, `M05 LOW`, `M06 LOW` — two disjoint LOW runs (`M02–M03`, `M05–M06`) plus elevated contrast, each run satisfying the `≥ 2`-period minimum.

Planned comparable pairs per history per horizon (5: 4 TRANSITION + 1 CONTROL):

- Weekly: BASELINE-A×ELEVATED-A, BASELINE-B×ELEVATED-A, BASELINE-A×ELEVATED-B, BASELINE-B×ELEVATED-B, plus control BASELINE-A×BASELINE-B.
- Monthly: RUN-A×ELEVATED-M4, RUN-B×ELEVATED-M4, RUN-A×ELEVATED-M6, RUN-B×ELEVATED-M6, plus control RUN-A×RUN-B.

Every planned pair is nonadjacent by construction: the two arms share neither files nor periods. Adjacent chronology is never a primary comparison (M6 found no surviving adjacent-week support on the old corpus).

Per planned history this yields 10 comparable pairs (5 weekly + 5 monthly); grid-wide 60 pairs over 6 histories. Pair counts are support accounting, not independent samples: pairs inside a history are dependent, and the within-history statistic is computed once per history.

## 6. Baseline, contrast, context, exposure, severity and duration provenance

- Baseline-to-baseline controls: the two disjoint planned baseline segments per horizon (weekly A×B, monthly RUN-A×RUN-B) are first-class planned pairs with the expectation that their E1 movement lies inside the within-period half-split band (N2) with a near-zero burden gap. A history without two qualifying control segments refuses (`REFUSED-NO-WEEKLY-CONTROL` / `REFUSED-NO-MONTHLY-CONTROL`).
- Baseline-to-elevated transitions: each baseline segment crosses each elevated block, so contrast cannot depend on one lucky pairing; the retained (post-balancing) burden gap must clear `≥ 0.10`, measured after balancing, never assumed.
- Common context and exposure: strata are the observable `(robot_id, program_id)` file cells; deterministic capped-`n` balancing keeps the first `min(n_baseline, n_contrast)` members per common stratum per arm ordered by `sha256('2407:{history}:{region}:{file_id}')` (M6 rule, null seed 2407). Mix drift is recomputed over the same frozen stratum universe before and after balancing. No label enters member selection; no artificial marker, injected tag, or scheduled artifact may distinguish arms — generator-arm provenance lives only in the annotation layer, never in the waveform (§8).
- Nuisance realism versus controlled contrast: nuisance channels (sensor noise, unit offsets, shift/diurnal context, regime structure) must be drawn from the same bounded envelope in every arm; the *only* designed difference between arms is the imposed burden schedule (file fraction plus per-file severity/area). The contrast envelope is bounded (elevated fraction `≤ 0.35`) so the task is a burden association, not an out-of-envelope detection.
- Within-file severity/duration provenance: every planned abnormal file carries per-file `severity ∈ [0, 1]`, an affected-channel mask, and `[start, end)` step provenance — the field the old corpus lacks. File prevalence alone is no longer the burden measure: each planned arm reports the fraction abnormal, the mean severity, the severity histogram, and the mean abnormal-step fraction.
- Dependence, overlap and restricted estimands: comparisons that share an arm are reported as `NONADJACENT-SHARED-SUPPORT` (the same class M6 disclosed for its sensitivity pairs). Common-support restriction changes the population: the primary estimand is the *restricted-common-support* association, reported alongside the full-population band gap so balancing attrition is visible and never hidden. If balancing removes the contrast, the verdict is LIMITED and the contrast is not rescued.

## 7. Endpoints, estimands, uncertainty, multiplicity and decision interpretation

- Endpoint hierarchy (unchanged from M3): `E1` primary; `E2` secondary; `E3`/`E4` descriptive decomposition; `N1`/`N2` descriptive-only; `C1` primary burden association; `C2` co-required mix-adjusted residual. Only `E1` feeds the disposition; `E2` can never create or block one.
- Estimands: the unit is the history. For each history: the within-history Spearman rank association between planned-pair E1 and planned burden (`C1`), and its mix-adjusted residual (`C2`), over the 5 planned comparable pairs per horizon. Encoder seeds are run replicates summarized by median, never independent data. No pooling across histories, seeds or horizons.
- Frozen encoder and reference custody: as-is (D7/N7); any refit needs a separately approved protocol. The frozen encoder is used as-is or not at all.
- Movement endpoints: `E1` unbiased squared MMD with a Gaussian kernel whose bandwidth is the median pairwise distance on the NEW causal reference pool; `E2` total-variation cluster occupancy with `K = 8` (4/16 sensitivity), prototypes on the new pool. Null controls: `N1` reference split-half null (crossing is necessary, never sufficient — G2 crossed 369/369), `N2` within-period half-split band; support abstentions (`INSUFFICIENT-REFERENCE`, `PARTIAL-PERIOD`, `INSUFFICIENT-PERIOD-SUPPORT`) stay in the denominator with published status, never silently dropped.
- Selection and multiplicity: every declared planned pair enters the association; no pair is selected, ranked or dropped by outcome. Auxiliary inferential statements are Holm-adjusted over the 4 declared primary contrasts (2 horizons × C1/C2) at `alpha = 0.05`; the disposition does not depend on a p-value. The exact rank reference is published beside every result: with 5 planned pairs the smallest two-sided permutation tail is `2/5! ≈ 0.0167`.
- Uncertainty: history-cluster bootstrap, `B = 2000`, seed 2407, descriptive at `n = 6`; pairs inside a history are dependent so no pair-level bootstrap is valid.
- Decision interpretation: the prospective triage is `PROMISING` / `NOT-PROMISING` / `INCONCLUSIVE` with the M3 promise/non-promise criteria carried over (median `rho_burden ≥ 0.5` and median `rho_residual ≥ 0.3` with consistency and sensitivity retention for PROMISING; the mirrored non-positive criteria for NOT-PROMISING; `INCONCLUSIVE` otherwise). A PROMISING disposition additionally requires that every promise criterion hold in BOTH the primary triple and the replication triple — replication agreement, not a second test. A PROMISING disposition proposes a separately authorized scaling study; it claims no specificity and releases nothing.
- Scale justification (research lower bound, §N5): 6 histories because each horizon disposition needs at least 3 histories with complete weekly/monthly support (mirroring the M5/M6 three-of-four bar) and the replication triple must independently agree. This is a support/replication lower bound; no statistical power is fabricated and every cross-history statement stays descriptive at `n = 6`.
- Cost formulas, assumptions and the forward gate: planned counts are `files/history = 220 + 25 × 40 = 1220`; total files `6 × 1220 = 7320`; required forwards `7320 × 3 = 21960` (+ 96 smoke = 22056). Against the inherited accounting (spent 52,789 of a strict 54,000 cap, 1,211 left, no reset) the planned inference exceeds headroom by 20,845 forwards: the gate is `INFERENCE-NOT-AUTHORIZED` and any future inference needs an explicit new owner-authorized budget before generation — never a reuse, reset, retry allowance or post-hoc top-up. Storage arithmetic (stated assumptions, not measurements): waveforms `7320 × 400 × 6 × 4 ≈ 70.3 MB`; embeddings `7320 × 128 × 4 × 3 ≈ 11.2 MB`; manifests `7320 × 2048 ≈ 15.0 MB`. No runtime estimate is presented as measured.
- The owner-reported 500–1000 files/day density is later business scale context. This protocol plans ≈ 5.7 files/day research density and must never be read as business-density validation, a scaling release, or evidence that the business density is supportable.

## 8. Identities, history roles, seeds, lineage and custody

- Planned identities are structural only: `{history}|week|calendar|W{k:02d}`, `{history}|month|calendar|M{k:02d}`, `{history}|REFERENCE|end_le_c1`, and nonadjacent research pairs `{history}|{horizon}|calendar|PAIR|{prev}_TO_{current}`. The `PREV_TO_CUR` pair key replaces the M3 single-key form, which would be ambiguous for nonadjacent arms.
- History roles: `S24C-ITER-0001-CONTROLLED-01…03` form the primary triple; `…04…06` the replication triple. No history may reuse, extend, mutate or re-read the immutable old corpus (`S18I-ITER-0005-DEVELOPMENT-01…04`); a colliding identity or the shared `S18I-ITER-0005` prefix refuses (`REFUSED-LINEAGE`), as does any non-frozen identity (`REFUSED-IDENTITY`).
- Seeds: proposed data seeds `42401–42406`, one per planned history. Any seed inside the certified old block (`32064–32079`) or assigned twice refuses (`REFUSED-SEED-COLLISION`). The label-blind member-selection null seed 2407 and the bootstrap seed 2407 are inherited unchanged.
- Lineage and custody: the old corpus (waveforms, manifests, caches, support digests) stays immutable and is never an input to planning, generation or inference. The new corpus carries its own `S24C-ITER-0001` namespace, generator lineage tag, config hash, per-file seeds and per-arm burden-schedule provenance. No label, file, mask, anomaly meta, episode or diagnostic may enter encoder inputs (the `MODEL_INPUT_FIELD_NAMES` / `DIAGNOSTIC_ONLY_FIELD_NAMES` separation is inherited unchanged).

## 9. Pre-generation freeze, structural checks and release prerequisites

Before any future generator implementation or waveform synthesis, the following freeze must be recorded and pass, in order; any failure refuses the release without retries, reseeding, threshold/band rescue or silent rescaling:

1. Design digest: the frozen `S24-CONTROLLED-DATA-v1` digest matches this document and the machine-readable `protocol_spec()`.
2. Real-`c1` nesting: on the actual planned calendar, every weekly baseline segment nests inside a LOW month and every weekly elevated block inside an ELEVATED month.
3. Density verification: the planned 40-file weekly density is reproducible under the burden-schedule planner; if not, the scale is re-planned under a new protocol version, never silently rescaled.
4. Seed/lineage non-collision: proposed data seeds collide with no certified range; no planned identity collides with the old corpus.
5. Role/seed/support matrix: all 60 planned pairs enumerated with their structural status; shortfalls refuse with closed reasons.
6. Budget authorization: an explicit owner-authorized forward budget covering the planned inference exists; the current 1,211-forward headroom does not.

Separate future releases, each needing explicit owner authorization: generator implementation (after §11 gaps close), corpus generation (after this freeze), and inference (after a new forward budget). Digiest persistence rule: the design digest must be persisted BEFORE any future outcome-producing data is created; this task verifies only planned identities, calendar, quotas, cost and consistency.

## 10. Interpretation limits and what a synthetic result would not establish

- A synthetic PROMISING result would mean the movement endpoint tracks the imposed burden schedule under controlled, matched, disjoint conditions in both triples — a warrant to *propose* a separately authorized real-data study, not evidence of real-world sensitivity, specificity, calibration, delay, false-alarm behavior, per-file accuracy, generalization, or deployability.
- A synthetic NOT-PROMISING or INCONCLUSIVE result would mean the endpoint fails to track even an imposed, bounded burden contrast — a reason to stop or redesign, never a claim about real data.
- No synthetic result can claim calibrated specificity, a detector operating point, a business performance threshold, or deployment readiness. The owner approves the protocol design, never a business performance threshold.

## 11. Exact generator feasibility and implementation gaps

Honest accounting: no existing accepted generator path implements this protocol's core requirement — a burden-schedule-driven planner that assigns per-file abnormal/normal outcomes, severity/area provenance and arm membership *before* waveform synthesis while holding nuisance context matched across arms. What exists and what is missing:

| Need | Status | Evidence |
|---|---|---|
| Per-file waveform synthesis with severity/area provenance (`AnomalyMeta` + mask) | AVAILABLE pattern | `synth.generator.SessionGenerator`, `synth.anomalies` families, `FileSample` schema |
| Burden-schedule-driven multi-history planner (fraction bands, stability runs, disjoint arms, buffer weeks, common-support robot/program quotas) | MISSING | No planner exists; the chronicled factory pipeline (`synth.chronicle`, `synth.scheduled`) drives burden from emergent health episodes, not from an imposed band schedule |
| Emergent-episode schedules as a substitute for imposed bands | EXPLICITLY NOT AVAILABLE | Health-episode timing cannot guarantee the `[0.02, 0.10]` / `[0.20, 0.35]` bands, the `≤ 0.05` stability range, or disjoint arm support; using it would silently re-tune the design |
| Within-file duration control at planned granularity | PARTIAL | Family-local duration knobs exist (`stuck_min/max_duration`, `reg_min_duration`) but no planner-level duration band with per-arm provenance |
| Robot/program roster control with matched common support | PARTIAL | `FleetConfig` (5 robots / 8 programs) exists but no planner enforces shared `(robot_id, program_id)` cells across arms; low-cell counts risk silent support failure |
| Lineage-separated multi-history factory calendars with non-colliding seeds | MISSING | No multi-history burden-scheduled calendar builder exists |
| Encoder/frozen-custody path for a new corpus | AVAILABLE pattern | `sprint24_weekly_monthly_execution` admission, cache identity, `gpu-forward` provenance |

No fake executable generator or stub is provided: the machine-readable module in this task freezes the design and its consistency checks only. Any future implementation must close the MISSING rows under a separate authorization and readiness review.

## 12. Plan acceptance coverage

| Acceptance item (`sprint-plans/sprint-24.md:182-184`) | Where this protocol satisfies it |
|---|---|
| Baseline→baseline controls and baseline→elevated transitions | §5 pairs, §6 control expectations |
| Matched observable context/exposure and separate support | §6 strata/capped-`n`/exposure floor; §9 freeze |
| Weekly AND monthly | §5 dual-horizon layout; §9 nesting check |
| Explicit estimands | §7 history-unit associations, no pooling |
| Calendar assumptions | §4 + `calendar_assumptions()` |
| Histories/roles/seeds/period identities | §5, §8 |
| Disjoint support and dependence limits | §5 nonadjacent pairs, §6 shared-support vocabulary |
| Baseline/contrast and context/exposure requirements | N1–N4 bands and quotas (§3), §6 |
| Event/severity/duration provenance | §6 per-file provenance requirement |
| Generator feasibility and exact implementation gaps | §11 table; no fake readiness |
| Lineage separate from immutable old corpus | §8, `validate_lineage` |
| Bounded minimum research scale with justification, not fabricated power | §7 scale justification + cost formulas |
| Forward/storage cost formulas with assumptions | §7 evaluated numbers |
| Known cumulative 52,789 / 54,000 cap (1,211 left), future budget before inference, no reset | §7 gate `INFERENCE-NOT-AUTHORIZED` |
| Frozen encoder/ref-bank as-is or separately approved refit | N7/D7 (§3, §7) |
| Compatible movement endpoints, null controls, support abstentions | §7 hierarchy and abstentions |
| Selection/multiplicity/decision interpretation | §7 no-selection, Holm, triage with dual-triple agreement |
| No DEV-04 or old embedding-based design selection | §0 + design provenance note; bands carried from M5/M6, not tuned to DEV-04 |
| Feasibility gaps explicit, no executable placeholders | §11 MISSING rows; module is design + checks only |
| Distinguish proposed numeric research defaults from owner gates | §3 tiers; §10 protocol-vs-threshold separation |
| Approve protocol, not business performance threshold | §10 |
| Pre-generation freeze and structural checks | §9 |
| Shortfall refusal without retries/reseeding/threshold rescue; separate future releases | §9 + refusal policy |
| Persist design digest before outcome-producing data; verify planned identities/calendar/quotas/cost/consistency | Design-freeze record + §9; CLI evidence |
| Explain what synthetic results would/wouldn't establish; no deployability claim | §2, §10 |

## 13. Reproduction (metadata-only design consistency smoke)

```bash
python experiments/sprint24_task7m7_controlled_data_protocol.py \
  --expected-protocol-sha256 <design digest> \
  --out artifacts/sprint-24/evidence/s24-7-m7-a01
```

The CLI fails closed on a design-digest mismatch, writes `design-freeze.json` before any validation, and records output digests in `run.json`. Planned schedule, quotas, lineage, seed and monthly-support boundary probes discriminate in-memory (overlap, shortfall, collision, cost refusal). Focused tests: `python -m pytest tests/representation/test_sprint24_controlled_data_protocol.py -q` (9 passed). Zero forwards, zero model loads, zero waveform reads, zero embedding/score/cache reads.

## 14. What is not authorized

This protocol authorizes no generator implementation, no corpus generation or execution, no inference, no forward budget, no retraining, no new window or reference lock, no threshold change, no Confirmation access, no Phase C execution and no protocol change. Any follow-on implementation, generation or inference study needs a separately approved prospective amendment, a readiness review, and (for inference) an explicit new forward budget. Until then Phase C stays held with this reason recorded.
