# Sprint 24 Task 7-M13 — Controlled dose-response and sample-size diagnostic on the existing embedding cache

- Task / attempts / agents: `S24-7-M13` / `S24-7-M13-A01` (`S24M13GoldA01`) produced the R1/R2 candidates; the owner paused the task and the authorized R3 candidate was completed by the replacement `S24-7-M13-A02` (`S24M13GoldA02`). Both worker records are preserved: `artifacts/sprint-24/task-7-m13-a01.md` (R1/R2 plus the pause handoff) and `artifacts/sprint-24/task-7-m13-a02.md` (R3). A third record, `artifacts/sprint-24/task-7-m13-a03.md`, covers the owner-authorized documentation-correction cycle `S24-7-M13-C2`/R1 that produced this version after the C1/R3 evidence FAIL; it corrects overstated sensitivity and nesting wording and records the post-readiness digest chronology without changing any measured value. Review status: C1/R1 and C1/R2 `changes_requested`, C1/R3 **FAIL** (`advance:false`, `owner_decision_required:true`); no assessment has passed and no commit, checkpoint or push exists. Role: Gold.
- Criterion: `docs/sprint-plans/sprint-24.md:232-238`. No quantitative success gate is assigned; a faithful negative or limited result satisfies execution acceptance.
- Frozen pre-outcome specification: `docs/sprint24-controlled-dose-response-specification-a01.json` (`S24-CONTROLLED-DOSE-RESPONSE-SPEC-A01`, sha256 `8a3681fc84e6187c5979f95172ab5e87773251efcca4f650c9cf7fbe2338ebe4`), written after the metadata-only census and before any M13 outcome. Its bytes are preserved verbatim under `artifacts/sprint-24/evidence/s24-7-m13-a01/spec/` with a custody record and were never edited. Four separately timestamped amendment records exist under `artifacts/sprint-24/evidence/s24-7-m13-a01/amendments/`: `A01`–`A03` (discovered after the R1 assessment) and `A04` (the R3 comparability/sensitivity results). One documentation erratum, `artifacts/sprint-24/evidence/s24-7-m13-a01/amendments/A04-ERRATUM-01.md`, corrects an overbroad sentence inside `A04` without editing `A04`'s bytes; it is not a specification amendment and is not loaded by the engine (§13).
- Metadata-only feasibility census: `artifacts/sprint-24/evidence/s24-7-m13-a01/census/census.json` (sha256 `a7a0c7553d3ed38f5a0cecc6c5915121160be7a3becdbc803328483b4329f574`), `metadata_only=true`, `embeddings_read=false`, `outcomes_computed=false`.
- Cache: the admitted `S24-7-M11` embeddings retrieved read-only for M12 and preserved under `artifacts/sprint-24/evidence/s24-7-m12-a01/cache-full` (36 files, digest-of-digests `ae7aa42d7661d95d9d292400f53f4ec76128fd9618acc75cc6c6fcf581aa02b3`). **0 new forwards; cumulative observed forwards remain 77,653.**
- Worker records: `artifacts/sprint-24/task-7-m13-a01.md` (A01) and `artifacts/sprint-24/task-7-m13-a02.md` (A02 replacement, R3). Evidence: `artifacts/sprint-24/evidence/s24-7-m13-a01/` (`census/`, `audit/` and `audit-repro/` = the R3 candidate, `audit-r1/`, `audit-repro-r1/`, `audit-r2/`, `audit-repro-r2/`, `smoke*`, `pilot/`, `spec/`, `amendments/`, `independent-verification.{py,json}`).
- This is **not** a revision of `S24-7-M11` or `S24-7-M12`, not independent confirmation of either, and not a temporal, business, performance or detection claim.

## 1. What was built and verified to the fail-closed boundary

- `src/representation/sprint24_controlled_dose_response.py` — the cache-only dose-response engine. It verifies the frozen specification digest, every pinned corpus/audit/preflight/freeze/contract digest, the canonical contract binding and every accepted M11/M12 product digest **before** any statistic; admits the cache through the accepted identity-strict public admission path; fits one reference per `(history, encoder seed)` on the causal `REFERENCE` pool only; builds the metadata census and refuses any drift from the recorded census statuses; constructs the frozen composition frame, abnormal sequence and fixed baseline arm; draws dose arms with duplicate-free, disjoint identity invariants; and measures E1/E2 for all seeds, K and doses.
- `experiments/sprint24_task7m13_controlled_dose_response.py` — the public entry point with `--stage census` (manifest + M9 audit only, no cache, no embeddings) and `--stage audit` (full cache-only path). It refuses a non-empty `--out`, records the exact table digests, and re-verifies cache and prior-product bytes after the run.
- `tests/representation/test_sprint24_controlled_dose_response.py` — 20 focused checks: capped largest-remainder apportionment (including the case where a naive allocator over-allocates), two-disjoint-arm capacity, family-deficit prefix proportionality and nesting, exact plan totals and context counts, duplicate-free and baseline-disjoint draws, both abstention branches, E1 equality with the accepted unbiased MMD, exact integer E2 quanta, movement-vs-null semantics, tie-aware Spearman, specification-byte refusal, foreign-amendment refusal, reference-arm exclusion and period selection, metadata-only census flags, the abnormal-headroom refusal, realized composition from the actual draws, realized non-nesting/overlap, the abnormal-only quantum/support rule, and the nested-prefix arm invariants (context quota, abnormal nesting, anchor identity).
- Refusal boundaries exercised on real inputs: a tampered specification copy raises `REFUSED-SPEC-DRIFT`; a foreign amendment digest raises `REFUSED-AMENDMENT`; a drifting census status raises `REFUSED-CENSUS-DRIFT`; a changed accepted product raises `REFUSED-DIGEST-DRIFT`; a changed cache digest-of-digests raises `REFUSED-CACHE-DRIFT`; a non-empty output directory is refused. There is no recapture fallback anywhere in the code path.
- The public entry point was exercised end-to-end twice on the real admitted cache with byte-identical statistical tables, and the R3 candidate adds three evidence-only tables: `abnormal-composition.json` (abnormal-only severity/fraction/duration per dose with the adjacent-dose TV quantum and support flag), `sensitivity.json` (measured nested-prefix identity arms versus the implemented per-dose sampling) and `shared-sequence-infeasibility.json` (metadata-only check of the literal size-independent sequence reading) (see §5 and §8).

## 2. Lineage, cache custody and forward accounting

- Pinned digests verified before computation: corpus manifest `9606648d…`, corpus audit `5fddabc1…`, G2 preflight `26f9e45d…`, canonical execution freeze `91d4ce26…` (file `be6e1058…`), prospective-contract canonical `f4f92a60…` (envelope `34031e8c…`). The live census/plan/custody binding recomputed and matched the frozen contract; the frozen 18-group identity plan was reproduced exactly. Every accepted M11/M12 product digest matched before and after the run.
- Admission: `ADMITTED`, required 24,840 rows, available 24,840, missing 0, 18 groups. No row was captured, added or substituted.
- The cache digest-of-digests is unchanged before and after the audit, and each of the 36 files keeps its recorded digest. The prior M11 movement/run records and the M11/M12 source, test and documentation files keep their accepted digests.
- Forwards: `forwards_executed_this_run = 0`, `model_loads = 0`, `checkpoint_opens = 0` (an audit hook counts every opened `.pt/.pth/.ckpt/.bin/.safetensors` path), `cuda_initialized = false` and zero CUDA memory allocated. `torch` is imported transitively by the accepted readiness/inference modules, so the record states that role explicitly; no model, checkpoint, optimizer or GPU context is constructed. Cumulative observed forwards stay **77,653**.
- Reference fits: exactly 18 (one per history × encoder seed, never per comparison), `n_reference = 220` causal files each, `KMeans(random_state=2407, n_init=10)` for `K ∈ {4, 8, 16}` fitted on the reference pool only. Evaluation mixtures, labels, families, severities and context fields never enter a fit.

## 3. Frozen design

### 3.1 Cells, source periods and pools

The 12 accepted `history × horizon` cells are read from the accepted M9 audit pair rows. The low-burden period group is the audit's baseline period set and the elevated group is the contrast period set of its `BASELINE-VS-ELEVATED` rows:

- week: low `{W05, W06, W07, W18, W19, W20}`, elevated `{W13, W14, W26, W27}`;
- month: low `{M02, M03, M05, M06}`, elevated `{M04, M07}`.

Two source-pool designs, each built per cell:

- `POOLED-CELL-PERIODS` — every period of the cell. A mixture therefore **spans periods and is not a calendar window**; source-period composition is a measured confounder.
- `LOW-PERIOD-SOURCE` — audit low-burden periods only, retaining their real low-burden faults. This is the period-composition-controlled sensitivity.

The causal `REFERENCE` arm (220 files per history) is never a source.

### 3.2 Sizes, doses and count rounding

- Sizes: two supported equal per-arm file counts per horizon, fixed from the census: week `{40, 60}`, month `{160, 250}`.
- Doses: `POOLED-CELL-PERIODS {0.00, 0.05, 0.10, 0.20, 0.30, 0.40}` and `LOW-PERIOD-SOURCE {0.00, 0.05, 0.10, 0.15}`.
- `n_abnormal = floor(p * N + 0.5)` exactly; `n_ordinary = N − n_abnormal`; the realized `n/N` is published at every dose.

### 3.3 Composition frame, abnormal sequence and fixed baseline

- Robot|program context quota: capacity `floor(ordinary/2)` per `robot|program` stratum, allocated by capped largest-remainder apportionment of `N` with the capacities as caps. Every arm therefore carries exactly the same robot|program counts, the fixed baseline arm and the zero-abnormal draw stay disjoint, and the context mix TV between any two arms is exactly zero. This apportionment is implemented in the M13 module, not inherited.
- Abnormal sequence: one deterministic sequence per `cell × design × size` (the per-stratum headroom depends on the size-specific context quota), ordered by family deficit against proportional availability (ties by family name; files inside a family by `sha256("S24-7-M13|sequence|" + file_id)`), with each file admitted only while its stratum keeps fewer abnormal files than its context headroom. Dose `p` uses the first `n_abnormal(p)` entries to fix its per-`(stratum, family)` and per-stratum **counts**, so the planned counts are nested and the family mix of every prefix is as proportional as the support allows; no family is silently dropped. The sequence does not supply the identities: the specification's literal `nesting` sentence asks for the abnormal **files** of dose `p` to be exactly those prefix entries, which the frozen per-dose sampling rule contradicts and the implementation does not satisfy (§13).
- Realized identity sets are **not** nested. Each dose resamples identities from its `(stratum, family)` candidate group under a dose-specific RNG token, so a lower-dose abnormal identity need not appear at a higher dose. On the declared reference repeat, 74 of 168 positive-dose transitions fail subset inclusion; the realized overlap is reported separately for the abnormal and ordinary parts across all repeats (§5). Family counts are exact by construction; severity, anomalous fraction and duration are attributes of the sampled identities and are measured, not fixed.
- Baseline: one fixed all-ordinary arm of exactly `N` files carrying the frame context quota, selected inside each stratum by hash order. Every dose arm is disjoint from it; dose 0 is a **separate** zero-abnormal draw with identical size/context/sampling semantics, and that comparison is the baseline-vs-baseline band.
- Sampling: per repeat, files are drawn without replacement inside each `(stratum, family)` abnormal group and each ordinary stratum group, with `numpy.random.default_rng` seeded from a frozen SHA-256 token and one `random((repeats, group_size))` matrix per group. Draws are shared by all three seeds and all K, and differ across doses.
- Repeats `400`; encoder seeds `181801/181802/181803`; `K ∈ {4, 8, 16}`.

### 3.4 Estimators and summaries

- **E1** — accepted unbiased squared MMD at the accepted causal-reference bandwidth (`sprint24_weekly_monthly_endpoints.mmd2_unbiased`), in squared-MMD units; it can be negative.
- **E2** — occupancy total-variation distance against the accepted causal-reference prototypes, stored as exact integer quanta `E2 = m/N`, for `K = 4, 8, 16`.
- Summaries: median, mean, q05, q95, min, max and the strictly-positive / exact-zero / strictly-negative repeat fractions; E2 also keeps its integer numerator.
- Movement: each dose is described against the dose-0 baseline-vs-baseline band of the same unit and seed — null median, null q95, median excess, fraction of repeats above the null q95, and whether the dose median is above the null q95.
- Trends: tie-aware Spearman rank correlation between resolved dose and the per-dose median, step counts, monotone-nondecreasing flag, number of doses above the null q95, and the full dose curve.
- Aggregates only per `horizon × size × design × estimator`; horizons, sizes and designs are never pooled, and no best dose, K, seed or size is selected.

## 4. Realized support and explicit abstentions

All 48 planned units (12 cells × 2 sizes × 2 designs) are accounted for; 40 resolve and 8 abstain with a closed reason:

| Reason | Units | Cells / sizes / design |
|---|---|---|
| `ABSTAIN-INSUFFICIENT-ABNORMAL` | 6 | all six month cells, `N = 250`, `LOW-PERIOD-SOURCE`: the max dose needs 38 abnormal files but the low-burden source pool holds 34–36 |
| `ABSTAIN-ABNORMAL-HEADROOM` | 2 | cell 04, week `N = 60`, low-period; cell 05, week `N = 40`, low-period: the required abnormal count cannot be placed inside the context headroom |

| Cell | Horizon | Size | Design | Pool | Ordinary | Abnormal | Quota | Max abnormal | Status |
|---|---|---|---|---|---|---|---|---|---|
| 01–06 | week | 40 | pooled | 400 | 344 | 56 | 40 | 16 | computed (6/6) |
| 01–06 | week | 60 | pooled | 400 | 344 | 56 | 60 | 24 | computed (6/6) |
| 01–06 | week | 40 | low-period | 240 | 228 | 12 | 40 | 6 | computed (5/6), cell 05 abstains |
| 01–06 | week | 60 | low-period | 240 | 228 | 12 | 60 | 9 | computed (5/6), cell 04 abstains |
| 01–06 | month | 160 | pooled | 1,023 | 899–900 | 123–124 | 160 | 64 | computed (6/6) |
| 01–06 | month | 250 | pooled | 1,023 | 899–900 | 123–124 | 250 | 100 | computed (6/6) |
| 01–06 | month | 160 | low-period | 692 | 656–658 | 34–36 | 160 | 24 | computed (6/6) |
| 01–06 | month | 250 | low-period | 692 | 656–658 | 34–36 | 250 | 38 | abstained (6/6) |

Accounting: 48 planned units (40 computed, 8 abstained), 720 planned dose × seed records (624 computed, 96 abstained), 2,880 planned estimator cells (2,496 computed, 384 abstained), 208 computed composition records. `all_planned_cells_accounted = true`.

## 5. Composition invariants, abnormal-only comparability and sequence-rule sensitivity

### 5.1 Composition under the implemented design

- **Context**: the robot|program counts of the fixed baseline and of every dose arm equal the frozen quota exactly in all 208 materialized arms (TV `0.0`); the independent verifier re-derived the counts from the arm identities and found zero failures.
- **Family (planned and exact)**: the *planned* abnormal family counts are the sequence-prefix counts, and every realized arm matches them exactly in every repeat of all 208 dose plans (the independent verifier recomputed the family counts from the arm identities: 0 failures). The abnormal family mix of the prefix differs from the proportional availability target by a median TV of `0.0` and a maximum of `0.333` (dose `0.15`); 20 of 208 plans have any non-zero deviation, all rounding of small counts. No family is dropped: all families present in a pool appear in the sequence.
- **Whole-arm normalization understated the abnormal-only question (R2 finding, corrected in R3)**: the R1/R2 realized-mix TV divided every bin count by the full arm `N`, so a severity/fraction change confined to the `n_abnormal` abnormal files was scaled by `n_abnormal / N` (`0.05–0.40`). That is why those medians read `0.0`–`0.025`: the dilution, not the composition, was small.
- **Abnormal-only severity, anomalous fraction and duration (R3)**: for every resolved unit and dose, the abnormal files of each repeat are normalized among themselves (the accepted M12 severity and anomalous-fraction bins; per-unit abnormal-duration quartile edges) and compared with the planned sequence-prefix mixture (`vs_planned_abnormal_only`), the maximum dose and the adjacent lower dose. Each adjacent comparison carries its empirical TV quantum `0.5·(1/n_current + 1/n_previous)` and is `supported` only when that quantum is at most `0.05`, the declared diagnostic resolution floor. All 208 resolved dose plans are published in `abnormal-composition.json`.

| Horizon | Size | Design | Supported / unsupported adjacent transitions | Median severity / fraction / duration TV (supported) | Median TV quantum (supported) |
|---|---|---|---|---|---|
| month | 160 | pooled | 18 / 6 | 0.0703 / 0.0625 / 0.0625 | 0.0260 |
| month | 250 | pooled | 18 / 6 | 0.0617 / 0.0433 / 0.0533 | 0.0167 |
| week | 60 | pooled | 6 / 18 | 0.0764 / 0.0625 / 0.0833 | 0.0486 |
| week | 40 | pooled | 0 / 24 | not measurable | — |
| month | 160 | low-period | 0 / 12 | not measurable | — |
| week | 40 | low-period | 0 / 10 | not measurable | — |
| week | 60 | low-period | 0 / 10 | not measurable | — |

- **Abstention where the resolution floor cannot be met**: the adjacent-dose abnormal-only comparison is unsupported for every week `N=40` pooled transition (quantum `0.0729–0.375`), every low-period week transition (`0.1389–0.375` at `N=60`, `0.2083–0.375` at `N=40`) and every month `N=160` low-period transition (`0.0521–0.0938`). Week `N=60` pooled supports only the last transition (`18→24` abnormal files, quantum `0.0486`); the month pooled sizes support the three larger transitions (quantum `0.0167–0.0260`) and not the first (`0.0585` at `N=250`, `0.0938` at `N=160`). **No controlled-dose statement is made for any unsupported cell**, and no comparison is re-labelled supported by relaxing the floor.
- **Reviewer example reproduced exactly**: cell 05, week, `N=60`, pooled, `.05→.10` — abnormal-only severity median TV `0.5` with quantum `0.25` (`supported=false`), fraction median TV `0.333`, duration `0.167`.
- **What the supported comparisons show**: among supported comparisons the median abnormal-only severity TV is `0.062–0.076`, anomalous fraction `0.043–0.063` and duration `0.053–0.083`, at or above the quantum. The abnormal severity/fraction/duration mixtures are therefore **not held fixed across doses**. Attribution is measured, not conjectured: in the week `N=60` units the realized mix equals its plan (median `vs_planned` TV `0.0`), so there the change is the planned prefix composition alone; in the month units the realized mix also deviates from its own plan (median `vs_planned` severity/fraction/duration `0.0365`/`0.0312`/`0.0417` at `N=160` and `0.040`/`0.020`/`0.0267` at `N=250`), so plan change and identity resampling are both present and are not separable on this support.
- **Duration comparability is not established**: the abnormal-duration distribution drifts by a median TV of `0.0533–0.0833` in the supported comparisons (more than the quantum) and cannot be assessed at all in the unsupported cells. Duration is neither held fixed nor shown comparable, and no dose comparison may be read as duration-controlled.
- **Composition conclusion (corrected)**: context and abnormal family composition are exact; abnormal severity, anomalous fraction and duration are sampled and drift across doses by `0.04–0.08` TV wherever the resolution floor permits measurement. Dose curves are conditional on realized composition, burden and abnormal-composition change are collinear in the pooled design (as burden and source period already are, §7), and no controlled-dose interpretation is offered where the comparison is unsupported.
- **Disjointness and true reuse**: no duplicate identity inside any draw and no identity shared by the fixed baseline and a comparison arm (0 failures in 208 arms × 400 repeats). Across doses the realized arms are **not** nested: 24 of 40 units have at least one positive-dose transition whose abnormal identities are not a superset of the previous dose (74 of 168 transitions at repeat 0; 27,745 of 67,200 transition × repeat checks fail). At the top dose the median abnormal overlap with the immediately lower dose is `0.667–0.730` and the median ordinary overlap with the zero-dose arm is `0.125–0.392` (depending on horizon/size/design). Across all 400 repeats a unit uses a median of 75.4% of its pool's distinct files; total slot draws are `400 × N × doses`.
- **Baseline**: every baseline record has exactly `N` all-ordinary identities with the frame stratum counts and the `ordinary: N` family count.

### 5.2 Sensitivity to the declared sequence and nesting rules

- **Measured nested-identity sensitivity (R3)**: at the largest pooled size per horizon (week `N=60`, month `N=250`; 12 units) each repeat's abnormal identities are rebuilt as group-wise `sha256` prefixes of the maximum-dose arm, one prefix per `(stratum, family)`, so the context quota, the abnormal family counts and the realized ordinary slots are unchanged. At the maximum dose every pair's prefix count equals its maximum-plan count, so that arm's identity set equals the implemented arm's for every repeat; the independent check confirms this at repeat 0, and the anchor is excluded from the aggregates: 180 records (`12 units × 5 positive doses × 3 seeds`), aggregates `n = 72` (`6 cells × 4 non-anchor doses × 3 seeds`). The arms are published in `sensitivity.json` with their repeat-0 identities and digests. The measured profile is deliberately narrow: pooled design only, the largest size per horizon only, and only the per-dose `E1` and `E2_K8` **medians** were recomputed — 180 of the 624 computed dose × seed records. No smaller size, no low-period design, no `K=4`/`K=16` statistic, and no per-seed curve or trend statistic was recomputed.
- **Measured effect of the nesting rule**: `|Δ median E1|` has a per-dose median of `1.4×10⁻⁵`–`3.5×10⁻⁵` and a maximum of `1.99×10⁻⁴` squared-MMD units; `|Δ median E2_K8|` is `0.0` in 144 of 180 records, is at most `0.0167` (exactly one `1/60` estimator quantum, in one record) and reaches `0.01` in that record only. Both changes are inside the baseline-vs-baseline dispersion at the same sizes (E1 null q05–q95 spans of `1.5×10⁻³`–`8×10⁻³`; E2_K8 null q05–q95 endpoints of `0.04`–`0.25`), but that comparison bounds magnitude, not direction: `9` of `144` adjacent per-seed `E1` median steps change sign between the two constructions (e.g. cell 01, month `N=250`, seed 181802, `.30→.40`: `+3.37×10⁻⁵` implemented vs `−2.45×10⁻⁵` nested). Step direction at this resolution is therefore sensitive to the identity rule, and it is the per-dose medians within this narrow profile that stay close, not every curve, null comparison or trend. The composition difference is reported separately: the abnormal-only mix of the nested construction differs from the implemented one by a median severity TV of `0.0–0.077` per dose (up to `0.333` at week `N=60`, dose `0.05`, where `n_abnormal = 3`), and by at most `0.04` for fraction and duration — a consequence of the rule at small `n`, not an estimator change.
- **The literal shared-sequence reading is not feasible as specified (R3, metadata only)**: the specification's `abnormal_sequence.scope` asks for one sequence per `cell × design`, independent of size, with length equal to the maximum planned `n_abnormal` at the largest supported size. Checking that largest-size sequence prefix against the frozen size-specific context quota gives 96 `(cell, design, size, dose)` checks: 50 feasible, 38 infeasible, 8 unavailable (the cells whose largest-size frame already abstains), with a maximum excess of 5 files. The infeasible checks are pooled week `N=40` (17 of 36), pooled month `N=160` (14 of 36) and low-period week `N=40` (7 of 16); the worst case is cell 06, month pooled, dose `0.40`, which needs `+1` file in `R01|P110`, `+2` in `R02|P110` and `+2` in `R04|P160` beyond the `N=160` quota. Each row and its exact excess strata are in `shared-sequence-infeasibility.json`; no arm was materialised for any check.
- **Disposition**: the per-size sequence remains the implemented alternative and is recorded as `A01`. The shared-sequence reading is reported as partly infeasible metadata evidence — not implemented, not silently substituted, and not claimed — while the nesting component of the declared rule is measured by the nested-identity sensitivity above. Where the literal shared sequence would violate the smaller size's quota, no arm exists under the frozen headroom rule and no result is claimed.

## 6. Measurement results

### 6.1 Baseline-vs-baseline null bands

`median [min, max]` over the resolved units × seeds; E1 in squared-MMD units, E2 in `[0, 1]`.

| Horizon | Size | Design | E1 | E2_K4 | E2_K8 | E2_K16 |
|---|---|---|---|---|---|---|
| week | 40 | pooled | −0.0027 [−0.0039, −0.0004] | 0.1250 [0.0750, 0.1750] | 0.2125 [0.1750, 0.2750] | 0.3125 [0.2750, 0.3750] |
| week | 40 | low-period | −0.0040 [−0.0061, 0.0037] | 0.1000 [0.0750, 0.2000] | 0.2000 [0.1750, 0.3000] | 0.3250 [0.2500, 0.3750] |
| week | 60 | pooled | −0.0021 [−0.0044, 0.0002] | 0.1000 [0.0667, 0.2000] | 0.1833 [0.1167, 0.2500] | 0.2583 [0.2167, 0.3333] |
| week | 60 | low-period | −0.0017 [−0.0046, 0.0022] | 0.1167 [0.0667, 0.1833] | 0.1667 [0.1167, 0.2167] | 0.2667 [0.2167, 0.3167] |
| month | 160 | pooled | −0.0002 [−0.0015, 0.0014] | 0.0688 [0.0437, 0.1062] | 0.1125 [0.0750, 0.1437] | 0.1688 [0.1375, 0.2062] |
| month | 160 | low-period | −0.0010 [−0.0016, 0.0007] | 0.0563 [0.0375, 0.1062] | 0.0844 [0.0750, 0.1250] | 0.1469 [0.1250, 0.1875] |
| month | 250 | pooled | −0.0003 [−0.0009, 0.0000] | 0.0580 [0.0320, 0.0840] | 0.0770 [0.0640, 0.1080] | 0.1160 [0.1060, 0.1560] |

Two disjoint all-ordinary draws of the same size differ by a TV of 0.08–0.33 depending on `N` and `K`. This band is the scale every movement statement below has to be read against; it shrinks with `N`, so weekly cells are much coarser than monthly cells.

### 6.2 E2 dose curves (seed median over the three encoder seeds)

| Horizon | Size | Design | dose 0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|---|---|---|
| week | 40 | pooled | 0.2125 | 0.2250 | 0.2250 | — | 0.2250 | 0.2250 | 0.2250 |
| week | 40 | low-period | 0.2000 | 0.2000 | 0.2250 | 0.2250 | — | — | — |
| week | 60 | pooled | 0.1833 | 0.1750 | 0.1667 | — | 0.1792 | 0.1833 | 0.1833 |
| week | 60 | low-period | 0.1667 | 0.1667 | 0.1667 | 0.1667 | — | — | — |
| month | 160 | pooled | 0.1125 | 0.1063 | 0.1031 | — | 0.1000 | 0.0938 | 0.0969 |
| month | 160 | low-period | 0.0844 | 0.0875 | 0.0844 | 0.0844 | — | — | — |
| month | 250 | pooled | 0.0770 | 0.0780 | 0.0800 | — | 0.0800 | 0.0820 | 0.0780 |

The complete per-cell dose curves at `K = 8` are below; the per-seed values and `K = 4, 16` are in `curves.json`.

**E2_K8 week N=40 pooled / low-period**

| Cell | 0 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|
| 01 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2250 |
| 02 | 0.2000 | 0.2000 | 0.2000 | 0.1750 | 0.1750 | 0.2000 |
| 03 | 0.1750 | 0.2000 | 0.1750 | 0.1750 | 0.1750 | 0.1750 |
| 04 | 0.2000 | 0.2250 | 0.2250 | 0.2250 | 0.2250 | 0.2250 |
| 05 | 0.2500 | 0.2250 | 0.2500 | 0.2750 | 0.2500 | 0.2250 |
| 06 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2250 |

Low-period (`N = 40`; cells 01, 02, 03, 04, 06 — cell 05 abstains): `0.2250/0.2000/0.1750/0.2000/0.2000` at dose 0 and `0.2250/0.2250/0.1750/0.1750/0.2250` at dose 0.15.

**E2_K8 week N=60 pooled**

| Cell | 0 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|
| 01 | 0.1500 | 0.1500 | 0.1667 | 0.1667 | 0.1833 | 0.1667 |
| 02 | 0.2000 | 0.1833 | 0.1667 | 0.1667 | 0.1500 | 0.1667 |
| 03 | 0.1667 | 0.1667 | 0.1667 | 0.1667 | 0.1500 | 0.1417 |
| 04 | 0.1833 | 0.1833 | 0.1833 | 0.1833 | 0.1833 | 0.1833 |
| 05 | 0.1833 | 0.1667 | 0.1833 | 0.1833 | 0.2000 | 0.1833 |
| 06 | 0.1500 | 0.1500 | 0.1333 | 0.1500 | 0.1833 | 0.1833 |

Low-period (`N = 60`; cells 01, 02, 03, 05, 06 — cell 04 abstains): dose 0 `0.1500/0.2167/0.1667/0.1667/0.1833`, dose 0.15 `0.1667/0.2000/0.1667/0.1500/0.1667`.

**E2_K8 month N=160 pooled**

| Cell | 0 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|
| 01 | 0.0813 | 0.0813 | 0.0813 | 0.0813 | 0.0875 | 0.0844 |
| 02 | 0.1313 | 0.1250 | 0.1250 | 0.1375 | 0.1438 | 0.1438 |
| 03 | 0.1063 | 0.1063 | 0.0938 | 0.1000 | 0.0875 | 0.0938 |
| 04 | 0.1125 | 0.1188 | 0.1125 | 0.1000 | 0.0938 | 0.0875 |
| 05 | 0.1125 | 0.1063 | 0.1000 | 0.1063 | 0.1000 | 0.0875 |
| 06 | 0.0938 | 0.0906 | 0.0938 | 0.1000 | 0.0938 | 0.1063 |

Low-period (`N = 160`, all six cells): dose 0 `0.0813/0.0813/0.0938/0.1000/0.1188/0.0813`, dose 0.15 `0.0813/0.0813/0.0875/0.0938/0.1063/0.0750`.

**E2_K8 month N=250 pooled**

| Cell | 0 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|
| 01 | 0.0960 | 0.0920 | 0.0880 | 0.0880 | 0.0920 | 0.0760 |
| 02 | 0.0740 | 0.0720 | 0.0720 | 0.0720 | 0.0720 | 0.0720 |
| 03 | 0.0720 | 0.0720 | 0.0680 | 0.0720 | 0.0720 | 0.0800 |
| 04 | 0.0720 | 0.0760 | 0.0720 | 0.0720 | 0.0720 | 0.0720 |
| 05 | 0.0880 | 0.0880 | 0.0880 | 0.0960 | 0.0960 | 0.0880 |
| 06 | 0.0720 | 0.0760 | 0.0760 | 0.0760 | 0.0760 | 0.0760 |

### 6.3 E1 dose curves (seed median over the three encoder seeds)

| Horizon | Size | Design | 0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|---|---|---|
| week | 40 | pooled | −0.00274 | −0.00284 | −0.00277 | — | −0.00278 | −0.00253 | −0.00219 |
| week | 40 | low-period | −0.00400 | −0.00365 | −0.00392 | −0.00420 | — | — | — |
| week | 60 | pooled | −0.00210 | −0.00176 | −0.00165 | — | −0.00115 | −0.00140 | −0.00197 |
| week | 60 | low-period | −0.00167 | −0.00154 | −0.00126 | −0.00169 | — | — | — |
| month | 160 | pooled | −0.00021 | −0.00014 | −0.00012 | — | −0.00003 | −0.00013 | −0.00017 |
| month | 160 | low-period | −0.00101 | −0.00106 | −0.00109 | −0.00108 | — | — | — |
| month | 250 | pooled | −0.00030 | −0.00034 | −0.00037 | — | −0.00026 | −0.00007 | −0.00021 |

Per-cell E1 seed-median curves are published in `curves.json` and summarised by the trend table below. The largest absolute pooled-design change is `≈ 0.001` squared-MMD units (week `N=60`, cell 05: `+0.00019 → +0.00252`); most cells move by `10⁻⁴`–`10⁻³`.

### 6.4 Movement above baseline variability

Across `2,016` movement records (`208` resolved unit–dose combinations × `3` seeds × `4` estimators, excluding the zero dose), **2** dose medians lie above their unit–seed null q95:

| Unit | Dose | Seed | Estimator | Median excess | Fraction of repeats above null q95 |
|---|---|---|---|---|---|
| cell 01, month `N=250`, pooled | 0.40 | 181803 | E2_K4 | +0.032 | 0.58 |
| cell 01, week `N=60`, pooled | 0.30 | 181803 | E1 | +0.00415 | 0.56 |

Every other dose median is at or below the baseline-vs-baseline q95. E2 records: `1 / 1,512` above; E1 records: `1 / 504` above.

### 6.5 Trend and direction (seed-median Spearman over the dose curve)

| Horizon | Size | Design | Estimator | Cells | Median seed-median ρ | Cells +/0/− | All-seeds positive | All-seeds negative | All-seeds zero |
|---|---|---|---|---|---|---|---|---|---|
| month | 160 | low | E1 | 6 | −0.300 | 2/0/4 | 1 | 2 | 0 |
| month | 160 | low | E2_K4 | 6 | −0.618 | 1/0/5 | 0 | 2 | 0 |
| month | 160 | low | E2_K8 | 6 | −0.632 | 0/0/6 | 0 | 1 | 0 |
| month | 160 | low | E2_K16 | 6 | −0.323 | 1/1/4 | 1 | 1 | 0 |
| month | 160 | pooled | E1 | 6 | +0.457 | 5/0/1 | 4 | 1 | 0 |
| month | 160 | pooled | E2_K4 | 6 | −0.641 | 2/0/4 | 2 | 2 | 0 |
| month | 160 | pooled | E2_K8 | 6 | −0.518 | 2/0/4 | 2 | 1 | 0 |
| month | 160 | pooled | E2_K16 | 6 | −0.559 | 1/0/5 | 0 | 2 | 0 |
| month | 250 | pooled | E1 | 6 | +0.629 | 4/0/2 | 2 | 1 | 0 |
| month | 250 | pooled | E2_K4 | 6 | +0.401 | 6/0/0 | 2 | 0 | 0 |
| month | 250 | pooled | E2_K8 | 6 | +0.417 | 4/0/2 | 1 | 0 | 0 |
| month | 250 | pooled | E2_K16 | 6 | +0.706 | 6/0/0 | 2 | 0 | 0 |
| week | 40 | low | E1 | 5 | +0.400 | 3/1/1 | 1 | 1 | 0 |
| week | 40 | low | E2_K4 | 5 | +0.282 | 2/0/2 | 0 | 0 | 0 |
| week | 40 | low | E2_K8 | 5 | +0.447 | 3/1/1 | 0 | 0 | 0 |
| week | 40 | low | E2_K16 | 5 | +0.060 | 3/0/2 | 1 | 0 | 0 |
| week | 40 | pooled | E1 | 6 | +0.486 | 4/0/2 | 2 | 1 | 0 |
| week | 40 | pooled | E2_K4 | 6 | −0.212 | 2/0/4 | 1 | 0 | 0 |
| week | 40 | pooled | E2_K8 | 6 | −0.236 | 2/0/4 | 0 | 1 | 0 |
| week | 40 | pooled | E2_K16 | 6 | −0.254 | 2/0/4 | 0 | 1 | 0 |
| week | 60 | low | E1 | 5 | +0.400 | 4/0/1 | 1 | 1 | 0 |
| week | 60 | low | E2_K4 | 5 | +0.087 | 3/0/2 | 0 | 0 | 0 |
| week | 60 | low | E2_K8 | 5 | −0.447 | 2/0/3 | 1 | 1 | 0 |
| week | 60 | low | E2_K16 | 5 | +0.474 | 3/0/2 | 1 | 2 | 0 |
| week | 60 | pooled | E1 | 6 | +0.629 | 5/0/1 | 4 | 0 | 0 |
| week | 60 | pooled | E2_K4 | 6 | +0.043 | 3/0/3 | 1 | 0 | 0 |
| week | 60 | pooled | E2_K8 | 6 | +0.107 | 3/0/3 | 2 | 0 | 0 |
| week | 60 | pooled | E2_K16 | 6 | −0.065 | 2/1/3 | 0 | 1 | 0 |

Reading the table: E1 carries a small positive dose association in every pooled configuration (median ρ `+0.46` to `+0.63`, with two to four of six cells positive under all three seeds) and no consistent association in the period-controlled low-period design (month `−0.30`; week `+0.40` with only one of five cells positive under all seeds). E2 trend signs are mixed in every configuration: the strongest values are month `N=250` pooled (`+0.40` to `+0.71`) and month `N=160` low-period (`−0.32` to `−0.63`), so the direction is not stable across sizes or designs. No K, seed, size or dose is selected as best.

## 7. Source-period composition and confounding

The pooled design's abnormal files are drawn from the same sequence across periods, and the corpus places most abnormal files in the elevated periods. As the dose rises the arm necessarily contains a larger share of elevated-period files:

| Horizon | Size | Design | Dose | Period-mix TV vs fixed baseline (median) | Low-burden-period file share (median) |
|---|---|---|---|---|---|
| week | 40 | pooled | 0.00 / 0.40 | 0.250 / 0.300 | 0.650 / 0.475 |
| week | 60 | pooled | 0.00 / 0.40 | 0.217 / 0.250 | 0.658 / 0.475 |
| week | 40–60 | low-period | 0.00 / 0.15 | 0.150–0.225 / 0.167–0.200 | 1.000 / 1.000 |
| month | 160 | pooled | 0.00 / 0.40 | 0.100 / 0.250 | 0.722 / 0.531 |
| month | 250 | pooled | 0.00 / 0.40 | 0.072 / 0.182 | 0.726 / 0.550 |
| month | 160 | low-period | 0.00 / 0.15 | 0.073 / 0.073 | 1.000 / 1.000 |

The low-period design keeps the source period set fixed (all arms are low-burden-period files) and its period-mix TV stays at the sampling level of the zero dose; the pooled design's period mix is a deterministic function of the dose. **Burden and source-period composition are therefore collinear in the pooled design.** Any pooled-design drift — including the small positive E1 association — cannot be attributed to burden rather than to the period-composition change, and the low-period sensitivity is the only period-controlled reading on this cache.

## 8. Independent verification

`artifacts/sprint-24/evidence/s24-7-m13-a01/independent-verification.py` does not import the M13 module. It re-derives the bandwidth (median pairwise distance) and the `K ∈ {4, 8, 16}` prototypes with its own `scipy`/`scikit-learn` path, uses `cdist` for the kernels and `1 − Σ min(p, q)` for the occupancy distance, and checks the stored evidence. All 22 checks pass:

| Check | Result |
|---|---|
| Cache identity (36 files, digest-of-digests) | PASS |
| Accepted M11/M12 products unchanged | PASS |
| Independent reference fits | PASS (18 groups) |
| Arm size and duplicate-free draws | PASS (208 arms) |
| Baseline/comparison disjointness | PASS (208 arms) |
| Context composition exact | PASS (208 arms) |
| Abnormal family composition exact | PASS (208 arms) |
| E2 exact integer quanta vs independent TV | PASS (1,872 values, 0 mismatches) |
| E1 vs independent explicit unbiased MMD | PASS (624 values, 0 mismatches) |
| Dose/count/composition invariants | PASS (208 levels) |
| Accounting completeness | PASS (720 records: 624 computed + 96 abstained) |
| Census matches support statuses | PASS (48 units) |
| Movement blocks consistent with null bands | PASS (2,016 records) |
| Stored quanta exact and integral | PASS (624 records) |
| Realized composition vs actual draws | PASS (208 doses, 0 mismatches; 71 severity / 63 fraction distributions differ from the sequence-prefix plan) |
| Composition tables distinguish plan and realized | PASS (208 records, 0 stale) |
| Realized non-nesting and overlap | PASS (40 units, 168 transitions, 74 subset failures at the reference repeat) |
| Abnormal-only composition vs actual abnormal identities | PASS (168 positive-dose arms, 0 mismatches) |
| Nested sensitivity repeat-0 identities, E1 and E2 | PASS (180 records, 0 failures) |
| Shared-sequence infeasibility internal consistency | PASS (96 checks, 0 failures; 50 feasible / 38 infeasible / 8 unavailable) |
| Reproducibility (two runs) | PASS (14 tables byte-identical) |
| Baseline records | PASS (40 records) |

Reproducibility: the `audit/` and `audit-repro/` tables — `census.json`, `support.json`, `curves.json`, `baseline.json`, `composition.json`, `abnormal-composition.json`, `sensitivity.json`, `shared-sequence-infeasibility.json`, `arms.json`, `trends.json`, `aggregates.json`, `accounting.json`, `repeats-keys.json` and `repeats.npz` (14 files) — are byte-identical; `lineage.json` is identical after removing the recorded fit wall times. The R3 correction adds evidence only: `curves.json`, `baseline.json`, `composition.json`, `arms.json`, `trends.json`, `aggregates.json`, `accounting.json` and `census.json` are byte-identical to the assessed R2 candidate (`audit-r2/`), and seven of those were already byte-identical to the R1 assessment (`audit-r1/`). `support.json` gains the per-unit abnormal-only block and `repeats.npz`/`repeats-keys.json` gain the 360 nested-sensitivity vectors (2,496 → 2,856 stored vectors); no estimator, summary, movement, trend, composition or abstention rule changed.

## 9. What this establishes and does not establish

- On this fixed cache, at fixed arm size, with the robot|program mix held exactly and the abnormal family composition exact while severity/anomalous fraction/duration are sampled and measured (§5.1): raising the abnormal-file share from 0 to 0.40 does **not** move the E2 prototype-occupancy distribution beyond the baseline-vs-baseline resampling band: 1 of 1,512 E2 dose medians exceeds its null q95, and almost every E2 dose curve moves by at most one or two estimator quanta (`1/N`). The null band itself is 0.06–0.33 depending on `N`, `K` and horizon (`K=4` smallest, `K=16` largest). This is not a controlled-composition claim: at the supported adjacent doses the abnormal-only severity/fraction/duration mixtures themselves drift by `0.04–0.08` TV, and at the smaller week and low-period designs no abnormal-only comparison reaches the diagnostic resolution floor at all.
- E1 changes are small (`≈ 10⁻⁴` to `≈ 10⁻³` squared-MMD units) and only consistently positive in the pooled design. Because the pooled design's source-period composition co-varies deterministically with the dose, that association is **confounded**; the period-controlled low-period design shows no consistent E1 or E2 dose trend. A small burden effect and a period-composition effect are not separable on this support.
- The design's sensitivity is diluted by construction: a mixture with abnormal share `p` carries an RKHS mean difference of order `p` (so E1 of order `p²`) and an occupancy difference of order `p` against the ordinary distribution, while the equal-`n` baseline-vs-baseline TV band at these sizes is of the same order. The measured flatness is therefore partly a power statement about this cache, these sizes and this estimator, not proof that burden has no representational effect.
- Repeats are Monte Carlo integration of the sampling distribution of one finite support; their quantiles are resampling variability, never confidence limits. Encoder seeds are run replicates, never independent data. Every dose shares the same fixed baseline and independently resamples identities from the same source pool: the **planned** abnormal counts are nested but the realized arms are not (74/168 positive transitions fail subset inclusion at repeat 0, with 0.67–0.73 abnormal overlap against the adjacent dose), so dose comparisons are dependent by shared support, not by identity nesting. Measured abnormal-only, the realized severity/fraction/duration composition deviates from its own plan by a median TV of `0.0` (week and low-period units, where the pair groups are nearly exhausted by the allocation) to `0.037`/`0.031`/`0.042` (month `N=160`), so dose comparisons are conditional on that composition.
- The sequence-rule sensitivity is measured, not assumed, but only within a bounded profile: at the two largest pooled sizes (`week N=60`, `month N=250`; 180 of the 624 computed dose × seed records), replacing the implemented per-dose identity sampling with the nested-prefix construction changes the `E1` median by at most `1.99×10⁻⁴` and the `E2_K8` median by at most one `1/60` quantum (`0.0167`, in one of 180 records). Those medians stay close, but directions do not: `9` of `144` adjacent `E1` median steps reverse sign between the two constructions (§5.2). The smaller sizes, the low-period design, `K=4`/`K=16`, per-seed curves and every trend statistic other than these medians were not recomputed, so no claim is made that any reported curve, null comparison or trend is identity-rule invariant. The literal size-independent shared sequence could not have been run as written: its largest-size prefix exceeds the smaller size's frozen context quota in 38 of 88 evaluable metadata checks (maximum excess 5 files), and no arm is claimed for those checks.
- No p-value, significance test, success gate, confidence interval, performance, detection-accuracy, business-density, causal or generalisation claim is produced. Nothing here is a calendar-window or business claim: the pooled mixtures deliberately span periods and the low-period mixtures are constructed, not observed, periods.

## 10. Explicit abstentions and the prospective supplemental design they imply

Two low-period units cannot be built at their planned maximum dose, and the whole `month / N=250 / low-period` block is unsupported:

- `ABSTAIN-INSUFFICIENT-ABNORMAL` (6 units): the maximum planned `0.15 × 250 = 38` abnormal files exceeds the 34–36 low-burden-period abnormal files available per history. A prospective supplemental corpus would need either more low-burden periods with `≈ 4–8` additional abnormal files per history, or a larger `N` with more low-burden abnormal supply.
- `ABSTAIN-ABNORMAL-HEADROOM` (2 units): the required abnormal files exist but their robot|program strata carry insufficient context headroom under the fixed `floor(ordinary/2)` quota. A prospective corpus would need more ordinary files in those strata (or a pre-declared larger stratum quota) to place 9 abnormal files at week `N=60` for cell 04 and 6 at week `N=40` for cell 05.

These are reported, not repaired: no replacement draw, reseeding, label substitution or synthetic file was used. A future design that wants the period-controlled high-dose cells must plan the corpus's low-burden abnormal supply and per-stratum ordinary supply accordingly; that is a new generation task, not part of M13.

## 11. Reproduction

From the repository root with the repository environment (cache already retrieved; the CLI refuses a non-empty `--out`):

```bash
python experiments/sprint24_task7m13_controlled_dose_response.py --stage census \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --out artifacts/sprint-24/evidence/s24-7-m13-a01/census-FRESH

python experiments/sprint24_task7m13_controlled_dose_response.py --stage audit \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --contract artifacts/sprint-24/evidence/s24-7-m10-a02/prospective-contract.json \
  --cache-root artifacts/sprint-24/evidence/s24-7-m12-a01/cache-full \
  --census artifacts/sprint-24/evidence/s24-7-m13-a01/census/census.json \
  --out artifacts/sprint-24/evidence/s24-7-m13-a01/audit-FRESH

python artifacts/sprint-24/evidence/s24-7-m13-a01/independent-verification.py
python -m pytest tests/representation/test_sprint24_controlled_dose_response.py -q
```

The cache retrieval (read-only, no remote edit) is the one recorded by M12:

```bash
ssh trietlm@192.168.99.214 'cd /tmp/s24-7-m11-a03-run/cache-full && sha256sum *' > remote-cache-sha256.txt
scp -r trietlm@192.168.99.214:/tmp/s24-7-m11-a03-run/cache-full artifacts/sprint-24/evidence/s24-7-m12-a01/
```

## 12. Runtime, storage and cost

- Audit wall time ≈ 788 s and 731 s for the two final R3 runs (the first R3 pair, preserved as `audit-r3-prea04/` and `audit-repro-r3-prea04/`, took ≈ 579 s and 540 s; `repeats = 400`, 18 fits, 208 resolved unit–dose combinations, 40-unit `KMeans` prediction on 24,840 rows, 624 E1 repeat vectors of length 400, plus realized/abnormal-only composition, overlap and the 180 nested-sensitivity E1 and E2 repeat vectors). A 5-repeat smoke took ≈ 141 s on the resumed host (identical tables to the 14 s paused-host smoke of the same code).
- Evidence per run: 15 JSON tables + `repeats.npz` (int64 E2 quanta and float64 E1 vectors, 2,856 stored vectors: 2,496 main + 360 nested-sensitivity). Preserved: `audit/` and `audit-repro/` (final R3 candidate, byte-identical tables, lineage records `A01`–`A04`), `audit-r3-prea04/` and `audit-repro-r3-prea04/` (the first R3 pair, run before `A04` existed; all 14 statistical tables byte-identical, only `lineage.json` and `run.json` differ), `audit-r1/`, `audit-repro-r1/`, `audit-r2/`, `audit-repro-r2/` (assessed R1/R2 candidates), `smoke/`, `smoke-r2/`, `smoke-r3/`, `smoke-r3b/`, `smoke-r3c/` (5-repeat smokes), `pilot/` (pre-final full run), `census/`, `spec/` and `amendments/`. The retrieved cache (36 files, 15.5 MB) is never committed; `artifacts/` is excluded from Git.
- **Deleted first smoke (custody gap, disclosed):** a first 5-repeat smoke directory was created after the spec freeze, then removed with `rm -rf …/evidence/s24-7-m13-a01/smoke` and re-created after two reporting fixes. Its pre-fix tables are not recoverable. It contained the same schema for 5 repeats with exactly the two defects later corrected (E2 movement fields left in quanta; family-mix TV target including the ordinary category); no fit, arm, estimator or reported number depends on it, and the corrected `smoke/` plus the pre-final `pilot/` are preserved.
- Forwards: **0 new**; cumulative observed cost remains 77,653. No GPU context, no model, no checkpoint open, no network compute.

## 13. Specification custody, amendments and pre-outcome chronology

- The original specification bytes are preserved verbatim at `artifacts/sprint-24/evidence/s24-7-m13-a01/spec/sprint24-controlled-dose-response-specification-a01.json` with `spec/custody.json` recording the digest, byte count and census digest. The module pins the digest and refuses any drift; amendments are separate timestamped records loaded and recorded by the engine.
- **Pre-outcome chronology (corrected).** Before the specification bytes existed, the manifest and M9 audit (metadata) were inspected, and one admitted cache group file was opened with numpy for a **schema check only**: the NPZ member names `['file_ids','embeddings']` and their shapes/dtypes (`(1380,) <U28`, `(1380, 128) float32`) plus the cache sidecar JSON were printed. Loading an NPZ member materialises that array, so the original specification's statement that no embedding array was read is too strong; no embedding value was printed, summarised, compared or used, and no distance, kernel, prototype, occupancy, E1 or E2 quantity was computed before the specification existed. The audit module and CLI also existed before the freeze, although the audit stage had never been executed. `no_outcome_computed: true` remains correct.
- **Amendments (four, separately timestamped; the original specification bytes are untouched).** Amendment records are under `artifacts/sprint-24/evidence/s24-7-m13-a01/amendments/`:
  - `A01` — the specification's `abnormal_sequence.scope` sentence says "one sequence per `cell × design`, independent of size", while the implementation builds it per `cell × design × size` because the per-stratum headroom depends on the size-specific context quota. The headroom/quota rules themselves are unchanged; the literal shared-sequence reading would materialise different arms. Its R3 metadata check now quantifies that reading: the largest-size sequence prefix violates the smaller size's quota in 38 of 88 evaluable checks, with a maximum excess of 5 files (§5.2). `A01.sensitivity_executed = false` refers to materialising those literal shared-sequence arms, not to the R3 metadata check or the nested-prefix sensitivity.
  - `A02` — the specification's `abnormal_sequence.nesting` says the abnormal **files** of dose `p` are the first `n_abnormal(p)` entries, so the abnormal **sets** are nested across doses. Each dose instead resamples identities from its `(stratum, family)` candidate set, so the realized abnormal file sets are not nested (74 of 168 positive transitions fail subset inclusion at repeat 0; adjacent-dose abnormal overlap `0.667–0.730`) and the realized severity/fraction/duration mix is not the sequence-prefix plan (71/208 severity and 63/208 fraction distributions differ at repeat 0). This is a deviation from the literal nesting requirement, recorded here rather than described as a count-only clarification: the frozen `sampling.without_replacement` rule and the frozen nesting clause cannot both hold at these group sizes, and the implementation follows the sampling clause. The sampling rule itself was not edited; the correction measures and publishes the realized composition and true overlap and stops describing the plan as the realized arm. The R3 abnormal-only measurement supersedes the diluted whole-arm statistic quoted in R2.
  - `A03` — the `evidence_of_absence` wording overstates the pre-freeze state (schema-only cache inspection; module implemented but unexecuted) and records the deleted first smoke described in §12.
  - `A04` — created for R3, after the R2 assessment and before the R3 submission: records the abnormal-only severity/fraction/duration comparability and its supported/unsupported split, the measured nested-prefix identity sensitivity, and the metadata-only shared-sequence infeasibility, with the numbers in §5.1 and §5.2. It changes no rule, grid, size, seed, repeat, estimator or abstention. Its `sensitivity` field states a universal invariance that the measurement does not support; those bytes are preserved unchanged and the overbroad sentence is superseded by the timestamped erratum `amendments/A04-ERRATUM-01.md` (corrected claim in §5.2 and §9). The erratum is a documentation record, not a fifth engine amendment, so stored lineage still lists `A01`–`A04`.
- **Candidate chronology after the R3 readiness message (disclosed, not repaired).** The R3 `ready_for_review` message and the preserved A02 worker record bound report sha256 `16066f3c675a617ace3ef7b93b0b766ec8e3870cae0851b51c6a47bebc3e1577`. Immediately after that message one documentation-only edit was applied to §5.2: a sentence asserting that the independent check confirms `nested[0.40] == draws[0.40]` "repeat by repeat" was replaced by the construction statement plus "the independent check confirms this at repeat 0". No numeric value, table, source, test, specification, amendment or evidence byte changed. That edit produced `ac2c8b6c5c5c21a23159a512e867630bdb5ee9a04742e9f9a215bbc7a7604570`, which is the report the E02 assessment captured and rejected; the A02 record's digest field is stale by exactly that edit and is preserved unedited, and the report bytes did not change after the E02 capture. This C2/R1 documentation correction therefore starts from the assessed bytes and produces a new report digest, recorded in `artifacts/sprint-24/task-7-m13-a03.md`; it changes no stored statistical output or lineage and is not presented as having preceded the runs.

- **Which rules changed and which deviations exist (precise statement).** No frozen rule, grid, size, seed, repeat, estimator, summary, movement, trend, composition or abstention rule was changed by any correction, and the original specification bytes are unchanged. Two literal requirements of the specification are nevertheless **not met** by the implemented design, and both are disclosed rather than renamed: (i) `abnormal_sequence.scope` asks for one sequence per `cell × design`, independent of size; the implementation builds it per size because the per-stratum headroom depends on the size-specific context quota (`A01`), and the literal shared-sequence reading is infeasible at the smaller pooled sizes in 38 of 88 evaluable metadata checks (§5.2). (ii) `abnormal_sequence.nesting` asks for the abnormal **files** of dose `p` to be the first `n_abnormal(p)` entries, and therefore for the abnormal **sets** to be nested; the frozen `sampling.without_replacement` rule resamples identities per dose inside each `(stratum, family)` group, so realized abnormal file sets are not nested (74 of 168 positive-dose transitions fail subset inclusion at the reference repeat). The plan's per-group counts are nested, but that is a consequence of using the prefix for counts, not satisfaction of the file-level requirement; the two frozen clauses cannot both hold at these group sizes, and the implementation follows the sampling clause (`A02`). (iii) The R2 correction added evidence fields only; the R3 correction adds three evidence tables, one sensitivity computation and one metadata feasibility check, and no previously reported statistical value changed. Confirmation: `census.json`, `curves.json`, `baseline.json`, `arms.json`, `trends.json`, `aggregates.json` and `accounting.json` are byte-identical across the assessed R1 candidate, the assessed R2 candidate (`audit-r2/`) and the R3 candidate (`audit/`); `composition.json` changed once, in R2, and is byte-identical between R2 and R3; `support.json` gains an evidence block and `repeats.npz`/`repeats-keys.json` gain the 360 sensitivity vectors.

## 14. Accounting summary

| Item | Planned | Computed | Abstained |
|---|---|---|---|
| Units (`cell × size × design`) | 48 | 40 | 8 |
| Dose × seed records | 720 | 624 | 96 |
| Estimator cells | 2,880 | 2,496 | 384 |
| Composition records | 208 | 208 | 0 |

`all_planned_cells_accounted = true`; every abstention carries a closed reason code and stays in the denominator. `cache_unchanged = true`, `prior_products_unchanged_after_run = true`, `forwards_executed_this_run = 0`, `cumulative_forwards_after_run = 77653`.
