# Sprint 24 Prospective Research Protocol v1 (frozen, pre-execution)

- Protocol ID: `S24-PROTO-v1`; status: **frozen candidate, pre-execution** (no model comparison has been run under it).
- Task: `S24-T3`, attempt `S24-T3-A01`, role Gold, cycle `S24-T3-C1` round 1.
- Criterion: `docs/sprint-plans/sprint-24.md:97`.
- Base commits: local `master` `7a2c6987830f405ea66a3d4d74f383f93c1f3f19` (Task 2 checkpoint, pushed to `origin/master`); remote compute/source `3780b718ffd663c8d563217afdc3f9131f886b29`.
- Support basis: Task 2 audit `artifacts/sprint-24/task-2.md` + machine inventory `artifacts/sprint-24/evidence/s24-t2/s24-t2-support-inventory.json` (SHA-256 `5718e827b7b2f83257f8a474bcb04248bc6da7342cba3819b3227351bd46ecfc`).
- Feasibility proof: `artifacts/sprint-24/evidence/s24-t3/s24-t3-feasibility-checker.py` + `s24-t3-feasibility-report.json` + `s24-t3-feasibility.log` (data-only, inventory-level, no waveforms/models/scores).
- Owner boundary: Sections 10–11 require explicit owner acceptance before any assessed review round. No numeric success gate is proposed (recommendation: descriptive paired estimates only).

This document is the executable reference for Tasks 4–10. Any change after owner acceptance needs a versioned amendment (`S24-PROTO-v2`) with rationale; silent edits are forbidden.

## 1. Source identity and immutability

The exact accepted Sprint 18 corpus is read-only:

- Roster `S18-ITER-0005`, binding digest `8e2705140d07d9c603989ec709de5ebb766e580a8619b37c7462f6dd2751dd3b`; source root `/home/trietlm/anomaly-representation-learning-s18t68-a02-worktree/data/generated/sprint18-iterative-v1/S18-ITER-0005-ASSESS-P-ALLOWANCE-V1` (1,648 files, 1,850,266,373 bytes).
- All 16 histories keep original role IDs, seeds (`32064`–`32079`), raw shards, annotations, manifests and custody. New views are versioned manifests referencing original IDs; no waveform/annotation/split rewrite.
- History-local IDs (`operation`, `episode`, `failure`) collide across histories (120/120 pairs). Every new manifest MUST key records by `(history_id, file_id)` / `(history_id, episode_id)` / `(history_id, failure_id)`. A bare `fail-robot-01-0001` or `op-000000` is an invalid key.
- Cross-history wall-clock concatenation is forbidden: the 16 histories are independent simulator realizations of the same nominal 450-day calendar (same-robot nominal interval overlap in 1,080/1,080 history-pair/robot cases). All chronological schedules run **inside one history**; histories are independent replicates.

## 2. Source-to-role views (manifest family `S24-VIEWS-v1`)

Adaptation and primary evaluation use **only the 4 Development histories** (`S18I-ITER-0005-DEVELOPMENT-01..04`). Fit and Calibration are provenance-only (Sections 5, 10). Confirmation is evaluation-only (Section 10).

Per-history clock (own `tmin`, own span; absolute times differ, bin fractions identical). Cutoffs are exact recorded bin edges from the Task 2 audit (no re-derivation at runtime):

| History | `c1` (bin3/bin4 edge, 40% time) | `c2` (bin6/bin7 edge, 70% time) |
|---|---|---:|
| DEVELOPMENT-01 | `12902229.909` s | `22578902.340` s |
| DEVELOPMENT-02 | `12897362.418` s | `22570384.231` s |
| DEVELOPMENT-03 | `12900372.819` s | `22575652.433` s |
| DEVELOPMENT-04 | `12900060.198` s | `22571624.329` s |

File membership by `end_time`: `V-ADAPT` = bins 0–3 (`end_time <= c1`); `E1/E2/E3` = bins 4/5/6 (`c1 < end_time <= c2`); `V-FUTURE` = bins 7–9 (`end_time > c2`); `F'` (new-fault scenario only) = bin 9.

| View | Support per history | Purpose |
|---|---|---|
| `V-ADAPT` | bins 0–3 (≈2,574–2,612 files; ab 538–624; eligible-healthy 750–821) | label-budget training pool (Task 7), initial head (Task 9) |
| `V-VAL` | fixed 60 files from `V-ADAPT` (12 ab / 48 normal), seed `2401`, removed from all training frames | threshold calibration + early stopping only |
| `V-TRIG` | fixed 100 files from `V-ADAPT` (20 ab / 80 normal), seed 2410, disjoint from `V-VAL`, `V-R0` and all budget frames | fine-tuning trigger decision only (Section 7); never training, stopping, threshold, or primary evaluation |
| `V-R0` | fixed 50 files from bins 0–1 (10 ab / 40 normal), seed `2403`, disjoint from `V-VAL` and budget frames | retention probe, reporting-only, never training/stopping |
| `V-SEQ-E1..E3` | bins 4/5/6 (each ≥122 ab, ≥162 eligible-healthy per history) | sequential experiences (Task 9) |
| `V-FUTURE` | bins 7–9 (≈1,930–1,969 files; ab 374–474; eligible-healthy 459–573) | future/forward evaluation |
| `V-CONF-EVAL` | all 4 Confirmation histories, whole-history | secondary evaluation-only scoring (frozen heads, no tuning) |
| `V-RESERVE-SLICE` | `robot-08` / `program-03` files inside any eval support | unseen-identity generalization slice (Section 9), not a regime claim |
| `V-NF-E4` / `V-NF-F` | bins 7–8 / bin 9 (new-fault scenario only) | post-release experience / future eval (Section 9) |

Training eligibility (frozen predicate, extends the B0 observable rule to abnormal rows): `not quarantined` AND `not censored` AND `robot_id != "robot-08"` AND `program_id != "program-03"` AND no manifest maintenance-window overlap. Normal rows passing it are supervised negatives; abnormal rows passing it are supervised positives. Reserve identities never enter any training/validation frame.

Grouping rule (frozen): group by `(history_id, robot_id, merged related-event interval)`; merge failure windows with same-robot overlapping degradation episodes (Task 2: 4,194 merged intervals corpus-wide, 250–280 per Development history); never cross a maintenance reset boundary; eligible-healthy files inside intervals move with their episode; abnormal files outside any interval attach to the nearest same-robot interval by `|file end_time − interval end|`; gap `> 7` days (604,800 s) or no same-robot interval → singleton group `(history_id, file_id)`. No file's group is ever split across adaptation/validation/evaluation frames; sampling is without replacement at file level and reported at episode level (Section 8).

## 3. Nested label budgets (Task 7)

Annotation unit = **file**. Effective independent episode counts (distinct `(history_id, merged-interval)` groups spanned) are reported per cell at runtime; they are an observed quantity, not a sampling target.

Sampling: stratified (abnormal/normal-passing) without replacement from `V-ADAPT` minus (`V-VAL` U `V-TRIG` U `V-R0`), seed `2402`, nested `B1 ⊂ B2 ⊂ B3`, identical ID sets across all methods:

| Budget | Files/history | Abnormal | Normal | Minimum pool margin (worst history) |
|---|---|---:|---:|---|
| `B1` | 40 | 8 | 32 | pool ab 538, eh 750 |
| `B2` | 120 | 24 | 96 | same pool |
| `B3` | 320 | 64 | 256 | pool ab ≥ 2×128 ✓, pool eh ≥ 2×512 ✓ (checker asserts) |

Full annotation accounting per cell: budget labels + 60 `V-VAL` labels + 100 `V-TRIG` labels (Task 7 trigger evaluation) + 50 `V-R0` probe labels + sequential revealed-label counts + replay memory (files). Validation/threshold/trigger/probe labels are costs, never free. The smallest budget is feasible without replacement or hidden oversampling (checker Check B).

## 4. Label delays and release policy

- Task 7: all `V-ADAPT` labels released once at `c1 + 86,400` s (period close + 1-day simulated annotator latency).
- Task 9 main scenario: experience `Ek` labels released at `end(Ek) + 86,400` s. Predict-before-update ordering is mandatory: the learner predicts `E(k+1)` and `V-FUTURE` **before** consuming `Ek` labels; next-period evaluation is never touched by the current update.
- New-fault scenario: withheld-family labels released at `c2 + 86,400` s (Section 9).
- A withheld-family scenario may use labels for declared offline scenario construction, but the learner sees only labels released by that scenario (no future-label use for sampling, balancing, threshold choice, replay selection or stopping).

## 5. Method matrix (frozen)

Primary representation: accepted B0-FULL seeds `181801/181802/181803` (identity-selected, not outcome-ranked). File embedding = existing 128-d file vector (`d_model = 128`); mechanics resolved by Task 5 through public entry points only.

| Method | Encoder | Head / training | Trainable params |
|---|---|---|---|
| `M0a/M0b/M0c` | frozen B0 (per seed) | no-new-label comparators: `S_pred`, `S_pop`, N4-CRN `S_pred` at frozen S18 q95 thresholds | 0 |
| `M1` | frozen B0 (per seed) | linear head 128→1, per-feature standardization fitted on the head-training subset only | 129 |
| `M2` | randomly initialized frozen encoder, same architecture, init seeds `241801/241802/241803` | identical head + protocol as `M1` | 129 |
| `M3` | same V1 architecture from scratch on identical revealed labels | identical head shape; all weights trainable; fixed fair budget (below) | 1,938,819 |
| `M4` | conditional, trigger-gated (Section 7) | partial fine-tune: sequence context encoder only | 793,473 |

Reference banks stay fixed (Fit bank 5,446 rows, rows SHA-256 `447b219e0c77f6d76ed54a4de7a0bee7c2b3a9a8142f61cbb434694ed73acb94`, bank SHA-256 `2d6e0a117feb7e332792d22e0ecd2d4ab64ef273290e9d9139f9e1efa2416750`; Calibration q95 `S_pred = 0.13165724277496332`, `S_pop = 5.98599042892456` on 1,877 healthy rows). `ConditionalBatchNorm` runs in eval mode (no refit). Online bank adaptation is excluded. N4 shares the frozen B0 representation; no score fusion.

## 6. Optimizer, compute and seeds (frozen, no search)

- `M1/M2` head: AdamW, lr `1e-3`, batch 128, ≤200 epochs, early stopping on `V-VAL` loss (patience 20). Head-init seed `2405`, shuffle seed `2406`.
- `M3` scratch: AdamW cosine `1.5e-3 → 1e-5`, batch 128, **fixed 100 epochs** (fair declared budget, not a strawman; identical labels ≠ identical total compute). Init seed `2408`, shuffle seed `2409`.
- Sequential updates (Task 9): same head config, ≤50 epochs, early stopping on frozen `V-VAL`; continual arms warm-start from previous head; cumulative reference re-initialized each experience (retrospective upper reference, not an online method, no future labels).
- Search budget: **zero** (single frozen config per method; no HPO, no architecture search).
- Seeds frozen: models `181801/2/3`; random encoders `241801/2/3`; validation `2401`; budgets `2402`; probes `2403`; replay reservoir `2404`; head init/shuffle `2405/2406`; bootstrap `2407`; scratch `2408/2409`.
- Compute log per cell: wall-time, device, trainable params, update/inference time, pretraining cost (B0: 100 epochs, 544,600 file-epochs, batch 128) vs adaptation cost. Pretraining cost reported separately; label-budget equality never implies compute equality.

## 7. Conditional fine-tuning trigger (frozen, pre-outcome)

Evaluated once in Task 7 on the `B3` cells. Trigger `M4` iff **both** hold on the frozen independent trigger slice `V-TRIG` only (file-level PR-AUC, macro over the 4 Development histories, paired on history-seed) -- never on primary evaluation (`V-FUTURE`/`V-CONF-EVAL`), never on `V-VAL` (stopping/threshold), never on any sequential experience:

1. `mean(M3 − M1) > 0.05` (material transfer gap), AND
2. `mean(M1) < 0.70` (absolute inadequacy of the frozen head).

Otherwise record `NOT_TRIGGERED` (an observed disposition, never an unrun positive claim). If triggered, `M4` scope = unfreeze sequence context encoder only (793,344 + 129 params), same `B3` labels/config, carried through the identical retention evaluation; reported separately from frozen-encoder transfer. These thresholds are **trigger mechanics for a bounded diagnostic**, not success gates — there is no passing score anywhere in this protocol (Section 11).

## 8. Metrics, uncertainty and reporting (frozen)

Per planned cell report: prevalence/class counts, labeling + replay memory costs, update/inference time, trainable params, pretraining vs adaptation compute. Publish all methods/budgets/seeds, failures and unsupported cells; no favorable-seed selection.

- File level: PR-AUC, ROC-AUC (undefined single-class endpoints reported as `unavailable`, never zero), recall and FPR at the declared operating point.
- Operating point: `M0*` at frozen S18 q95 (continuity); `M1/M2/M3(/M4)` at Youden's J on frozen `V-VAL` — the **rule** is shared and prospective, numeric values differ per (history, method, budget). `V-VAL`/`V-CONF-EVAL` labels never tune anything else.
- Event level (where structure supports): event recall = fraction of failure windows with max file score ≥ τ; false-alarm episodes per 100 eval-support days (above-τ runs outside merged intervals, same-robot gap < 1 day merged); detection delay = first above-τ file `end_time` − degradation onset (clamped ≥ 0), median + IQR among recalled only; non-recalled events are censored (counted in recall, excluded from delay). File-level FPR is never substituted for event false alarms.
- Sequential matrix (Task 9): after **every** update report retention (`V-R0`), current-experience heldout, next-period support and `V-FUTURE`. Forgetting per history = best past `V-R0` score − current `V-R0` score (same fixed probe); new-task acquisition = post- vs pre-update heldout on `Ek`; forward = `V-FUTURE`; false alarms after each update. Probe results are reporting-only, never model-selection signals.
- Uncertainty: paired method differences on matched (history, encoder-seed) units; bootstrap over the **history** unit (`B = 2000`, seed `2407`); `n = 4` Development histories is an explicit small-history limitation — wide intervals are expected and reported as-is; a within-arm interval is never read as a paired interval.
- Replay: buffer `K = 256` files, class-balanced reservoir (cap 128 normal / 128 abnormal), past-only, eviction seed `2404`; no future example ever enters replay. Cumulative-label reference uses all revealed labels under the declared per-experience compute budget.

## 9. New-fault / new-regime / identity scenarios (frozen)

- New-fault scenario `S-NF` (withheld family `F* = missing_event`): excluded from every Task 7 and `E1–E3` training/validation frame; labels released at `c2 + 86,400` s; post-release experience `V-NF-E4` = bins 7–8 (all eligible labels incl. `F*`); future eval `V-NF-F` = bin 9; `F*`-stratified metrics reported pre/post release. Inventory basis: 134–148 abnormal files/history, first appearance < `c1`, last appearance > `c2` (checker Check D). Runtime guardrail: if `F*` abnormal < 20 in `V-NF-E4` or < 10 in `V-NF-F` in any history → that cell is `UNSUPPORTED` (reported with counts, never fabricated).
- New-regime at regime-type level: **unsupported** — all six regime types (`active`, `idle`, `periodic`, `ramp_down`, `ramp_up`, `recovery`) are present in the 5,446 Fit-healthy pretraining rows. No categorical new-regime cell exists.
- Declared family `duration_anomaly`: **unsupported** — zero files in all 16 histories.
- Unseen-identity slice `V-RESERVE-SLICE` (`robot-08`: 164–206 abnormal/history; `program-03`: 194–227 abnormal/history in Development): reported as deployment-identity generalization via the normalization fallback, explicitly **not** a new-regime result. Runtime guardrail: slice reported only when the eval period holds ≥ 10 reserve abnormal files per history, else `UNSUPPORTED` for that slice.
- A repeated ordinary batch of an already-labeled family is repeated support, never a new fault. `family_only_in_confirmation` is empty — no Confirmation-only adaptation target exists.

## 10. Secondary-use boundary (requires owner acceptance, no S18 alteration)

- Development histories were forward-scored and inspected in Sprint 18. Adaptation + primary evaluation on Development is **secondary research on an exposed benchmark**, not fresh independent Confirmation. Primary claims are bounded to within-benchmark label-efficient transfer and sequential retention; no fresh-generalization claim follows from any manifest name.
- Confirmation stays **evaluation-only**: frozen heads scored once on `V-CONF-EVAL`; no adaptation, tuning, checkpoint selection or threshold fitting on Confirmation. Confirmation is reported as secondary context, not as an independent success test.
- Fit/Calibration rows are provenance-only (pretraining/bank/threshold fitting already done in S18); no Sprint 24 adaptation touches Fit rows.
- Claim language is fixed: results are **sequential label adaptation with prior unlabeled exposure** (encoder trained 100 epochs on Fit-healthy rows spanning the full timeline; Development waveforms forward-exposed during S18 scoring). The strictly prospective end-to-end temporal claim is **not available**; a strictly causal claim would need a separately approved training design, not concealment.
- Sprint 18 verdict (`UNRESOLVED`), gates, lock and records are unchanged and unreopened.

## 11. Tolerances (recommendation: none numeric)

No numeric performance target, success gate, recovery conjunction or forgetting tolerance is proposed or approved. All comparisons are descriptive paired estimates with uncertainty and stated limits (Section 8); negative and unsupported findings are valid outcomes. The `0.05 / 0.70` values in Section 7 are predeclared diagnostic-branch mechanics inside this frozen protocol, not performance tolerances — their acceptance as part of the protocol is requested in the owner gate (task-3 owner items O3–O4).

## 12. Runtime and custody contract (for Tasks 4–10)

- Implement against existing repo conventions (`src/representation/*` public entry points, `sprint18_harness` permission checks); no new framework unless Task 4 justifies it.
- Cache/run identity includes checkpoint SHA, bank SHA, source support digest and preprocessing hash. Runtime banks fixed (Section 5 digests).
- Every view manifest records parent lineage (source manifest SHA, config hash, seed, affected IDs, purpose, output hash) with an unchanged-baseline comparison; new generation needs a prospective amendment.
- Execute each planned (method, budget, seed, experience) cell once; preserve failed attempts and any explicitly approved operational corrections; no evaluation-driven expansion of search.
- Simulator labels are an explicitly simulated feedback source, not evidence of real annotator quality; no simulator variable becomes a predictor input.

## 13. Owner acceptance record (Phase A Task 3 protocol only)

- Received via Main (owner ask; auto-selected responses are owner answers per standing user instruction): **O1 Duyet ranh gioi nay; O3 Duyet descriptive + trigger; O5 Duyet gioi han support** — recorded verbatim as delivered.
- Approved: (a) Development secondary reuse for adaptation + temporal evaluation (no fresh-generalization claim); Confirmation evaluation-only with no tuning/update/selection; exposure-qualified sequential label-adaptation claim (not strict prospective end-to-end, not fresh Confirmation); Sprint 18 `UNRESOLVED` verdict/gates/lock/records unchanged. (b) No numeric success/forgetting gate — descriptive paired estimates with uncertainty only. (c) `V-TRIG` separate 100-file seed-`2410` validation-only branch mechanics `mean(M3-M1) > 0.05` AND `mean(M1) < 0.70`, bounded context-encoder fine-tune on trigger. (d) Unsupported frozen: new-regime-type and `duration_anomaly` unsupported; `robot-08`/`program-03` as guardrailed unseen-identity slice only.
- Scope: this approval covers the Phase A Task 3 frozen protocol only. It grants **no** Phase B/C execution release (Tasks 4–10 machinery and runs remain separately authorized).
- No protocol choice changed on approval; Sections 1–12 stand as frozen pre-execution.
