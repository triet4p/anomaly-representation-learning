# Sprint 17 — Controlled Suspect-Component Architecture Ablation

## Sprint Goal

Run a prospectively frozen, controlled architecture ablation over the seven qualified Sprint 16 SUSPECT components, using the accepted Sprint 15 Candidate 7 `MEASURABLE` data-generation methodology, to determine whether one-component replacements or predeclared multi-component combinations reproducibly improve anomaly-signal retention over the unchanged architecture baseline.

**Status:** PAUSED before Batch E / Task 23. Tasks 1–22 are complete. K1–K9 all remain `VALID_NEGATIVE`; the strongest Development result is K9 at ΔP+W +0.0432, below the frozen +0.05 recovery threshold. The independent Batch D evidence review passed with zero actionable findings at `artifacts/sprint-17/review-batch-d.md`, including exact K-map/provenance, 27/27 per-seed deltas, 18/18 interaction aggregates, and all frozen gates. Results retain `MEASURABLE_WITH_USER_WAIVER`; no recovery is claimed. Confirmation and Sprint 15 Sealed histories remain untouched. Stop before Task 23; resumption requires new explicit user instruction.

## Decision Context

Sprint 16 closed `UNRESOLVED`: C2/C3/C5/C6/C7/C8/C9 are qualified/provisional `SUSPECT`, C1 is `UNRESOLVED / NOT PRESENT`, C4 is `UNRESOLVED`, and no component has a valid replicated R3/BOTTLENECK path. Sprint 17 therefore tests explicit replacement hypotheses; it does not assume any suspect is defective.

The comparison baseline `B0` is the unchanged accepted representation-to-score architecture, retrained, calibrated, and refit on the same Sprint 17 roles, model seeds, optimizer-step budget, and checkpoint-selection rule as every trainable alternative. Old weights or a restored old reference bank are not the fairness baseline because they were fitted on different data.

C1 and C4 remain fixed at `B0` behavior. They are not ablation axes because Sprint 16 did not mark them `SUSPECT`. Adding C1/C4 experiments requires a separately reviewed protocol amendment before any outcome is inspected.

## Frozen Data Principle

Sprint 17 must use the exact accepted Sprint 15 Candidate 7 generation methodology: the `sprint15-v7` physics, scheduler, allocator, causal windows, balanced P/W/A quotas, nuisance envelopes, reset/quarantine/censoring rules, and structural/measurability gates. It must create fresh, disjoint Sprint 17 Fit, Calibration, Development, and Confirmation histories under a prospectively frozen seed namespace so architecture selection does not reuse Sprint 15 Confirmation outcomes.

- Fit: representation fitting and healthy-reference rows only.
- Calibration: the same frozen quantile-threshold procedure only; no evaluation or architecture selection.
- Development: individual-alternative comparison and deterministic per-component selection.
- Confirmation: untouched until the individual and combination matrix is frozen; one-shot final comparison.
- Sprint 15 Sealed H-SEAL-37..40 remain forbidden to open, score, probe, enumerate, or use.
- Every root is generated once. No retry, replacement, omission, pooling for rescue, or outcome-selected seed is allowed.
- The frozen Sprint 15 observable probe and measurability thresholds must certify the new Development and Confirmation roles before model conclusions are eligible.

## Replacement Registry

Each single-component arm changes exactly one principal mechanism relative to `B0`. Necessary shape adapters must be minimal, declared, parameter-counted, and held constant across the paired comparison. Labels, anomaly masks, failure categories, severity, and simulator hidden state remain post-hoc diagnostics only.

| Component | Sprint 16 lead | Alternative A | Alternative B |
|---|---|---|---|
| C2 patchification | Short/boundary anomaly dilution; C2/C6 joint caveat | `C2-A overlap`: fixed-width windows at 50% overlap, preserving valid-length, padding, starts, and file identity | `C2-B multi-offset`: two fixed-width offset grids with unit-mass support weights so duplicated timesteps do not gain pooling weight |
| C3 local encoder | Observable-to-local-latent gap; bypass unavailable without redesign | `C3-A TCN`: parameter-matched residual depthwise-separable dilated 1D convolutional patch encoder | `C3-B patch-attention`: parameter-matched lightweight within-patch Transformer with explicit position and padding masks |
| C5 objective/masking | Objective sensitivity without replicated retrain win | `C5-A structured-predict`: fixed total mask ratio with channel-time blocks and multi-horizon EMA latent prediction | `C5-B redundancy-control`: retain EMA masked prediction but replace file-level InfoNCE with VICReg-style variance/covariance regularization |
| C6 file pooling | Alternative fragility; production mean showed no systematic loss | `C6-A statistics`: valid-patch mean-plus-standard-deviation pooling projected to the existing file-embedding dimension | `C6-B attention`: gated attention pooling over valid patches, with no anomaly-derived input or target |
| C7 geometry/reference bank | Directional bank-swap lead failed replication | `C7-A robust-shrinkage`: per-condition robust location plus Ledoit-Wolf-style covariance shrinkage with the frozen hierarchy/fallback | `C7-B local-density`: condition-aware kNN/local-density score with robust scale fitted only from Fit healthy-reference rows |
| C8 scorer | Alternative fragility; production scorer showed no measured readability loss | `C8-A robust-residual`: non-query-conditioned standardized Huber prediction-residual energy for `S_pred` | `C8-B cosine-residual`: non-query-conditioned cosine distance between predicted and target latents for `S_pred` |
| C9 patch-to-file aggregation | Median alternative failed; production tail appeared healthy | `C9-A top-k`: fixed-fraction top-k mean over valid patch scores | `C9-B contiguous-tail`: maximum fixed-duration contiguous-window mean, using patch starts/valid lengths |

C7 alternatives affect `S_pop`; C8 alternatives affect `S_pred`. The two scores must remain independently reported. Fusion is outside Sprint 17.

## Ablation Matrix

### Individual arms

Run `B0` plus all 14 registered alternatives individually. Trainable arms use the same frozen model seeds, Fit data, optimizer steps, schedule, checkpoint rule, and parameter envelope. C7/C8/C9 arms must reuse eligible frozen latent/prediction caches when doing so avoids redundant computation without changing semantics.

### Deterministic per-component selection

Development outcomes select at most one alternative per suspect for combination construction. The rule is frozen before any model outcome:

1. reject arms violating data, provenance, score-separation, finite-output, localization, or nuisance/background gates;
2. rank remaining A/B arms by paired Development `P+W` macro improvement over `B0`;
3. break ties by the worse of P and W improvement, then history-block LCB, then smaller parameter/FLOP delta, then lexical arm ID;
4. if neither arm is valid, the component is omitted from combinations and recorded as unresolved; never invent a replacement after outcomes.

Selection is a Development decision only. Confirmation is not opened until the selected arm IDs and all combination configs are frozen.

### Predeclared multi-component combinations

Build combinations only from the deterministically selected per-component arms:

| ID | Components | Interaction tested |
|---|---|---|
| K1 | C2 + C3 | Patch support × local representation |
| K2 | C3 + C5 | Local representation × learning pressure |
| K3 | C5 + C6 | Learning pressure × file readout |
| K4 | C6 + C7 | File embedding × healthy geometry |
| K5 | C7 + C8 | Geometry × score readability |
| K6 | C8 + C9 | Patch score × file aggregation |
| K7 | C2 + C3 + C5 + C6 | Complete upstream representation stack |
| K8 | C7 + C8 + C9 | Complete downstream scoring stack |
| K9 | C2 + C3 + C5 + C6 + C7 + C8 + C9 | End-to-end selected replacement stack |

This bounded adjacent-pair/stack matrix replaces an unconstrained full factorial search. Every combination uses the exact single-arm implementation; no combo-specific tuning is allowed.

## Common Evaluation Contract

- Primary: tie-aware event `P+W` macro AUROC as the unweighted mean under common support.
- Required companions: P macro, W macro, per-history values, history-block bootstrap `B=2000` with seed `20260202` and 95% LCB, directional count above 0.55, severity ordering, localization, unaffected-background stability, and category slices.
- Calibration: the same fixed quantile rule applied per arm using Calibration only. No Development/Confirmation threshold tuning.
- Nuisance gate: FAR `<= 0.05` per robot-day and `Delta FAR <= +0.02` versus paired `B0`, with the accepted <=2-day grouping and reset split.
- Improvement must be reported as paired arm-minus-`B0` deltas on identical histories, events, supports, model seeds, and metric code. Pooled-only or one-seed wins do not count.
- A Confirmation recovery requires positive `P+W` improvement on at least 3/4 histories, post-arm LCB `> 0.55`, no negative P or W macro regression beyond the frozen tolerance, and both nuisance limits. Task 1 freezes the exact minimum effect and non-inferiority tolerance before outcomes.
- Raw/handcrafted observable features remain a diagnostic ceiling, never a model input or production detector.
- No calibrated risk, probability-quality, deployment threshold, fleet prevalence, or independent sealed-performance claim is allowed.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done.

### Batch A — Prospective contract, data, and harness

- [x] **Task 1 — Freeze the Sprint 17 ablation protocol.** Record `B0`, the 14 immutable single arms, K1–K9, model seeds, parameter/compute envelope, exact minimum-effect/non-inferiority thresholds, metrics, promotion rule, stopping rules, and artifact schema before outcomes. Evidence: `experiments/sprint17-ablation-protocol-v2.md` (v1 preserved as superseded history), `artifacts/sprint-17/task-1.md`. (Done: v2 corrects the B0 schema representation to `arm_kind=baseline`, `one_principal_change=none`, `component_set=[]`; focused registry/schema/boundary checks passed. No outcomes or Sealed content inspected.)
- [x] **Task 2 — Bind fresh Sprint 17 data roles.** Freeze a collision-free seed namespace and Fit/Calibration/Development/Confirmation roster that delegates to the accepted `sprint15-v7` generation methodology without changing its physics, scheduler, allocator, quotas, nuisance, or causal rules. Evidence: `experiments/sprint17-role-binding-v2.json` (v1 preserved as superseded history), `artifacts/sprint-17/task-2.md`. (Done: 16 literal role/seed bindings and canonical digest recorded; prior seed bands explicitly enumerated; no roots materialized.)
- [x] **Task 2A — Freeze the prospective protocol-v3 restart.** Preserve the complete v2 failure record; keep the accepted Sprint 15 Candidate 7 method, gates, ablation registry, metrics, and Sealed prohibition unchanged; authorize exactly one entirely new 16-role roster with no reused v2 seed, reserve block, per-seed replacement, retry, or favorable selection. Evidence: `experiments/sprint17-ablation-protocol-v3.md`, `artifacts/sprint-17/task-2a.md`. (Done: v3 amendment and focused schema/history/boundary checks passed; v2 bytes and fail-closed artifacts remain preserved.)
- [x] **Task 2B — Bind the all-new protocol-v3 role roster.** Freeze all 16 new role IDs, seeds, paths, permissions, prior-band exclusions, generator digests, and canonical binding digest before preflight; no root or outcome may be created. Evidence: `experiments/sprint17-role-binding-v3.json`, `artifacts/sprint-17/task-2b.md`. (Done: exact `2800..2815` roster, 10 explicit exclusion bands, 11/11 method bytes, and digest `075868b22c028a981cc63ffe37b8d29720cd324c3e2c7254ce688678758230ba` verified.)
- [x] **Task 2C — Pass the protocol-v3 amendment evidence review.** Verify v2 failure preservation, explicit user authorization, exact generator-method continuity, full-roster replacement rather than single-seed rescue, prospectiveness, one-shot/no-reserve rules, Sealed non-touch, and zero actionable findings before Task 3 resumes. Evidence: `artifacts/sprint-17/review-protocol-v3.md`. (Done: PASS with zero actionable findings; non-actionable INFO notes N1–N3 are recorded faithfully in the review artifact.)
- [x] **Task 3 — Preflight and materialize the protocol-v3 frozen roots once.** After Task 2C passes, prove the entirely new roster feasible, then generate every authorized root once with no retry/replacement/omission and record manifests, hashes, runtime, and role isolation. The protocol-v2 blocker remains preserved at `artifacts/sprint-17/task-3.md`; protocol-v3 evidence goes to `artifacts/sprint-17/task-3-v3.md`. (Done: single rejection-only preflight `PREFLIGHT-PASS 16/16` over `2800..2815` persisted before assertion at `artifacts/sprint-17/task3-v3-preflight.json`; all 16 authorized roots materialized exactly once under `data/generated/sprint17-ablation-v3/` with per-root hashes/runtimes in `artifacts/sprint-17/task3-v3-materialization.json`; focused integrity verification passed with Confirmation untouched.)
- [x] **Task 4 — Certify structural and observable measurability.** Preserve the original protocol-v3 `NOT_MEASURABLE` result and raw evidence. Task 4B passed with zero actionable findings, so the qualified closure is exactly `MEASURABLE_WITH_USER_WAIVER`: Development and Confirmation pass every inherited gate, while the exact Design-only P-share exception remains disclosed. Evidence: `artifacts/sprint-17/task-4.md` (original immutable record), `artifacts/sprint-17/task-4-v4.md` (qualified closure).
- [x] **Task 4A — Freeze the user-authorized Task 4 bypass.** Create protocol v4 as a post-observation waiver for exactly `H-S17-V3-DESIGN-03 / cohort_mix_15_60 / P / 91÷149=0.6107382550 > 0.60`; preserve every raw result, root, generator rule, arm, metric, and all other gates unchanged. The waiver must never be described as an unqualified `MEASURABLE` result. Evidence: `experiments/sprint17-ablation-protocol-v4.md`, `artifacts/sprint-17/task-4a.md`. (Done: v4 schema requires exact `MEASURABLE_WITH_USER_WAIVER` qualification and rejects missing, altered, or unqualified waiver identity; Task 4B PASS is recorded in `artifacts/sprint-17/review-task4-waiver.md`.)
- [x] **Task 4B — Pass the Task 4 waiver evidence review.** Verify explicit user authorization, exact one-cell scope, preservation of the original failure, zero data/root changes, unchanged Development/Confirmation gates, mandatory `MEASURABLE_WITH_USER_WAIVER` labeling, and zero actionable findings before Task 5. Evidence: `artifacts/sprint-17/review-task4-waiver.md`. (Done: fresh review PASS with zero actionable findings; non-actionable notes and unverified limits are recorded.)
- [x] **Task 5 — Implement the common ablation harness.** Add registered arm configs, one-principal-change enforcement, common seeds/budgets/checkpoint selection, cache provenance, paired evaluation, and fail-fast score-separation checks. Evidence: `artifacts/sprint-17/task-5.md`. (Done: `src/representation/sprint17_ablation.py` provides the execution-free registry/parity/provenance/eligibility/support contract; focused tests and public dry smoke pass; no training or data access occurred.)
- [x] **Task 6 — Pass Batch A evidence review.** Verify protocol prospectiveness, exact generator-method continuity, fresh-role isolation, Sealed non-touch, harness invariants, and zero actionable findings. Evidence: `artifacts/sprint-17/review-batch-a.md`. (Done: PASS with zero actionable findings; execution paused at the explicit pre-GPU boundary before Task 7.)

### Batch B — Unchanged architecture baseline

- [x] **Task 7 — Train and evaluate `B0` on Development.** Retrain the unchanged architecture on frozen Fit across all model seeds, refit its healthy bank on Fit, calibrate by the frozen rule, and produce Development metrics/caches. Evidence: `artifacts/sprint-17/task-7.md`. (Done: 3 seeds × 300 steps @ `8c15f02` on RTX 4060 Ti; Fit bank 5040 rows; q95 calibration; S_pred P+W 0.668/0.666/0.682, S_pop 0.688/0.663/0.691; schema-validated `VALID_NEGATIVE`; Task 7 evidence review PASS with zero actionable findings.)
- [x] **Task 8 — Audit baseline reproducibility and support.** Confirm deterministic config identity, seed-complete runs, finite outputs, common event support, score separation, parameter/compute accounting, and measurable oracle ceiling. Evidence: `artifacts/sprint-17/task-8.md`. (Done: 60/60 mechanical checks PASS plus CPU checkpoint-reload rescore PASS; oracle ceiling 0.7981 vs B0 0.6720; Task 8 evidence review PASS with zero actionable findings.)
- [x] **Task 9 — Pass Batch B evidence review.** Require an eligible, reproducible `B0` before any alternative comparison. Evidence: `artifacts/sprint-17/review-batch-b.md`. (Done: PASS with zero actionable findings; Task 7–8 evidence supports an eligible reproducible `B0`; Sprint paused before Batch C / Task 10.)

### Batch C — Individual suspect replacements on Development

- [x] **Task 10 — Execute C2 patchification alternatives.** Compare `C2-A` and `C2-B` individually against `B0`; keep every downstream mechanism unchanged and preserve timestep-to-patch localization. Evidence: `artifacts/sprint-17/task-10.md`. (Done: both arms 3 seeds × 300 steps @ `5e6582c` on RTX 4060 Ti; paired Development deltas vs hash-verified B0; C2-A VALID_NEGATIVE parity control ΔP+W −0.000000 S_pred; C2-B VALID_NEGATIVE ΔP+W −0.0044 S_pred / +0.0047 S_pop with background-stability fail on DEV-01/03/04; support `800fc825…` exact match; Task 11 next.)
- [x] **Task 11 — Execute C3 local-encoder alternatives.** Compare `C3-A` and `C3-B` individually with parameter/compute accounting and identical training/evaluation support. Evidence: `artifacts/sprint-17/task-11.md`. (Done: both arms 3 seeds × 300 steps @ `7eb63e7` on RTX 4060 Ti; paired Development deltas vs hash-verified B0; C3-A VALID_NEGATIVE ΔP+W −0.0101 S_pred / −0.0086 S_pop; C3-B VALID_NEGATIVE ΔP+W −0.0153 S_pred / −0.0033 S_pop with background-stability fail on DEV-03/04; support `800fc825…` exact match; Task 12 next.)
- [x] **Task 12 — Execute C5 objective/masking alternatives.** Compare `C5-A` and `C5-B` individually while keeping architecture, total mask ratio, inputs, and model-selection basis frozen except for the declared objective mechanism. Evidence: `artifacts/sprint-17/task-12.md`. (Done: both arms 3 seeds × 300 steps @ `8dea7fd` on RTX 4060 Ti; paired Development deltas vs hash-verified B0; C5-A VALID_NEGATIVE ΔP+W +0.0323 S_pred (4/4 histories positive but below +0.05) with background-stability fail on DEV-04; C5-B VALID_NEGATIVE dormant-λ parity control ΔP+W +0.000011 S_pred; support `800fc825…` exact match; Task 13 next.)
- [x] **Task 13 — Execute C6 pooling alternatives.** Compare `C6-A` and `C6-B` individually over valid patches while holding learned patch representations and downstream evaluation fixed. Evidence: `artifacts/sprint-17/task-13.md`. (Done: both arms 3 seeds × 300 steps @ `5ee3e04` on RTX 4060 Ti; paired Development deltas vs hash-verified B0; C6-A VALID_NEGATIVE ΔP+W +0.0090 S_pred / +0.0168 S_pop with background-stability fail on DEV-04; C6-B VALID_NEGATIVE ΔP+W -0.0155 S_pred / -0.0041 S_pop with background-stability fail on DEV-01/04; support `800fc825...` exact match; Task 13 evidence review pending, Task 14 next.) Compare `C6-A` and `C6-B` individually over valid patches while holding learned patch representations and downstream evaluation fixed. Evidence: `artifacts/sprint-17/task-13.md`.
- [x] **Task 14 — Execute C7 geometry/bank alternatives.** Compare `C7-A` and `C7-B` individually using only Fit healthy-reference rows and explicit bank row/source provenance; keep `S_pop` separate. Evidence: `artifacts/sprint-17/task-14.md`. (Done: both arms train-free @ `5eee50e` on RTX 4060 Ti, frozen B0 step-300 states, fidelity 0.0; S_pred bitwise B0 (deltas exactly 0.0); C7-A VALID_NEGATIVE ΔP+W S_pop -0.0128 (all seeds/histories negative); C7-B VALID_NEGATIVE ΔP+W S_pop -0.0016 (mixed); support `800fc825...` exact match; Task 15 next.) Compare `C7-A` and `C7-B` individually using only Fit healthy-reference rows and explicit bank row/source provenance; keep `S_pop` separate. Evidence: `artifacts/sprint-17/task-14.md`.
- [x] **Task 15 — Execute C8 scorer alternatives.** Compare `C8-A` and `C8-B` individually with frozen representations/targets and keep `S_pred` separate from `S_pop`. Evidence: `artifacts/sprint-17/task-15.md`. (Done: both arms train-free @ `9d5c8e3` on RTX 4060 Ti, frozen B0 step-300 states, fidelity ≤4.1e-05; S_pop bitwise B0 (deltas exactly 0.0); C8-A VALID_NEGATIVE ΔP+W S_pred +0.0044 with background-stability fail on DEV-03; C8-B VALID_NEGATIVE ΔP+W S_pred -0.0004 (near-parity) with marginal background-stability fail on DEV-04; support `800fc825...` exact match; Task 16 next.) Compare `C8-A` and `C8-B` individually with frozen representations/targets and keep `S_pred` separate from `S_pop`. Evidence: `artifacts/sprint-17/task-15.md`.
- [x] **Task 16 — Execute C9 aggregation alternatives.** Compare `C9-A` and `C9-B` individually over identical valid patch scores and preserve localization/duration accounting. Evidence: `artifacts/sprint-17/task-16.md`. (Done: both arms train-free @ `55af8f2` on RTX 4060 Ti, frozen B0 step-300 states, fidelity ≤4.1e-05; S_pop bitwise B0 (deltas exactly 0.0); C9-A VALID_NEGATIVE ΔP+W S_pred −0.0087 (uniform negative seeds, DEV-03 −0.0284) with background-stability fail on DEV-04; C9-B VALID_NEGATIVE ΔP+W S_pred −0.0073 (mixed seeds −0.016/+0.008) with background-stability fail on DEV-04; support `800fc825…` exact match; Task 17 next.)
- [x] **Task 17 — Select one Development arm per suspect.** Apply only the frozen deterministic rule, publish every paired single-arm result and invalid arm, and freeze the selected arm IDs without opening Confirmation. Evidence: `artifacts/sprint-17/task-17.md`. (Done: mechanical audit over the 7 hashed frozen summaries under protocol v4 §5/§5.1; selected {C2-A, C3-A, C5-A, C6-A, C7-A (level-5 lexical over a bitwise-exact tie), C8-A, C9-B}; 0 unresolved; no recovery (max ΔP+W +0.0323); no execution, no Confirmation/Sealed; Task 18 next.)
- [x] **Task 18 — Pass Batch C evidence review.** Verify all 14 individual comparisons, one-component isolation, complete negative results, selection mechanics, and zero actionable findings. Evidence: `artifacts/sprint-17/review-batch-c.md`. (Done: PASS with zero actionable findings; 14/14 arms `VALID_NEGATIVE`, 28/28 per-seed delta means and 7/7 selections independently recomputed, all input hashes matched, and Batch C authorized to close before the mandatory Task 19 stop.)

### Batch D — Predeclared interactions on Development

- [x] **Task 19 — Freeze selected combination configs.** Materialize K1–K9 configs from the exact selected single-arm implementations; no combo-specific hyperparameter change or outcome-selected omission. Evidence: `artifacts/sprint-17/task-19.md`. (Done: K1=C2-A+C3-A, K2=C3-A+C5-A, K3=C5-A+C6-A, K4=C6-A+C7-A, K5=C7-A+C8-A, K6=C8-A+C9-B, K7=C2-A+C3-A+C5-A+C6-A, K8=C7-A+C8-A+C9-B, K9=all seven selected arms frozen in `experiments/sprint17-task19-combinations.json` via `src/representation/sprint17_combinations.py` with provenance hashes and canonical digests; 10 focused contract tests plus 9 adjacent harness tests pass; deterministic rematerialization demonstrated; no K execution, no Confirmation/Sealed contact; Task 20 not started.)
- [x] **Task 20 — Execute adjacent-pair combinations.** Run K1–K6 on Development with the common seeds/budgets and paired `B0` comparison. Evidence: `artifacts/sprint-17/task-20.md`. (Done: K1 ΔP+W −0.0101, K2 +0.0141, K3 +0.0295, K4 +0.0090, K5 +0.0044, K6 −0.0062 S_pred (all `VALID_NEGATIVE`, 4/4 directional, support `800fc825…` exact); K1/K5 S_pred and K5 S_pop reproduce contributing singles bitwise per seed; all schema-valid; two verified execution commits `6a86a02` (K1–K4) + `1886729` (K5–K6) with disjoint fix effects; no Confirmation/Sealed contact; Task 21 not started.)
- [x] **Task 21 — Execute stack combinations.** Run K7–K9 on Development and report interaction gains relative to `B0` and the contributing single arms. Evidence: `artifacts/sprint-17/task-21.md`. (Done: K7 ΔP+W +0.0354, K8 −0.0062, K9 +0.0432 S_pred (all `VALID_NEGATIVE`, 4/4 directional, support `800fc825…` exact); K8 reproduces K5/K6 bitwise per seed; all schema-valid at single execution commit `ddc77c1`; no Confirmation/Sealed contact; Task 22 not started.)
- [x] **Task 22 — Pass Batch D evidence review.** Verify combination provenance, absence of combo-specific tuning, paired interaction arithmetic, complete negative results, and zero actionable findings. Evidence: `artifacts/sprint-17/review-batch-d.md`. (Done: PASS with high confidence and zero actionable findings; 9/9 K maps, 27/27 per-seed deltas, 9/9 macro means, 18/18 interaction aggregates, 52/52 selected-single cross-checks, provenance, support, compute, fidelity, and negative recovery verdicts independently verified. Batch D may close and the sprint must pause before Task 23.)

### Batch E — Independent Confirmation

- [ ] **Task 23 — Lock the Confirmation matrix.** Record hashes for `B0`, all 14 singles, K1–K9, metric code, thresholds, checkpoints, banks, model seeds, and untouched Confirmation roots before scoring. Evidence: `artifacts/sprint-17/task-23.md`.
- [ ] **Task 24 — Confirm `B0` and all individual alternatives.** Score `B0` plus all 14 single arms once on Confirmation and report paired per-history/per-seed results without refitting, reselecting, or pooling for rescue. Evidence: `artifacts/sprint-17/task-24.md`.
- [ ] **Task 25 — Confirm all multi-component combinations.** Score K1–K9 once on Confirmation under the same lock and report component-interaction deltas. Evidence: `artifacts/sprint-17/task-25.md`.
- [ ] **Task 26 — Publish the component and interaction matrix.** Report every Development and Confirmation arm, metric, LCB, nuisance result, compute cost, invalidity, and residual uncertainty; distinguish single-component recovery from interaction-only recovery. Evidence: `artifacts/sprint-17/task-26.md`.
- [ ] **Task 27 — Pass Batch E evidence review.** Verify one-shot Confirmation, matrix completeness, no adaptive reuse, no sealed contact, exact metric/support parity, and zero actionable findings. Evidence: `artifacts/sprint-17/review-batch-e.md`.

### Batch F — Verdict and sprint gate

- [ ] **Task 28 — Issue the bounded ablation verdict.** Return exactly one of `NO_REPRODUCIBLE_RECOVERY`, `SINGLE_COMPONENT_RECOVERY`, `INTERACTION_RECOVERY`, or `UNRESOLVED`, with eligible replacement configs and explicit non-production limits. Evidence: `artifacts/sprint-17/task-28.md`.
- [ ] **Task 29 — Pass sprint-wide differential deep review and finalize.** Deep-review the complete Sprint 17 above accepted batch evidence; correct and re-review every actionable finding before updating final plan state. Evidence: `artifacts/sprint-17/deep-review-final.md`, `artifacts/sprint-17/task-29.md`.

## Acceptance Criteria

1. The accepted Sprint 15 Candidate 7 generation methodology is unchanged; only a fresh, disjoint Sprint 17 role/seed binding is added prospectively.
2. Development and Confirmation histories independently satisfy every inherited Sprint 15 structural and observable `MEASURABLE` gate. The original Design-only failure `H-S17-V3-DESIGN-03 / cohort_mix_15_60 / P=91/149=0.6107382550 > 0.60` remains preserved and is bypassed only by the explicit user-authorized protocol-v4 waiver; Sprint 17 must therefore use the qualified label `MEASURABLE_WITH_USER_WAIVER`, never unqualified `MEASURABLE`.
3. Sprint 15 Sealed H-SEAL-37..40 are never opened, enumerated, scored, probed, or used.
4. `B0`, all 14 single arms, and K1–K9 use identical eligible data roles, model seeds, optimizer-step budgets, checkpoint rules, and metric code except for declared component changes.
5. Every single arm changes one principal mechanism; adapters and parameter/compute deltas are explicit.
6. Development alone selects component arms; Confirmation remains untouched until the complete matrix is frozen.
7. All registered arms, failures, negative results, per-history values, uncertainty, nuisance effects, and compute costs are reported without favorable omission or pooled rescue.
8. `S_pred` and `S_pop` remain separate; labels and simulator state remain post-hoc only.
9. A recovery verdict satisfies the frozen paired improvement, replication, LCB, P/W non-inferiority, nuisance, and background gates on Confirmation.
10. Every batch evidence review and the final sprint-wide differential deep review pass with zero unresolved actionable findings.
11. No result is described as production recovery, calibrated risk, fleet prevalence, sealed generalization, or causal proof of Sprint 16's provisional SUSPECT labels.

## Scientific Stop Rules

- Any generator-method, role, seed, manifest, or frozen-probe mismatch blocks model execution; do not silently regenerate or substitute data. The sole exception is the protocol-v4 user waiver for the already-recorded `H-S17-V3-DESIGN-03` Design P-share gate; it does not waive any Development, Confirmation, probe, provenance, model, or nuisance gate.
- If `B0` is not reproducible and eligible, stop alternatives and repair the baseline contract first.
- An arm that changes more than one undeclared principal mechanism is invalid, not approximately comparable.
- No failed or weak alternative may be replaced after Development outcomes without a prospectively reviewed protocol version and a fresh sprint decision.
- Confirmation may be opened only after Task 23 locks every arm and artifact hash.
- A Development win that does not replicate on Confirmation is a negative result.
- Interaction recovery does not retroactively prove any individual component defective.
- If no arm passes, close `NO_REPRODUCIBLE_RECOVERY` or `UNRESOLVED`; do not automatically authorize Sprint 18.

## Explicit Non-Goals

- Rewriting C1 normalization or C4 context encoding in this sprint.
- Exhaustive architecture or hyperparameter search.
- Full factorial enumeration beyond K1–K9.
- Fusing `S_pred` and `S_pop`.
- Using handcrafted features as model inputs.
- Opening Sprint 15 Sealed histories.
- Calibrated-risk, probability-quality, deployment-threshold, early-warning-utility, or production-readiness claims.
- Reopening Sprint 4 or rewriting Sprint 15/16 accepted evidence.

## Notes / Blockers

- Sprint 16's `UNRESOLVED`, zero-BOTTLENECK conclusion remains authoritative historical evidence.
- This new sprint is authorized by explicit user instruction as a diagnostic ablation despite the prior empty production-recovery scope.
- Task 1 must freeze the remaining numeric tolerances and model-seed budget before any new generated or model outcome is inspected.
- Protocol v2 Task 3 is permanently `BLOCKED / FAIL-CLOSED`: the unchanged frozen `sprint15-v7` preflight returned `PREFLIGHT-FAIL` at 15/16, with sole failure `H-S17-DEV-02` (DEVELOPMENT seed 2709), reason `quota-shortfall`; exact CSP quotas are infeasible. No Sprint 17 root was materialized. Preserve `artifacts/sprint-17/task-3.md` and `artifacts/sprint-17/task3-preflight-diagnostic.json` unchanged as the v2 failure record.
- The user explicitly selected a prospective protocol-v3 restart instead of closing Sprint 17. V3 must abandon the entire v2 roster, freeze exactly one all-new 16-role block before preflight, and retain the exact Sprint 15 Candidate 7 method and all scientific gates. Replacing only seed 2709, screening reserve blocks, or using a favorable preflight outcome is forbidden.
- Batch C Tasks 10–18 are complete. All 14 registered alternatives were reported as `VALID_NEGATIVE`; Task 17 froze `{C2-A, C3-A, C5-A, C6-A, C7-A, C8-A, C9-B}` with zero unresolved components and no recovery claim; Task 18 evidence review passed with zero actionable findings. Batch D Tasks 19–22 are complete: K1–K9 were frozen without combination-specific tuning, all nine executed on Development and remained `VALID_NEGATIVE`, and the independent Batch D review passed with zero actionable findings. The sprint is paused before Batch E / Task 23; resumption requires new explicit user instruction. Confirmation and Sprint 15 Sealed histories remain untouched.
- On 2026-09-14 the user explicitly authorized bypassing the Task 4 blocker. This is recorded as a post-observation protocol-v4 waiver, not a prospective pass: only `H-S17-V3-DESIGN-03 / cohort_mix_15_60 / P=91/149` is waived. No history may be regenerated, omitted, replaced, pooled, or modified; every downstream result must disclose `MEASURABLE_WITH_USER_WAIVER`.
