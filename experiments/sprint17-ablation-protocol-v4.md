# Sprint 17 Controlled Suspect-Component Ablation Protocol v4

**Status: FROZEN POST-OBSERVATION — protocol-v3 outcome remains BLOCKED / NOT_MEASURABLE; this v4 document freezes one user-authorized waiver only.**
**Protocol ID:** `sprint17-ablation-v4`
**Scope:** Task 4A–4B waiver amendment only. Protocol v3 remains the immutable scientific contract; this document records one post-observation qualification and does not authorize any new data, root, probe, model, GPU, or Sealed activity.

The preserved v3 amendment supersedes no history: v1, v2, and v3 remain byte-preserved historical protocols and evidence. The sole v4 semantic change is the user-authorized S0 exact one-cell waiver recorded in §0; the v3 roster restart is historical context only. All DGP, quota, gate, architecture, metric, model, budget, promotion, stopping, and Sealed rules remain unchanged.


## 0. Protocol-v4 post-observation amendment — exact user waiver

**This is a post-observation amendment, not a prospective pass.** The original protocol-v3 result remains true under its original rules: `BLOCKED / NOT_MEASURABLE` because `H-S17-V3-DESIGN-03` failed `cohort_mix_15_60` (`P=91/149=0.610738255033557 > 0.60`). Protocol v4 records the user's explicit authorization to waive that one already-observed cell; it does not rewrite, rerun, retune, or retroactively pass protocol v3.

The sole waiver identity is immutable:

| Field | Exact authorized value |
|---|---|
| Waiver ID | `S17-V4-WAIVER-H-S17-V3-DESIGN-03-COHORT_MIX_15_60-P-91-149` |
| History | `H-S17-V3-DESIGN-03` |
| Role | `DESIGN` |
| Gate | `cohort_mix_15_60` |
| Cohort | `P` |
| Observed numerator | `91` |
| Observed denominator | `149` |
| Observed share | `91/149 = 0.610738255033557` |
| Original upper bound | `0.60` |

Only this exact tuple is waived. Any other history, role, gate, cohort, numerator, denominator, observed share, or threshold is unwaived and remains subject to the original fail-closed rule. The original failure and raw evidence are hash-linked and immutable: `artifacts/sprint-17/task-4.md` (sha256 `48fc4409b7739ac09fb40288a985b01eda71b88df9a1a85a56acf0bed27d4bd3`), `task4-structural.json` (sha256 `8b1db8a5dd36029c338d6e8ea87a91d2c1f4cd6dab9e5832353fe8d96b429cc1`), and `task4-probe.json` (sha256 `16ffa69c2d1ac752a2f8b8a9d21e12971d43b142745601f7a96e9d2b448356ed`).

Protocol v4 changes no root, seed, history, generator, probe, model result, arm, metric, support, threshold, Development gate, Confirmation gate, or Sealed rule. It preserves the exact Sprint 15-v7 DGP, the 16 v3 roots/binding, B0, 14 singles, K1--K9, model seeds/budget, metric/effect/nuisance gates, `S_pred`/`S_pop` separation, promotion/Confirmation lock, and no-retry rules from v3 below. The qualified downstream state is **exactly** `MEASURABLE_WITH_USER_WAIVER`; unqualified `MEASURABLE` is forbidden.
No root, seed, history, generator, probe, or model result may be changed, regenerated, replaced, omitted, pooled, rescored, or retuned under this amendment.

Task 4 may close and Task 5 may begin only after Task 4B returns **zero actionable findings**. Until then Task 4A is the only completed waiver task; Task 4B and Task 4 remain pending, Task 5+ remains pending, and the pre-GPU boundary remains in force. This amendment does not authorize Task 7, any GPU connection, Sealed access, formatter/linter/project-wide validation, or data/model execution.

### 0.1 Machine-checkable downstream qualification

Every downstream phase/arm result must contain the exact top-level `eligibility` object required by the v4 schema in §7. Its `qualification` must be `MEASURABLE_WITH_USER_WAIVER` and its `waiver_id` and `waiver_scope` must equal the exact constants below. Missing, altered, generalized, or unqualified eligibility is invalid; this requirement applies equally to B0, every single arm, and K1--K9 in Development and Confirmation. The schema leaves B0/single/K arm-kind, component-set, score-branch, metric, and gate behavior unchanged.

```json
{
  "qualification": "MEASURABLE_WITH_USER_WAIVER",
  "waiver_id": "S17-V4-WAIVER-H-S17-V3-DESIGN-03-COHORT_MIX_15_60-P-91-149",
  "waiver_scope": {
    "history_id": "H-S17-V3-DESIGN-03",
    "role": "DESIGN",
    "gate": "cohort_mix_15_60",
    "cohort": "P",
    "observed_numerator": 91,
    "observed_denominator": 149,
    "observed_share": 0.610738255033557,
    "original_upper_bound": 0.6
  }
}
```

## 0.2 Protocol-v3 amendment and permanent v2 failure

Protocol v2 is immutable historical evidence, not a candidate to repair. Its unchanged Sprint 15 Candidate 7 rejection-only preflight over the fixed `2700..2715` roster returned `PREFLIGHT-FAIL` at `15/16`: the sole failure was `H-S17-DEV-02` (seed `2709`), exact-CSP `quota-shortfall` (`P/P1<12, P/P2<12, W/W1<12, W/W2<12, A/A1<8, A/A2<8`; HiGHS status 8). The first timeout and the completed diagnostic are retained in `artifacts/sprint-17/task-3.md` and `artifacts/sprint-17/task3-preflight-diagnostic.json`; no v2 root was materialized. This is a permanent fail-closed v2 result, not permission to replace seed 2709.

The user explicitly authorized and selected one prospective protocol-v3 restart after that frozen v2 failure. The sole scientific amendment is abandonment of the complete v2 roster and prospective binding of exactly one complete all-new 16-role roster. The accepted Sprint 15 Candidate 7 generator method, B0, exact 14 singles, K1--K9, model seeds, 300-step budget, parameter/FLOP envelopes, numeric gates, metric schema, independent `S_pred`/`S_pop` branches, promotion/Confirmation/stopping rules, and absolute Sealed prohibition are unchanged. Protocol/schema IDs, v3 role IDs, and v3 paths are versioned only to identify this restart.

V3 has no reserve, screening, per-seed replacement, retry, omission, pooling rescue, preflight-based seed selection, root creation, generation, outcome inspection, model execution, or Task 3+ authorization in Tasks 2A--2B. The current boundary stops after Task 6; Task 7 and GPU connection remain pending. Any v3 preflight miss must stop without another replacement.

## 1. Lineage, boundary, and permitted data

The data-generating process (DGP) is exactly the accepted Sprint 15 Candidate 7 method, protocol `sprint15-benchmark-protocol-v7`: the same physics, scheduler, allocator, causal windows, balanced P/W/A quotas, nuisance envelope, reset/quarantine/censoring rules, deterministic subtype stratification, and structural/observable `MEASURABLE` gates. Sprint 17 adds only a fresh, disjoint role/seed binding below. No Sprint 15 generator byte, setting, quota, or gate is changed.

Accepted evidence grounding this choice is `artifacts/sprint-15/task-20.md` §§1–5 (accepted Candidate 7, role permissions, frozen metric contract, and no-Sealed boundary), `experiments/sprint15-benchmark-protocol-v7.md` §§1–5 (method and quotas), and `artifacts/sprint-16/task-2.md` (prospective one-boundary contract and bounded execution discipline). The accepted Sprint 15 method remains the sole data method; Sprint 17 is case-control evaluation and cannot support fleet-prevalence or production claims.

### 1.1 Fresh Sprint 17 roles and seeds

The following role topology and binding constraints were frozen prospectively by protocol v3 and are retained unchanged by v4. The v3 binding froze exactly one complete, all-new 16-role roster: data seeds `2800..2815`, assigned one-to-one with `H-S17-V3-{DESIGN,FIT,CAL,DEV,CONF}-NN` IDs. The v2 roster (`2700..2715`) is permanently failed history and no v2 role ID, seed, directory, root, or outcome may be reused. Every bound root is generated once, retained even if it fails, and is never retried, replaced, omitted, pooled for rescue, or regenerated. There is no reserve block, per-seed swap, screening, favorable preflight selection, or outcome-selected seed.

| Role | Exact count | Permitted use |
|---|---:|---|
| Design | 4 histories | Structural and observable certification only; never model fitting or final comparison |
| Fit | 3 histories | Representation fitting and healthy-reference rows only |
| Calibration | 1 history | Per-arm frozen quantile threshold only; no architecture selection |
| Development | 4 histories | Paired individual-arm comparison and deterministic single-arm selection |
| Confirmation | 4 histories | Untouched one-shot final comparison after the complete matrix lock |

The required fresh IDs follow `H-S17-V3-{DESIGN,FIT,CAL,DEV,CONF}-NN` with two-digit ordinals and the required data seed binding is one-to-one across the 16 histories. The role directories are exactly `data/generated/sprint17-ablation-v3/{DESIGN,FIT,CALIBRATION,DEVELOPMENT,CONFIRMATION}/<history-id>`. No literal binding, root, or history is valid until Task 2B records it. Sprint 15 `H-SEAL-37..40` remain absolutely prohibited: Sprint 17 never opens, enumerates, hashes, probes, scores, loads, or uses any Sealed root or content. A Sealed name may appear only in this prohibition and in machine-checkable no-touch assertions. No Sprint 17 Sealed role is created.
The machine binding is `experiments/sprint17-role-binding-v3.json`; its canonical digest is `sha256(canonical_json_utf8_without_binding_digest)` (sorted keys, compact separators, UTF-8, excluding only `binding_sha256`) with frozen value `075868b22c028a981cc63ffe37b8d29720cd324c3e2c7254ce688678758230ba`. This digest and all 11 method-byte expectations must be recomputed immediately before any later preflight; a mismatch voids the binding.


The Sprint 15 Candidate 7 generation method's exact P/W/A construction is retained: construction targets `P=24` (`P1=12,P2=12`), `W=24` (`W1=12,W2=12`), `A=16` (`A1=8,A2=8`), and at least 48 deterministic negative controls per history. Design and Confirmation must independently pass inherited hard floors and promotion targets; Fit and Calibration must pass their downstream support floors. Labels, failure categories, masks, severity, and simulator hidden state are post-hoc diagnostics only and never model inputs.

### 1.2 Structural and observable gates

Before any model execution, every Design, Development, and Confirmation history must have, independently: `P>=10`, `W>=10`, `A>=8`, total positives `>=30`, negative controls `>=25`, evaluable robot-days `>=150`, at least 6 positive-contributing robots, at least 6 negative-contributing robots, at least 2 programs in each P and W cohort, each cohort share in `[0.15,0.60]`, robot positive concentration `<=0.35`, robot negative-control concentration `<=0.40`, and program concentration `<=0.60` in P and W. The fresh binding must also reserve at least one complete robot identity, one complete program identity, and one eligible subtype from each predictable P and W cohort in both Development and Confirmation, as required by the accepted gate topology. Design promotion targets are `P>=13`, `W>=13`, `A>=10`, total positives `>=38`, controls `>=32`, and robot-days `>=188`. At least 80% of P and W events require the accepted seven-day lead-support predicates.

Fit must provide at least 200 verified-healthy files and 6,000 valid patches in aggregate. Calibration must provide at least 40 verified-healthy eligible rows. No confirmation, sealed, quarantined, censored, maintenance-overlapping, or non-healthy row may enter Fit or Calibration. The frozen Sprint 15 observable probe/measurability checks certify Development and Confirmation before model conclusions are eligible; a probe or structural miss blocks execution and cannot be repaired by model outcomes.


### 1.3 Exhaustive seed exclusion bands

The v3 data block `2800..2815` is disjoint from every prior/test/candidate/v2 band: Sprint 14 accepted history `742..797`; Sprint 14 documented test-generation `910..933`; Sprint 15 accepted Candidate 1 `1000..1017`, Candidate 2 `1100..1117`, Candidate 3 `1200..1217`, Candidate 4 `1300..1317`, Candidate 5 `1400..1417`, Candidate 6 `1500..1517`, Candidate 7 `1600..1617`; and failed v2 roster `2700..2715`. It is also disjoint from model seeds `171701,171702,171703`. No other seed block exists or is reserved for this protocol.

## 2. Baseline, immutable registry, and one-principal-change rule

`B0` is the unchanged accepted representation-to-score architecture, retrained and recalibrated on Sprint 17 Fit/Calibration under the same data roles, model seeds, optimizer-step budget, checkpoint rule, score code, and support as every trainable alternative. Old Sprint 15/16 weights or a restored old reference bank are not the fairness baseline. C1 normalization and C4 context behavior are fixed exactly as B0 and are not registry axes.

The registry below is immutable. It contains exactly 14 single alternatives and no others. Every single arm changes exactly one named principal mechanism; all downstream/upstream mechanisms, labels, masks, severity, hidden state, and evaluation support remain unchanged. Minimal shape adapters are permitted only where required by the named replacement, must be declared in its config, and count toward the envelope.

| Arm ID | Principal mechanism | Exact frozen replacement |
|---|---|---|
| `C2-A` | C2 patchification | Fixed-width windows at 50% overlap; preserve valid length, padding, starts, and file identity |
| `C2-B` | C2 patchification | Two fixed-width offset grids with unit-mass support weights; duplicated timesteps do not gain pooling weight |
| `C3-A` | C3 local encoder | Parameter-matched residual depthwise-separable dilated 1-D convolutional patch encoder |
| `C3-B` | C3 local encoder | Parameter-matched lightweight within-patch Transformer with explicit position and padding masks |
| `C5-A` | C5 objective/masking | Fixed total mask ratio with channel-time blocks and multi-horizon EMA latent prediction |
| `C5-B` | C5 objective/masking | Retain EMA masked prediction; replace file-level InfoNCE with VICReg-style variance/covariance regularization |
| `C6-A` | C6 file pooling | Valid-patch mean plus standard deviation, projected to the existing file-embedding dimension |
| `C6-B` | C6 file pooling | Gated attention over valid patches with no anomaly-derived input or target |
| `C7-A` | C7 geometry/reference bank | Per-condition robust location plus Ledoit-Wolf-style covariance shrinkage with the frozen hierarchy/fallback |
| `C7-B` | C7 geometry/reference bank | Condition-aware kNN/local-density score with robust scale fitted only from Fit healthy-reference rows |
| `C8-A` | C8 scorer | Non-query-conditioned standardized Huber prediction-residual energy for `S_pred` |
| `C8-B` | C8 scorer | Non-query-conditioned cosine distance between predicted and target latents for `S_pred` |
| `C9-A` | C9 patch-to-file aggregation | Fixed-fraction top-k mean over valid patch scores |
| `C9-B` | C9 patch-to-file aggregation | Maximum fixed-duration contiguous-window mean using patch starts and valid lengths |

C7 alternatives affect only `S_pop`; C8 alternatives affect only `S_pred`. All result files carry both independent score branches. Fusion, score substitution, handcrafted-feature input, and C1/C4 changes are forbidden.

### 2.1 Immutable combinations

K arms are formed only after Development selects at most one valid A/B arm per suspect. The selected implementation is copied exactly; no combination-specific tuning, adapter, seed, checkpoint, or omission is permitted.

| ID | Exact selected components | Interaction |
|---|---|---|
| `K1` | C2 + C3 | Patch support × local representation |
| `K2` | C3 + C5 | Local representation × learning pressure |
| `K3` | C5 + C6 | Learning pressure × file readout |
| `K4` | C6 + C7 | File embedding × healthy geometry |
| `K5` | C7 + C8 | Geometry × score readability |
| `K6` | C8 + C9 | Patch score × file aggregation |
| `K7` | C2 + C3 + C5 + C6 | Complete upstream representation stack |
| `K8` | C7 + C8 + C9 | Complete downstream scoring stack |
| `K9` | C2 + C3 + C5 + C6 + C7 + C8 + C9 | End-to-end selected replacement stack |

If neither A nor B for a component is valid, that component is omitted from combinations and recorded as unresolved; no replacement is invented. This is the only permitted omission and is determined solely by the frozen validity rule, never by a favorable outcome.

## 3. Model, optimizer, checkpoint, and compute parity

The model-seed set is exactly `M17 = [171701,171702,171703]` (three seeds), frozen before any outcome. Three is the accepted independent-fit cardinality precedent (the three Fit histories in `artifacts/sprint-15/task-20.md` §2), and the set prevents a one-seed claim while keeping the bounded intervention budget. The numeric IDs are a fresh namespace disjoint from the data seeds. Every trainable arm (`B0`, C2-A/B, C3-A/B, C5-A/B, C6-A/B, and any K arm containing one of these) runs all three seeds. The seed is a model-initialization/optimizer seed, not a data seed. C7/C8/C9-only alternatives may reuse eligible Sprint 17 latent/prediction caches for each of the three seed identities, but must report cache provenance and consume zero additional optimizer steps; cache reuse cannot change semantics.

For every trainable arm and model seed: exactly 300 optimizer steps; same optimizer class, learning-rate schedule, batch construction, precision, device class, gradient clipping, weight decay, and initialization policy as B0; no warm-start from another arm; no extra gradient run; no early stopping. This 300-step cap is the accepted Sprint 16 tiny-retrain budget (`artifacts/sprint-16/task-1.md` §4 and `experiments/sprint16-attribution-protocol-v5.md` §2), chosen to keep the controlled intervention bounded. The checkpoint is the state after exactly step 300. Step-0 or intermediate checkpoints are diagnostic only and cannot be selected.

B0 parameter count and reference FLOPs are measured once by the common harness at the frozen reference shape, before any alternative outcome is read. For each arm and seed, including all adapters:

- `params_total <= ceil(1.05 * B0.params_total)` and `abs(params_total-B0.params_total)/B0.params_total <= 0.05`;
- `flops_per_reference_sample <= ceil(1.10 * B0.flops_per_reference_sample)` and `abs(flops_per_reference_sample-B0.flops_per_reference_sample)/B0.flops_per_reference_sample <= 0.10`;
- batch size, reference input length, valid-mask semantics, and precision are identical to B0.

A missing count, zero/negative count, envelope breach, undeclared adapter, or measurement mismatch invalidates the arm before ranking. No width/depth/step/batch/precision change may be used to rescue an envelope breach. The envelope percentages are conservative bounded changes, grounded in Sprint 16's one-boundary and fixed-budget contract (`artifacts/sprint-16/task-2.md` §§1–2); they are not a license for architecture search.

All arms fit representations and healthy references only on the three Fit histories; each arm calibrates exactly once from its Calibration history by the rule below. Development and Confirmation never refit representation, bank, standardizer, or thresholds.

## 4. Calibration, scoring, support, and metrics

Each arm independently fits the same global Fit-healthy standardization and its declared healthy reference procedure, records a provenance token, and applies it without refit. Its operating threshold is the 95th percentile of its Calibration healthy eligible scores, computed once per score branch with the repository's deterministic quantile implementation. Calibration is never used for evaluation, selection, or tuning. `S_pred` (context/prediction mismatch) and `S_pop` (population/geometry mismatch) are separate estimands; neither is fused or used as a hidden fallback. A missing branch is an invalid result, not permission to substitute the other branch.

The primary estimand is tie-aware event AUROC for `P+W`, defined per history as the unweighted mean of P AUROC and W AUROC on common support, then macro-averaged across the four histories. Every arm reports paired arm-minus-B0 deltas on identical history, event, file, patch, seed, and support keys. No pooled-only estimate, one-seed win, favorable subset, or rethresholded score counts.

Required metrics for each branch and for the primary `P+W` readout are:

1. P macro, W macro, and P+W macro AUROC; per-history values; paired deltas; and history-block bootstrap `B=2000`, RNG seed `20260202`, 95% LCB equal to the 2.5th percentile (the accepted Sprint 15/Sprint 16 convention in `artifacts/sprint-15/task-20.md` §4).
2. Directional count: number of the four histories with absolute P+W AUROC strictly `>0.55`; report numerator and denominator, and require `>=3/4` for recovery. The arm's post-score primary LCB must be strictly `>0.55`.
3. P recall `>=0.50` with median lead `>=1.0` day and W recall `>=0.25` with median lead `>=0.5` day at the arm's frozen Calibration threshold; report per history. Abrupt A is reported separately and has no minimum ranking gate, preserving the accepted handoff.
4. Severity ordering (within P and W), localization/duration accounting, valid-patch and event support, and category slices. These are mandatory finite diagnostic fields; severity ordering is descriptive and cannot by itself promote an arm.
5. Unaffected-background stability ratio `|median(background)-median(Fit healthy)| / max(pooled SD, STD_FLOOR)`, with `STD_FLOOR` taken from the shared metric implementation, per history and 95% bootstrap interval; require point ratio `<=0.10` per history. Background support and all exclusions are explicit.
6. Nuisance false-alert rate (FAR) per robot-day using the accepted `<=2-day` grouping and reset split; require FAR `<=0.05` per history and paired arm-minus-B0 `Delta FAR <= +0.02` per history. The `+0.02` tolerance is the accepted Sprint 15/Sprint 16 nuisance limit, not a post-outcome choice.

Metric code, tie handling, support intersection, standardization, aggregation, and bootstrap are common and hash-recorded. C9 changes only the declared patch-to-file aggregation; all other metric definitions remain identical.

### 4.1 Exact paired effect and non-inferiority gates

These values are frozen before any result:

- **Minimum paired effect:** Confirmation `Delta_PW = P+W_macro(arm) - P+W_macro(B0) >= +0.05` (strictly no rounding before comparison).
- **P non-inferiority:** `Delta_P >= -0.02`.
- **W non-inferiority:** `Delta_W >= -0.02`.

The `+0.05` minimum effect is the accepted Sprint 16 bounded-movement magnitude used to distinguish meaningful intervention movement from small sensitivity (`experiments/sprint16-attribution-protocol-v2.md` §7 and `artifacts/sprint-16/task-2.md`); the `-0.02` P/W tolerance is deliberately no looser than the accepted `Delta FAR <= +0.02` nuisance allowance (`artifacts/sprint-15/task-20.md` §4 and this protocol §4). These are paired macro point estimates on the four-history common support, not confidence-bound substitutions. Development uses the same values for reporting, but selection remains the deterministic ranking rule below and does not turn a negative arm into a positive claim.

## 5. Validity, negative results, promotion, and Confirmation lock

An arm is `VALID` only if its role/seed/protocol tags, root and code digests, support keys, score separation, finite outputs, parameter/FLOP accounting, one-principal-change declaration, calibration provenance, and all required metric fields pass. It is `NEGATIVE` when valid but fails the effect, replication, LCB, P/W non-inferiority, nuisance, background, or other recovery gate. It is `FAILED` when execution terminates with a declared runtime/training failure. It is `INVALID` when any contract invariant is violated or data/metric output is missing/non-finite. `INCONCLUSIVE` is reserved for a validly attempted arm whose required estimand cannot be computed despite sufficient structural support. Every status and reason is retained; failed, negative, and invalid arms are never silently omitted.

A mismatch, NaN/Inf, empty required support, single-class AUROC, score-branch collision, role reuse, unauthorized data row, adapter/envelope breach, undeclared mechanism, cache mismatch, or metric-code/hash mismatch is fail-closed invalidity. There is no retry, replacement, pooling rescue, favorable-seed deletion, discretionary threshold, or post-outcome schema relaxation.

### 5.1 Development promotion rule

After all 14 single-arm Development outputs (including invalid/failed outputs) are recorded, rank only valid A/B arms within each component, descending by paired Development `Delta_PW`. Ties are broken, in order, by: (1) larger `min(Delta_P, Delta_W)`; (2) larger history-block P+W LCB; (3) smaller absolute parameter delta; (4) smaller absolute FLOP delta; (5) lexical arm ID. Select at most one arm per suspect. A valid negative arm may be selected if it ranks highest; this is a deterministic design selection, not evidence of recovery. If neither arm is valid, select none and record the component unresolved. Development never opens or scores Confirmation.

### 5.2 Confirmation eligibility and recovery rule

Confirmation is ineligible until all 14 single-arm Development results, all statuses/reasons, selected IDs, all nine K definitions, metric-code digest, threshold rules, model seeds, checkpoint hashes, bank/cache hashes, and data-role manifests are locked in a Task 23 lock record. Once locked, each B0, single arm, and K1–K9 is scored once on the four untouched Confirmation histories. No refit, reselect, retune, rerun, pooling rescue, or matrix change is permitted.

A single-arm or combination **recovery** requires all of: `Delta_PW >= +0.05`; positive Delta_PW on at least 3/4 histories; primary post-arm LCB `>0.55`; directional count `>=3/4`; `Delta_P >= -0.02`; `Delta_W >= -0.02`; P/W recall-and-lead gates; per-history FAR `<=0.05` and `Delta FAR <=+0.02`; and per-history background stability ratio `<=0.10`. If any gate fails, the result remains negative or invalid according to status, even if another metric improves. An interaction-only recovery is reported separately and does not retroactively establish an individual component defect.

## 6. Scientific stopping rules and non-goals

1. Any DGP, role, seed, manifest, frozen-probe, code, support, or provenance mismatch blocks model execution; do not regenerate or substitute.
2. If any Design/Development/Confirmation structural or observable gate fails, stop the affected path; never interpret model outcomes as a repair.
3. If B0 is not reproducible and eligible on all model seeds, stop all alternatives and repair the baseline contract under a reviewed protocol amendment.
4. An arm changing more than one undeclared principal mechanism is invalid, not approximately comparable.
5. A failed/weak alternative is never replaced after Development outcomes. A new arm requires a new prospective protocol version.
6. Confirmation opens only after the complete Task 23 lock; any lock mismatch bars Confirmation permanently for v4.
7. A Development win that fails Confirmation is a negative result. No Sealed history is opened to resolve a tie or ambiguity.
8. If no arm meets the frozen gates, issue only the bounded Sprint 17 verdict permitted by the plan (`NO_REPRODUCIBLE_RECOVERY` or `UNRESOLVED` as applicable); do not authorize production recovery or Sprint 18 automatically.
9. No claim of calibrated risk, probability quality, deployment threshold, fleet prevalence, sealed generalization, causal proof of Sprint 16 SUSPECT labels, or production readiness is permitted.

## 7. Machine-checkable artifact contract

Every phase/arm result is one JSON document validated against the JSON Schema below (`$id = "sprint17-ablation-result-v4"`). The schema is closed at every contract object (`additionalProperties: false`); unknown keys are rejected. JSON `null` is allowed only where the schema explicitly permits it for a failed, invalid, or inconclusive result. Numeric values must be finite JSON numbers; IDs and enums are case-sensitive.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "sprint17-ablation-result-v4",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_id", "protocol_id", "phase", "arm_id", "arm_kind", "component_set", "status", "invalid_reasons", "provenance", "config", "compute", "support", "scores", "metrics", "gates", "eligibility"],
  "properties": {
    "schema_id": {"const": "sprint17-ablation-result-v4"},
    "protocol_id": {"const": "sprint17-ablation-v4"},
    "phase": {"enum": ["development", "confirmation"]},
    "arm_id": {"enum": ["B0", "C2-A", "C2-B", "C3-A", "C3-B", "C5-A", "C5-B", "C6-A", "C6-B", "C7-A", "C7-B", "C8-A", "C8-B", "C9-A", "C9-B", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9"]},
    "arm_kind": {"enum": ["baseline", "single", "combination"]},
    "component_set": {"type": "array", "minItems": 0, "maxItems": 7, "uniqueItems": true, "items": {"enum": ["C2", "C3", "C5", "C6", "C7", "C8", "C9"]}},
    "status": {"enum": ["VALID_POSITIVE", "VALID_NEGATIVE", "FAILED", "INVALID", "INCONCLUSIVE"]},
    "invalid_reasons": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
    "provenance": {
      "type": "object", "additionalProperties": false,
      "required": ["data_protocol", "role_binding_sha256", "data_roles", "data_seeds", "model_seeds", "fit_root_sha256", "calibration_root_sha256", "evaluation_root_sha256", "metric_code_sha256", "checkpoint_sha256", "cache_sha256", "parent_arm_ids"],
      "properties": {
        "data_protocol": {"const": "sprint15-benchmark-protocol-v7"},
        "data_roles": {"type": "array", "minItems": 1, "items": {"type": "string", "pattern": "^H-S17-V3-(DESIGN|FIT|CAL|DEV|CONF)-[0-9]{2}$"}},
        "role_binding_sha256": {"$ref": "#/$defs/hash"},
        "data_seeds": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0, "maximum": 2147483647}},
        "model_seeds": {"const": [171701, 171702, 171703]},
        "fit_root_sha256": {"$ref": "#/$defs/hash"},
        "calibration_root_sha256": {"$ref": "#/$defs/hash"},
        "evaluation_root_sha256": {"$ref": "#/$defs/hash"},
        "metric_code_sha256": {"$ref": "#/$defs/hash"},
        "checkpoint_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
        "cache_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
        "parent_arm_ids": {"type": "array", "items": {"enum": ["B0", "C2-A", "C2-B", "C3-A", "C3-B", "C5-A", "C5-B", "C6-A", "C6-B", "C7-A", "C7-B", "C8-A", "C8-B", "C9-A", "C9-B"]}, "uniqueItems": true}
      }
    },
    "config": {
      "type": "object", "additionalProperties": false,
      "required": ["one_principal_change", "adapter_description", "optimizer_steps", "checkpoint_step", "calibration_quantile", "score_branches"],
      "properties": {
        "one_principal_change": {"enum": ["none", "C2", "C3", "C5", "C6", "C7", "C8", "C9"]},
        "adapter_description": {"type": "string"},
        "optimizer_steps": {"const": 300},
        "checkpoint_step": {"const": 300},
        "calibration_quantile": {"const": 0.95},
        "score_branches": {"const": ["S_pred", "S_pop"]}
      }
    },
    "compute": {
      "type": "object", "additionalProperties": false,
      "required": ["b0_params", "params_total", "params_delta_fraction", "b0_flops_per_reference_sample", "flops_per_reference_sample", "flops_delta_fraction", "envelope_pass"],
      "properties": {
        "b0_params": {"type": "integer", "minimum": 1},
        "params_total": {"type": "integer", "minimum": 1},
        "params_delta_fraction": {"type": "number", "minimum": -0.05, "maximum": 0.05},
        "b0_flops_per_reference_sample": {"type": "integer", "minimum": 1},
        "flops_per_reference_sample": {"type": "integer", "minimum": 1},
        "flops_delta_fraction": {"type": "number", "minimum": -0.10, "maximum": 0.10},
        "envelope_pass": {"type": "boolean"}
      }
    },
    "support": {
      "type": "object", "additionalProperties": false,
      "required": ["common_support_sha256", "per_history", "excluded_rows"],
      "properties": {
        "common_support_sha256": {"$ref": "#/$defs/hash"},
        "per_history": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"$ref": "#/$defs/history_support"}},
        "excluded_rows": {"type": "integer", "minimum": 0}
      }
    },
    "scores": {
      "type": "object", "additionalProperties": false, "required": ["S_pred", "S_pop"],
      "properties": {"S_pred": {"$ref": "#/$defs/score_branch"}, "S_pop": {"$ref": "#/$defs/score_branch"}}
    },
    "metrics": {
      "type": "object", "additionalProperties": false,
      "required": ["primary_PW", "P", "W", "A_companion", "severity_ordering", "localization", "background", "category_slices", "bootstrap"],
      "properties": {
        "primary_PW": {"$ref": "#/$defs/primary_metric"},
        "P": {"$ref": "#/$defs/cohort_metric"},
        "W": {"$ref": "#/$defs/cohort_metric"},
        "A_companion": {"$ref": "#/$defs/companion_metric"},
        "severity_ordering": {"type": "object", "additionalProperties": false, "required": ["P_spearman", "W_spearman"], "properties": {"P_spearman": {"$ref": "#/$defs/four_values"}, "W_spearman": {"$ref": "#/$defs/four_values"}}},
        "localization": {"type": "object", "additionalProperties": false, "required": ["present", "per_history"], "properties": {"present": {"type": "boolean"}, "per_history": {"$ref": "#/$defs/four_values"}}},
        "background": {"type": "object", "additionalProperties": false, "required": ["stability_ratio_per_history", "far_per_robot_day", "delta_far_vs_B0"], "properties": {"stability_ratio_per_history": {"$ref": "#/$defs/four_values"}, "far_per_robot_day": {"$ref": "#/$defs/four_values"}, "delta_far_vs_B0": {"$ref": "#/$defs/four_values"}}},
        "category_slices": {"type": "object", "additionalProperties": {"type": "object"}},
        "bootstrap": {"type": "object", "additionalProperties": false, "required": ["replicates", "seed", "lcb_percentile"], "properties": {"replicates": {"const": 2000}, "seed": {"const": 20260202}, "lcb_percentile": {"const": 2.5}}}
      }
    },
    "gates": {
      "type": "object", "additionalProperties": false,
      "required": ["structural", "observable", "finite_and_support", "score_separation", "compute_envelope", "minimum_effect", "P_noninferiority", "W_noninferiority", "nuisance", "background_stability", "eligible", "recovery"],
      "properties": {
        "structural": {"type": "boolean"}, "observable": {"type": "boolean"}, "finite_and_support": {"type": "boolean"}, "score_separation": {"type": "boolean"}, "compute_envelope": {"type": "boolean"}, "minimum_effect": {"type": "boolean"}, "P_noninferiority": {"type": "boolean"}, "W_noninferiority": {"type": "boolean"}, "nuisance": {"type": "boolean"}, "background_stability": {"type": "boolean"}, "eligible": {"type": "boolean"}, "recovery": {"type": "boolean"}
      }
    },
    "eligibility": {
      "type": "object", "additionalProperties": false,
      "required": ["qualification", "waiver_id", "waiver_scope"],
      "properties": {
        "qualification": {"const": "MEASURABLE_WITH_USER_WAIVER"},
        "waiver_id": {"const": "S17-V4-WAIVER-H-S17-V3-DESIGN-03-COHORT_MIX_15_60-P-91-149"},
        "waiver_scope": {"const": {"history_id":"H-S17-V3-DESIGN-03","role":"DESIGN","gate":"cohort_mix_15_60","cohort":"P","observed_numerator":91,"observed_denominator":149,"observed_share":0.610738255033557,"original_upper_bound":0.6}}
      }
    }
  },
  "allOf": [
    {"if": {"properties": {"arm_id": {"const": "B0"}}}, "then": {"properties": {"arm_kind": {"const": "baseline"}, "component_set": {"const": []}, "config": {"properties": {"one_principal_change": {"const": "none"}}}}}},
    {"if": {"properties": {"arm_id": {"enum": ["C2-A", "C2-B", "C3-A", "C3-B", "C5-A", "C5-B", "C6-A", "C6-B", "C7-A", "C7-B", "C8-A", "C8-B", "C9-A", "C9-B"]}}}, "then": {"properties": {"arm_kind": {"const": "single"}, "component_set": {"minItems": 1, "maxItems": 1}}}},
    {"if": {"properties": {"arm_id": {"enum": ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9"]}}}, "then": {"properties": {"arm_kind": {"const": "combination"}, "component_set": {"minItems": 2, "maxItems": 7}}}}
  ],
  "$defs": {
    "hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "finite_or_null": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    "four_values": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"$ref": "#/$defs/finite_or_null"}},
    "history_support": {"type": "object", "additionalProperties": false, "required": ["history_id", "p", "w", "a", "controls", "robot_days"], "properties": {"history_id": {"type": "string", "pattern": "^H-S17-V3-(DESIGN|DEV|CONF)-[0-9]{2}$"}, "p": {"type": "integer", "minimum": 0}, "w": {"type": "integer", "minimum": 0}, "a": {"type": "integer", "minimum": 0}, "controls": {"type": "integer", "minimum": 0}, "robot_days": {"type": "number", "minimum": 0}}},
    "score_branch": {"type": "object", "additionalProperties": false, "required": ["status", "threshold", "provenance"], "properties": {"status": {"enum": ["PRESENT", "NOT_APPLICABLE", "MISSING"]}, "threshold": {"$ref": "#/$defs/finite_or_null"}, "provenance": {"$ref": "#/$defs/hash"}}},
    "primary_metric": {"type": "object", "additionalProperties": false, "required": ["point", "lcb95", "delta_vs_B0", "directional_count", "per_history"], "properties": {"point": {"$ref": "#/$defs/finite_or_null"}, "lcb95": {"$ref": "#/$defs/finite_or_null"}, "delta_vs_B0": {"$ref": "#/$defs/finite_or_null"}, "directional_count": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 4}, {"type": "null"}]}, "per_history": {"$ref": "#/$defs/four_values"}}},
    "cohort_metric": {"type": "object", "additionalProperties": false, "required": ["point", "lcb95", "delta_vs_B0", "recall_per_history", "median_lead_days_per_history"], "properties": {"point": {"$ref": "#/$defs/finite_or_null"}, "lcb95": {"$ref": "#/$defs/finite_or_null"}, "delta_vs_B0": {"$ref": "#/$defs/finite_or_null"}, "recall_per_history": {"$ref": "#/$defs/four_values"}, "median_lead_days_per_history": {"$ref": "#/$defs/four_values"}}},
    "companion_metric": {"type": "object", "additionalProperties": false, "required": ["point", "lcb95", "per_history"], "properties": {"point": {"$ref": "#/$defs/finite_or_null"}, "lcb95": {"$ref": "#/$defs/finite_or_null"}, "per_history": {"$ref": "#/$defs/four_values"}}}
  }
}
```

The literal schema contract requires exactly one top-level document per `(phase, arm_id)`; all 15 arm IDs (`B0` plus 14 singles) in Development and all 24 IDs (`B0`, 14 singles, K1–K9) in Confirmation; both score branches with explicit status (never silent substitution); four history slots in every per-history metric; `invalid_reasons` for every status; bootstrap constants exactly `2000/20260202/2.5`; and every gate boolean even when false. B0 is explicitly `arm_kind="baseline"`, `config.one_principal_change="none"`, and `component_set=[]`; schema conditionals reject B0 with a non-empty component set or non-baseline kind. Singles require one component and `arm_kind="single"`; K arms require at least two components and `arm_kind="combination"`. `NOT_APPLICABLE` is permitted only for an unchanged branch of a cache-only arm while its paired branch is `PRESENT`; a `MISSING` branch makes the arm invalid. Artifact filenames are `artifacts/sprint-17/results/<phase>/<arm_id>.json` and aggregate index `artifacts/sprint-17/results/<phase>/index.json`, with hashes recorded in task evidence and the lock record.

## 8. Required provenance and reporting

Every result and evidence record must identify protocol/schema IDs, repository commit and dirty-tree state, runtime, arm/config digest, DGP/probe/metric-code digests, all role IDs and data/model seeds, root and cache hashes, checkpoint hash and step, parameter/FLOP counts, support/exclusion counts, threshold source, both score branches, all per-history metrics, bootstrap constants, gate booleans, status, and reasons. Reports must include every negative, failed, invalid, and inconclusive arm; no favorable omission. Sprint 17 artifacts remain under ignored `artifacts/`; source protocol and plan are the only tracked deliverables here.

This v4 document is a post-observation waiver amendment. It contains no new Sprint 17 model, data, score, or outcome and does not inspect or authorize access to Sprint 15 Sealed histories. Protocol-v3 remains the original `NOT_MEASURABLE` record; only the exact v4 waiver tuple may qualify downstream state as `MEASURABLE_WITH_USER_WAIVER`.
