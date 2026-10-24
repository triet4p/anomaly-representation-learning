# Sprint 18 Full-Training Representation Protocol v1

**Protocol ID:** `sprint18-full-training-protocol-v1`  
**Status:** FROZEN PROSPECTIVELY by Task 1; **not independently reviewed or execution-authorized**. Task 2 is a separate required contract-review gate.  
**Scope:** the ordered Sprint 18 eight-arm, fresh-role, 100-epoch comparison and its common scoring, decision, and provenance contract.  
**Decision boundary:** no Sprint 18 roots, coefficient outcomes, model outcomes, or Confirmation scores are part of this freeze.

## 1. Authority, versioning, and source lineage

This document freezes the common contract proposed by `experiments/sprint18-full-training-protocol.md`. The four N1–N4 specifications named in §4 are incorporated byte-for-byte by their SHA-256 digests; the unversioned drafts remain preserved as source inputs. The exact Sprint 17 K-member identities are inherited from the accepted Sprint 17 lock, not reconstructed from result scores. The accepted Sprint 15 Candidate 7 generator method is reused for new Sprint 18 roles only; no old role, root, seed, or outcome is reused.

The frozen Sprint 18 plan is the ordered authority for task boundaries. Task 1 freezes this common protocol; Task 2 is an independent prospective contract review; later tasks own role binding, roots, harness, pilots, execution, and result locks. This freeze does not claim Task 2 passed and authorizes none of those later actions by itself.

The common protocol governs role permissions, the shared training schedule, coefficient-selection rules, compute limits, scoring, recovery gates, result schema, and verdict precedence. Each mechanism specification governs its named mechanism. If a spec and this common contract conflict, execution is blocked pending an independently reviewed prospective amendment; an executor must not choose an interpretation. Any semantic change requires a new protocol/spec version and a fresh review before the affected data or outcome is accessed. No post-outcome edit, coefficient-grid expansion, arm substitution, score change, or threshold change is permitted.

### Accepted contract lineage

- Sprint 15 generation method: `experiments/sprint15-benchmark-protocol-v7.md`, SHA-256 `a1fc09db2e246ed79d0595aec953a7788fd1b47c4981fbe1d92017d944b8d7b6`.
- Sprint 15 normative measurability boundary: `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md`, SHA-256 `d294762331ded4fd213f6870e563e32e17e12c632d7cb11a80373478ede34e1c`.
- Sprint 17 accepted ablation contract: `experiments/sprint17-ablation-protocol-v4.md`, SHA-256 `35a0526b4b26b6003589fa2498b075abd9c3172a1dedf443ebdb396cc6df4482`.
- Sprint 17 accepted arm/configuration lock: `experiments/sprint17-task23-confirmation-lock.json`, file SHA-256 `3b2f14ffb9d4b3d548df2d02158f8c23e0808475dedf69e84877df7a2ae34bf4`, canonical `lock_sha256` `14fab303d78725af061aaa639a558f84b262c76d5b918042618a109d61afd039`.

Sprint 15's Candidate 7 methodology is inherited without changing its calendar-first construction, causal anchor/window predicates, quota/allocation method, nuisance bounds, observable definitions, or relevant thresholds. This is a new Sprint 18 role binding, not reuse of Sprint 15 or Sprint 17 data. The final Sprint 17 qualification `MEASURABLE_WITH_USER_WAIVER` is not a Sprint 18 waiver: every new Sprint 18 history must meet the applicable inherited structural and observable gates without that waiver. A miss stops the affected Sprint 18 path; no waiver, replacement, retry, or favorable subset is implied.

## 2. Scientific question and limits

The study asks whether the relative K3/K7/K9 hypotheses persist under a full, active prediction-plus-contrastive schedule, and whether one of four isolated N1–N4 mechanisms improves the same fresh benchmark relative to `B0-FULL`. It does not estimate the causal effect of training duration alone, establish that any Sprint 17 component was defective, authorize a combined mechanism, or make production, fleet-prevalence, calibrated-risk, deployment-threshold, real-data, or sealed-generalization claims.

All labels, event cohorts/subtypes, failure times, severity, anomaly masks, simulator state, and future information are post-hoc diagnostics only. They MUST NOT be model inputs, loss targets, coefficient-selection signals, training filters other than the verified-healthy eligibility predicate, checkpoint selectors, or score-selection inputs. N1 may use only its frozen robot/program relation metadata; N4 may use only the context input permitted by its spec.

## 3. Fresh role topology and information permissions

Task 3 binds a collision-free, fresh, whole-history roster and canonical binding digest before any roots are created. The fixed role names/cardinalities are:

| Role | Frozen history IDs | Count | Permitted use |
|---|---|---:|---|
| Design | `H-S18-DES-01`–`H-S18-DES-04` | 4 | Structural/measurability qualification; healthy-only coefficient-pilot retention evaluation. Never final fitting. |
| Fit | `H-S18-FIT-01`–`H-S18-FIT-03` | 3 | Verified-healthy representation fitting, EMA targets, Fit reference banks/standardizers, Fit-only auxiliary fitting and cross-fitting. |
| Calibration | `H-S18-CAL-01` | 1 | Verified-healthy per-arm/per-seed q95 thresholds only. |
| Development | `H-S18-DEV-01`–`H-S18-DEV-04` | 4 | Complete descriptive comparison and diagnostics after all configs are frozen. No coefficient, arm, mask, checkpoint, branch, or threshold selection. |
| Confirmation | `H-S18-CONF-01`–`H-S18-CONF-04` | 4 | One-shot locked final comparison only. |

The role sets are disjoint, complete, and fixed before generation. Task 3 supplies exact fresh data seeds, generator-byte digests, paths, and role-manifest bindings; Task 1 does not invent or pre-empt that binding. No reserve role, alternate seed block, retry, replacement, omission, favorable seed selection, cross-role pooling, or role reassignment is allowed. The Candidate 7 generation methodology is applied only to these 16 roles; no Sealed role is part of the Sprint 18 roster.

Sprint 18 MUST NOT enumerate, list, derive paths for, open, hash, probe, load, score, or otherwise contact Sprint 15 Sealed histories or their contents. Sprint 18 does not run the inherited EG6 Sealed evaluation and makes no claim that EG6 or any independent sealed-performance criterion passed. The prohibition includes using a Sealed root as an input to a tool or status check.

Before representation execution, the fresh roles must pass the applicable inherited Sprint 15 structural and observable qualification: EG0–EG3 as applicable, all four Design promotion targets and concentration limits (EG2), Fit/Calibration healthy support floors, and fixed-probe observability on both four-history Development and four-history Confirmation blocks using the unchanged EG4 thresholds. Confirmation structural generalization must satisfy the unchanged EG5 floors and `[0.75, 1.25]` median stability bands. These checks use the inherited probe fitted on Fit healthy rows and thresholded on Calibration healthy rows. A Design, Development, or Confirmation miss is a stop, not a license to inherit the Sprint 17 waiver or adjust the generator/probe after outcomes.

- The fixed-probe values are explicit: on each four-history Development and Confirmation block require macro `P+W >=0.65`, history-block `LCB95 >0.50`, at least 3/4 histories with `P+W >0.55`, macro P `>=0.70`, macro W `>=0.60`, P recall/median lead `>=0.50`/`>=1.0` day, W recall/median lead `>=0.25`/`>=0.5` day, and FAR `<=0.05` on every history. Abrupt A remains a reported companion without a minimum. No pooled or favorable-history rescue is allowed.

## 4. Immutable eight-arm registry and mechanism digests

Exactly these eight IDs are registered. IDs, principal mechanisms, and membership cannot be edited or pruned based on Development.

| Sprint 18 ID | Exact frozen identity | Representation training |
|---|---|---:|
| `B0-FULL` | Unchanged Sprint 17 B0 architecture/scorers under this common 100-epoch contract. | 3 model seeds × 100 Fit epochs |
| `K3-FULL` | Exact Sprint 17 K3 = `C5-A + C6-A`; no member substitution or combination-specific tuning. | 3 × 100 |
| `K7-FULL` | Exact Sprint 17 K7 = `C2-A + C3-A + C5-A + C6-A`. | 3 × 100 |
| `K9-FULL` | Exact Sprint 17 K9 = `C2-A + C3-A + C5-A + C6-A + C7-A + C8-A + C9-B`. | 3 × 100 |
| `N1-HCC` | B0 plus the one hierarchical conditional contrastive mechanism; replaces B0 InfoNCE as stated by the HCC spec. | 3 × 100 |
| `N2-MSP` | B0 plus fixed fine/4-patch/8-patch latent prediction; retains the common InfoNCE objective. | 3 × 100 |
| `N3-CCRP` | B0 plus the declared cross-channel relational latent-prediction objective; retains the common InfoNCE objective. | 3 × 100 |
| `N4-CRN` | Exact final `B0-FULL` representation/EMA/caches plus the Fit-only context-dependent scale model; zero representation optimizer steps. | 0 representation epochs; 3 fixed 50-epoch scale fits |

The K membership is the accepted Sprint 17 lock membership, whose source hash is given in §1. The old K implementations and component semantics remain exact; only the common Sprint 18 training duration/schedule, fresh roles, and shared evaluation contract change. Sprint 17 weights, banks, thresholds, and roots are never warm-started or reused.

The four new-mechanism specifications are frozen at these exact bytes:

| Arm | Immutable input path | SHA-256 |
|---|---|---|
| `N1-HCC` | `experiments/sprint18-hierarchical-conditional-contrastive-spec.md` | `bf757fa4cda85e8ea5e743928bf405294c6bc4c6f16c67d54b8edef763a69d51` |
| `N2-MSP` | `experiments/sprint18-multiscale-latent-prediction-spec.md` | `7b92cd5c9d7f468dd95e46fa544fcd57d0a76cf0638536560a5d4e4d5439e72d` |
| `N3-CCRP` | `experiments/sprint18-cross-channel-relational-prediction-spec.md` | `1c0f570606b665f0976bc8161f0d1dd3b2cc2d482b0f861b5dbc2975bd4c7cfd` |
| `N4-CRN` | `experiments/sprint18-context-residual-normalization-spec.md` | `fd0e1722ea4f248a354086bbdf44bc486d741a9cfb018f68ef08277827d7954d` |

Any byte or semantic change to one of these specs breaks the frozen digest binding and blocks the affected path pending a new reviewed version. A spec name alone is not identity.

## 5. Full-training optimizer, deterministic schedule, and checkpoints

### 5.1 Seeds, inputs, and epoch accounting

The fixed representation model seeds are `[181801, 181802, 181803]`; the N4 scale fits use the same three paired seed IDs. These are a new model-seed namespace, distinct from Sprint 17's `[171701, 171702, 171703]` and from the Task 3 data-seed binding. All seven representation-training arms use all three seeds. For a given seed, common modules use seed-matched fresh initialization across arms where tensor shapes permit; arm-specific new parameters use deterministic substreams derived from that model seed and a stable module/arm label. No Sprint 17 model state is restored. Python, NumPy, CPU tensor, and accelerator RNG states used by the training process are captured and restored coherently.
Initialization substreams are exact: seed each process RNG with the listed model seed; derive every module sub-seed as `int.from_bytes(SHA256(UTF8(key))[:8], "big") mod (2^63−1)`, where a shared module key is `sprint18-init-v1|<model_seed>|shared|<module_path>` and an arm-specific module key is `sprint18-init-v1|<model_seed>|<arm_id>|<module_path>`. Module paths are stable qualified names recorded in the config. No runtime-generated or outcome-dependent seed is allowed.

A representation epoch is one deterministic permutation of every eligible verified-healthy Fit file exactly once, batch size 128 files, `drop_last=false`; the final short batch is retained. The same model seed, epoch, Fit roster, batch order, and B0 view/augmentation realization are paired across arms wherever each frozen mechanism permits it. No non-healthy row enters an optimizer update. Global optimizer steps are exactly `100 × ceil(N_fit_files / 128)` per trainable arm/seed and are recorded from the bound Fit support; do not substitute Sprint 17's step count or pad a short epoch. The model-seed list and all derived initialization/sampler seeds are recorded in the config lock.
The epoch permutation is exact: sort canonical Fit file IDs lexically, compute `epoch_seed = int.from_bytes(SHA256(UTF8("sprint18-fit-order-v1|<model_seed>|<1-based-epoch>|<fit_role_sha256>"))[:8], "big") mod (2^63−1)`, seed a CPU `torch.Generator` with it, and apply one `torch.randperm` to that sorted list. Use these boundaries for every arm sharing that model seed; any mechanism-specific patch/mask order is derived from the frozen spec and recorded deterministic substream, never from labels or score values.
The existing B0 two-view augmentation algorithm remains unchanged. Its deterministic per-file/view stream key is `sprint18-view-v1|<model_seed>|<1-based-epoch>|<canonical_file_id>|<view_index>`; derive the per-view sub-seed with the same SHA-256/first-eight-byte/big-endian/modulo rule above. This keeps seed-matched views aligned without sharing mutable RNG state across files or arms.

### 5.2 Representation optimizer and epoch schedules

For `B0-FULL`, K3/K7/K9, and N1/N2/N3, use the same base optimizer contract:

| Setting | Frozen value |
|---|---:|
| Optimizer | AdamW |
| Batch size | 128 files; retain final partial batch |
| Base learning rate | `1.5e-3` |
| Weight decay | `0.0` |
| Global gradient clip | L2 norm `1.0` before optimizer step |
| Precision | float32; no mixed precision |
| Epochs | exactly 100 complete Fit epochs |
| EMA decay | `0.996` |
| B0 total unique masked-patch ratio | `0.40`, with each mechanism's frozen allocation semantics |

Learning-rate values are constant within an epoch. For 1-based epoch `e` and `lr_max = 1.5e-3`, use:

- `e = 1..5`: `lr(e) = lr_max × (0.1 + 0.9 × (e − 1) / 4)` (epoch 1 is `0.1 × lr_max`; epoch 5 is `lr_max`).
- `e = 6..100`: `lr(e) = 1e-5 + 0.5 × (lr_max − 1e-5) × [1 + cos(π × (e − 5) / 95)]` (continuous from the epoch-5 maximum to exactly `1e-5` at epoch 100).

The common contrastive coefficient is epoch-based and constant within each epoch. For `L_max` equal to the frozen selected coefficient:

- epochs 1–10: `λ(e) = 0` exactly;
- epochs 11–30: `λ(e) = L_max × (e − 10) / 20` (linear, reaching `L_max` at epoch 30);
- epochs 31–100: `λ(e) = L_max` exactly.

Thus the primary checkpoint includes exactly 70 complete epochs at the full active coefficient. N1 uses this schedule for `lambda_hcc` and replaces, rather than adds to, InfoNCE. N2/N3 retain the selected common InfoNCE coefficient plus their own selected auxiliary coefficient. B0/K3/K7/K9 share the same selected InfoNCE coefficient. No objective, mask, affinity, temperature, projection width, batch size, or architecture parameter may be altered to rescue a failed balance or compute gate.

### 5.3 Coherent checkpoint rule

For each representation-training seed, save diagnostic checkpoints at global step 300 when reached and after epochs 10, 30, 60, and 100. If step 300 is not reached by the exact 100-epoch schedule, record `NOT_REACHED`; do not pad training. If a requested checkpoint coincides with another checkpoint, store one byte-identical checkpoint and reference it in both ledger entries. Save the epoch-100 state as the sole primary checkpoint and only eligible Confirmation checkpoint. Learning-curve checkpoints are descriptive and cannot select, replace, or rescue the primary state.

Every checkpoint is one coherent atomic state: online model, EMA target, optimizer, LR scheduler, global step, completed epoch, effective loss-coefficient schedule state, deterministic sampler position, all RNG states, initialization/config digests, and any arm-specific train-time modules. Partial restore, late optimizer metadata with early weights, missing EMA/RNG state, or mismatched schedule/config makes the arm `INVALID`. No early stopping, best-checkpoint search, or resume from an incoherent checkpoint is allowed.

Build each arm's declared final reference bank and Fit-derived standardizers from that arm's final Fit state only. Calibration fits no representation or bank; it sets q95 thresholds only. Development and Confirmation never refit, recalibrate, re-standardize, or change any checkpoint.

N4 binds to the exact final `B0-FULL` checkpoint and cache for the same seed, detaches features/targets as required by its spec, performs zero representation updates, and fits only its fixed scale model on Fit-healthy records. Its three-way two-Fit-history/one-held-out Fit-history cross-fit is diagnostic; after the spec's finite/capacity gates pass, its one final scale model is fit on all Fit-healthy records for exactly 50 epochs with AdamW, lr `1e-3`, weight decay `1e-4`, batch size 512 patch records, clip `1.0`, and final-epoch-50 checkpoint. N4's fixed model has hidden width 32, at most 100,000 parameters, and Fit-only bounds `v_floor=max(q05(e_Fit),1e-6)`, `v_cap=max(q95(e_Fit),10×v_floor)`. Development healthy `v_cap` saturation above 20% invalidates N4; bounds/architecture/epochs cannot be retuned.

## 6. Healthy-only coefficient grids and mechanical balance gates

Coefficient selection is a prospective, healthy-only pilot stage. It runs only after the applicable pre-execution evidence gate; pilots train on Fit healthy rows only and evaluate retention on Design healthy rows only. Anomaly/event labels and all anomaly metrics are inaccessible to the balancing procedure. Design is never included in final model, bank, or scale-model fitting. Pilot traces and the chosen values are locked before Development is scored. Pilot runs are not the final 100-epoch runs and cannot produce primary model outcomes.

### 6.1 Frozen grids and selection order

The finite grids, in ascending order, are:

| Objective / arm(s) | Candidate maximum coefficient |
|---|---|
| Common InfoNCE on `B0-FULL`, copied unchanged to B0/K3/K7/K9/N2/N3 | `lambda_max ∈ [0.025, 0.05, 0.10]` |
| N1-HCC, replacing InfoNCE | `lambda_hcc_max ∈ [0.025, 0.05, 0.10]` |
| N2-MSP coarse objective | `alpha ∈ [0.10, 0.25, 0.50]` |
| N3-CCRP relational objective | `beta ∈ [0.10, 0.25, 0.50]` |

Selection order is fixed: (1) select the common InfoNCE maximum on B0 from its grid and copy that value to every named recipient without retuning; (2) select N1's HCC maximum from its grid; (3) with the common InfoNCE value active, select N2 `alpha`; (4) with the common InfoNCE value active, select N3 `beta`. N2 uses `L_coarse=0.5 L_group4 + 0.5 L_group8`; N3 uses its frozen `L_channel`. No coefficient is selected for N4.

For every grid selection, all three model seeds run the same fresh initialization, same Fit batches, same epoch-based schedule through epoch 30, and same frozen objective/masks as the matched pilot. The no-auxiliary comparison uses the same architecture, seed, and schedule with the coefficient under test set to zero; for N2/N3 the already-selected common InfoNCE coefficient remains active when testing only `alpha`/`beta`. For common InfoNCE and N1, the matched prediction-only reference has the relevant contrastive coefficient zero. The last five ramp epochs are 26–30.

### 6.2 Gradient and healthy-retention calculation

Let `theta` be all trainable online local-patch and context-encoder parameters shared by `L_pred` and the objective under test, including shared trainable normalization affine parameters; exclude EMA parameters and objective-specific projection, HCC, coarse, channel, and N4 scale heads. At every global optimizer step divisible by 40 during epochs 26–30, compute gradients without changing optimizer/scheduler/RNG state:

`r_aux = ||grad_theta(c(e) × L_aux)||_2 / (||grad_theta(L_pred)||_2 + 1e-12)`.

Here `c(e)` is the scheduled effective coefficient and `L_aux` is the one objective under test (InfoNCE, HCC, coarse loss, or channel loss). Use concatenated L2 norms over the exact same `theta`. For each seed, compute median and NumPy-linear-interpolated 90th percentile over that seed's collected ramp observations. Every seed must have at least one observation; missing diagnostics fail closed.

A candidate is balance-valid only if, for **each of the three seeds**:

1. every recorded raw loss, weighted loss, gradient, ratio, and embedding statistic is finite;
2. median `r_aux` is in the closed interval `[0.15, 0.50]` and 90th-percentile `r_aux <= 1.0`;
3. on each of the four Design histories, the median across embedding dimensions of file-embedding population standard deviation over Design-healthy files is `>= 0.05`;
4. on each Design history, the candidate's mean held-out healthy fine prediction loss is `<= 1.10 ×` its matched no-auxiliary pilot's mean loss.

Select the smallest grid value passing **every** condition for **every** seed. The common InfoNCE value selected on B0 is copied to K3/K7/K9/N2/N3; each recipient must then pass the same finite/gradient/embedding/retention checks at that copied coefficient with its own frozen architecture/objective. A copied coefficient is not retuned per arm. N2/N3 `alpha`/`beta` selection occurs with that fixed InfoNCE active. If a grid has no valid value, or any required copied value fails its recipient gate, stop the affected path. Do not expand/recenter a grid, interpolate, select a larger candidate, alter the batcher, or use a Development/Confirmation result to repair it. The complete traces, candidate pass/fail vector, chosen value, and canonical digest are part of the coefficient lock.

Record at least once per epoch: raw `L_pred`, each raw auxiliary loss, weighted auxiliary contribution, effective coefficient, both shared-encoder gradient norms, `r_aux`, embedding variance/effective rank, valid support, LR, epoch, and global step. Each active coefficient must equal its frozen maximum for epochs 31–100; a zero/nonfinite auxiliary contribution or failure to activate invalidates the intended full-training claim.

## 7. Compute envelope and measurement rules

Compute is measured by one common harness and one PyTorch runtime from the locked project environment, pinned and hash-recorded before coefficient pilots. Use `torch.profiler` with `with_flops=True` and `record_shapes=True`; count a multiply and an add as two FLOPs for a fused multiply-add. The measurement input is the exact complete Fit-healthy file support from the frozen Fit role, with its actual valid lengths, masks, and deterministic arm-specific objective views. Profile the complete deterministic 100-epoch forward/backward/gradient-clip/optimizer/EMA work without retaining model outcomes; run all scheduled objective/mask epochs where epoch-dependent work changes. N4 compute includes its three Fit cross-fit scale-model runs and final all-Fit scale-model run, each under the fixed 50-epoch contract. Inference is profiled on the same fixed Fit-healthy support with the ordinary uncorrupted per-branch inference/scoring path. No score value or anomaly outcome is read for compute accounting.

Parameter count is the sum of `numel()` for unique `torch.nn.Parameter` objects on the deployed primary inference path (shared/tied parameters counted once); exclude optimizer state, non-inference EMA targets, fixed buffers, caches, and reference-bank rows, but include every added scorer parameter such as N4's scale head. Report `S_pred` and `S_pop` inference FLOPs separately; the envelope must pass for each applicable branch. Training FLOPs include each declared online/EMA forward, objective, backward, clipping, optimizer, and EMA update; divide complete-run total by the exact number of Fit-healthy files. For N4 include all scale-fit work described above in that denominator.

The limits are:

| Arm(s) | Inference parameter / FLOP envelope | Training compute envelope |
|---|---|---|
| B0-FULL | Reference baseline. | `1.00 ×` B0 by definition. |
| K3/K7/K9 | Each inference parameter count and reference-sample scoring FLOPs `<= ceil(1.10 × B0-FULL)`. | Total 100-epoch Fit training FLOPs per Fit file `<= 1.25 × B0-FULL`. |
| N1-HCC | Primary inference parameter count and FLOPs equal B0 exactly; relation-matrix/objective overhead is counted in training. | Total 100-epoch Fit training FLOPs per Fit file `<= 1.25 × B0-FULL`. |
| N2-MSP | Parameters and primary fine-scale scoring FLOPs `<= ceil(1.10 × B0-FULL)`; coarse diagnostic scores are reported separately. | Total 100-epoch Fit training FLOPs per Fit file `<= 1.25 × B0-FULL`. |
| N3-CCRP | Parameters and ordinary uncorrupted primary scoring FLOPs `<= ceil(1.10 × B0-FULL)`; corrupted auxiliary forward is training-only. | Total 100-epoch Fit training FLOPs per Fit file `<= 2.25 × B0-FULL`. |
| N4-CRN | Frozen representation and `S_pop` exactly equal B0; scale model `<=100,000` parameters; combined scoring FLOPs `<= ceil(1.10 × B0-FULL)`. | Fixed 50-epoch scale-fit FLOPs per Fit file `<=0.25 ×` the paired B0 100-epoch Fit-training FLOPs per Fit file. |

For N4, `S_pop` equality is bitwise on identical cached inputs; model/cache/checkpoint hashes bind each scale fit to its paired B0 seed. The common input, counter, and limits may not be altered to rescue an arm. Any envelope breach is `INVALID`; no width/depth/epoch or mechanism reduction is permitted.

## 8. Score definitions, support, and required metrics

### 8.1 Independent score branches and calibration

Every arm reports two independent, explicitly named branches: `S_pred` (context/prediction mismatch) and `S_pop` (population/geometry mismatch). The branches have separate scores, thresholds, provenance, metrics, and gates. No concatenation, weighted sum, max/min, calibration pooling, threshold borrowing, or outcome-selected branch switch is permitted. N2 coarse residuals and N3 channel residuals are diagnostics only; neither can enter/fuse with a primary branch. N4's one declared `S_pred` is the frozen context-normalized residual in its spec, while `S_pop` remains bitwise B0.

For each arm and model seed, fit banks/standardizers only on that arm's eligible Fit-healthy state and compute each branch's operating threshold once as the `0.95` quantile of that branch's eligible Calibration-healthy file scores, using the inherited deterministic NumPy linear interpolation. Store each threshold and source hash. No Calibration anomaly row selects a threshold. Development and Confirmation do not refit or rethreshold.

Use the inherited tie-aware event AUROC and eligibility/aggregation implementation. Event score is the maximum file score over its eligible event-window files; a control score is the maximum over eligible control-window members. Per history, P and W AUROCs are computed on frozen common support and `P+W = (P_AUROC + W_AUROC) / 2`; the primary macro is the unweighted mean of the four per-history `P+W` values. A is reported separately as a descriptive companion with no minimum predictive threshold. All arm-minus-B0 deltas are paired on identical history, event, file, valid-patch, seed, branch, and support keys. No pooled-only estimate, seed deletion, favorable-history omission, or support change is valid.

Common support is fixed from the role manifests and inherited eligibility predicates, not per-arm score availability. The same support hash is used by every arm and score branch within each phase. A missing, nonfinite, single-class, or mismatched required value is not silently excluded: it makes the affected result invalid. Report eligible/excluded rows and all inherited exclusions.

### 8.2 Metrics and aggregation

For every branch, arm, model seed, and evaluation history, report P, W, P+W, A AUROC; paired deltas from the same seed/history B0; P/W recall and median first-alert lead at the frozen q95 threshold; false-alert episodes and FAR per evaluated robot-day; paired delta FAR; unaffected-background stability ratio and uncertainty; localization/duration accounting; severity ordering; P1/P2/W1/W2/A1/A2 slices; valid support; and mechanism-specific diagnostics. Report all 3×4 seed/history cells. Do not treat model seeds as independent histories.

For gate arithmetic, first pair within each seed/history, then average the three seed values within each of the four histories. The recovery macro is the unweighted four-history mean. The directional count is the number of histories whose seed-averaged absolute P+W AUROC is strictly `>0.55`. Positive-effect replication is the number of histories whose seed-averaged paired `Delta_PW` is strictly positive. The primary 95% LCB is a history-block bootstrap over the four histories: resample four history units with replacement, compute the macro of the seed-averaged per-history P+W values, `B=2000`, RNG seed `20260202`, and LCB the 2.5th percentile (NumPy linear quantile). Seeds are kept paired and are not resampled as histories. Report seed-specific intervals descriptively only; never call them an aggregate gate CI.

At each frozen threshold require and report per history after averaging seed-specific values: P event recall `>=0.50` and median first-alert lead `>=1.0` day; W event recall `>=0.25` and median first-alert lead `>=0.5` day; false-alert episodes/FAR `<=0.05` per evaluated robot-day under the inherited `<=2`-day grouping and reset split; paired arm-minus-B0 `Delta FAR <=+0.02`; and unaffected-background stability `<=0.10`. The stability ratio uses the accepted Sprint 17 common metric definition, `|median(background) − median(Fit healthy)| / max(pooled SD, STD_FLOOR)`, with the existing shared metric's `STD_FLOOR` and support implementation pinned in the metric-code lock. Report the background point estimate, its history-block uncertainty, support, and exclusions. No per-seed CI substitutes for the history-block gate.


For each branch, the descriptive background 95% uncertainty interval is the `[2.5, 97.5]` percentile history-block bootstrap of the four seed-averaged per-history stability ratios, `B=2000`, seed `20260202`, NumPy linear quantiles. It does not relax the per-history `<=0.10` gate.

### 8.3 Frozen Confirmation recovery conjunction

Apply the following conjunction separately to **each** registered score branch of each comparison arm against the corresponding B0 branch on the same Confirmation support. A branch passes only when every condition is true, without rounding before comparison:

1. paired macro `Delta_PW >= +0.05`;
2. strictly positive seed-averaged `Delta_PW` on at least 3 of 4 histories;
3. post-arm history-block `LCB95 > 0.55`;
4. absolute directional count `>=3/4` histories with P+W AUROC `>0.55`;
5. paired macro `Delta_P >= -0.02`;
6. paired macro `Delta_W >= -0.02`;
7. P recall/lead gates pass on all four histories;
8. W recall/lead gates pass on all four histories;
9. FAR `<=0.05` and paired `Delta FAR <=+0.02` on all four histories;
10. unaffected-background stability ratio `<=0.10` on all four histories;
11. data, support, score separation, finite-output, checkpoint, coefficient, and compute validity gates pass.

A recovery is branch-qualified and must name `(arm_id, score_branch)`; values never cross branches. Development does not apply recovery selection or delete arms. Every valid registered arm is reported on Development and remains in the eight-arm Confirmation lock. Only epoch 100 is eligible for a recovery claim. All eligible locked arms are evaluated once on Confirmation; no refit, retune, reselect, rescore, rerun, threshold change, support change, or pooling rescue follows opening Confirmation.

## 9. Arm validity, failures, and bounded verdict states

Every arm/phase has exactly one status: `VALID`, `FAILED`, or `INVALID`. `FAILED` means the declared execution could not complete; `INVALID` means a contract, provenance, support, finite-value, score-separation, checkpoint, coefficient, or compute invariant failed. Valid negative metrics remain `VALID`; poor AUROC is not by itself an invalidity. Preserve every status, reason, and partial output. No replacement, retry, dropped seed/history, or favorable omission is allowed.

The final Sprint 18 verdict is exactly one of:

- `FULL_TRAINING_RECOVERY`: the complete eight-arm comparison is valid and at least one K3/K7/K9 branch passes the §8.3 conjunction, while no N1–N4 branch passes.
- `NEW_MECHANISM_RECOVERY`: the complete comparison is valid and at least one N1–N4 branch passes, while no K3/K7/K9 branch passes.
- `MIXED_RECOVERY`: the complete comparison is valid and at least one legacy K arm and at least one N arm each pass on named independent branches.
- `NO_REPRODUCIBLE_RECOVERY`: all eight arms are validly evaluated on the locked roles and no comparison branch passes every gate.
- `UNRESOLVED`: a mandatory benchmark/role/provenance/metric gate fails, B0 is not valid, any registered arm/seed cannot be validly completed, the coefficient path is unsupported, Confirmation is not fully lockable/evaluable, or another frozen invariant prevents the complete bounded comparison.

`UNRESOLVED` takes precedence over any partial branch pass: if the full eight-arm comparison is incomplete, report partial metrics but do not call them a Sprint 18 recovery. A valid but weak arm is never pruned; a failed/invalid arm is never rescued. A relative branch recovery does not establish that training duration caused the difference or authorize production, calibrated risk, fleet prevalence, or Sealed claims.

## 10. Result schema and immutable Confirmation lock

Each `(phase, arm_id)` result is one JSON object validated against the closed Draft 2020-12 schema below (`additionalProperties:false` at every contract object). The matrix contains all eight exact IDs in §4 for both `development` and `confirmation`, including any `FAILED`/`INVALID` rows. Every score branch is mandatory; all arrays are canonical-order 3 model seeds and 4 evaluation histories. Invalid/not-run scalar metrics are JSON `null` with an explicit status/reason, never a fabricated number. A `VALID` row must have finite required values, all expected seeds/histories, complete support, and no invalid reason. The separate final verdict object uses exactly the five states in §9 and carries the eligible `(arm_id, branch)` pairs plus all invalid/unresolved paths.

Before any Confirmation score is opened, lock and hash: this protocol, all four spec bytes, role manifests/binding, all eight arm configs and selected coefficient traces, seeds/samplers, metric and compute-counter code, common support, every epoch-100 checkpoint and EMA state, N4 scale states, Fit banks/standardizers, Calibration thresholds, and the complete eligible/failure status matrix. Confirmation opens once only after the lock is independently reviewed. No artifact is changed or replaced after opening.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "sprint18-full-training-result-v1",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_id", "protocol_id", "phase", "arm_id", "status", "failure_reasons", "provenance", "configuration", "seed_runs", "support", "scores", "metrics", "branch_gates"],
  "properties": {
    "schema_id": {"const": "sprint18-full-training-result-v1"},
    "protocol_id": {"const": "sprint18-full-training-protocol-v1"},
    "phase": {"enum": ["development", "confirmation"]},
    "arm_id": {"enum": ["B0-FULL", "K3-FULL", "K7-FULL", "K9-FULL", "N1-HCC", "N2-MSP", "N3-CCRP", "N4-CRN"]},
    "status": {"enum": ["VALID", "FAILED", "INVALID"]},
    "failure_reasons": {"type": "array", "uniqueItems": true, "items": {"type": "string", "minLength": 1}},
    "provenance": {
      "type": "object", "additionalProperties": false,
      "required": ["protocol_sha256", "role_binding_sha256", "fit_role_sha256", "calibration_role_sha256", "evaluation_role_sha256", "config_sha256", "coefficient_lock_sha256", "metric_code_sha256", "compute_counter_sha256", "mechanism_spec_sha256"],
      "properties": {
        "protocol_sha256": {"$ref": "#/$defs/hash"},
        "role_binding_sha256": {"$ref": "#/$defs/hash"},
        "fit_role_sha256": {"$ref": "#/$defs/hash"},
        "calibration_role_sha256": {"$ref": "#/$defs/hash"},
        "evaluation_role_sha256": {"$ref": "#/$defs/hash"},
        "config_sha256": {"$ref": "#/$defs/hash"},
        "coefficient_lock_sha256": {"$ref": "#/$defs/hash"},
        "metric_code_sha256": {"$ref": "#/$defs/hash"},
        "compute_counter_sha256": {"$ref": "#/$defs/hash"},
        "mechanism_spec_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]}
      }
    },
    "configuration": {
      "type": "object", "additionalProperties": false,
      "required": ["model_seeds", "optimizer", "batch_size", "batch_size_unit", "learning_rate", "weight_decay", "clip_norm", "precision", "representation_epochs", "scale_epochs", "ema_decay", "lambda_info_max", "lambda_hcc_max", "alpha", "beta", "primary_checkpoint_epoch"],
      "properties": {
        "model_seeds": {"const": [181801, 181802, 181803]},
        "optimizer": {"const": "AdamW"},
        "batch_size": {"type": "integer", "minimum": 1},
        "batch_size_unit": {"enum": ["files", "patch_records"]},
        "learning_rate": {"type": "number", "exclusiveMinimum": 0},
        "weight_decay": {"type": "number", "minimum": 0},
        "clip_norm": {"const": 1.0},
        "precision": {"const": "float32"},
        "representation_epochs": {"enum": [0, 100]},
        "scale_epochs": {"enum": [0, 50]},
        "ema_decay": {"const": 0.996},
        "lambda_info_max": {"anyOf": [{"enum": [0.025, 0.05, 0.1]}, {"type": "null"}]},
        "lambda_hcc_max": {"anyOf": [{"enum": [0.025, 0.05, 0.1]}, {"type": "null"}]},
        "alpha": {"anyOf": [{"enum": [0.1, 0.25, 0.5]}, {"type": "null"}]},
        "beta": {"anyOf": [{"enum": [0.1, 0.25, 0.5]}, {"type": "null"}]},
        "primary_checkpoint_epoch": {"enum": [50, 100]}
      }
    },
    "seed_runs": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"$ref": "#/$defs/seed_run"}},
    "support": {
      "type": "object", "additionalProperties": false,
      "required": ["common_support_sha256", "evaluation_histories", "per_history", "excluded_rows"],
      "properties": {
        "common_support_sha256": {"$ref": "#/$defs/hash"},
        "evaluation_histories": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "string", "pattern": "^H-S18-(DEV|CONF)-0[1-4]$"}},
        "per_history": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"$ref": "#/$defs/history_support"}},
        "excluded_rows": {"type": "integer", "minimum": 0}
      }
    },
    "scores": {
      "type": "object", "additionalProperties": false, "required": ["S_pred", "S_pop"],
      "properties": {"S_pred": {"$ref": "#/$defs/score_branch"}, "S_pop": {"$ref": "#/$defs/score_branch"}}
    },
    "metrics": {
      "type": "object", "additionalProperties": false, "required": ["S_pred", "S_pop"],
      "properties": {"S_pred": {"$ref": "#/$defs/branch_metrics"}, "S_pop": {"$ref": "#/$defs/branch_metrics"}}
    },
    "branch_gates": {
      "type": "object", "additionalProperties": false, "required": ["S_pred", "S_pop"],
      "properties": {"S_pred": {"$ref": "#/$defs/gates"}, "S_pop": {"$ref": "#/$defs/gates"}}
    }
  },
  "allOf": [
    {
      "if": {"properties": {"arm_id": {"const": "N4-CRN"}}},
      "then": {"properties": {"configuration": {"properties": {
        "representation_epochs": {"const": 0}, "scale_epochs": {"const": 50},
        "primary_checkpoint_epoch": {"const": 50}, "batch_size": {"const": 512},
        "batch_size_unit": {"const": "patch_records"}, "learning_rate": {"const": 0.001},
        "weight_decay": {"const": 0.0001}
      }}}},
      "else": {"properties": {"configuration": {"properties": {
        "representation_epochs": {"const": 100}, "scale_epochs": {"const": 0},
        "primary_checkpoint_epoch": {"const": 100}, "batch_size": {"const": 128},
        "batch_size_unit": {"const": "files"}, "learning_rate": {"const": 0.0015},
        "weight_decay": {"const": 0.0}
      }}}}
    },
    {
      "if": {"properties": {"arm_id": {"enum": ["B0-FULL", "K3-FULL", "K7-FULL", "K9-FULL"]}}},
      "then": {
        "properties": {
          "configuration": {
            "properties": {
              "lambda_info_max": {"type": "number"},
              "lambda_hcc_max": {"type": "null"},
              "alpha": {"type": "null"},
              "beta": {"type": "null"}
            }
          }
        }
      }
    },
    {
      "if": {"properties": {"arm_id": {"const": "N1-HCC"}}},
      "then": {
        "properties": {
          "configuration": {
            "properties": {
              "lambda_info_max": {"type": "null"},
              "lambda_hcc_max": {"type": "number"},
              "alpha": {"type": "null"},
              "beta": {"type": "null"}
            }
          }
        }
      }
    },
    {
      "if": {"properties": {"arm_id": {"const": "N2-MSP"}}},
      "then": {"properties": {"configuration": {"properties": {"lambda_info_max": {"type": "number"}, "lambda_hcc_max": {"type": "null"}, "alpha": {"type": "number"}, "beta": {"type": "null"}}}}}
    },
    {
      "if": {"properties": {"arm_id": {"const": "N3-CCRP"}}},
      "then": {"properties": {"configuration": {"properties": {"lambda_info_max": {"type": "number"}, "lambda_hcc_max": {"type": "null"}, "alpha": {"type": "null"}, "beta": {"type": "number"}}}}}
    },
    {
      "if": {"properties": {"arm_id": {"const": "N4-CRN"}}},
      "then": {
        "properties": {
          "configuration": {"properties": {"lambda_info_max": {"type": "null"}, "lambda_hcc_max": {"type": "null"}, "alpha": {"type": "null"}, "beta": {"type": "null"}}},
          "scores": {"properties": {"S_pop": {"required": ["S_pop_bitwise_equal_to_B0"], "properties": {"S_pop_bitwise_equal_to_B0": {"const": true}}}}}
        }
      }
    },
    {
      "if": {"properties": {"status": {"const": "VALID"}}},
      "then": {"properties": {"failure_reasons": {"maxItems": 0}}}
    },
    {
      "if": {"properties": {"status": {"enum": ["FAILED", "INVALID"]}}},
      "then": {"properties": {"failure_reasons": {"minItems": 1}}}
    }
  ],
  "$defs": {
    "hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "finite_or_null": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    "seed_run": {
      "type": "object", "additionalProperties": false,
      "required": ["model_seed", "representation_epochs_completed", "scale_epochs_completed", "optimizer_steps", "primary_checkpoint_sha256", "ema_state_sha256", "optimizer_scheduler_rng_sha256", "fit_bank_sha256", "calibration_sha256", "coefficient_trace_sha256", "compute"],
      "properties": {
        "model_seed": {"enum": [181801, 181802, 181803]},
        "representation_epochs_completed": {"type": "integer", "minimum": 0, "maximum": 100},
        "scale_epochs_completed": {"type": "integer", "minimum": 0, "maximum": 50},
        "optimizer_steps": {"type": "integer", "minimum": 0},
        "primary_checkpoint_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
        "ema_state_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
        "optimizer_scheduler_rng_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
        "fit_bank_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
        "calibration_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
        "coefficient_trace_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
        "compute": {"$ref": "#/$defs/compute"}
      }
    },
    "history_support": {
      "type": "object", "additionalProperties": false,
      "required": ["history_id", "p", "w", "a", "controls", "robot_days"],
      "properties": {
        "history_id": {"type": "string", "pattern": "^H-S18-(DEV|CONF)-0[1-4]$"},
        "p": {"type": "integer", "minimum": 0}, "w": {"type": "integer", "minimum": 0},
        "a": {"type": "integer", "minimum": 0}, "controls": {"type": "integer", "minimum": 0},
        "robot_days": {"type": "number", "minimum": 0}
      }
    },
    "score_branch": {
      "type": "object", "additionalProperties": false,
      "required": ["calibration_quantile", "thresholds_by_seed", "calibration_role_sha256", "score_provenance_sha256", "independent"],
      "properties": {
        "calibration_quantile": {"const": 0.95},
        "thresholds_by_seed": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"$ref": "#/$defs/finite_or_null"}},
        "calibration_role_sha256": {"$ref": "#/$defs/hash"},
        "score_provenance_sha256": {"$ref": "#/$defs/hash"},
        "independent": {"const": true}, "S_pop_bitwise_equal_to_B0": {"type": ["boolean", "null"]}
      }
    },
    "history_metric": {
      "type": "object", "additionalProperties": false,
      "required": ["history_id", "P_AUROC", "W_AUROC", "PW_AUROC", "A_AUROC", "delta_P", "delta_W", "delta_PW", "P_recall", "P_median_lead_days", "W_recall", "W_median_lead_days", "FAR_per_robot_day", "delta_FAR", "background_stability_ratio", "localization", "severity_ordering", "subtype_slices"],
      "properties": {
        "history_id": {"type": "string", "pattern": "^H-S18-(DEV|CONF)-0[1-4]$"},
        "P_AUROC": {"$ref": "#/$defs/finite_or_null"}, "W_AUROC": {"$ref": "#/$defs/finite_or_null"},
        "PW_AUROC": {"$ref": "#/$defs/finite_or_null"}, "A_AUROC": {"$ref": "#/$defs/finite_or_null"},
        "delta_P": {"$ref": "#/$defs/finite_or_null"}, "delta_W": {"$ref": "#/$defs/finite_or_null"}, "delta_PW": {"$ref": "#/$defs/finite_or_null"},
        "P_recall": {"$ref": "#/$defs/finite_or_null"}, "P_median_lead_days": {"$ref": "#/$defs/finite_or_null"},
        "W_recall": {"$ref": "#/$defs/finite_or_null"}, "W_median_lead_days": {"$ref": "#/$defs/finite_or_null"},
        "FAR_per_robot_day": {"$ref": "#/$defs/finite_or_null"}, "delta_FAR": {"$ref": "#/$defs/finite_or_null"},
        "background_stability_ratio": {"$ref": "#/$defs/finite_or_null"}, "localization": {"$ref": "#/$defs/finite_or_null"},
        "severity_ordering": {"type": "object", "additionalProperties": false, "required": ["P", "W"], "properties": {"P": {"$ref": "#/$defs/finite_or_null"}, "W": {"$ref": "#/$defs/finite_or_null"}}},
        "subtype_slices": {"type": "object", "additionalProperties": false, "required": ["P1", "P2", "W1", "W2", "A1", "A2"], "properties": {"P1": {"$ref": "#/$defs/finite_or_null"}, "P2": {"$ref": "#/$defs/finite_or_null"}, "W1": {"$ref": "#/$defs/finite_or_null"}, "W2": {"$ref": "#/$defs/finite_or_null"}, "A1": {"$ref": "#/$defs/finite_or_null"}, "A2": {"$ref": "#/$defs/finite_or_null"}}}
      }
    },
    "branch_metrics": {
      "type": "object", "additionalProperties": false,
      "required": ["per_seed", "aggregate"],
      "properties": {
        "per_seed": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "additionalProperties": false, "required": ["model_seed", "per_history"], "properties": {"model_seed": {"enum": [181801, 181802, 181803]}, "per_history": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"$ref": "#/$defs/history_metric"}}}}},
        "aggregate": {
          "type": "object", "additionalProperties": false,
          "required": ["P_macro", "W_macro", "PW_macro", "PW_lcb95", "delta_P_macro", "delta_W_macro", "delta_PW_macro", "directional_count", "positive_delta_history_count", "background_uncertainty"],
          "properties": {
            "P_macro": {"$ref": "#/$defs/finite_or_null"}, "W_macro": {"$ref": "#/$defs/finite_or_null"},
            "PW_macro": {"$ref": "#/$defs/finite_or_null"}, "PW_lcb95": {"$ref": "#/$defs/finite_or_null"},
            "delta_P_macro": {"$ref": "#/$defs/finite_or_null"}, "delta_W_macro": {"$ref": "#/$defs/finite_or_null"},
            "delta_PW_macro": {"$ref": "#/$defs/finite_or_null"},
            "directional_count": {"type": ["integer", "null"], "minimum": 0, "maximum": 4},
            "positive_delta_history_count": {"type": ["integer", "null"], "minimum": 0, "maximum": 4},
            "background_uncertainty": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"$ref": "#/$defs/finite_or_null"}}
          }
        }
      }
    },
    "compute": {
      "type": "object", "additionalProperties": false,
      "required": ["parameters_inference", "inference_flops_per_fit_file", "training_flops_per_fit_file", "parameter_ratio_vs_B0", "inference_flop_ratio_vs_B0", "training_flop_ratio_vs_B0", "envelope_pass"],
      "properties": {"parameters_inference": {"type": ["integer", "null"], "minimum": 0}, "inference_flops_per_fit_file": {"$ref": "#/$defs/finite_or_null"}, "training_flops_per_fit_file": {"$ref": "#/$defs/finite_or_null"}, "parameter_ratio_vs_B0": {"$ref": "#/$defs/finite_or_null"}, "inference_flop_ratio_vs_B0": {"$ref": "#/$defs/finite_or_null"}, "training_flop_ratio_vs_B0": {"$ref": "#/$defs/finite_or_null"}, "envelope_pass": {"type": ["boolean", "null"]}}
    },
    "gates": {
      "type": "object", "additionalProperties": false,
      "required": ["recovery_applied", "minimum_effect", "positive_history_replication", "lcb", "directional", "P_noninferiority", "W_noninferiority", "P_recall_lead", "W_recall_lead", "FAR", "background_stability", "support_and_provenance", "compute", "score_separation", "recovery"],
      "properties": {"recovery_applied": {"type": "boolean"}, "minimum_effect": {"type": ["boolean", "null"]}, "positive_history_replication": {"type": ["boolean", "null"]}, "lcb": {"type": ["boolean", "null"]}, "directional": {"type": ["boolean", "null"]}, "P_noninferiority": {"type": ["boolean", "null"]}, "W_noninferiority": {"type": ["boolean", "null"]}, "P_recall_lead": {"type": ["boolean", "null"]}, "W_recall_lead": {"type": ["boolean", "null"]}, "FAR": {"type": ["boolean", "null"]}, "background_stability": {"type": ["boolean", "null"]}, "support_and_provenance": {"type": "boolean"}, "compute": {"type": "boolean"}, "score_separation": {"type": "boolean"}, "recovery": {"type": ["boolean", "null"]}}
    }
  }
}
```

For a `development` row set `recovery_applied=false` and all recovery-only gate values to `null`. For a valid `confirmation` comparison row other than B0, set `recovery_applied=true`; each branch's `recovery` is a boolean from the full §8.3 conjunction. B0 and any `FAILED`/`INVALID` confirmation row set `recovery_applied=false` and all recovery-only gate values to `null`. B0 has no arm-minus-itself recovery claim. Status, exact seed/history ordering, arm-specific epochs/coefficient applicability, common-support identity, all gate arithmetic, and the cross-object relationship between summaries and cells are additionally validated by the result validator; JSON Schema does not replace those semantic checks.


The final verdict is a separate closed JSON object validated against:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "sprint18-verdict-v1",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_id", "protocol_id", "verdict", "eligible_recoveries", "unresolved_paths", "role_binding_sha256", "development_matrix_sha256", "confirmation_lock_sha256", "confirmation_matrix_sha256"],
  "properties": {
    "schema_id": {"const": "sprint18-verdict-v1"},
    "protocol_id": {"const": "sprint18-full-training-protocol-v1"},
    "verdict": {"enum": ["FULL_TRAINING_RECOVERY", "NEW_MECHANISM_RECOVERY", "MIXED_RECOVERY", "NO_REPRODUCIBLE_RECOVERY", "UNRESOLVED"]},
    "eligible_recoveries": {
      "type": "array", "uniqueItems": true,
      "items": {
        "type": "object", "additionalProperties": false, "required": ["arm_id", "score_branch"],
        "properties": {
          "arm_id": {"enum": ["K3-FULL", "K7-FULL", "K9-FULL", "N1-HCC", "N2-MSP", "N3-CCRP", "N4-CRN"]},
          "score_branch": {"enum": ["S_pred", "S_pop"]}
        }
      }
    },
    "unresolved_paths": {"type": "array", "items": {"type": "string", "minLength": 1}},
    "role_binding_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
    "development_matrix_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
    "confirmation_lock_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]},
    "confirmation_matrix_sha256": {"anyOf": [{"$ref": "#/$defs/hash"}, {"type": "null"}]}
  },
  "$defs": {
    "hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"}
  }
}
```

Verdict semantics remain those of §9: an `UNRESOLVED` record has no eligible recovery claim and lists the concrete blocking paths; each other state lists every branch-qualified recovery and no unresolved path. `NO_REPRODUCIBLE_RECOVERY` requires a complete valid eight-arm Confirmation matrix and an empty eligible-recovery list.

## 11. Required evidence and no-execution boundary

Required downstream records are: the independently reviewed contract, fresh role binding, role qualification, common harness/config/metric/compute hashes, coefficient traces and mechanical choices, coherent per-seed checkpoints, Fit/Calibration provenance, complete Development matrix, complete pre-Confirmation lock, one-shot Confirmation matrix, and final bounded verdict. Each ordered batch requires its own zero-actionable evidence review; final completion additionally requires the sprint-wide zero-actionable differential deep review. This protocol is not that review.

No data generation, model training, scoring, or Confirmation work is part of Task 1. Sprint 18 status and checklist ownership remain with Main. No permanent test is required for the protocol-only freeze; its document/schema integrity is verified by the targeted checks recorded in `artifacts/sprint-18/task-1.md`.
