# Sprint 18 N2-MSP — Multi-Scale Latent Prediction Specification

**Status:** DRAFT — freeze with the Sprint 18 protocol before execution.
**Arm:** `N2-MSP`
**Base:** `B0-FULL`
**Principal change:** Add coarse temporal latent targets at two fixed patch spans while preserving B0's fine EMA latent prediction and independent scores.

## 1. Hypothesis

A single patch-level target may reward short-range predictability while under-constraining healthy structure that spans longer intervals. N2-MSP asks whether one representation can predict masked EMA latents at fine, 4-patch, and 8-patch scales without reconstructing the waveform.

This is not C5-A's multi-horizon target. Horizons change which positions are predicted; N2-MSP changes the temporal support over which the target latent is defined.

## 2. Fixed scales and target construction

Use B0 patch order, valid lengths, starts, padding masks, and file identity. The scales are:

- fine: one patch;
- coarse-4: four consecutive valid patch positions;
- coarse-8: eight consecutive valid patch positions.

For an eligible coarse group `G`, construct a stop-gradient teacher target from the EMA patch latents:

`z_G = LayerNorm(sum_{t in G} valid_weight_t × z_target_t / sum valid_weight_t)`.

`valid_weight_t` is the valid-timestep fraction of patch `t`; padded timesteps have zero weight. A group is eligible only when all declared member positions exist and total valid weight is positive.

The student group predictor receives only contextual outputs at masked member positions:

`z_hat_G = MLP_scale(LayerNorm(weighted_mean_{t in G}(h_context_t)))`.

Use separate two-layer predictors for coarse-4 and coarse-8; hidden and output width equal `d_model`. Target and student group heads share no parameters. Teacher targets are stop-gradient.

## 3. Leakage-free mask allocation

The total unique masked-patch ratio remains exactly `0.40` subject to integer rounding already used by B0. Allocate the mask budget prospectively:

- 50% of masked positions: B0 fine-mask sampler;
- 25%: non-overlapping contiguous coarse-4 groups;
- 25%: non-overlapping contiguous coarse-8 groups.

Coarse groups cannot overlap each other or fine-selected positions. Selection is deterministic from the common masking seed. If the file is too short to satisfy a coarse quota, reassign the infeasible quota to the next smaller scale (`8 -> 4 -> fine`) and record the reassignment. Never pad a group with visible or fabricated tokens.

Every member of a selected coarse group is masked from the online encoder before context encoding. A coarse target is never constructed from a partially visible group. The EMA target branch may encode the unmasked source, as in B0, but its outputs remain stop-gradient and inaccessible to the online context path.

## 4. Loss

Fine prediction uses B0's loss over every masked patch. Coarse losses use the same normalized latent-distance family:

`L_coarse = 0.5 × L_group4 + 0.5 × L_group8`.

`L_total = L_fine + alpha × L_coarse + lambda(step) × L_InfoNCE`.

- Select `alpha` from `[0.10, 0.25, 0.50]` using the common Design-only gradient-balance rule.
- InfoNCE uses the coefficient selected once on `B0-FULL`; N2 does not retune it.
- B0 augmentations, Conditional BatchNorm, context encoder, projection head, and EMA decay remain unchanged.
- Missing support at one scale is excluded from that scale's denominator, never represented as zero loss.

## 5. Inference behavior

Primary `S_pred` remains the original fine patch-level context mismatch, with B0 localization and aggregation. Coarse-4 and coarse-8 mismatch are required diagnostics but are not fused into `S_pred`, thresholds, or promotion metrics during Sprint 18. `S_pop` is unchanged except through the learned representation.

This restriction separates the representation-learning question from a score-fusion question.

## 6. Required behavioral checks

1. The union of unique masked positions equals the frozen total-mask count.
2. Every coarse target group is fully hidden from the online context path.
3. Coarse target weights ignore padding and sum to one on eligible support.
4. Short-file quota reassignment follows exactly `8 -> 4 -> fine` and is recorded.
5. Teacher targets are EMA and stop-gradient at all scales.
6. `alpha=0` reproduces B0 loss/scoring semantics within declared tolerance when given the same mask realization.
7. Fine `S_pred` shape, patch starts, valid lengths, and localization mapping match B0.
8. Coarse predictors add no raw-waveform decoder and cannot access labels or simulator state.
9. Scale-specific loss denominators are explicit and never include unsupported files as zeros.
10. Parameters/FLOPs of both scale heads are measured against `B0-FULL`.

## 7. Required diagnostics

Report per scale and seed:

- eligible groups and quota reassignments by file-length bucket;
- raw and weighted loss;
- gradient norm ratio to fine prediction;
- residual distribution by P/W/A subtype and duration, post-hoc only;
- representation effective rank and file-embedding variance;
- primary fine-scale AUROC plus descriptive coarse-scale AUROC without fusion;
- localization support and duration accounting at each scale.

A macro improvement concentrated only in long files or one scale must be disclosed. Coarse diagnostic superiority does not authorize changing the primary score after Development.

## 8. Forbidden shortcuts

- Do not average visible and masked tokens into the same coarse student input.
- Do not increase total mask ratio beyond `0.40`.
- Do not replace fine prediction with coarse prediction.
- Do not select group sizes or `alpha` using anomaly outcomes.
- Do not fuse coarse scores into the primary metric in Sprint 18.
- Do not describe this arm as waveform reconstruction.
