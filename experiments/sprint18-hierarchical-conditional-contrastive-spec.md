# Sprint 18 N1-HCC — Hierarchical Conditional Contrastive Specification

**Status:** DRAFT — freeze with the Sprint 18 protocol before execution.
**Arm:** `N1-HCC`
**Base:** `B0-FULL`
**Principal change:** Replace same-file-only InfoNCE target semantics with a soft hierarchical relation target aligned with Conditional BatchNorm.

## 1. Intended relation

The metadata prior is ordered, not categorical identity:

1. two augmented views of the same file: strongest relation;
2. different files with the same `(robot, program)`: strong relation;
3. different files with the same robot and different programs: weak relation;
4. different robots: no positive relation is asserted.

`No positive relation` does not mean a pair is known to be semantically opposite. Such samples remain denominator alternatives in the contrastive distribution but receive no positive target mass.

Program equality across different robots does not define an additional relation level. Labels, anomaly categories, failure time, severity, and simulator state never define pairs.

## 2. Soft-target contrastive loss

Each Fit file produces the same two semantics-preserving augmented views as B0. Let normalized projected file embeddings be `u_i`, and let `j != i` index other views in the batch.

Define unnormalized target affinity:

| Pair relation | Affinity |
|---|---:|
| other view of the same source file | `1.0` |
| different source file, same robot and same program | `0.5` |
| different source file, same robot and different program | `0.1` |
| different robot | `0.0` |

For each anchor, normalize positive affinity:

`q_ij = affinity(i,j) / sum_{k != i} affinity(i,k)`.

The same-file paired view guarantees a non-zero denominator. Predicted relation probabilities are:

`p_ij = exp(cos(u_i,u_j)/tau) / sum_{k != i} exp(cos(u_i,u_k)/tau)`.

The symmetric arm loss is:

`L_HCC = mean_i[-sum_{j != i} q_ij log(p_ij)]`.

Use B0's projection dimension, L2 normalization, temperature, augmentation family, and two-view symmetry. `N1-HCC` replaces `L_InfoNCE`; it does not add both losses.

The affinities are fixed constants, not tunable outcomes. Only `lambda_hcc_max` is selected by the common Design-only balance rule.

## 3. Common deterministic batching and relation support

Use the exact B0 seeded epoch permutation, batch boundaries, final-batch rule, and two-view construction. N1 must not introduce relation-aware oversampling, grouping, duplication, or reordering: otherwise batching would become a second principal change.

For every batch and epoch, record:

- anchors with a cross-file same-`(robot,program)` relation;
- anchors with a cross-file same-robot/different-program relation;
- anchors supported only by their same-file view;
- target entropy and relation mass by level.

Every eligible Fit file must appear exactly once per epoch, with batch membership and order identical to `B0-FULL` under the same model seed and epoch. Both cross-file relation levels must receive non-zero support over every complete epoch. If natural common batches provide no support for either level, the arm is structurally unsupported; do not change the sampler after inspection.

## 4. Conditional BatchNorm interaction

Conditional BatchNorm remains byte-equivalent to B0. HCC does not add a second metadata-conditioning path into the patch or context encoder. Metadata is used only to:

- select the existing Conditional BatchNorm bucket/fallback; and
- construct the contrastive target affinities above.

The hierarchy is exactly `(robot, program) -> robot -> global fallback`. Unknown or under-supported groups follow B0's frozen fallback behavior. HCC must not create a program-only-across-robots fallback.

## 5. Loss and schedule

`L_total = L_pred + lambda_hcc(step) × L_HCC`.

- Prediction loss, EMA target, stop-gradient behavior, masking, and `S_pred` scoring are unchanged from `B0-FULL`.
- `lambda_hcc(step)` uses the common 10-epoch zero warmup, 20-epoch linear ramp, and 70-epoch plateau.
- Select `lambda_hcc_max` from `[0.025, 0.05, 0.10]` using only the protocol's gradient-balance rule.
- Do not alter temperature, projection width, augmentation intensity, model width, or batch size to rescue a failed coefficient candidate.

## 6. Required behavioral checks

1. Same-file affinity is always `1.0` and appears exactly once per anchor.
2. Same `(robot, program)` and same-robot/different-program relations receive `0.5` and `0.1` respectively.
3. Different-robot and program-only pairs receive zero positive mass.
4. Target distributions sum to one and remain finite for every anchor.
5. Reordering batch rows changes no relation matrix after applying the same permutation to outputs.
6. Epoch batch membership/order is identical to B0 and deterministic by seed/epoch.
7. Conditional BatchNorm parameters and fallback decisions match B0 for identical inputs.
8. Labels and simulator-only fields cannot enter the relation builder API.
9. `lambda_hcc=0` reproduces B0 joint-loss semantics within declared numerical tolerance.
10. Both relation levels obtain non-zero empirical support; otherwise report the arm structurally unsupported rather than silently degrading to InfoNCE.

## 7. Evaluation diagnostics

In addition to the common metrics, report on Fit, Development, and Confirmation without fitting on the latter two:

- cosine similarity distributions for same file, same `(robot,program)`, same robot/different program, and different robot;
- ordering rate `sim(same RP) > sim(same R) > sim(unrelated)` on supported triplets;
- embedding variance and effective rank;
- per-condition reference-bank coverage;
- background stability by robot/program support tier.

These are diagnostics, not alternative promotion targets. Primary inference remains separate `S_pred` and `S_pop`.

## 8. Forbidden interpretations and implementation shortcuts

- Do not treat every file other than the anchor as a known semantic negative.
- Do not make all files in one robot/program share one target embedding.
- Do not use anomaly labels or temporal proximity to create positives.
- Do not replace Conditional BatchNorm with metadata embeddings in this arm.
- Do not tune affinities after viewing AUROC.
- Do not claim that improved contrastive loss alone improves anomaly detection.
