# Sprint 18 N1-HCC Specification v2 Amendment

**Specification ID:** `sprint18-n1-hcc-spec-v2`  
**Status:** PROSPECTIVE AMENDMENT FOR INDEPENDENT REVIEW — not execution-authorized.  
**Arm:** `N1-HCC`  
**Base specification:** `experiments/sprint18-hierarchical-conditional-contrastive-spec.md`, SHA-256 `bf757fa4cda85e8ea5e743928bf405294c6bc4c6f16c67d54b8edef763a69d51`.  
**Base protocol:** `experiments/sprint18-full-training-protocol-v1.md`, SHA-256 `9ae3d43f94a6194bae602d68067958bdf7d11cf47e801dd328e87b22e7907cf6`.

This version changes only the Conditional BatchNorm parity wording in §4 and behavioral check 7 of the bound v1 specification. All other N1 clauses are inherited unchanged from the exact base-specification digest above. The v2 protocol amendment defines this file as the sole N1 specification for a v2 contract; do not combine it with a v1 N1 specification.

## Replacements

Replace the first sentence of v1 §4 with:

> N1 uses the exact B0 Conditional BatchNorm implementation and configuration: the same module layout, normalization/affine behavior, initialization substream, condition-bucket mapping, and unknown/under-supported fallback mapping. With matched pre-optimizer parameters and identical inputs/conditions, the initial N1 Conditional BatchNorm outputs and fallback decisions match B0. Conditional BatchNorm parameters remain trainable. After optimization begins, learned affine parameters and running state are expected to evolve with each arm's own gradients and are not required to remain equal to B0. HCC adds no second metadata-conditioning path into the patch or context encoder. Metadata is used only to select the existing Conditional BatchNorm bucket/fallback and construct the frozen contrastive relation affinities.

Replace v1 behavioral check 7 with:

> 7. Before the first optimizer update, the N1 Conditional BatchNorm configuration, initialized state, and fallback mapping match B0 for identical inputs/conditions; verify the mapping on known, unknown, and under-supported groups. Confirm that the parameters remain trainable. Do not require post-update parameter, running-state, or output equality with B0.

The inherited prohibition on a program-only-across-robots fallback remains in force. This amendment does not freeze Conditional BatchNorm, change its fallback hierarchy, modify the HCC objective, or alter any common training, coefficient, compute, scoring, or decision gate.