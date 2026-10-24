# Sprint 18 N2-MSP Specification v2 Amendment

**Specification ID:** `sprint18-n2-msp-spec-v2`  
**Status:** PROSPECTIVE AMENDMENT FOR INDEPENDENT REVIEW — not execution-authorized.  
**Arm:** `N2-MSP`  
**Base specification:** `experiments/sprint18-multiscale-latent-prediction-spec.md`, SHA-256 `7b92cd5c9d7f468dd95e46fa544fcd57d0a76cf0638536560a5d4e4d5439e72d`.  
**Base protocol:** `experiments/sprint18-full-training-protocol-v1.md`, SHA-256 `9ae3d43f94a6194bae602d68067958bdf7d11cf47e801dd328e87b22e7907cf6`.

This version replaces only v1 §3 and behavioral checks 1 and 4. All other N2 clauses are inherited unchanged from the exact base-specification digest above. The v2 protocol amendment defines this file as the sole N2 specification for a v2 contract; do not combine it with a v1 N2 specification.

## Replacement for §3 — Leakage-free mask allocation

For each Fit file and representation epoch, form `V`, the B0 patch indices that are not padding, sorted by original B0 patch index; let `n = |V|`. Define the exact unique-mask budget as `M = min(n, int(round(0.40 × n)))`, using Python's built-in ties-to-even `round`, matching the B0 total-count rule. For `n=0`, `M=0` and no scale target is emitted.

Allocate this integer budget to fine, coarse-4, and coarse-8 by choosing non-negative integers `(F, g4, g8)` satisfying `F + 4g4 + 8g8 = M` that minimize the integer score

`S = 4 × (2F − M)^2 + (16g4 − M)^2 + (32g8 − M)^2`.

If several allocations attain the minimum, choose the lexicographically greatest `(F, g4, g8)` (fine first, then coarse-4, then coarse-8). Thus `F`, `4g4`, and `8g8` are the deterministic nearest feasible integer allocation to the declared 50%/25%/25% position shares, while every coarse quota is a whole number of complete groups. This rule replaces the v1 unspecified “integer rounding already used by B0” for the three-scale split; the total `M` remains the B0 0.40 count.

Derive one file/epoch mask seed as

`int.from_bytes(SHA256(UTF8("sprint18-n2-mask-v2|<model_seed>|<1-based-epoch>|<fit_role_sha256>|<canonical_file_id>"))[:8], "big") mod (2^63−1)`.

Initialize one `numpy.random.default_rng(seed)` per file/epoch. Candidate groups of length `L` are every window of `L` adjacent original B0 patch indices whose positions are all in `V`, enumerated by ascending first patch index. Selected groups may not overlap.

1. Let `cap8 = sum(floor(run_length/8))` over maximal consecutive runs in `V`. Place `q8 = min(g8, cap8)` coarse-8 groups. Consider candidate starts in `rng.permutation(number_of_candidates)` order. Accept a group only if it is free and, after tentatively selecting it, the remaining free consecutive runs can still fit every unplaced group up to `q8` (`sum(floor(run_length/8)) >= groups_still_needed`). This seeded, capacity-preserving scan fills `q8` whenever feasible and otherwise has no order-dependent false shortfall.
2. Reassign every unplaced coarse-8 group to two coarse-4 groups. The coarse-4 request is `g4 + 2 × (g8 − q8)`. From the remaining positions, let `cap4 = sum(floor(free_run_length/4))`; place `q4 = min(coarse4_request, cap4)` coarse-4 groups using the next `rng.permutation(number_of_candidates)` order and the same capacity-preserving acceptance rule for groups of four. Any unplaced coarse-4 positions fall through to fine scale.
3. The fine count is exactly `f = M − 8q8 − 4q4`. Apply the frozen B0 fine-mask algorithm to the remaining free valid indices, with its integer target overridden to exactly `f`; do not recompute the global `0.40` ratio. The override changes only the target count and eligible-index pool. Use the same per-file generator after the two coarse-candidate permutations. When `f=0`, select no fine positions.

For this fine-mask call, keep the B0 `MaskingConfig` composition fixed at random/info/block fractions `(0.34, 0.33, 0.33)`, information-stratum fractions `(0.20, 0.30, 0.30, 0.20)` in stable/transition/dynamic/extreme order, and block-size range `[2, 6]`. Normalize each fraction vector, apportion integer component/stratum quotas by largest remainder, and resolve equal remainders in the listed order.

Run the existing B0 selection steps on the remaining free indices, in this order: (a) choose the random quota uniformly without replacement using `rng.choice`; (b) compute the unchanged B0 patch-information scores on the original patch data, form linear 25/50/75-percentile strata over the remaining free indices using the B0 boundaries (stable `<=q25`, transition `(q25,q50]`, dynamic `(q50,q75]`, extreme `>q75`), and choose each stratum quota uniformly without replacement; if a stratum is short, fill its unfilled quota uniformly from the still-free indices; (c) for the block quota, repeatedly draw `take = min(remaining_quota, available_count, max(1, rng.integers(2, 7)))`, draw `start = rng.integers(0, available_count - take + 1)`, and choose that length-`take` slice from the ascending array of still-free indices, removing it before the next draw. If any defensive shortfall remains, fill it from the lowest remaining original B0 patch indices. No B0 selection step may select a coarse-group member or exceed `f`. Record the fine target and the realized B0 random/info/block counts.
Attribute any defensive fill to the first positive component in random/info/block order, matching B0 composition accounting.

Here a free consecutive run is a maximal sequence of adjacent original B0 patch indices still eligible after prior groups are removed. The capacity rule uses `sum(floor(run_length/L))`, the exact maximum number of disjoint length-`L` groups in those runs. This makes candidate selection deterministic without sacrificing a feasible quota. If there are too few valid runs for an 8-group, its eight positions are offered to two 4-groups; if those cannot be placed, their remaining positions become fine targets. Record, per file, `n`, `M`, `(F,g4,g8)`, realized fine/coarse counts, and every 8→4 and 4→fine reassignment.

The union of fine and group members must contain exactly `M` distinct valid positions. Every selected coarse group is fully hidden from the online encoder before context encoding; no group is padded or formed across a gap. The inherited EMA-target, stop-gradient, loss, coefficient, inference, and no-leakage clauses remain unchanged.

## Replacements for behavioral checks

Replace check 1 with:

> 1. Per file/epoch, the unique mask union has exactly `M = min(n, int(round(0.40 × n)))` positions; verify the integer allocation, disjoint group membership, and realized fine/coarse counts against §3.

Replace check 4 with:

> 4. Reproduce the same allocation and seeded group order for identical inputs/seeds; verify every infeasible 8-group is offered to two 4-groups before any remaining positions fall through to fine scale, and record both fallback counts by file.