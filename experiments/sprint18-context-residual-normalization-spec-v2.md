# Sprint 18 N4-CRN Specification v2 Amendment

**Specification ID:** `sprint18-n4-crn-spec-v2`  
**Status:** PROSPECTIVE AMENDMENT FOR INDEPENDENT REVIEW — not execution-authorized.  
**Arm:** `N4-CRN`  
**Base specification:** `experiments/sprint18-context-residual-normalization-spec.md`, SHA-256 `fd0e1722ea4f248a354086bbdf44bc486d741a9cfb018f68ef08277827d7954d`.  
**Base protocol:** `experiments/sprint18-full-training-protocol-v1.md`, SHA-256 `9ae3d43f94a6194bae602d68067958bdf7d11cf47e801dd328e87b22e7907cf6`.

This version replaces only v1 §3's bound-estimation scope, v1 §4's fit-order/step-accounting clauses, and behavioral checks 3–4 to make their scope explicit. All other N4 clauses are inherited unchanged from the exact base-specification digest above. The v2 protocol amendment defines this file as the sole N4 specification for a v2 contract; do not combine it with a v1 N4 specification.

## Replacement for §3 — Fit-scope residual bounds

For each model seed, retain the same eligible Fit-healthy residual records and eligibility predicate as v1. Use the three Fit histories in the frozen role binding, ordered by canonical history ID. For diagnostic cross-fit fold `h`, train and score using the other two histories only. Compute that fold's bounds from the residuals in those two in-fold training histories only:

`v_floor(h) = max(float(numpy.quantile(E_train(h), 0.05, method="linear")), 1e-6)`

`v_cap(h) = max(float(numpy.quantile(E_train(h), 0.95, method="linear")), 10 × v_floor(h))`.

The held-out history contributes no residual to either fold bound. Use the same NumPy linear-interpolation convention as the common protocol. Record both bounds, the exact training-history IDs, and the residual-record count for each fold. After all three diagnostic folds pass the inherited capacity and finite-output gates, recompute `v_floor(final)` and `v_cap(final)` from the eligible Fit-healthy residuals in all three Fit histories using the same formulas; only these all-Fit bounds are used by the final all-Fit scale model for Calibration/Development/Confirmation scoring. Do not reuse a fold bound for the final fit, pool held-out residuals into a fold bound, or use non-Fit residuals.

## Replacement for §4 — Exact record order and optimizer steps

Each of the three diagnostic scale fits uses the two in-fold Fit histories; the final scale fit uses all three Fit histories. For each fit scope, create exactly one eligible scale-training record for each eligible cached Fit-healthy patch record already defined by v1. Sort the canonical record list by `(history_id lexical ascending, canonical_file_id lexical ascending, original B0 patch_index ascending)` before shuffling. Do not duplicate or drop records to make batches even.

For each model seed and each 1-based epoch `e` in `1..50`, derive a separate order seed from

`int.from_bytes(SHA256(UTF8("sprint18-n4-scale-order-v2|<model_seed>|<scope>|<comma-joined sorted in-fold Fit history IDs>|<e>"))[:8], "big") mod (2^63−1)`,

where `<scope>` is `fold:<heldout_history_id>` for a diagnostic fold and `final` for the all-Fit model. Seed a CPU `torch.Generator` with this value and apply exactly one `torch.randperm(N_scope, generator=generator)` to that scope's canonical record list. Train each resulting permutation once in contiguous batches of at most 512 records, preserving the final partial batch (`drop_last=false`). Thus every eligible in-scope record is consumed exactly once each epoch, and there is no cross-history fold contamination.

For a fit scope with `N_scope > 0` eligible records, the exact optimizer-step count is `50 × ceil(N_scope / 512)`. Each diagnostic fold and the final fit has its own count and counter. The exact per-seed N4 scale-update total is

`50 × (ceil(N_fold1/512) + ceil(N_fold2/512) + ceil(N_fold3/512) + ceil(N_all/512))`,

with `N_foldh` the count from the two in-fold histories and `N_all` the count from all three histories. Report all four record counts, per-epoch step counts, final global step, fold order seed rule, and final order seed rule. Keep the inherited fixed 50 epochs, AdamW settings, clip norm, initialization rule, capacity gates, no-early-stopping rule, and no-Development/Confirmation-selection rule. If an existing support/finite gate is not met, apply the inherited invalid/unsupported disposition; do not fabricate records or a padded batch.

## Replacements for behavioral checks

Replace check 3 with:

> 3. Fit-history folds are disjoint and deterministic; for each fold, bounds and training records use only its two in-fold Fit histories, and the held-out history is used only for its diagnostic prediction.

Replace check 4 with:

> 4. Each fold's recorded bounds equal its in-fold-only `numpy.quantile(..., method="linear")` values; final bounds equal the same quantiles over all three Fit histories. Record complete canonical/shuffled record order, retained partial-batch counts, and exact optimizer-step counts per fold and final fit.