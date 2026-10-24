# Sprint 18 N3-CCRP Specification v2 Amendment

**Specification ID:** `sprint18-n3-ccrp-spec-v2`  
**Status:** PROSPECTIVE AMENDMENT FOR INDEPENDENT REVIEW — not execution-authorized.  
**Arm:** `N3-CCRP`  
**Base specification:** `experiments/sprint18-cross-channel-relational-prediction-spec.md`, SHA-256 `1c0f570606b665f0976bc8161f0d1dd3b2cc2d482b0f861b5dbc2975bd4c7cfd`.  
**Base protocol:** `experiments/sprint18-full-training-protocol-v1.md`, SHA-256 `9ae3d43f94a6194bae602d68067958bdf7d11cf47e801dd328e87b22e7907cf6`.

This version replaces only v1 §2's channel assignment and v1 §4's patch sampling, plus the corresponding deterministic-selection behavioral checks. All other N3 clauses are inherited unchanged from the exact base-specification digest above. The v2 protocol amendment defines this file as the sole N3 specification for a v2 contract; do not combine it with a v1 N3 specification.

## Replacement for §2 — Deterministic channel assignment

Within each 1-based epoch, construct the final eligible relational target-key set after the sampling/resampling rule in §4 below. Sort those keys by `(canonical_file_id lexical ascending, original B0 patch_index ascending)` across the entire Fit epoch, and assign a zero-based global ordinal starting at zero. With `C=6`, assign exactly one held-out channel by

`c = (ordinal + model_seed + 1-based_epoch) mod 6`.

The ordinal does not reset at a file, batch, or optimizer step. Record the epoch's final target count and counts for all six channels; the round-robin counts must differ by at most one. The corrupted-input construction, channel-ID embedding, and target behavior in v1 §2–§3 are unchanged.

## Replacement for §4 — Exact patch sampling and resampling

For each Fit file/epoch, let `V` be the valid non-padding patch indices sorted by original B0 patch index and `n=|V|`. The requested count is `Q=0` when `n=0`; otherwise `Q=min(n, max(1, int(round(0.25 × n))))`, using Python's built-in ties-to-even `round`.

Derive the file/epoch selection seed as

`int.from_bytes(SHA256(UTF8("sprint18-n3-patch-v2|<model_seed>|<1-based-epoch>|<fit_role_sha256>|<canonical_file_id>"))[:8], "big") mod (2^63−1)`.

Initialize one `numpy.random.default_rng(seed)` and compute one permutation of `V`. The first `Q` positions are the initial relational sample, independent of the temporal mask. Remove any initial positions whose entire temporal input is already masked. For each removed position, in initial-sample order, replace it with the next position in the same permutation after the initial `Q` that is not fully temporally masked; scan each candidate at most once. Stop when `Q` unmasked positions have been retained or the permutation is exhausted. The final count is therefore `min(Q, number_of_valid_not_fully_temporally_masked_positions)`; if it is zero, record the file as unsupported for that batch. No replacement may duplicate a retained key or include a fully masked patch.

Apply §2's global ordinal/channel formula only after all per-file resampling is complete. Record `n`, `Q`, initial selections rejected as fully masked, replacements, final target count, unsupported files, and final channel counts per epoch. The relational target selection remains independent of anomaly labels and outcome information.

## Replacements for behavioral checks

Replace check 1 with:

> 1. Each final relational target has exactly one held-out channel; the selected-key order, zero-based epoch-global ordinal, and channel formula reproduce the recorded per-channel counts, which differ by at most one.

Replace the sampling/resampling clause in the existing unsupported-case check with:

> For fixed seed, epoch, file, and temporal-mask realization, the initial permutation, rejected positions, ordered replacements, and final set are exactly reproducible; final target count equals `min(Q, unmasked valid count)`, and no fully temporally masked patch enters the loss denominator.