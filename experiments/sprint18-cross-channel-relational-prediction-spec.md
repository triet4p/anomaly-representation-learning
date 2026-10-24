# Sprint 18 N3-CCRP — Cross-Channel Relational Prediction Specification

**Status:** DRAFT — freeze with the Sprint 18 protocol before execution.
**Arm:** `N3-CCRP`
**Base:** `B0-FULL`
**Principal change:** Add a self-supervised auxiliary task that predicts the held-out channel's EMA latent contribution from the other channels and temporal context.

## 1. Hypothesis

A healthy file may be identifiable by stable cross-channel coupling, lag, and relative response even when each individual channel remains within a plausible marginal range. B0 can represent such relations but its objective does not explicitly require them. N3-CCRP tests that requirement without reconstructing raw samples.

No physical channel grouping is assumed. All six channels use one shared mechanism and symmetric sampling.

## 2. Two-path training forward

Each training batch has:

1. the unchanged B0 forward for fine EMA prediction and file-level InfoNCE;
2. one auxiliary relational forward over the same files.

For each epoch, list selected relational patch keys in the common deterministic file/patch order. Assign held-out channel `c = (ordinal + model_seed + epoch) mod C`; this round-robin rule makes channel counts differ by at most one while remaining deterministic. Record counts by channel.

Construct the student auxiliary input by replacing the selected channel samples in that patch with zero **after the common input normalization**. Do not alter the other five channels. The held-out channel ID is supplied only to the relational predictor through a learned channel-ID embedding; it is not injected into the B0 encoder.

The B0 local and context encoders process this corrupted auxiliary view. The relational predictor consumes:

- the contextual latent at the selected patch;
- the learned channel-ID embedding;
- valid length and padding mask through the existing mask-aware path only.

## 3. Held-out channel contribution target

Do not add an independently drifting channel encoder. Reuse the common EMA local patch encoder so the target network remains trained by B0's main online/EMA path.

For the original normalized patch `x` and selected channel `c`, compute with the stop-gradient EMA local encoder:

`z_full = LayerNorm(local_target(x))`

`z_without_c = LayerNorm(local_target(zero_channel(x, c)))`

The held-out channel contribution target is:

`delta_z_c = normalize(z_full - z_without_c)`.

If `||z_full - z_without_c||_2 < 1e-6`, the position is explicitly unsupported for the relational loss and excluded from its denominator.

The student predictor maps `[context_latent_from_zero_channel_view; channel_embedding_c]` to `d_model`:

`delta_z_hat_c = normalize(predictor(context_latent, channel_id))`.

Use cosine distance `1 - dot(delta_z_hat_c, delta_z_c)` over supported selected patches. No decoder predicts waveform values. The target path is EMA and stop-gradient; the student cannot access `z_full`, `z_without_c`, their difference, or original samples from the held-out channel patch.

## 4. Patch sampling and loss

Sample relational targets from 25% of valid patch positions per file, with at least one when the file has valid patches. Relational selection is independent of B0's temporal masking; if a selected patch is already fully temporally masked, resample deterministically from remaining valid positions. If none remains, the file contributes no relational target and is recorded as unsupported for that batch.

`L_total = L_pred + beta × L_channel + lambda(step) × L_InfoNCE`.

- Select `beta` from `[0.10, 0.25, 0.50]` using the common Design-only gradient-balance rule.
- Use the InfoNCE coefficient selected once on `B0-FULL`; do not retune it for N3.
- The B0 forward, mask ratio, projection head, Conditional BatchNorm, EMA patch target, and primary scorers remain unchanged.
- The auxiliary forward may increase compute but cannot change the B0 forward's inputs or score definition.

## 5. Inference behavior

Primary `S_pred` and `S_pop` use the ordinary uncorrupted B0 inference path. Cross-channel residuals are emitted as post-hoc diagnostics by channel and patch but are not fused into either primary score during Sprint 18.

This arm tests whether auxiliary relational pressure improves the shared representation. It does not simultaneously test a new anomaly scorer.

## 6. Required behavioral checks

1. Exactly one channel is held out per selected relational patch.
2. Zero replacement occurs after normalization and only in the auxiliary forward.
3. The main B0 forward remains byte-equivalent when given the same seed and parameters.
4. Channel selection is deterministic, balanced, and independent of anomaly labels.
5. Both target terms use the common EMA local encoder; no untrained auxiliary target encoder exists.
6. Target contribution latents are stop-gradient and near-zero targets are excluded explicitly.
7. The student path cannot access original samples or target latents from the held-out channel patch.
8. Relational targets never overlap a fully temporal-masked patch.
9. `beta=0` removes auxiliary gradients and reproduces B0 optimization/scoring semantics within declared tolerance.
10. No raw-waveform reconstruction tensor or loss is created.
11. Unsupported short/padded cases are excluded with explicit counts, not encoded as zero loss.
12. Additional parameters and double-forward FLOPs are measured and checked against the frozen Sprint 18 envelope.

## 7. Required diagnostics

Report by seed, history, and channel:

- target counts and unsupported counts;
- raw/weighted channel loss and gradient ratio;
- contribution-target variance, near-zero-target exclusions, and predictor effective rank;
- cross-channel contribution residual distributions on healthy background and P/W/A post-hoc slices;
- primary score changes by subtype, especially relation/lag-sensitive anomalies where such labels already exist for diagnostics;
- runtime, memory, and FLOP overhead.

A gain limited to one channel must be reported rather than averaged away. A lower auxiliary loss alone is not evidence of improved anomaly detection.

## 8. Forbidden shortcuts

- Do not use fixed physical channel groups not declared here.
- Do not expose the original held-out channel to the student through another view or cached feature.
- Do not change primary inference to require channel corruption.
- Do not fuse channel residuals after viewing Development.
- Do not tune `beta` or channel sampling with anomaly outcomes.
- Do not interpret predictability between channels as causality.
