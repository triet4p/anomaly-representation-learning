# Sprint 18 N4-CRN — Context-Dependent Residual Normalization Specification

**Status:** DRAFT — freeze with the Sprint 18 protocol before execution.
**Arm:** `N4-CRN`
**Base:** frozen final `B0-FULL` checkpoints and prediction caches
**Principal change:** Normalize each fine prediction residual by a Fit-healthy context-dependent expected residual scale.

## 1. Hypothesis

Healthy contexts differ in prediction difficulty. A fixed global residual scale can treat naturally variable transitions as anomalous and stable contexts as ordinary. N4-CRN tests whether a context-only estimate of healthy residual variance improves `S_pred` without modifying the representation.

This differs from Sprint 17 C8-A: C8-A applied a Fit-derived robust global standardization. N4 predicts a scale for each patch from its permitted context.

## 2. Frozen representation boundary

- Reuse the exact final `B0-FULL` online model, EMA target, patch mapping, and cached Fit/Calibration/Development/Confirmation prediction pairs.
- Execute zero new representation optimizer steps.
- Detach every feature and residual before fitting the scale model.
- Leave the reference bank and `S_pop` bitwise identical to `B0-FULL` on common cached inputs.

The scale model cannot update the encoder, context model, predictor, EMA target, or bank.

## 3. Scale-model inputs and target

For masked patch `t`, define the B0 residual target:

`e_t = mean_d((normalize(z_target_t) - normalize(z_pred_t))^2)`.

The scale model receives only detached online context `h_t` available to the B0 predictor before observing `z_target_t`. It cannot receive:

- target latent or any function of it;
- residual `e_t` as an input;
- waveform samples from the masked target;
- anomaly labels, cohort, subtype, severity, failure time, or simulator state;
- Development or Confirmation statistics.

The fixed model is:

`log_v_t = clamp(Linear_2(GELU(Linear_1(LayerNorm(h_t)))), log(v_floor), log(v_cap))`,

with hidden width 32 and scalar output. Use no metadata embedding: the context has already passed through B0 Conditional BatchNorm.

Fit-only bounds are:

- `v_floor = max(quantile_0.05(e_Fit), 1e-6)`;
- `v_cap = max(quantile_0.95(e_Fit), 10 × v_floor)`.

Bounds are computed once from eligible Fit healthy residuals per model seed and recorded.

## 4. Fitting contract

Fit the scale model with the B0 representation frozen using Gaussian residual NLL:

`L_scale = mean_t[0.5 × (e_t / exp(log_v_t) + log_v_t)]`.

Fixed optimizer contract:

| Setting | Value |
|---|---:|
| optimizer | AdamW |
| learning rate | `1e-3` |
| weight decay | `1e-4` |
| epochs | 50 |
| batch size | 512 patch records |
| gradient clip | `1.0` |
| checkpoint | final epoch 50 |

No early stopping or Development-based model selection.

Use three-way Fit-history cross-fitting for diagnostics: fit on two Fit histories and predict the held-out Fit history, rotating all three. After capacity and finite-output checks pass, fit one final scale model on all Fit healthy records under the same fixed contract for Calibration/Development/Confirmation scoring. Cross-fit results do not select architecture or hyperparameters.

## 5. Score

The normalized patch score is:

`S_pred_patch_CRN = e_t / (exp(log_v_t) + 1e-6)`.

Use B0's unchanged valid-patch filtering, timestep localization, and patch-to-file aggregation. Fit no threshold on Fit. Set the arm's operating threshold once as q95 of eligible Calibration healthy file scores, exactly as in the common protocol.

Do not combine raw B0 residual and normalized residual. `N4-CRN` has one declared `S_pred`; raw B0 `S_pred` remains a diagnostic comparator.

## 6. Anti-explain-away safeguards

1. Representation and predictor are frozen.
2. Target latent and residual are excluded from model inputs.
3. `log_v` is bounded by Fit-only residual quantiles.
4. Report the fraction of patches at each bound.
5. Report Spearman correlation between predicted scale and residual on cross-fit healthy records.
6. Report anomaly residual and scale separately post-hoc; a lower normalized anomaly score caused solely by inflated predicted scale must remain visible.
7. Fail the arm if more than 20% of eligible Development healthy patches saturate `v_cap`; do not raise the cap after outcomes.

The saturation gate is a structural scoring-validity gate, not a performance gate.

## 7. Required behavioral checks

1. Encoder, predictor, EMA target, bank, and raw residual caches are unchanged from `B0-FULL`.
2. Scale-model inputs exclude target/residual/waveform and prohibited labels by API construction.
3. Fit-history folds are disjoint and deterministic.
4. Bounds use Fit healthy residuals only and match recorded quantiles.
5. Calibration sets q95 once; Development and Confirmation never refit the scale model or threshold.
6. `S_pop` is bitwise equal to B0 on identical inputs.
7. Patch starts, valid lengths, file IDs, and localization map are preserved.
8. All scale outputs are finite and inside the frozen bounds.
9. A constant predicted scale produces a monotonic rescaling of B0 patch residuals in a synthetic check.
10. Checkpoint and cache hashes bind the scale model to the exact B0 seed state.

## 8. Required diagnostics

Report per seed/history:

- Fit cross-fold NLL, scale–residual correlation, and saturation;
- scale distributions by robot/program support tier;
- raw residual, predicted scale, and normalized residual separately;
- background stability and FAR changes;
- P/W/A and subtype changes;
- localization and duration accounting;
- exact `S_pop` parity with B0.

Improved background stability with materially reduced P/W recall is not recovery. Calibration alone cannot be described as better representation learning.

## 9. Forbidden shortcuts

- Do not jointly train the scale head with B0 in Sprint 18.
- Do not feed the target latent or residual into the scale head.
- Do not tune bounds, architecture, or epochs on Development/Confirmation.
- Do not fuse normalized and raw scores.
- Do not allow predicted variance to be unbounded.
- Do not claim N4 improves the representation; it is a scoring-calibration mechanism.
