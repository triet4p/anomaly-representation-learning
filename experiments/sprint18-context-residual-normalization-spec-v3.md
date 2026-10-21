# Sprint 18 N4-CRN Specification v3 Amendment

**Specification ID:** `sprint18-n4-crn-spec-v3`
**Status:** PROSPECTIVE AMENDMENT FOR INDEPENDENT REVIEW — not execution-authorized, not certified.
**Arm:** `N4-CRN`
**Base specification:** `experiments/sprint18-context-residual-normalization-spec.md`, SHA-256 `fd0e1722ea4f248a354086bbdf44bc486d741a9cfb018f68ef08277827d7954d`.
**v2 amendment (historical, immutable):** `experiments/sprint18-context-residual-normalization-spec-v2.md`, SHA-256 `4d1c8097c9cc27d3fbeb2e2a6e70a29a7bc63d7899646a9c712a315b39d9ffe3`.
**Base protocol:** `experiments/sprint18-full-training-protocol-v2.md`, SHA-256 `e1dcd17914e920520ad0d8f50e185874b95414e6a4fe907903619daf145622d2`.

This version replaces only the initialization rule and behavioral check 11 of the base v1 specification as inherited through the v2 amendment, and adds the initialization/guard provenance required by the recovered v1 failure. All other v1 and v2 clauses are inherited unchanged: residual definition, context-only inputs, Fit-only quantile bounds, the hard clamp, Gaussian residual NLL, AdamW `1e-3`/`1e-4`, 50 epochs, batch at most 512 records, global clip `1.0`, the frozen `sprint18-n4-scale-order-v2` order rule, exact `50 × ceil(N_scope/512)` step accounting, saturation diagnostics, the 20% `v_cap` gate, Fit-only roles, and the no-Development/Confirmation-selection rule.

The v1 and v2 specification files remain byte-for-byte historical records and are not reinterpreted. This v3 file is the operative N4 mechanism specification only for a prospective v3 execution. A normative v3 run additionally requires a new Task14 lock version that binds this spec digest; until that lock version exists, no normative v3 capacity result may be claimed.

## Replacement for the inherited initialization rule

> **Initialization (v3).** For each fit scope, after the frozen seeded construction (`init_n4_scale_model`, forked CPU RNG, no caller-visible RNG side effect) and before the first optimizer step, set the output-layer bias to `b2 = log(v_ref)` where `v_ref = (1/N_S) Σ_{i∈S} e_i` is computed in float64 over the exact Fit-only in-scope training records (the two in-fold histories for a diagnostic fold; all three Fit histories for the final model). Only `out.bias` is written; `out.weight`, `hidden.weight`, `hidden.bias`, `norm.weight` and `norm.bias` retain their seeded values, and the last layer is never zeroed. Reject a non-finite or non-positive `v_ref`. The anchor is the initialization of the scalar offset, not a claim that every predicted `log_v` equals `log(v_ref)`.

The frozen forward remains exactly
`log_v = clamp(W₂·GELU(W₁·LayerNorm(h) + b₁) + b₂, log(v_floor), log(v_cap))`,
including the hard clamp, so a hard-bound saturation diagnostic remains meaningful.

## Replacement for behavioral check 11 — fail-closed startup guards

Guard contract identity: `n4-v3-guards-v1`.

### Guard A — initialized interiority (forward-only, before the optimizer exists)

Inputs: the anchored scale model, the canonical in-scope `contexts` already materialized for training (exact canonical `(history_id, file_id, patch_index)` order), `v_floor`, `v_cap`.

Structural preconditions (fail closed): exactly 4,417 parameters (the frozen deployed head at latent width 128) and no more than the 100,000-parameter envelope; zero registered buffers; no Dropout/BatchNorm-family layer; `log(v_cap) > log(v_floor)`.

Computation: under `torch.no_grad()`, iterate `contexts` in contiguous chunks of at most 512, call the unchanged clamped forward, and accumulate the returned `log_v` count, minimum, maximum, and the frozen saturation counts with `log_v <= log(v_floor) + 1e-6` and `log_v >= log(v_cap) - 1e-6`.

Predicate (one shot, no retry): `frac_at_floor == 0` and `frac_at_cap == 0` over the entire in-scope pool, i.e. every initialized record is strictly interior by more than `1e-6` log units. There is no minimum-margin threshold and no outcome-dependent tolerance.

State protection: Guard A runs before `torch.optim.AdamW` is constructed. It reads parameters only, writes no parameter/buffer/`.grad`/mode/optimizer/checkpoint state, draws no RNG, and does not call the sampler. The extra forward is the only added cost and is bounded to one chunk of at most 512 records. It cannot be replaced by the first batch, which covers at most 512 of up to 66,989 records, and it is not a speculative gradient probe.

Failure: raise `N4ScaleGuardError(guard_id="n4-scale-init-interiority", diagnostics)` with zero optimizer updates applied.

### Guard B — first normal backward (read-only, before the unchanged clip/step)

Placement: immediately after the sole normal `loss.backward()` for the batch at `global_step == 0`, and before the unchanged `clip_grad_norm_(..., 1.0)` and `optimizer.step()`. It is a no-op branch on every later step.

Predicates: every named parameter has a non-`None` gradient; every gradient element is finite; the total number of exactly nonzero gradient elements across all parameters is at least one. Individual zero coordinates and whole zero tensors are legitimate and are recorded as diagnostics only: per-record clamp zeros, GELU dead units, and unused context coordinates are expected. There is no per-tensor, derivative-fraction, or magnitude threshold.

Forbidden in the check: any in-place gradient operation, `optimizer.zero_grad`, a second `backward`, `torch.autograd.grad`, hooks, loss rescaling, or re-clipping. The check is read-only and the frozen clip is not a finiteness guard: a pre-clip check is the only behavior-preserving location.

Failure: raise `N4ScaleGuardError(guard_id="n4-scale-first-step-gradients", diagnostics)`; because the raise precedes clipping and stepping, no clipping, weight decay, parameter change, AdamW state creation, or RNG draw occurs.

### Prohibitions

Both guards and their calling fit/runner MUST NOT: run a speculative extra forward/backward for gradients, accumulate probe gradients, or add an extra `zero_grad`; change the loss, loss scaling, clipping, bounds, learning rate, weight decay, epoch count, batch size, order-seed rule, saturation definition, or gate; draw or reseed RNG; call the sampler; change model mode or write parameters/buffers; read or write optimizer or checkpoint state; retry, reinitialize, reseed, resample, fall back to another parameterization, adjust bounds, catch and continue, or reuse an aborted model after a guard failure. The guards are unconditional.

## Diagnostics and provenance

Recorded JSON-serializably in every scope result and in the scale checkpoint envelope:

- `scale_parameterization = "fit-mean-anchored-bias-v1"`, `guard_contract = "n4-v3-guards-v1"`.
- `init_anchor`: `{v_ref, log_v_ref, record_count}` computed in float64.
- `init_interiority`: `{record_count, log_v_min, log_v_max, log_floor, log_cap, floor_margin, cap_margin, frac_at_floor, frac_at_cap, tolerance}`.
- `first_step_gradients`: `{total_nonzero, total_nonfinite, missing_grad_parameters, tensors: [{name, numel, nonzero, zero, nonfinite, norm}]}`.

On failure the runner records `scale_guard_failure = {guard_id, message, diagnostics}` and terminates; `N4ScaleGuardError` carries `.guard_id` and `.diagnostics` so a guard failure is distinguishable from a schema or validation error. A failed scope applies zero optimizer updates and writes no checkpoint.

## Scale-checkpoint and binding cutover

- New scale checkpoints are written as a versioned envelope `{"schema": "s18-n4-scale-checkpoint-v2", "parameterization": "fit-mean-anchored-bias-v1", "guard_contract": "n4-v3-guards-v1", "fit_scope": "fold:<heldout_history_id>" | "final", "training_history_ids": [...], "v_floor": ..., "v_cap": ..., "initialization": {...}, "state_dict_sha256": ..., "state_dict": {...}}`.
- The strict scale-checkpoint loader accepts only that schema, parameterization, and guard contract, recomputes the state digest, and refuses a bare legacy `state_dict` (the rejected v1 form), unknown parameterization/guard ids, stale scope/history/bound bindings, and unsafe bytes. This refusal is specific to the N4 scale checkpoint and is not the generic Sprint18 coherent-checkpoint path.
- The coherent Sprint18 N4 checkpoint (`arm_specific_state`) records the same `scale_parameterization`, `guard_contract`, and initialization diagnostics; a coherent N4 checkpoint lacking them is refused as a legacy v1 dead-zone checkpoint.
- The N4 cache-binding digest includes `scale_parameterization` and fails closed on a missing or unknown value. Old v1 A03 digests remain historical evidence and are not valid v3 bindings.
- The nine A03 scale checkpoints and their `results.json`/`capacity.json` entries are reclassified `rejected_failed_dead_zone_v1`. They are never loaded or consumed by the v3 path, and their bytes are not edited.
- The run identity/schemas are v3. The authorized source commit and runner digest remain external trusted-CLI/external-binding values; no future commit is frozen in source.

## Provenance of this amendment

Insertion-only repair selected by the accepted RCA and design record: the deterministic initialized pre-clamp output lay above the hard cap on every A03 fold, so the loss gradient was exactly zero and AdamW decoupled decay was the only parameter motion. For a constant scale, `L(v) = 0.5·(mean(e)/v + log v)` has its exact optimum at `v = mean(e)`, so `log(mean(e))` is the structural, Fit-only interior anchor. The guards convert any residual initialization dead zone or missing learning signal into a fail-closed abort with zero updates instead of a silent decay-only fit.

Historical v1/v2 records, the original failed normative evidence, and the A01 diagnostic overrun disclosure remain immutable. This amendment makes no claim that the clamp dead zone is globally eliminated, no CPU/CUDA equivalence claim, and no capacity/certification claim.
