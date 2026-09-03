# Task 21 Summary — Explicit Contrastive Warmup

## Changed symbols

- `V1Config.contrastive_warmup_steps` is a validated non-negative integer with default `0`.
- `ProgressiveLambda` accepts `warmup_steps`, returns exactly `0.0` for every step `t <= warmup_steps`, starts its linear ramp at `warmup_steps + 1`, reaches `lambda_max` after `ramp_steps`, and stays monotonic thereafter. A zero-duration ramp becomes `lambda_max` immediately after warmup.
- `RepresentationTrainer` builds its default criterion from `V1Config`, propagating contrastive maximum weight, warmup, ramp, temperature, and prediction weight. Explicit caller-supplied criteria remain unchanged.
- Checkpoints already serialize `model.config.to_dict()`; the new field is therefore persisted and remains part of strict configuration compatibility checks. The checkpoint test verifies round-trip metadata.

## Default and compatibility behavior

The new warmup default is `0`, preserving the prior schedule's behavior at step `0` and its configured ramp endpoint. With the default, the first positive training step begins the ramp, while a configured positive warmup holds contrastive learning at zero through the requested boundary.

## Verification

Command:

```text
uv run pytest tests/representation/test_contrastive_criterion.py tests/representation/test_trainer.py tests/representation/test_checkpoint.py -q
```

Result: `12 passed` (7 existing PyTorch nested-tensor warnings).

The focused tests cover negative warmup validation, the steps immediately before/at/after warmup, ramp completion and monotonicity, trainer λ logging across warmup boundaries, and checkpoint metadata preservation.
