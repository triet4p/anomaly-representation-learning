# Task 18 Summary — Controllable Batched Contrastive View Generation

## Status

Complete. This remediation addresses static augmentation, device placement, and serialized per-sample view encoding in `V1RepresentationModel._view_embeddings`.

## Changed targets

- `src/representation/model.py`: replaced per-forward RNG recreation with a persistent NumPy generator seeded from `V1Config.seed`; added `get_extra_state`, `set_extra_state`, and `reset_view_rng` controls so augmentation state advances and serializes with model state. Both augmented views are now collated to the batch maximum patch count and encoded in batched calls. View patch/padding/valid tensors are allocated on the input patches' device and dtype, with NumPy conversion explicitly transferred to that device.
- `tests/representation/test_model.py`: added tests for changing training views, deterministic fresh seeded models, RNG state-dict round-trip, batched view shape behavior, and conditional CUDA device/dtype correctness.
- `docs/sprint-plans/sprint-2.md`: appended completed Atomic Task 18.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_model.py -q
```

Result: `5 passed, 1 skipped in 8.62s`. The skipped test is the conditional CUDA check because CUDA is unavailable. Five existing PyTorch nested-tensor warnings were emitted by the pre-norm sequence encoder.

Coverage verifies changing training views, reproducibility for identically seeded fresh models, serialized RNG continuation, homogeneous batched view output, and model/device dtype behavior when CUDA is available.

## Residual risks

- CUDA/MPS execution is structurally supported by explicit `.to(device, dtype)` transfers but was not hardware-exercised because CUDA is unavailable in this environment.
- View augmentation uses one persistent generator for the model instance; callers should use `reset_view_rng(seed)` when intentionally restarting a deterministic augmentation stream.
- NumPy augmentation remains CPU-side by design; only resulting tensors are transferred before neural encoding.
