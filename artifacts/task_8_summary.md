# Task 8 Summary — V1 Representation Model

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 8 marked `[x]`; Task 9 marked `[~]`).

## Changed targets

- `src/representation/model.py`: added `V1RepresentationModel`, composing patch encoder, variable-length context encoder, EMA target branch, masked latent predictor, valid-mask file pooling, and same-file contrastive view embeddings via `make_contrastive_views()`. The target branch is stop-gradient and no raw reconstruction head is exposed.
- `src/representation/__init__.py`: exported `V1RepresentationModel`.
- `tests/representation/test_model.py`: added focused forward coverage for homogeneous C=3 and C=6 batches with mixed file lengths, output shapes, target stop-gradient, valid-only pooling, prediction-mask alignment, contrastive views, reconstruction-head absence, and label independence.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_model.py -q
```

Result: `3 passed in 17.67s` (PyTorch emitted the existing nested-tensor optimization warning from the pre-norm sequence encoder).

The focused suite exercised both supported channel layouts in separate homogeneous minibatches, variable file lengths, target stop-gradient, mask-aware file pooling, same-file contrastive views, and metadata non-consumption. No project-wide tests, formatter, linter, or build was run.

## Deviations / risks

- Mixed C=3/C=6 tensors remain intentionally rejected by the batch contract; both channel modes are tested through separate homogeneous minibatches.
- The model computes optional contrastive view embeddings when `file_samples` metadata is available; a manually assembled tensor-only batch returns the required core outputs without views.
- EMA updates remain an explicit trainer responsibility after the context optimizer step.
