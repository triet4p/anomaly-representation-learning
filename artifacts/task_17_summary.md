# Task 17 Summary — Masked Predictor Context Remediation

## Status

Complete. This remediation closes the semantic defect identified in review: masked predictions now use cross-attention to visible local context rather than only a global average and position-wise MLP.

## Changed targets

- `src/representation/layers/predictor.py`: replaced masked-position point-wise prediction with multi-head cross-attention. Mask/position queries attend only to visible, valid contextual patch states; all-masked rows use a safe sentinel solely to avoid attention NaNs; visible/invalid outputs remain zero and aligned.
- `tests/representation/test_predictor.py`: added a behavioral test showing a masked query responds to a nearby visible-context perturbation while remaining invariant to padded/invalid-token perturbations.
- `docs/sprint-plans/sprint-2.md`: appended and completed Atomic Task 17.
- `docs/PLAN.md`: records Sprint 2 complete including the remediation.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_predictor.py tests/representation/test_model.py -q
```

Result: `7 passed in 4.75s` with 3 existing nested-tensor warnings from the pre-norm sequence encoder.

The focused tests verified target dimensions/alignment, mask intersection, visible/invalid exclusion, context-only gradients, and the new local-context response/invariance behavior. Model composition for both C=3/C=6 modes also passed. No broad suite, formatter, linter, or build was run after this remediation.

## Residual risks

- The predictor attends to every visible valid patch, with positional query/key features providing local sequence geometry; an explicit relative-distance bias is not introduced.
- Existing final-gate evidence predates this predictor remediation; the focused predictor/model suite above was rerun after the fix. A future owner may rerun the full final gate if required.
