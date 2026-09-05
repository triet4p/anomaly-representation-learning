# Sprint 9 Task 1 — Preserve the restored reference bank

## Root cause

The inference notebook unconditionally overwrote the restored checkpoint bank: both branches of `if bank.embeddings is None: ... else: ...` called `bank.fit(...)` on the 64-file reference batch, discarding the fitted 8192-row bank (Sprint 6 evidence: `normal train reference rows 64`). Extraction already used the restored bank as-is; only inference (and its contract test) assumed refit.

## Fix (default preserve + explicit opt-in + provenance)

- `src/representation/inference.py`: new `prepare_reference_bank(bank, reference_embeddings, *, refit=False) -> str` — fits only when the bank is empty or `refit=True`, returns `"restored"`/`"refit"` for output provenance. Exported from `representation` (+`__all__`).
- `notebooks/infer_v1_representation.ipynb`: `REFIT_REFERENCE_BANK` flag (`V1_REFIT_BANK`, default false) in the config cell; scoring cell calls the helper; final print records `bank source: restored|refit` with row count; markdown bullet documents the default. No LSP server exists in this environment (no LSP tool); callers traced by repo-wide search (`bank.fit(` now occurs only in the helper, training code, and tests).
- `src/representation/embedding_extraction.py`: extraction manifest `checkpoint` metadata gains `"reference_source": "restored-checkpoint"` (free-form dict; additive).
- `tests/representation/test_notebooks.py`: bank-fit contract updated to the helper call shape (semantics preserved: reference embeddings, no labels).

## Tests

- `tests/representation/test_inference.py`: `test_prepare_reference_bank_prefers_restored_over_refit` (default keeps restored embeddings, flag refits, empty bank fits) and `test_score_batch_never_mutates_the_scoring_bank` (score uses restored rows; S_pop == 1.0 on the crafted fixture).
- Focused: `test_inference.py` + `test_notebooks.py` + `test_extraction_finalization.py` + `test_geometry_contracts.py` → **19 passed**. No formatters/linters/project-wide suites.
