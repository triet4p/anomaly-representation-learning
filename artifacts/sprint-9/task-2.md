# Sprint 9 Task 2 — Diagnose and repair timestep localization

## Diagnosis (from executed outputs + probes, no rerun)

- Measured degenerate shape (`infer-v1-representation.executed.ipynb`, `cell4` stream outputs): val first trace len 736 with 320 exact zeros + 416 nonzero over only 21 unique values; test first trace len 382 with 128 zeros + 254 nonzero over 11 unique values.
- Explained component (by design): unmasked patches contribute exact `0.0` via `errors.masked_fill(~prediction_mask, 0.0)` in `RepresentationInference.score_batch` — traces are sparse by construction since only masked target patches are scored.
- Unexplained-remainder bound: the long bit-identical runs require many adjacent patches to share bit-identical MSE, i.e. constant predictions *and* targets across those patches (saturation on the normal data) and/or a mask/patch misalignment in the run's source tree. The exact vintage cannot be recovered — the gate `src.zip`/`src/` trees were deleted in Sprint 6 and no src hash is recorded in the outputs — so no single historical line is claimed. Mask placement itself was exonerated on current code (seeds 6/7 scatter correctly with exact 0.40/0.39 ratios and 6/6/6 composition).

## Fix at the source boundary

- `src/synth/patchify.py::patch_to_timestep_scores`: fail-fast coverage validation — 1-D equal lengths, non-negative starts/valid lengths, every covered patch starts inside `[0, T)` (padding slots `-1/0` still contribute nothing, behavior unchanged). The misalignment class that silently shifts localization now raises `ValueError` naming the violated invariant.
- Verified the live path is sound: a synthetic 8-step spike in a 40-step file through real `collate → model → score_batch` localizes (peak inside 16:24, background exactly 0, deterministic across calls).

## Tests

- `tests/synth/test_patchify.py::test_timestep_aggregation_rejects_misaligned_coverage`: mismatched lengths, negative lengths, out-of-range and negative-covered starts all raise; padding slots contribute nothing.
- `tests/representation/test_inference.py::test_timestep_scores_localize_synthetic_spike_deterministically`: spike trace length 40, rerun-identical, argmax inside the spike window, zero background, positive window mean.
- Focused: `test_inference.py` (7) + `test_patchify.py` (12, incl. pre-existing aggregation round-trip) → all pass. Rerun-trace verification lands in Task 3/4 outputs (printed per-split timestep arrays).
