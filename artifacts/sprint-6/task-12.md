# Sprint 6 Task 12 — Push, pull, and rerun corrected notebooks

## Commit chain (no force/reset/clean; user work still unstaged)

- `a90adb4` server-native workflow + Kaggle removal (executed first run).
- `c9d726f` legacy-manifest fleet gate with recorded precheck (unblocked pre-fleet production data).
- `69d665e` extraction import restore (fixed first-run `NameError: Path`).
- `a326e48` Tasks 8–10 corrections (budgeted pairs, blocked distances, manifest validation, bundle deletion) — **executed corrected source commit**.
- Followed by an evidence/results commit below; identified as evidence-only (no source).

## Server rerun (PowerShell OpenSSH key auth, `uv run --no-sync`)

- Pre-pull blocker fixed without overwriting: the first run's untracked outputs collided with committed review files, so `server-geometry-run-1/` was renamed to `server-geometry-run-1-prev/` (preserved), then fast-forward pull to `a326e48` (verified via `git rev-parse`).
- Execution copies rebuilt from corrected canonicals with concrete paths (`SRC_DIR`, dataset, checkpoint, cache/analysis dirs); prep asserts each substitution exactly once, cells compile, outputs cleared, no `/kaggle` text.
- Extraction: exit 0, 12.4 s (16:08:56→16:09:08 +07), device `cuda`, 5000 records, manifest-linked cache with recorded `fleet_precheck`.
- Analysis: exit 0, 20.2 s (16:09:17→16:09:37 +07), 4/4 cells with outputs, zero errors; 12 figures, 75k neighbor rows, diagnostics bundle.
- Corrected metrics: same-class 4999 @ 0.4162, different-class 0, margin null; reference distance 3.5745183471514523 (matches prior run to 1 ulp — blocked summation reorder only); rank 127, eff-rank 17.84, anisotropy 26.48, not collapsed; `S_pop` 7.3446 ± 0.7053, `S_pred` 0.0735 ± 0.0306.

## Refreshed local set

- `experiments/20260905/server-geometry-run-1/` refreshed (executed notebooks, logs, manifests, analysis metrics, `pca.png`); large outputs stay server-side uncommitted. Verified executed notebooks: all code cells executed, zero error outputs.

## Limitations

- Same as task-7: all-normal head sample; single seed/device; torch version skew; legacy fleet skip recorded.
