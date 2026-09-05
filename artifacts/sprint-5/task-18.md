# Sprint 5 Task 18 — Mixed-device `index_add` root fix

## User-reported failure (ground truth, not rerun)

`RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0 (wrapper_CUDA_index_add)` in `geometry_extraction.ipynb` on Kaggle GPU.

## Root cause (traced notebook → source, exact symbol)

Failing symbol: `torch.index_add` in `ConditionalBatchNorm._update_statistics` (`src/representation/layers/normalization.py`), reached via `extract_embeddings` → `BoundedEmbeddingExtractor.extract` → `_process_batch` → `V1RepresentationModel.forward` → `conditional_norm(...)` (the first device-sensitive op in `forward`).

Contributing chain (all three required):

1. `load_checkpoint` never changes train/eval mode, and `extract()` never called `model.eval()` — the freshly built model stayed in **training mode**, so `forward` entered `_update_statistics`.
2. `extract()` did `self.model.to('cuda')` (params **and** registered `_BucketStatistics` buffers move to CUDA), but `_process_batch` fed a **CPU** batch: it calls `inference.model(unmasked_batch)` directly, bypassing `RepresentationInference._move_batch` (only `score_batch` moves batches).
3. `_update_statistics` assumed co-located devices: `torch.index_add(stats.value_sum [cuda], 0, buckets [cpu], per_sample_sum [cpu])` → the reported error (CUDA accumulator, CPU index). CPU-only runs never trigger it, which is why local runs passed.

No LSP server is available in this environment (no LSP tool in inventory); callers were traced with repo-wide reference search instead.

## Fix (source invariant, no avoidable copies)

- `src/representation/layers/normalization.py::_update_statistics`: each level now co-locates the index and all three sources onto the accumulator device (`buckets.to(device)`, `per_sample_*.to(device)`). `.to()` is a no-op returning self when already aligned, so single-device train/infer pays zero copies; only the small per-batch tensors copy on the mixed path.
- Same file, `forward` selection/aggregation reads (the adjacent reductions): `stats.sample_count[buckets]` gather and the `value_count`/`value_sum`/`value_square_sum` gathers now move the small index to the stats device and the gathered means/variances to the input device/dtype (`mean.to(device=means.device, dtype=means.dtype)`), again no-ops when aligned.
- `src/representation/embedding_extraction.py::extract`: added `self.model.eval()` after `self.model.to(device)` — diagnostics must be deterministic and must not mutate fleet statistics (`score_batch` already re-asserts eval per call).
- Same file, `_process_batch`: the unmasked batch now goes through `inference._move_batch` before the direct model call, mirroring `score_batch`.
- No Kaggle special-casing: the fix is device-general, not input-specific.

## Neighbor audit (extraction-reached aggregation/pooling/reduction paths)

- `NormalReferenceBank.score`: already co-locates (queries → CPU for `cdist`, result back to query device) — unchanged.
- `V1RepresentationModel._pool_file`: weights derive from `latents` (`valid.to(latents.dtype)`) — same device by construction — unchanged.
- `score_batch` timestep block: explicit `.cpu()` conversions — unchanged.
- `baselines/rvq.py`: not referenced by `V1RepresentationModel`, trainer, or extraction — out of reach, unchanged.
- `criterion.py` (`arange(..., device=...)`), predictor/sequence-encoder `_positions`, `model.py` patch staging (explicit `device=`): already device-explicit — unchanged.

## Regression coverage (focused only; no real inference/notebook execution)

New `tests/representation/test_device_placement.py` (3 tests + 1 CUDA test):

- `test_update_statistics_accumulates_exact_counts_and_sums` — CPU numerics: exact per-bucket counts/masked sums.
- `test_eval_forward_is_deterministic_and_freezes_statistics` — CPU: bit-identical eval forwards, accumulators frozen.
- `test_cuda_model_with_cpu_batch_matches_colocated_numerics` — exact Kaggle replication (CUDA train-mode model, CPU batch) with numerical parity vs an all-CPU twin; **skipped here** (`torch.cuda.is_available()` is `False` on this CPU-only torch 2.10.0 host).
- Honesty note: a meta-device variant was tried for deterministic mixed-device coverage but discarded — `torch.index_add` does not enforce device checks against `meta`, so it passed on pre-fix code (verified via temporary revert) and proved nothing; copying meta→CPU is impossible, so no deterministic mixed-device test exists without a second data-bearing device.

Results: `test_device_placement.py` + `test_normalization.py` → **7 passed, 1 skipped (CUDA)** (2 new CPU tests pass; all 5 existing normalization tests pass unmodified with the fix). No formatters/linters/project-wide suites; no notebook execution.
## Packaging

- Changed source synced into `exports/kaggle-20260905-01/src/` (`layers/normalization.py`, `embedding_extraction.py`); full repo↔export `src/*.py` parity re-verified (52 files).
## Final packaging (via Task 19 regen)

- `src.zip`: 108092 bytes, SHA-256 `db3d7abffc4e142760dd344c1170095432b5ebb55f2d1fb72cb4098161b13510`.
- Outer `kaggle-20260905-01.zip`: 245395 bytes, SHA-256 `d8ab83f490eba9db08973fb9b255c58750e1d6663919c3e5484b6d03718d3795`.

## Changed files

- `src/representation/layers/normalization.py` (reductions co-located)
- `src/representation/embedding_extraction.py` (`eval()` + batch device move)
- `tests/representation/test_device_placement.py` (new)
- `exports/kaggle-20260905-01/src/{representation/layers/normalization.py,representation/embedding_extraction.py}`, `src.zip` (+ manifest/outer ZIP via Task 19)
