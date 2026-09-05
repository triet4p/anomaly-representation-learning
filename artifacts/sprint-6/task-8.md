# Sprint 6 Task 8 — Bound and balance geometry analysis

## Pair budget (same/different-class starvation)

- Root cause: `compute_separation_metrics` filled one shared `max_pairs` budget in lexicographic order, so normal-first data consumed all 10,000 pairs on same-class normals before any different-class pair → `negatives` empty, margin `None`.
- Fix (`src/representation/geometry_analysis.py`): deterministic per-class budgets — `positive_cap = (max_pairs+1)//2`, `negative_cap = max_pairs//2` (total still ≤ `max_pairs`). Quotas split evenly across eligible groups (same-class) and sorted label-pairs (cross-class). Pair endpoints are selected by strided ordinal take with vectorized triangular-index conversion (`_strided_pair_take`, `_lex_endpoints`) and grid divmod for cross pairs (`_sample_same_class`, `_sample_cross_class`); similarities computed vectorized. No pair list is ever materialized; no RNG (deterministic by construction).
- Existing 2-normal/2-abnormal fixture still yields same=2/different=4 (quotas exceed available pairs, all taken).

## Reference distances (broadcast temporary)

- Root cause: `(chunk[:,None,:] - normal[None,:,:])**2` materialized a Q×M×D float64 temp (~1.3 GiB at 256×5000×128).
- Fix: `_reference_distances` uses `||q-r||² = ||q||² + ||r||² − 2q·r` over query blocks (256) and reference blocks (1024) — bounded ~2 MB working set — with running top-k merge, self-exclusion by global column, and `maximum(...,0)` clamp before `sqrt`. Mean divisor unchanged (`N × k`).
- Tests: direct-formula equivalence (approx — block summation order changes last-bit roundoff), singleton (`None`), self-exclusion exactness (crafted mean 10.0), block-size invariance (approx across, exact rerun), triangular-index brute-force match.

## Coverage

- New tests in `tests/representation/test_geometry_analysis.py`: normal-first mixed labels yield both similarities + margin with `labelled_pair_count ≤ max_pairs` and exact rerun equality; the four distance tests above.
- Focused run with the Task 9 + notebook suites: 39 passed, 1 skipped (CUDA unavailable). No formatters/linters/project-wide suites.
