# Sprint 18 Isolated Feasibility Study v2 Infrastructure Continuation Amendment

**Study ID:** `sprint18-isolated-feasibility-study-v1`  
**Amendment ID:** `sprint18-isolated-feasibility-study-v2-infrastructure-continuation`  
**Status:** Prospective Task 58 correction and independent review only. This amendment does not itself authorize continuation.  
**Base contract:** [`sprint18-isolated-feasibility-study-v1.md`](sprint18-isolated-feasibility-study-v1.md), SHA-256 `1252707032aabfd95d99d1944140e57f8eff3610f7c8d35810e649c65a276bd0`.

## 1. Amendment boundary

This amendment changes persistence, provenance guards, and the versioned runner only. It does not change the v1 study question, profiles, configuration, seeds, diagnostics, allocation semantics, no-write guarantee, or interpretation. The fixed design remains seeds `918000–918015` in ascending order, with `sprint15-v7` followed by `sprint18-c2` for each seed (32 unique profile-seed evaluations total). The original 47-file source/config closure remains SHA-256 `2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834`; reference config hashes remain v7 `7de530a28d23` and C2 `720e31e94681`.

The v1 runner and outputs remain historical and immutable. The original runner is archived byte-for-byte as `experiments/sprint18_task58_feasibility-v1.py` (SHA-256 `d65f01ee703d2f89b46c5b0138b3690f811edce304878204d9b1cbf14fa83111`). The corrected continuation runner is `experiments/sprint18_task58_feasibility-v2.py`. The v1 one-shot output directory is never opened for writing; all continuation evidence goes to the separate, fixed `artifacts/sprint-18/isolated-feasibility-v2/` directory.

## 2. Immutable v1 evidence and exact import

The v1 attempt is consumed and remains `RUNNING`, not resumable. Its attempt marker SHA-256 is `e4efabee11096ab19de0aa1f37f9b5f7f2743c1aa16c02e5d02a79584e1874e8`; its one-row result SHA-256 is `d9f45f6cb40040281f0b4570653f2c06e9ec5135330fc8507dc505558c2df2e6`; and the row SHA-256 is `36da034b9fa0288310811f5c49212ac7869b9d486a1720b66659ca8d38652204`. The row is exactly `(918000, sprint15-v7)`. The original v1 freeze digest is `d0707a898dee7ddf8b3aacf153cba2cea53db6e7b1f142bf081ccee469bbf037`.

Before claiming the new output directory, the v2 runner MUST verify the byte digests and `RUNNING` state of both v1 files, the unchanged v1 freeze/source closure, the sole row's canonical seal and expected coordinate, and the partial summary. It rejects any changed, corrupt, completed, mismatched, extra, missing, or reordered provenance. The imported row is used directly and is never regenerated or edited. The original attempt/result bytes are never rewritten.

## 3. Fixed continuation and one-shot output

After a fresh independent Task 58 evidence-review PASS only, the v2 runner imports the verified v7 seed `918000` row and evaluates exactly these remaining 31 fixed coordinates, in order: `(918000, sprint18-c2)`, then both profiles in v1 order for each seed `918001–918015`. Completion requires exactly 32 rows in the original fixed order, 32 unique coordinates, 32 rows with `evaluation_started=true`, and valid row seals. No seed/profile may be selected, replaced, screened, resampled, rerun, or reordered.

The new marker/result pair records the v1 attempt/result/row digests, the v2 freeze/source/config digests, imported and remaining budgets, the last sealed row digest, and the hash of the exact result bytes. Each result and marker update is atomically replaced; the result is written before the marker that binds its digest. A partial or completed v2 output consumes the sole continuation attempt: an existing output directory blocks all re-execution and there is no resume mode. The runner also blocks if the frozen forbidden generated-data root exists. It writes no model, Sealed, benchmark, or generated-data outputs.

## 4. Disposable persistence smoke

`--smoke` uses only disposable fixture seed `93058`. It builds and seals a fixture v7 row through the actual metric path, persists a temporary v1-shaped partial record, imports that row through the same continuation verifier, evaluates only the fixture C2 row, and exercises the shared result/attempt marker persistence loop through a complete fixture pair. It checks the final summary, both row hashes, marker-to-result digest binding, exact imported-row preservation, unchanged source fixture bytes, duplicate-attempt blocking, and byte-identical v1 source attempt/result evidence. It does not evaluate any study seed or create the fixed continuation output directory.
The smoke actively verifies rejection of mismatched result digests, a completed prior attempt, mismatched attempt provenance, corrupt source closure, missing or extra imported rows, reordered coordinates, and an invalid row seal. On the disposable completed fixture it also verifies rejection of extra, missing, reordered, and unstarted continuation evaluations, then confirms the duplicate attempt is blocked without changing the persisted result or marker.


## 5. Commands and gate

Disposable smoke:

```bash
uv run --locked python experiments/sprint18_task58_feasibility-v2.py --smoke
```

Only after fresh Task 58 independent evidence-review PASS, the one permitted continuation command is:

```bash
uv run --locked python experiments/sprint18_task58_feasibility-v2.py --continue --review-passed --expected-freeze <freeze-sha256-from-reviewed-task-58-artifact>
```

The runner rejects a missing review acknowledgement or a freeze digest that differs from the reviewed v2 source/document freeze. Do not run the continuation while Task 58 is pending review. Do not invoke the historical v1 runner.

The v2 freeze digest is SHA-256 over `b"SPRINT18-ISOLATED-FEASIBILITY-CONTINUATION-FREEZE-v2\n"`, followed by the raw 32-byte SHA-256 of this amendment, the raw 32-byte SHA-256 of the corrected v2 runner, then canonical JSON containing the sorted 47-file source closure and the historical v1 provenance hash map. The runner release binding below freezes the corrected runner, source-closure, reference config, and historical provenance hashes. The task artifact records the observed amendment hash and final combined freeze digest.

<!-- release-binding -->
```json
{
  "runner_sha256": "d5c4dfe3039bf82ec7eec05bf81f083966b81098ed54813b3e1c3d34c35f50b6",
  "source_closure_sha256": "2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834",
  "profile_config_hashes": {
    "sprint15-v7": "7de530a28d23",
    "sprint18-c2": "720e31e94681"
  },
  "historical_provenance": {
    "runner_sha256": "d65f01ee703d2f89b46c5b0138b3690f811edce304878204d9b1cbf14fa83111",
    "contract_sha256": "1252707032aabfd95d99d1944140e57f8eff3610f7c8d35810e649c65a276bd0",
    "attempt_sha256": "e4efabee11096ab19de0aa1f37f9b5f7f2743c1aa16c02e5d02a79584e1874e8",
    "result_sha256": "d9f45f6cb40040281f0b4570653f2c06e9ec5135330fc8507dc505558c2df2e6",
    "row_sha256": "36da034b9fa0288310811f5c49212ac7869b9d486a1720b66659ca8d38652204",
    "freeze_sha256": "d0707a898dee7ddf8b3aacf153cba2cea53db6e7b1f142bf081ccee469bbf037",
    "source_closure_sha256": "2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834"
  }
}
```

## 6. Interpretation boundary

The continuation is infrastructure-only: the valid existing v7 seed `918000` row is imported by exact digest; only the 31 remaining pre-bound coordinates are evaluated. The resulting 32-row block remains exploratory and descriptive. It does not establish benchmark qualification, causality, a population pass probability, or independent Confirmation.
