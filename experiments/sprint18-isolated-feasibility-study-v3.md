# Sprint 18 Isolated Feasibility Study v3 Interruption-Continuation Amendment

**Study ID:** `sprint18-isolated-feasibility-study-v1`  
**Amendment ID:** `sprint18-isolated-feasibility-study-v3-interruption-continuation`  
**Status:** Prospective Task 58 freeze only. This amendment authorizes disposable smoke verification, not study execution. A fresh independent Task 58 evidence-review PASS and a separately assigned Task 59 worker are required before continuation.  
**Base contract:** [`sprint18-isolated-feasibility-study-v1.md`](sprint18-isolated-feasibility-study-v1.md), preserved under the historical v1/v2 freezes.

## 1. Amendment boundary

This amendment adds only a versioned interruption-continuation runner, its evidence guards, and a separate v3 evidence destination. It does not change the frozen v7/C2 profiles, generator, allocation, diagnostics, no-write requirements, 47-file source/config closure, or interpretation. The fixed study remains 16 ascending seeds (`918000–918015`) with `sprint15-v7` followed by `sprint18-c2`, for exactly 32 unique persisted coordinates.

The v1 and v2 runners, contracts, attempts, and results are historical and immutable inputs. The v3 runner must verify their pinned hashes before claiming its own output. It imports every sealed row from the v2 result as-is; it must not regenerate or revise any imported coordinate. The 28 imported coordinates end at `(918013, sprint18-c2)`. The only coordinates eligible for new evaluation, in order, are:

1. `(918014, sprint15-v7)`
2. `(918014, sprint18-c2)`
3. `(918015, sprint15-v7)`
4. `(918015, sprint18-c2)`

The v2 worker was interrupted during or immediately after computing `(918014, sprint15-v7)`. That computation was not persisted. It may have started and its in-memory work was discarded; therefore the 32-coordinate budget means 32 unique persisted coordinates, not 32 total computation starts. The v3 evidence and any later report must preserve this distinction.

## 2. Frozen v2 evidence and v1 provenance

The v2 source closure remains 47 files with SHA-256 `2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834`; reference config hashes remain v7 `7de530a28d23` and C2 `720e31e94681`. The v3 runner delegates the frozen static closure, configuration, metric, row-seal, and summary logic to the byte-verified v2 implementation; it does not edit production code or introduce a new generator path.

Before creating v3 output, the runner verifies all of the following:

- v1 runner, contract, attempt, result, and imported-row provenance against the hashes frozen in the historical v2 amendment;
- v2 runner SHA-256 `d5c4dfe3039bf82ec7eec05bf81f083966b81098ed54813b3e1c3d34c35f50b6` and contract SHA-256 `8b8e28e489c393cd0ccf33d5844bab9a1d5e562904d671f5b94c122fc4b5ec42`;
- v2 attempt SHA-256 `9bde33a9656f9b0a4d70215f28a44d44fd382f8413eb4fe06bd3366f36040ba1`, result SHA-256 `2fee65aa8af915db021d63138bb2748172402f125c66478ae4ee803216a6c25a`, and last imported row seal `0157b3c980eee4f2697ef0572107d4f7ae7f6989def2662e1a316732d84d9b05`;
- the v2 freeze digest `4a34ab48b76b02305a8267c138f6cddaf5f72550d763ef39dfc49d46c4ded535`, exact `RUNNING` partial-attempt state (28 completed, 4 remaining), and result/attempt digest linkage;
- exactly 28 ordered, unique, completed rows with valid row seals and `no_write=true`, `generation_writes=false`; the first v2 row must equal the exact imported v1 row;
- consistent v1 provenance, result source closure, and `sealed_inputs_opened=false`.

Any changed bytes, corrupt row, duplicate or reordered coordinate, inconsistent marker, wrong prior attempt, changed closure/config, or mismatched freeze blocks before the v3 output directory is claimed. v3 writes only `artifacts/sprint-18/isolated-feasibility-v3/{attempt,result}.json`; an existing v3 output consumes the one-shot attempt and blocks retry/resume.

## 3. v3 persistence and budget

The v3 attempt marker begins by binding the 28 imported rows and four-coordinate remainder. Each evaluation is sealed and the result file is atomically written before the attempt marker that binds its digest, using the already-reviewed v2 persistence primitives. Completion requires exactly 32 ordered unique rows, all marked `evaluation_started=true`, with all original 28 row objects preserved exactly and only the four fixed coordinates evaluated by the existing v2 metric path.

The result records both historical provenance chains, v3 freeze/source/config digests, imported and remaining coordinates, per-row seals, `no_write=true`, and the explicit persisted-coordinate budget caveat above. No generated-data root may be created; no model or Sealed input may be accessed. A partial v3 output is one-shot and is not resumable.

## 4. Disposable persistence/import/finalization smoke

The v3 runner's `--smoke` uses the real immutable v2 attempt/result for import validation, then exercises the same v3 one-shot claim, row ordering/sealing, atomic result/marker persistence, summaries, and finalization in a temporary directory outside the project. It uses marked disposable row projections from the existing v2 rows rather than invoking the generator again. It verifies that all four missing coordinates are visited exactly once in the disposable path, the imported 28 row objects remain equal, the completed fixture has 32 persisted coordinates, duplicate output is blocked, and v1/v2 evidence remains byte-identical. The fixture output is discarded and is never study evidence.

The smoke also checks rejection boundaries for a mismatched freeze, mismatched prior-attempt digest/state, mismatched prior-result digest, corrupt imported row seal, reordered imported rows, duplicate persisted coordinates, and duplicate v3 output. `generator_invoked=false` and no persistent v3 output/generated-data root are required smoke observations.

## 5. Commands and independent review gate

Disposable smoke only:

```bash
uv run --locked python experiments/sprint18_task58_feasibility-v3.py --smoke
```

The runner requires both explicit acknowledgement flags and the exact freeze digest recorded in the reviewed Task 58 artifact:

```bash
uv run --locked python experiments/sprint18_task58_feasibility-v3.py --continue --review-passed --expected-freeze <v3-freeze-sha256-from-reviewed-task-58-artifact>
```

Do not run this continuation command as part of Task 58. It is permitted only after a fresh independent Task 58 evidence-review PASS and assignment of a dedicated Task 59 worker. Do not invoke or rerun the v1 or v2 runner.

The v3 freeze is SHA-256 over `b"SPRINT18-ISOLATED-FEASIBILITY-CONTINUATION-FREEZE-v3\n"`, followed by the raw 32-byte SHA-256 digests of this contract and the v3 runner, then canonical JSON containing the sorted 47-file source closure, the exact v1 historical provenance map, and the pinned v2 runner/contract/attempt/result/freeze/last-row hashes. The runner release binding below pins the executable, closure, and both evidence lineages.

<!-- release-binding -->
```json
{
  "runner_sha256": "f74c1e182b7bb28c40808edf35f9505d5f2ed564d9f5f6ff9289ad14d174988c",
  "source_closure_sha256": "2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834",
  "historical_v1_provenance": {
    "runner_sha256": "d65f01ee703d2f89b46c5b0138b3690f811edce304878204d9b1cbf14fa83111",
    "contract_sha256": "1252707032aabfd95d99d1944140e57f8eff3610f7c8d35810e649c65a276bd0",
    "attempt_sha256": "e4efabee11096ab19de0aa1f37f9b5f7f2743c1aa16c02e5d02a79584e1874e8",
    "result_sha256": "d9f45f6cb40040281f0b4570653f2c06e9ec5135330fc8507dc505558c2df2e6",
    "row_sha256": "36da034b9fa0288310811f5c49212ac7869b9d486a1720b66659ca8d38652204",
    "freeze_sha256": "d0707a898dee7ddf8b3aacf153cba2cea53db6e7b1f142bf081ccee469bbf037",
    "source_closure_sha256": "2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834"
  },
  "v2_provenance": {
    "runner_sha256": "d5c4dfe3039bf82ec7eec05bf81f083966b81098ed54813b3e1c3d34c35f50b6",
    "contract_sha256": "8b8e28e489c393cd0ccf33d5844bab9a1d5e562904d671f5b94c122fc4b5ec42",
    "attempt_sha256": "9bde33a9656f9b0a4d70215f28a44d44fd382f8413eb4fe06bd3366f36040ba1",
    "result_sha256": "2fee65aa8af915db021d63138bb2748172402f125c66478ae4ee803216a6c25a",
    "freeze_sha256": "4a34ab48b76b02305a8267c138f6cddaf5f72550d763ef39dfc49d46c4ded535",
    "last_record_sha256": "0157b3c980eee4f2697ef0572107d4f7ae7f6989def2662e1a316732d84d9b05"
  }
}
```

## 6. Interpretation boundary

The v3 amendment only completes the already-frozen descriptive study block by importing 28 sealed rows and evaluating the four unpersisted coordinates. The resulting 32-row study remains exploratory and descriptive; it does not establish benchmark qualification, causality, population pass probability, independent confirmation, or authorization for a model run or production preflight.
