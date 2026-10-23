# Sprint 18 Task 14 lock v4 — Task 30 prediction-cache binding correction

- **Status:** prospective candidate; **no self-approval**. Independent source-evidence review, a separate Bronze exact-snapshot checkpoint and non-force source/sync are required before any normative runtime.
- **Task / attempt:** `S18-T30` / `S18-T30-A04`
- **Base:** `74ed62b0c0c4b3ef027bb1d473b76033d79b30d9` (`refs/heads/master`), parent `83710aa65d201cf3f37d7f9961368589932f6ec9`.
- **New lock:** [`experiments/sprint18-task14-config-lock-v4.json`](./sprint18-task14-config-lock-v4.json), schema `s18-task14-config-lock-v4`, lock version 4.
- **Preserved history:** the v3 prospective lock (`sprint18-task14-config-lock-v3.json`, raw `5ce105b4…`, core `c6c3fe84…`), the v2 prospective lock (raw `6dd815c5…`, core `4b399874…`) and the v1 lock (raw `c4c1a8bd…`, core `6eb108b2…`) are byte-identical and are chained by digest, never edited.

## Why a new lock version

The v3 lock was reviewed (`S18T30SR02` PASS), checkpointed (`S18T30SCP01`, commit `74ed62b0…`) and synchronized, and it pins the Task 30 runner bytes `e9237327…`. The first fresh certified-DATA CUDA smoke (`S18T30A03`) then failed closed at the first B0 restore: the runner required the **same** fixture key `prediction_cache_sha256` to equal both the accepted cache-head digest (`_b0_seed_identity`) and the raw SHA-256 of the cache-file bytes (`_restore_b0`). The accepted Task 23 A04 caches carry distinct values for every seed (`18345f86…` vs `bbed6278…`, `e035b2b7…` vs `1a1d534d…`, `1d4a2a74…` vs `385c0a50…`), so no binding document could satisfy both gates. That is a source defect, not a DATA, device or capacity fault.

The correction necessarily changes the Task 30 runner bytes and the lock-consumer bytes, both of which the current lock pins. Editing the v3 lock in place would rewrite the artifact accepted by `S18T30SCP01`; the established pattern is a new version that preserves the previous bytes and supersedes exactly the corrected scope. Hence v4.

## What changes

### 1. Task 30 prediction-cache binding (the correction)

`experiments/sprint18_task30_n4_runner.py` now defines one explicit convention with **two distinct per-seed identities** in the external fixture:

| Fixture key | Semantics | Compared against |
|---|---|---|
| `prediction_cache_file_sha256` | raw SHA-256 of the prediction-cache file bytes | the actual cache file at restore and in preflight |
| `prediction_cache_inner_sha256` | digest carried inside the cache head | the cache head and the accepted B0 record's `prediction_cache.cache_sha256` |

- `_validate_fixture` requires exactly the frozen fixture key set, all three seeds, and well-formed SHA-256 values. The superseded ambiguous key `prediction_cache_sha256` is refused **by name**; it is never interpreted as either identity.
- `_b0_restore_identity` is the single restore-level identity gate: it verifies the cache file bytes, the cache-head digest, the accepted-record agreement, the cache record count and the B0 checkpoint bytes. Both `run_preflight` (all three seeds, before any fit) and `_restore_b0` (runtime) call it, so a contradiction can no longer pass preflight.
- `_cache_pools` continues to derive the N4 records from the same accepted cache; the semantic cache and its residuals are unchanged.
- A regression test covers the real Task 23 shape (file digest ≠ head digest) with a real serialized cache through the real loader: correct acceptance, and separate refusal of a swapped file digest, a swapped head digest, a record mismatch, corrupted cache bytes, a legacy single-key fixture, and a drifted checkpoint.

### 2. Lock consumer

`experiments/sprint18_task13b_n4_capacity.py` loads and validates the current lock without falling back to older versions and validates the complete **v4 → v3 → v2 → v1** digest chain. The historical v1/v2/v3 lock bytes and core digests are pinned and must match exactly. The consumer now returns the superseded v3 and preserved v2 identities in its record so the runtime preflight records the full chain.

### 3. Reviewed implementation pin set

The pin set remains exactly the six required paths; only the Task 30 runner pin and the capacity-runner identity are refreshed:

| Path | SHA-256 |
|---|---|
| `experiments/sprint18-context-residual-normalization-spec-v3.md` | `e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2` (unchanged) |
| `experiments/sprint18_task30_n4_runner.py` | `f24e825ed242942045f7c69c0daac08a8a8374f897914acafddbca0902bac3f9` (corrected) |
| `src/representation/__init__.py` | `9c3c857a76acebea08f4d6eebede4af7799024cdac2f7193c2d0601fdbcbf09b` (unchanged) |
| `src/representation/crn.py` | `7902dd9b7209b2ad9f7ef0f0e6f7316eb13c7db20237bc04974336b863bb9251` (unchanged) |
| `src/representation/sprint18_checkpoint.py` | `fce263df093bc622ba0a69bfac0824941fd6a0d1c4c9ab02bcbcff62432aaef2` (unchanged) |
| `tests/representation/test_sprint18_crn.py` | `2203d3f0b74be7ddb9f1f693b1321834b9078e86d990b6764996187218809186` (unchanged) |

`runner_identity` binds `experiments/sprint18_task13b_n4_capacity.py` at the corrected bytes `985ac3b9…` (the digest at the base `74ed62b0…` is `bd02e1c1…`). The superseded v3 runner pin `e9237327…` is retained under `review.superseded_v3_implementation_pins`, alongside the A04 correction review block.

### 4. Tests

`tests/representation/test_sprint18_task14_lock_v3.py` is superseded by `tests/representation/test_sprint18_task14_lock_v4.py`, which validates the v4 lock, the v4 → v3 → v2 → v1 chain, the refreshed pin, the superseded v3 pins, and the unchanged recipe/capacity/record contract; the stale-lock test refuses v3, v2 and v1. `tests/representation/test_sprint18_task30_runner.py` gains the binding regression above.

## Historical source binding

- The v3 lock bytes `5ce105b4…` / core `c6c3fe84…` and its six implementation pins (including runner `e9237327…`) remain exactly as reviewed by `S18T30SR02` and committed by `S18T30SCP01` at `74ed62b0…`.
- The locked Task 23 A04 ledger `3540ef09…`, the B0 epoch-100 checkpoints `9292d7c5…` / `8ae58da9…` / `79edbdf3…`, the raw cache identities `18345f86…` / `e035b2b7…` / `1d4a2a74…`, and the historical Task13B capacity evidence `6169fa12…` are unchanged. **No B0 rerun and no DATA change.**

## What does not change

- The frozen scientific recipe: 50 epochs, batch ≤512 patch records, `sprint18-n4-scale-order-v2`, AdamW 1e-3/1e-4, clip 1.0, Fit-only linear q05/q95 bounds, 20 % cap gate, zero representation updates, no early stopping or Development selection. No numeric, digest-meaning or guard change.
- The certified N4 scope: 66,989 eligible masked Fit records, per-history 22,328/23,098/21,563, folds 44,661/43,891/45,426, fold steps 4,400/4,300/4,450, final 6,550 steps.
- The v3 mechanism amendment and the v1/v2 specifications.
- The final B0 checkpoint/cache identities, the Fit/Calibration/Development rosters and role permissions.
- Every arm other than N4-CRN; the historical v1 lock continues to bind B0/K3/K7/K9/N1/N2/N3 and the B0 strict restores.
- The capacity certification remains `historical_certification_at_original_snapshot` at `f40667a`; the Task 30 runtime must still re-execute the three folds on the executed source.

## Neutrality evidence

- The correction changes only the external cache-identity schema, the preflight restore-level proof and the lock/lock-consumer pins. Model construction, strict restore, Fit-bank derivation, Calibration q95, public N4 scoring and `S_pop` are untouched.
- The accepted caches' inner head digest is now consumed exactly where it was already recorded (the accepted B0 record), and the raw file digest exactly where bytes are read; neither check is removed or weakened.
- The runner's internal `task_id`/`attempt_id`/`agent_id` output labels remain the frozen source labels written when the runner was authored; they are not an execution-provenance claim.
- `n4_scale_scope`, `n4_recipe`, `locked_configs`, `config_sha256`, `common_contract`, `compute_envelopes`, `coefficient_selections` and `final_b0_binding` are carried forward unchanged except for the recomputed derived lock/core and arm-config digests required by the new version.

## Required gates before any normative run

1. Independent source-evidence review of this lock, the six pinned paths, the amendment, the correction record and the local nonnormative proof.
2. Separate Bronze exact-snapshot checkpoint of the reviewed source.
3. Non-force source push and isolated fast-forward sync.
4. Fresh runtime worker: real certified-DATA CUDA preflight + bounded smoke, then in-run capacity recertification and the three-seed final fit/q95/Development path.

No certified DATA, CUDA, capacity or full-runtime claim is made by this document or by the local nonnormative proof.
