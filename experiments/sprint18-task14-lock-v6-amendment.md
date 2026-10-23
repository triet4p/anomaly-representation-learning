# Sprint 18 Task 14 lock v6 — Task 30 Fit-bank row ordering

- **Status:** prospective candidate; **no self-approval**. Independent source-evidence review, a separate Bronze exact-snapshot checkpoint and non-force source/sync are required before any normative runtime.
- **Task / attempt:** `S18-T30` / `S18-T30-A08`
- **Base:** `7735669a8f603271e1b65a3caa68e09429eb2625` (`refs/heads/master`), parent `430cbbd4d5ced2f8bdc7926ff00f48bc0b124590`.
- **New lock:** [`experiments/sprint18-task14-config-lock-v6.json`](./sprint18-task14-config-lock-v6.json), schema `s18-task14-config-lock-v6`, lock version 6, file `b4d1f435…`, core `5f65f907…`.
- **Preserved history:** the v5 prospective lock (`sprint18-task14-config-lock-v5.json`, raw `feeff6ab…`, core `e48455b0…`), the v4 prospective lock (raw `edbd65f3…`, core `3151a4ea…`), the v3 prospective lock (raw `5ce105b4…`, core `c6c3fe84…`), the v2 prospective lock (raw `6dd815c5…`, core `4b399874…`) and the v1 lock (raw `c4c1a8bd…`, core `6eb108b2…`) are byte-identical and are chained by digest, never edited.

## Why a new lock version

The v5 lock was reviewed (`S18T30SR04` PASS), checkpointed (`S18T30SCP03`, commit `7735669a…`) and synchronized, and it pins the Task 30 runner bytes `19331f57…`. The fresh certified-DATA CUDA preflight (`S18T30A07`) passed, the focused actual-CUDA restore regression passed, and the bounded real-CUDA smoke then failed closed at the Fit-bank gate:

```
DRIVER_FATAL seed 181801: refit Fit bank differs from the accepted B0 bank digest
```

The diagnosis proved the rows, checkpoint, forward latents and environment were identical: `fit_fit_reference_bank` hashes the embedding matrix **in row order**, the accepted B0 bank was built from the globally `file_id`-sorted Fit support (`fit_rows_ordered` in the Task23 A04 runner), and the Task 30 runner passed the certified loader's **per-history concatenation** instead. Re-sorting the same 5,446 rows by `file_id` reproduced the accepted digest exactly (`2d6e0a11…`), while the runner path produced `852bf7e4…`. No stored bank exists in the checkpoint, restored object or prediction cache, so a refit is required; only its input order was wrong.

The correction necessarily changes the Task 30 runner bytes and the lock-consumer bytes, both of which the v5 lock pins. Editing the v5 lock in place would rewrite the artifact accepted by `S18T30SCP03`; the established pattern is a new version that preserves the previous bytes and supersedes exactly the corrected scope. Hence v6.

## What changes

### 1. Canonical Fit-bank row ordering (the correction)

`experiments/sprint18_task30_n4_runner.py`:

- adds `_canonical_fit_rows`, which returns the Fit support in the accepted B0 bank order — global ascending `file_id` — and refuses duplicate file IDs (certified Fit file IDs are unique across the three Fit histories, so the sort is a total order);
- `_load_supports` canonicalizes the loaded Fit rows **once**, before returning them, so every consumer of Fit rows (bank, compute probes, probe row) observes the identical order the accepted B0 digest binds;
- `_fit_bank` additionally verifies, without allocating a second list, that the sequence it receives is strictly ascending and unique, and fails closed with an explicit message if a future caller reintroduces a non-canonical order.

The bank mathematics are untouched: each row is still embedded independently with its own frozen mask seed; only the row order of the resulting matrix changes, and it now matches the accepted Task23 A04 construction exactly.

### 2. Correlated consumer audit (minimal)

- `_fit_bank` is the only refit/bank gate; `_run_capacity_folds` builds no reference bank (it consumes N4 scale records), so no other bank input exists in the Task 30 path.
- Calibration scoring consumes the single Calibration history's rows (unchanged manifest order) and `fit_calibration_q95` is order-invariant; the N4 in-scope pools are sorted by `(history_id, file_id, patch_index)` in the capacity consumer.
- `fit_fit_reference_bank`'s `source_rows_sha256` already sorts by `(actual_history_id, file_id)` and was equal to the accepted record (`447b219e…`); it is not order-sensitive and is unchanged.
- The compute-block probes now receive the same canonical Fit sequence, and the single probe row is the canonical first row rather than a redundant re-sort.
- No broad feature work or unrelated cleanup was performed.

### 3. Lock consumer

`experiments/sprint18_task13b_n4_capacity.py` keeps loading only the current lock (no fallback) and now validates the complete **v6 → v5 → v4 → v3 → v2 → v1** digest chain. The v5/v4/v3/v2/v1 lock bytes and core digests are pinned and must match exactly. The consumer record returns the superseded v5 identity plus the preserved `historical_v4_lock`, `historical_v3_lock`, `historical_v2_lock` and `historical_v1_lock` identities, so runtime preflight records the full chain. Nothing was removed or relaxed; the added link is strictly checked.

### 4. Reviewed implementation pin set

The pin set remains exactly the six required paths; only the Task 30 runner pin and the capacity-runner identity are refreshed:

| Path | SHA-256 |
|---|---|
| `experiments/sprint18-context-residual-normalization-spec-v3.md` | `e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2` (unchanged) |
| `experiments/sprint18_task30_n4_runner.py` | `4bb560e179cf385a95a0eca4a603827cc642e9109063687c82d04ea98ae85322` (corrected) |
| `src/representation/__init__.py` | `9c3c857a76acebea08f4d6eebede4af7799024cdac2f7193c2d0601fdbcbf09b` (unchanged) |
| `src/representation/crn.py` | `7902dd9b7209b2ad9f7ef0f0e6f7316eb13c7db20237bc04974336b863bb9251` (unchanged) |
| `src/representation/sprint18_checkpoint.py` | `fce263df093bc622ba0a69bfac0824941fd6a0d1c4c9ab02bcbcff62432aaef2` (unchanged) |
| `tests/representation/test_sprint18_crn.py` | `2203d3f0b74be7ddb9f1f693b1321834b9078e86d990b6764996187218809186` (unchanged) |

`runner_identity` binds `experiments/sprint18_task13b_n4_capacity.py` at its cutover bytes `e534fb08…`. The superseded v5 runner pin `19331f57…` is retained under `review.superseded_v5_implementation_pins`; the v4 (`f24e825e…`), v3 (`e9237327…`) and v2 pins remain in their preserved blocks.

### 5. Tests

`tests/representation/test_sprint18_task14_lock_v5.py` is superseded by `tests/representation/test_sprint18_task14_lock_v6.py`, which validates the v6 lock, the v6 → v5 → v4 → v3 → v2 → v1 chain, the refreshed pin, the superseded v5 v4/v3 pins, and the unchanged recipe/capacity/record contract; the stale-lock test refuses v5, v4, v3, v2 and v1.

`tests/representation/test_sprint18_task30_runner.py` gains a behavioural bank-ordering regression that uses a real tiny B0 model and a real saved/restored `experiment` checkpoint:

- the three Fit histories carry interleaved file IDs, so the certified loader's per-history order differs from the canonical global `file_id` order;
- per-history input, a shuffled permutation and the canonical list all canonicalize to the identical row sequence; duplicates are refused;
- the actual `fit_fit_reference_bank` embeddings equal the directly recomputed per-row embeddings in canonical order, and the shuffled/canonical banks have identical tensors and `bank_sha256`;
- the raw per-history bank keeps the same `source_rows_sha256` but a different `bank_sha256`, which is the A07 defect;
- Calibration `S_pred`/`S_pop` maps and both q95 thresholds are identical for the canonical and shuffled paths.

## Historical source binding

- The v5 lock bytes `feeff6ab…` / core `e48455b0…` and its six implementation pins (including runner `19331f57…`) remain exactly as reviewed by `S18T30SR04` and committed by `S18T30SCP03` at `7735669a…`.
- The v4/v3/v2/v1 lock bytes, the v3 mechanism amendment, the locked Task 23 A04 ledger `3540ef09…`, the B0 epoch-100 checkpoints `9292d7c5…` / `8ae58da9…` / `79edbdf3…`, the raw cache identities `18345f86…` / `e035b2b7…` / `1d4a2a74…`, and the historical Task13B capacity evidence `6169fa12…` are unchanged. **No B0 rerun and no DATA change.**

## What does not change

- The frozen scientific recipe: 50 epochs, batch ≤512 patch records, `sprint18-n4-scale-order-v2`, AdamW 1e-3/1e-4, clip 1.0, Fit-only linear q05/q95 bounds, 20 % cap gate, zero representation updates, no early stopping or Development selection. No numeric, digest-meaning or guard change.
- The certified N4 scope: 66,989 eligible masked Fit records, per-history 22,328/23,098/21,563, folds 44,661/43,891/45,426, fold steps 4,400/4,300/4,450, final 6,550 steps.
- The v3 mechanism amendment and the v1/v2 specifications.
- The final B0 checkpoint/cache identities, the Fit/Calibration/Development rosters and role permissions.
- Every arm other than N4-CRN; the historical v1 lock continues to bind B0/K3/K7/K9/N1/N2/N3 and the B0 strict restores.
- The capacity certification remains `historical_certification_at_original_snapshot` at `f40667a`; the Task 30 runtime must still re-execute the three folds on the executed source.

## Neutrality evidence

- The correction changes only the row order fed to the reference-bank fit, plus the per-row consumer order. The accepted B0 refit digest is now reproduced exactly from the same rows, checkpoint and final B0 model.
- The canonicalization helper is a stable sort of unique keys (`sorted` on unique `file_id` values), so it is deterministic and independent of input order; it allocates one dict and one list per support load, which is the minimum needed to produce the canonical order.
- `_fit_bank`'s added order check is a read-only scan with no allocation beyond loop-local names.
- The runner's internal `task_id`/`attempt_id`/`agent_id` output labels remain the frozen source labels written when the runner was authored; they are not an execution-provenance claim.
- `n4_scale_scope`, `n4_recipe`, `locked_configs`, `config_sha256`, `common_contract`, `compute_envelopes`, `coefficient_selections` and `final_b0_binding` are carried forward unchanged except for the recomputed derived lock/core and arm-config digests required by the new version.

## Required gates before any normative run

1. Independent source-evidence review of this lock, the six pinned paths, the amendment, the correction record and the local nonnormative proof.
2. Separate Bronze exact-snapshot checkpoint of the reviewed source.
3. Non-force source push and isolated fast-forward sync.
4. Fresh runtime worker: real certified-DATA CUDA preflight + bounded smoke (the Fit-bank gate must pass), then in-run capacity recertification and the three-seed final fit/q95/Development path. No normative CUDA run may occur before this gate.

The local proof is a tiny CPU-only mechanics proof (real model, real checkpoint, tiny cache, bank, q95 and public N4 scoring); it is not certified-DATA, CUDA, capacity or full-runtime evidence. The real CUDA regression and bounded smoke must be re-executed by the post-gate runtime.
