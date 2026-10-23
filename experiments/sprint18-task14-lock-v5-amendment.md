# Sprint 18 Task 14 lock v5 — Task 30 strict-restore probe device alignment

- **Status:** prospective candidate; **no self-approval**. Independent source-evidence review, a separate Bronze exact-snapshot checkpoint and non-force source/sync are required before any normative runtime.
- **Task / attempt:** `S18-T30` / `S18-T30-A06`
- **Base:** `430cbbd4d5ced2f8bdc7926ff00f48bc0b124590` (`refs/heads/master`), parent `74ed62b0c0c4b3ef027bb1d473b76033d79b30d9`.
- **New lock:** [`experiments/sprint18-task14-config-lock-v5.json`](./sprint18-task14-config-lock-v5.json), schema `s18-task14-config-lock-v5`, lock version 5.
- **Preserved history:** the v4 prospective lock (`sprint18-task14-config-lock-v4.json`, raw `edbd65f3…`, core `3151a4ea…`), the v3 prospective lock (raw `5ce105b4…`, core `c6c3fe84…`), the v2 prospective lock (raw `6dd815c5…`, core `4b399874…`) and the v1 lock (raw `c4c1a8bd…`, core `6eb108b2…`) are byte-identical and are chained by digest, never edited.

## Why a new lock version

The v4 lock was reviewed (`S18T30SR03` PASS), checkpointed (`S18T30SCP02`, commit `430cbbd4…`) and synchronized, and it pins the Task 30 runner bytes `f24e825e…`. The fresh certified-DATA CUDA preflight (`S18T30A05`) then passed, and the bounded real-CUDA smoke failed closed inside `_run_capacity_folds` with:

```
RuntimeError: Expected all tensors to be on the same device, but got weight is on cuda:0,
different from other tensors on cpu (when checking argument in method wrapper_CUDA__native_layer_norm)
```

Both strict-restore forward-parity probes built probe context rows on CPU (the certified cache loader yields `.cpu()` rows) and forwarded them through a scale head deployed on the run device, while the restored head also stayed on CPU. `_save_and_restore_scale` carries the identical unguarded pattern and would have failed one step later. The certified Task13B runner already implements the correct contract at `experiments/sprint18_task13b_n4_capacity.py:1249-1254`.

The correction necessarily changes the Task 30 runner bytes and the lock-consumer bytes, both of which the current lock pins. Editing the v4 lock in place would rewrite the artifact accepted by `S18T30SCP02`; the established pattern is a new version that preserves the previous bytes and supersedes exactly the corrected scope. Hence v5.

## What changes

### 1. Strict-restore probe device alignment (the correction)

`experiments/sprint18_task30_n4_runner.py` gains one shared helper, `_probe_scale_parity`, used by **both** envelope probes:

- the probe context is stacked as before and moved only when its device differs from the live head's actual device; an already-aligned CPU probe is never copied;
- the strict-restored head is moved to that same actual device for the forward and returned to its origin afterwards, again without a copy when devices already match;
- the passive (B0 residual) targets stay on CPU because `n4_scale_nll` consumes CPU log-variance and the caller detaches/copies the outputs to CPU exactly as before.

Both call sites keep their two strict checks and messages unchanged:

| Site | Checks preserved |
|---|---|
| `_save_and_restore_scale` | `forward_diff == 0.0` **and** live NLL == restored NLL (`strict scale restore forward/NLL parity failed`) |
| `_run_capacity_folds` | `forward_diff == 0.0` **and** live NLL == restored NLL (separate fold forward and NLL parity failures) |

No error suppression, no device fallback, no CPU substitution, and no change to the tolerance (exact equality). The probes still fail closed on any parity mismatch.

### 2. Correlated-site inspection

Every other new Task 30 tensor/device boundary was inspected and is already aligned:

- `_save_and_restore_coherent` constructs and restores the model and scale head directly on the run device.
- `_restore_b0` moves the strict-restored model to the run device before use.
- `_n4_score_row` / `_n4_score_rows` consume the public `n4_score_files` consumer, which co-locates the collated batch onto the deployed model device and fails closed when the scale head device differs.
- `_fit_final_scope` and `_compute_block` pass `device=` into `fit_n4_scale_scope`, which deploys the head itself; the compute-block public-consumer probe deploys both heads on the run device.

Only the two probe blocks carried the defect; no broad cleanup was applied.

### 3. Lock consumer

`experiments/sprint18_task13b_n4_capacity.py` keeps loading only the current lock (no fallback) and now validates the complete **v5 → v4 → v3 → v2 → v1** digest chain. The v4/v3/v2/v1 lock bytes and core digests are pinned and must match exactly. The consumer record now returns the superseded v4 identity plus the preserved `historical_v3_lock`, `historical_v2_lock` and `historical_v1_lock` identities, so runtime preflight records the full chain. Nothing was removed or relaxed; the added link is strictly stronger.

### 4. Reviewed implementation pin set

The pin set remains exactly the six required paths; only the Task 30 runner pin and the capacity-runner identity are refreshed:

| Path | SHA-256 |
|---|---|
| `experiments/sprint18-context-residual-normalization-spec-v3.md` | `e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2` (unchanged) |
| `experiments/sprint18_task30_n4_runner.py` | `19331f57e0d7b307085bd8e69f67740f693448096c82adede89b207cff39c36f` (corrected) |
| `src/representation/__init__.py` | `9c3c857a76acebea08f4d6eebede4af7799024cdac2f7193c2d0601fdbcbf09b` (unchanged) |
| `src/representation/crn.py` | `7902dd9b7209b2ad9f7ef0f0e6f7316eb13c7db20237bc04974336b863bb9251` (unchanged) |
| `src/representation/sprint18_checkpoint.py` | `fce263df093bc622ba0a69bfac0824941fd6a0d1c4c9ab02bcbcff62432aaef2` (unchanged) |
| `tests/representation/test_sprint18_crn.py` | `2203d3f0b74be7ddb9f1f693b1321834b9078e86d990b6764996187218809186` (unchanged) |

`runner_identity` binds `experiments/sprint18_task13b_n4_capacity.py` at its cutover bytes. The superseded v4 runner pin `f24e825e…` is retained under `review.superseded_v4_implementation_pins`; the older v3 (`e9237327…`) and v2 pins remain in their preserved blocks.

### 5. Tests

`tests/representation/test_sprint18_task14_lock_v4.py` is superseded by `tests/representation/test_sprint18_task14_lock_v5.py`, which validates the v5 lock, the v5 → v4 → v3 → v2 → v1 chain, the refreshed pin, the superseded v4 pins, and the unchanged recipe/capacity/record contract; the stale-lock test refuses v4, v3, v2 and v1.

`tests/representation/test_sprint18_task30_runner.py` gains a device-parameterized regression that exercises **both** restores:

- `_save_and_restore_scale` and `_run_capacity_folds` are run on `cpu` (always) and on a **real CUDA device** when one is available (honest skip otherwise);
- each case fits a real tiny scale head on the target device while the probe records keep their certified-loader CPU placement, so the pre-fix code would raise the observed device mismatch;
- the CPU path additionally loads the written envelope into a fresh head and drives the public `n4_score_files` consumer, proving the actual checkpoint artifact feeds public scoring.

## Historical source binding

- The v4 lock bytes `edbd65f3…` / core `3151a4ea…` and its six implementation pins (including runner `f24e825e…`) remain exactly as reviewed by `S18T30SR03` and committed by `S18T30SCP02` at `430cbbd4…`.
- The v3/v2/v1 lock bytes, the v3 mechanism amendment, the locked Task 23 A04 ledger `3540ef09…`, the B0 epoch-100 checkpoints `9292d7c5…` / `8ae58da9…` / `79edbdf3…`, the raw cache identities `18345f86…` / `e035b2b7…` / `1d4a2a74…`, and the historical Task13B capacity evidence `6169fa12…` are unchanged. **No B0 rerun and no DATA change.**

## What does not change

- The frozen scientific recipe: 50 epochs, batch ≤512 patch records, `sprint18-n4-scale-order-v2`, AdamW 1e-3/1e-4, clip 1.0, Fit-only linear q05/q95 bounds, 20 % cap gate, zero representation updates, no early stopping or Development selection. No numeric, digest-meaning or guard change.
- The certified N4 scope: 66,989 eligible masked Fit records, per-history 22,328/23,098/21,563, folds 44,661/43,891/45,426, fold steps 4,400/4,300/4,450, final 6,550 steps.
- The v3 mechanism amendment and the v1/v2 specifications.
- The final B0 checkpoint/cache identities, the Fit/Calibration/Development rosters and role permissions.
- Every arm other than N4-CRN; the historical v1 lock continues to bind B0/K3/K7/K9/N1/N2/N3 and the B0 strict restores.
- The capacity certification remains `historical_certification_at_original_snapshot` at `f40667a`; the Task 30 runtime must still re-execute the three folds on the executed source.

## Neutrality evidence

- The correction changes only the device placement of two probe forwards. Saved bytes, strict state digests, bounds, NLL values, fold envelope digests and the public scoring path are unchanged when the probe devices already match (the CPU case), and the CUDA case now evaluates the identical check instead of raising.
- The restored head is returned to its origin, so downstream code observes exactly the same module placement as before the probe.
- The runner's internal `task_id`/`attempt_id`/`agent_id` output labels remain the frozen source labels written when the runner was authored; they are not an execution-provenance claim.
- `n4_scale_scope`, `n4_recipe`, `locked_configs`, `config_sha256`, `common_contract`, `compute_envelopes`, `coefficient_selections` and `final_b0_binding` are carried forward unchanged except for the recomputed derived lock/core and arm-config digests required by the new version.

## Required gates before any normative run

1. Independent source-evidence review of this lock, the six pinned paths, the amendment, the correction record and the local nonnormative proof.
2. Separate Bronze exact-snapshot checkpoint of the reviewed source.
3. Non-force source push and isolated fast-forward sync.
4. Fresh runtime worker: real certified-DATA CUDA preflight + bounded smoke, then in-run capacity recertification and the three-seed final fit/q95/Development path. No normative CUDA run may occur before this gate.

No certified DATA, CUDA, capacity or full-runtime claim is made by this document or by the local nonnormative proof; the local host has a CPU-only build, so the CUDA regression case is honestly skipped here and must run in the post-gate runtime.
