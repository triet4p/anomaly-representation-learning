# Sprint 18 Task 14 lock v3 — N4-CRN implementation-snapshot and in-scope record amendment

- **Status:** prospective candidate; **no self-approval**. Independent source-evidence review, a separate Bronze exact-snapshot checkpoint and non-force source/sync are required before any normative runtime.
- **Task / attempt:** `S18-T30` / `S18-T30-A02`
- **Base:** `83710aa65d201cf3f37d7f9961368589932f6ec9` (`refs/heads/master`), parent `ec2d13a2e6cca3846be2a1f66eb00b83bbefe0cc`.
- **New lock:** [`experiments/sprint18-task14-config-lock-v3.json`](./sprint18-task14-config-lock-v3.json), schema `s18-task14-config-lock-v3`, lock version 3.
- **Preserved history:** the v2 prospective lock (`sprint18-task14-config-lock-v2.json`, raw `6dd815c5…`, core `4b399874…`) and the v1 lock (`sprint18-task14-config-lock.json`, raw `c4c1a8bd…`, core `6eb108b2…`) are byte-identical and are chained by digest, never edited.

## Why a new lock version

The v2 lock pins the reviewed implementation snapshot at `44f56b1`. Accepted Task 26 commit `b31c91b` later added 18 lines and deleted 0 lines in `src/representation/__init__.py` (K9 downstream exports in the import block and `__all__`). The reviewed commit remains an ancestor of every T23–T30 base, and `crn.py`, `sprint18_checkpoint.py` and `test_sprint18_crn.py` were unchanged at the T30 base, but the fail-closed v2 activation predicate refused every later base because one pinned byte path had legitimately advanced. Replacing the predicate with recipe-only acceptance, re-pinning incidental tests, or running under an invalid lock would each void the campaign.

The v3 lock therefore supersedes **only**:

1. the reviewed implementation-snapshot pins, and
2. the N4 in-scope record contract that the coherent-checkpoint consumer and the result schema actually validate.

## What changes

### Reviewed implementation scope

The lock binds these exact worktree bytes under `n4_v3_activation.reviewed_implementation.paths`:

- `experiments/sprint18-context-residual-normalization-spec-v3.md` (unchanged v3 mechanism amendment)
- `experiments/sprint18_task30_n4_runner.py` (new Task 30 runner)
- `src/representation/__init__.py` (current accepted bytes, including the additive Task 26 K9 exports)
- `src/representation/crn.py` (certified record-scope constants and the optional caller-owned AdamW hook)
- `src/representation/sprint18_checkpoint.py` (N4 count/step validators)
- `tests/representation/test_sprint18_crn.py` (coherent-checkpoint regression coverage)

The validator requires **exactly** this pin set: dropping, adding, or changing a pin is refused. `runner_identity` binds `experiments/sprint18_task13b_n4_capacity.py` at the observed bytes.

### In-scope record contract

An N4 training record is one **eligible masked valid Fit patch record**, not every valid Fit patch:

| Quantity | Value |
|---|---:|
| Fit valid patches (out of scope) | 167,419 |
| N4 eligible masked Fit records | **66,989** |
| Per history FIT-01 / FIT-02 / FIT-03 | 22,328 / 23,098 / 21,563 |
| Two-history fold counts | 44,661 / 43,891 / 45,426 |
| Final 50-epoch optimizer steps | **6,550** |
| Fold optimizer steps | 4,400 / 4,300 / 4,450 |

The counts are frozen once in `representation.crn` (`N4_SCALE_HISTORY_RECORD_ROSTER`, `N4_SCALE_FIT_RECORDS`, `n4_scale_scope_steps`) and consumed by the lock validator, the coherent-checkpoint sampler/arm validators and the full-training result schema. Normative runners re-derive them from the bound Task23 A04 prediction caches before any fit and abort on any difference. The coherent arm state additionally cross-checks its final count against the sampler's eligible rows and each fold count against the certified history split, so the stale 167,419 / 16,350 scope is refused, not silently accepted.

### Normative artifact and next-stage contract

The normative N4 artifact is a per-seed coherent `sprint18-coherent-checkpoint-v2` checkpoint that binds:

- the exact paired final epoch-100 B0 online/EMA bytes and checkpoint digest,
- the fitted v3 scale head (`fit-mean-anchored-bias-v1`, `n4-v3-guards-v1`, `s18-n4-scale-checkpoint-v2`),
- the three recertified Fit-history fold state digests,
- the certified in-scope record counts and 6,550 final steps,
- the Fit-only q05/q95 bounds and residual-record digest,
- zero representation steps and the separate `n4_scale` optimizer scope.

The v3 scale envelope remains a retained artifact. The runner strict-restores the coherent artifact and re-scores the complete Calibration support through the public `n4_score_files` consumer, reproducing the live N4 `S_pred` q95 exactly; the `S_pop` threshold remains the accepted B0 threshold. It also re-scores a bounded Development slice identically. Each seed record carries `capacity_recertification`, `coherent_checkpoint`, `restored_artifact_proof` and `next_stage` blocks for Tasks 31 and 33.

### Capacity-proof applicability

The Task13B A02 nine-fold PASS (`experiments/sprint18-task13-n4-capacity-v3-a02.json`, `6169fa12…`) is **historical certification at its original executed snapshot** (`f40667a`, under the v2 lock); the lock carries `status = historical_certification_at_original_snapshot` and `runtime_revalidation = pending_source_gate_then_in_run_fold_recertification`. **It is not transferred to the changed source.** The lock binds the certified evidence and its exact fold checkpoint digests, and the full runner must re-execute all three folds per seed *before* the final fit. Each recertified fold envelope must reproduce the certified digest exactly; a mismatch stops the run. Bounded smoke runs exercise the same fold code without the digest equality gate and are explicitly non-normative.

## What does not change

- The frozen scientific recipe: 50 epochs, batch ≤512 patch records, `sprint18-n4-scale-order-v2`, AdamW 1e-3/1e-4, clip 1.0, Fit-only linear q05/q95 bounds, 20 % cap gate, zero representation updates, no early stopping or Development selection.
- The v3 mechanism amendment and v1/v2 historical specifications.
- The final B0 checkpoint/cache identities, Fit/Calibration/Development rosters and role permissions.
- Every other arm: the historical v1 lock continues to bind B0/K3/K7/K9/N1/N2/N3 and the B0 strict restores.
- Historical evidence, failure dispositions and the v2 `reviewed_implementation` pins (retained under `review.neutrality_review.superseded_v2_implementation_pins`).

## Neutrality evidence

- `git diff 44f56b1..83710aa -- src/representation/__init__.py` is 18 insertions / 0 deletions: K9 downstream imports and `__all__` entries only.
- `src/representation/crn.py` changes freeze the already-certified record counts and add an optional caller-owned AdamW parameter to `fit_n4_scale_scope`; the default path and every numerical operation are unchanged.
- `src/representation/sprint18_checkpoint.py` and `src/representation/sprint18_harness.py` changes correct N4 count/step expectations only; they add cross-field consistency checks and do not alter the recipe.
- The N4 sampler seed depends only on model seed, scope, sorted Fit histories and epoch, not on the lock core digest, so fold numerics remain reproducible across the lock-version change.

## Required gates before any normative run

1. Independent source-evidence review of the v3 lock, the six pinned paths, this amendment and the worker records.
2. Separate Bronze exact-snapshot checkpoint of the reviewed source.
3. Non-force source push and isolated fast-forward sync.
4. Fresh runtime worker: real certified-DATA CUDA preflight + bounded smoke, then in-run capacity recertification and the three-seed final fit/q95/Development path.

No certified DATA, CUDA, capacity or full-runtime claim is made by this document or by the local nonnormative smoke.
