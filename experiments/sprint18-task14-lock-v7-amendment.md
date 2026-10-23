# Sprint 18 Task 14 lock v7 — certified-capacity identity comparison

- **Status:** prospective candidate; **no self-approval**. Independent source-evidence review, a separate Bronze exact-snapshot checkpoint and non-force source/sync are required before any normative runtime.
- **Task / attempt:** `S18-T30` / `S18-T30-A10`
- **Base:** `7735669a8f603271e1b65a3caa68e09429eb2625` (`refs/heads/master`), parent `430cbbd4d5ced2f8bdc7926ff00f48bc0b124590`.
- **New lock:** [`experiments/sprint18-task14-config-lock-v7.json`](./sprint18-task14-config-lock-v7.json), schema `s18-task14-config-lock-v7`, lock version 7, file `69ac1218…`, core `c4d33157…`.
- **Preserved history:** the v6 prospective lock (`sprint18-task14-config-lock-v6.json`, raw `b4d1f435…`, core `5f65f907…`), the v5 prospective lock (raw `feeff6ab…`, core `e48455b0…`), the v4 prospective lock (raw `edbd65f3…`, core `3151a4ea…`), the v3 prospective lock (raw `5ce105b4…`, core `c6c3fe84…`), the v2 prospective lock (raw `6dd815c5…`, core `4b399874…`) and the v1 lock (raw `c4c1a8bd…`, core `6eb108b2…`) are byte-identical and are chained by digest, never edited.

## Why a new lock version

The v6 lock carries the reviewed A08 Fit-bank row-ordering correction and was independently reviewed (`S18T30SR05` PASS), but it pins the Task 30 runner bytes `4bb560e1…` and the lock-consumer bytes `e534fb08…`. Its review recorded one load-bearing defect in the *normative* capacity gate:

> `_run_capacity_folds` `matches_certified_task13b_fold_digest` compares `torch.save` container bytes against the certified fold digest.

The A08 proof itself documents the contradiction: for one identical payload, `.pt` container bytes vary with `torch.save` serialization while every semantic state digest is stable. The normative gate therefore tested **serialization identity**, not **scientific identity**: a scientifically identical recertification could be refused on container bytes, and container coincidence could never substitute for the payload comparison. Because the runtime re-executes the nine folds on the changed source before any final fit, this had to be corrected before the long recertification campaign.

The correction necessarily changes the Task 30 runner and the lock-consumer bytes, both pinned by v6. Editing v6 in place would rewrite a reviewed artifact; the established pattern is a new version that preserves the previous bytes and supersedes exactly the corrected scope. Hence v7.

## Serialization custody vs scientific identity (the corrected contract)

The contract is now explicit and two-part, and both parts are load-bearing:

1. **Custody — the historical artifact is authenticated.** The certified Task13B A02 record names each retained fold envelope by its documented custody root (`evidence.remote_run_root`, `/home/trietlm/s18-t13b-v3-certa02-runs/`) plus the certified relative path (`checkpoints/<fold>.pt`). The runtime hashes the historical file bytes and refuses unless they equal the certified container digest recorded for that fold (`folds[].checkpoint_sha256`). It then strict-loads that file through the canonical `s18-n4-scale-checkpoint-v2` contract and requires the recomputed canonical state digest to equal the certified payload digest (`folds[].scale_state_sha256`, also bound to `scale_checkpoints[].state_dict_sha256`). The historical raw digests are never rewritten and the historical file bytes are never replaced.
2. **Identity — the recertified fold equals the authenticated payload exactly.** The recertified canonical scale-state digest (`state_sha256(state_dict)`) and every declared identity field must equal the authenticated historical payload's values, with no tolerance, no loose equivalence and no missing-field fallback:

| Group | Fields |
|---|---|
| config | `checkpoint_schema`, `checkpoint_parameterization`, `scale_parameterization`, `guard_contract`, `scale_parameters` |
| bounds | `bounds.v_floor`, `bounds.v_cap`, `quantiles.q05`, `quantiles.q95` |
| guards / initialization | `init_interiority`, `first_step_gradients`, `init_anchor` |
| seed / fold / support | `model_seed`, `fold_index`, `scope`, `heldout_history_id`, `training_history_ids`, `record_count`, `heldout_count` |
| sampler | `epoch_order_seeds`, `order_seed_rule` |
| budget | `optimizer_steps`, `per_epoch_steps` |
| gate | `cap_gate_pass_train` |
| payload | canonical `state_dict` digest (`scale_state_sha256`) |

The **recertified** `torch.save` container digest is recorded per fold as a custody observation (`certified_container_matches`, aggregate `recertified_containers_equal_certified`) and never gates transfer. A container-value coincidence cannot pass: the payload digest and every identity field are still compared.

A changed weight, bound, support, guard, initialization, sampler seed/order, step budget or provenance value is a scientific difference and fails closed with the specific field named. An unreachable historical envelope is a stated prerequisite failure, never a silent pass and never a substituted digest.

## What changes

### 1. Certified-capacity identity comparison (the correction)

`experiments/sprint18_task30_n4_runner.py`:

- adds `_CERTIFIED_FOLD_IDENTITY_FIELDS`, the explicit closed identity set above;
- adds `_authenticate_certified_fold`, which resolves the historical envelope from the certified record, authenticates its raw bytes against the certified container digest, strict-loads it through `load_n4_scale_checkpoint`, authenticates the canonical payload digest, and cross-checks the envelope's own declared scope/history/parameterization/guard/bounds/initialization against the certified fold record;
- adds `_require_identical_certified_fold`, which refuses the first identity or payload-digest difference exactly;
- `_run_capacity_folds` now looks up the certified record for the fitted fold (seed + held-out history), compares under the normative path only, and records both custody and identity evidence per fold; the old container-equality gate is removed, not aliased;
- `_run_seed` passes the certified fold records (with their custody paths) instead of a digest list, and refuses a certified roster that does not match the Fit roster order;
- preflight verifies the seed × fold roster covers exactly the three Fit held-out histories before any fitting.

### 2. Lock consumer

`experiments/sprint18_task13b_n4_capacity.py` keeps loading only the current lock (no fallback) and now validates the complete **v7 → v6 → v5 → v4 → v3 → v2 → v1** digest chain. Every historical lock's bytes and core digest are pinned. The returned consumer record carries the superseded v6 identity plus `historical_v5_lock`, `historical_v4_lock`, `historical_v3_lock`, `historical_v2_lock` and `historical_v1_lock`, so runtime preflight records the full chain.

The consumer also projects the certified capacity evidence the runtime needs, after cross-checking it against the same SHA-pinned certification file:

- `envelope_root` — the documented custody root taken from the certified record (never hardcoded in source);
- `folds` — the nine certified fold records with custody path/container digest and the exact payload/identity fields, cross-checked against `scale_checkpoints` (path, container digest, payload digest, held-out history, optimizer steps, parameterization, schema);
- `identity_contract` — validated against the lock's declared `authenticated_historical_container_and_payload_with_exact_identity`, so the separation itself is load-bearing;
- the envelope path is refused unless it is a retained relative `checkpoints/…` path with no traversal.

Nothing was removed or relaxed.

### 3. Reviewed implementation pin set

The pin set remains exactly the six required paths; only the Task 30 runner pin and the capacity-runner identity are refreshed:

| Path | SHA-256 |
|---|---|
| `experiments/sprint18-context-residual-normalization-spec-v3.md` | `e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2` (unchanged) |
| `experiments/sprint18_task30_n4_runner.py` | `ad7fd8616aebd70390d32baeb7ad604d218d1ce07a45b98441349062609d92ba` (corrected) |
| `src/representation/__init__.py` | `9c3c857a76acebea08f4d6eebede4af7799024cdac2f7193c2d0601fdbcbf09b` (unchanged) |
| `src/representation/crn.py` | `7902dd9b7209b2ad9f7ef0f0e6f7316eb13c7db20237bc04974336b863bb9251` (unchanged) |
| `src/representation/sprint18_checkpoint.py` | `fce263df093bc622ba0a69bfac0824941fd6a0d1c4c9ab02bcbcff62432aaef2` (unchanged) |
| `tests/representation/test_sprint18_crn.py` | `2203d3f0b74be7ddb9f1f693b1321834b9078e86d990b6764996187218809186` (unchanged) |

`runner_identity` binds `experiments/sprint18_task13b_n4_capacity.py` at its cutover bytes `b439b508…`. The superseded v6 runner pin `4bb560e1…` is retained under `review.superseded_v6_implementation_pins`, alongside the already-preserved v5 (`19331f57…`), v4, v3 and v2 pins.

### 4. Tests

`tests/representation/test_sprint18_task14_lock_v6.py` is superseded by `tests/representation/test_sprint18_task14_lock_v7.py` (its A08 bytes are preserved verbatim in the A10 evidence root), which validates the v7 lock, the v7 → v6 → v5 → v4 → v3 → v2 → v1 chain, the refreshed pins, the superseded v6/v5 provenance, and the unchanged recipe/capacity/record contract; the stale-lock test refuses v6, v5, v4, v3, v2 and v1.

`tests/representation/test_sprint18_task30_runner.py` gains the behavioural certified-capacity regressions:

- `test_capacity_fold_certification_authenticates_payload_not_container_bytes` fits real tiny folds, writes their v3 envelopes, re-serializes the identical payload into a different `torch.save` container, and proves the recertification passes while every container digest differs and every payload digest matches;
- `test_capacity_fold_certification_refuses_scientific_tamper` proves a coherently re-digested envelope with changed weights, a changed bound quantile, a changed qualified-support count, substituted historical bytes and an unreachable custody path are each refused with the specific field/custody message;
- the existing fold restore/device test is migrated to the nonnormative comparison contract (no certified comparison in a bounded smoke).

## Historical source binding

- The v6 lock bytes `b4d1f435…` / core `5f65f907…`, the v5 lock bytes `feeff6ab…` / core `e48455b0…` and their implementation pins remain exactly as reviewed by `S18T30SR05` and authored by `S18T30A08`.
- The certified Task13B A02 capacity evidence `6169fa12…`, its nine fold container/payload digests, the v3 mechanism amendment, the locked Task 23 A04 ledger `3540ef09…`, the B0 epoch-100 checkpoints `9292d7c5…` / `8ae58da9…` / `79edbdf3…`, the raw cache identities `18345f86…` / `e035b2b7…` / `1d4a2a74…` and the A07 cleanup/custody records are unchanged. **No B0 rerun and no DATA change.**
- The retained historical envelopes (nine × 22,501 B) live only under the documented server custody root recorded in the certification file. Their raw digests (`ba0a7ff8…` for seed 181801 fold 0, etc.) are never rewritten.

## What does not change

- The frozen scientific recipe: 50 epochs, batch ≤512 patch records, `sprint18-n4-scale-order-v2`, AdamW 1e-3/1e-4, clip 1.0, Fit-only linear q05/q95 bounds, 20 % cap gate, zero representation updates, no early stopping or Development selection. No numeric, digest-meaning or guard change.
- The certified N4 scope: 66,989 eligible masked Fit records, per-history 22,328/23,098/21,563, folds 44,661/43,891/45,426, fold steps 4,400/4,300/4,450, final 6,550 steps.
- The v3 mechanism amendment and the v1/v2 specifications.
- The final B0 checkpoint/cache identities, the Fit/Calibration/Development rosters and role permissions.
- Every arm other than N4-CRN; the historical v1 lock continues to bind B0/K3/K7/K9/N1/N2/N3 and the B0 strict restores.
- The capacity certification remains `historical_certification_at_original_snapshot` at `f40667a`; the Task 30 runtime must still re-execute the three folds on the executed source and now compares them by scientific identity.

## Neutrality evidence

- The correction changes only how the already-required fold recertification is compared. The fold fits, bounds, sampler, budget and gate are untouched.
- The authenticated payload comparison is strictly stronger than the container-equality test for scientific tampering: equal container bytes with different payload content are impossible to construct through the canonical loader, while different container bytes with equal payload content are now accepted as intended.
- The custody path is taken from the SHA-pinned certification record, not from a CLI flag or source constant, so it cannot be repointed without changing the certified bytes.
- The runner's internal `task_id`/`attempt_id`/`agent_id` output labels remain the frozen source labels written when the runner was authored; they are not an execution-provenance claim.
- `n4_scale_scope`, `n4_recipe`, `locked_configs`, `config_sha256`, `common_contract`, `compute_envelopes`, `coefficient_selections` and `final_b0_binding` are carried forward unchanged except for the recomputed derived lock/core and arm-config digests required by the new version.

## Required gates before any normative run

1. Independent source-evidence review of this lock, the six pinned paths, the amendment, the correction record and the local nonnormative proof.
2. Separate Bronze exact-snapshot checkpoint of the reviewed source.
3. Non-force source push and isolated fast-forward sync.
4. Fresh runtime worker: real certified-DATA CUDA preflight (the new roster/custody checks execute there), bounded smoke, then in-run capacity recertification and the three-seed final fit/q95/Development path. The historical envelope custody root must exist on the runtime host; if it does not, the run fails closed with the exact prerequisite path instead of inventing digests.

The local proof is a tiny CPU-only mechanics proof (real model, real fold fits, real envelope save/strict restore, independent re-serialization, canonical bank/q95 and public N4 scoring); it is not certified-DATA, CUDA, capacity or full-runtime evidence.
