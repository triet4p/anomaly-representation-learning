# Sprint 18 Task 14 lock v8 — certified envelope custody path

- **Status:** prospective candidate; **no self-approval**. Independent source-evidence review, a separate Bronze exact-snapshot checkpoint and non-force source/sync are required before any normative runtime.
- **Task / attempt:** `S18-T30` / `S18-T30-A11`
- **Base:** `39b51d5475f14794caa98842162fb587491d7289` (`refs/heads/master`), parent `7735669a8f603271e1b65a3caa68e09429eb2625`.
- **New lock:** [`experiments/sprint18-task14-config-lock-v8.json`](./sprint18-task14-config-lock-v8.json), schema `s18-task14-config-lock-v8`, lock version 8, file `8258a439…`, core `e4482efb…`.
- **Preserved history:** the v7 prospective lock (`sprint18-task14-config-lock-v7.json`, raw `69ac1218…`, core `c4d33157…`), the v6 prospective lock (raw `b4d1f435…`, core `5f65f907…`), the v5 prospective lock (raw `feeff6ab…`, core `e48455b0…`), the v4 prospective lock (raw `edbd65f3…`, core `3151a4ea…`), the v3 prospective lock (raw `5ce105b4…`, core `c6c3fe84…`), the v2 prospective lock (raw `6dd815c5…`, core `4b399874…`) and the v1 lock (raw `c4c1a8bd…`, core `6eb108b2…`) are byte-identical and are chained by digest, never edited.

## Why a new lock version

The accepted v7 source authenticated the historical Task13B envelope and compared the recertified canonical payload and identity exactly. Its custody directory, however, was projected from the certification record as `evidence.remote_run_root`:

```
evidence.remote_run_root = /home/trietlm/s18-t13b-v3-certa02-runs/
fold.checkpoint_path      = checkpoints/<seed>-fold<n>-<history>.pt
joined                    = /home/trietlm/s18-t13b-v3-certa02-runs/checkpoints/<...>.pt
```

The certified A02 launch actually ran with `--out-dir "$RUN/full"` (retained in `run-pilot.sh`, `RUN=/home/trietlm/s18-t13b-v3-certa02-runs`), and the certification record captures that absolute output directory as `evidence.remote_full = /home/trietlm/s18-t13b-v3-certa02-runs/full/`. The nine retained envelopes therefore live at:

```
/home/trietlm/s18-t13b-v3-certa02-runs/full/checkpoints/<...>.pt
```

The v7 derivation was one directory too high, so the safe runtime preflight reported all nine files unreachable even though they exist with exactly the certified raw container digests. Because the corrected derivation and the new preflight authentication change the Task 30 runner and the lock-consumer bytes, both pinned by v7, the established pattern is a new version that preserves the previous bytes and supersedes exactly the corrected scope. Hence v8.

## Exact custody derivation (no scan, no fallback)

`certified_envelope_root` in `experiments/sprint18_task13b_n4_capacity.py` now derives the directory only from the SHA-pinned certification record:

1. `evidence.remote_run_root` and `evidence.remote_full` are both required absolute, slash-terminated, traversal-free directory strings. The runtime record is POSIX; the validation accepts an absolute directory in the host's native form so local Windows proofs exercise the same code.
2. The output directory must be a strict child of the recorded run root (`remote_full.startswith(remote_run_root)` and not equal to it).
3. Every `evidence.remote_retained_server_side` entry must live under that child (`full/checkpoints/*.pt ...` for this record).
4. The exact recorded output directory string is returned. There is no directory scan, no fallback to `remote_run_root`, no CLI repointing and no record rewrite; a missing or inconsistent field fails closed.

The runner then joins the returned directory with the certified relative `checkpoints/...` path. The nine files and the certified raw container/payload digests are unchanged; only the storage directory is corrected. The envelope root is not part of the identity-field set and was never part of the scientific comparison, so the recipe, bounds, guards, support, sampler, budget and payload comparison remain identical.

## Nine-file preflight authentication

`run_preflight` now authenticates the complete seed × fold roster before any fitting:

- the roster must be exactly the nine certified `(model_seed, fold_index, heldout_history_id)` records matching the Fit roster;
- each historical envelope must exist at the derived exact path, authenticate against its certified raw container digest, strict-load through `s18-n4-scale-checkpoint-v2`, authenticate its canonical `state_dict_sha256` against the certified payload digest, and match the closed identity record;
- the accepted custody observations (`envelope_path`, `container_sha256`, `payload_sha256`) are recorded in the preflight payload under `capacity_custody_authentication`.

A missing, substituted or wrongly rooted envelope fails before the long Fit campaign instead of after expensive fits. `_run_seed` re-authenticates its own three folds before recertifying them, using the same path helper.

## What changes

`experiments/sprint18_task13b_n4_capacity.py`:

- adds `certified_envelope_root` and loads the v8 lock only (no fallback), validating the complete **v8 → v7 → v6 → v5 → v4 → v3 → v2 → v1** digest chain, with every historical lock's bytes and core digest pinned;
- returns the superseded v7 identity plus `historical_v6_lock`, `historical_v5_lock`, `historical_v4_lock`, `historical_v3_lock`, `historical_v2_lock` and `historical_v1_lock`;
- projects `envelope_root` from `evidence.remote_full` (the recorded launch output directory) instead of `evidence.remote_run_root`.

`experiments/sprint18_task30_n4_runner.py`:

- adds `_certified_fold_records` (the only path derivation: certified root + certified relative path);
- adds `_authenticate_certified_folds` and calls it in `run_preflight` before any fitting, recording the nine custody observations;
- `_run_seed` uses the same helper and keeps its per-seed roster check.

Nothing about the recipe, bounds, guards, sampler, budget, capacity criterion or the certified identity-field set changed.

## Reviewed implementation pins

The pin set remains the same six required paths; only the Task 30 runner pin is refreshed (and the capacity-runner identity):

| Path | SHA-256 |
|---|---|
| `experiments/sprint18-context-residual-normalization-spec-v3.md` | `e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2` (unchanged) |
| `experiments/sprint18_task30_n4_runner.py` | `a1110f77b06a56c452269fd468b65678b8cf9623c37df319772fee64a1bd4453` (corrected) |
| `src/representation/__init__.py` | `9c3c857a76acebea08f4d6eebede4af7799024cdac2f7193c2d0601fdbcbf09b` (unchanged) |
| `src/representation/crn.py` | `7902dd9b7209b2ad9f7ef0f0e6f7316eb13c7db20237bc04974336b863bb9251` (unchanged) |
| `src/representation/sprint18_checkpoint.py` | `fce263df093bc622ba0a69bfac0824941fd6a0d1c4c9ab02bcbcff62432aaef2` (unchanged) |
| `tests/representation/test_sprint18_crn.py` | `2203d3f0b74be7ddb9f1f693b1321834b9078e86d990b6764996187218809186` (unchanged) |

`runner_identity` binds `experiments/sprint18_task13b_n4_capacity.py` at `471de1d5…`. The superseded v7 runner pin `ad7fd861…` is retained under `review.superseded_v7_implementation_pins`, alongside the already-preserved v6 (`4bb560e1…`), v5 (`19331f57…`), v4, v3 and v2 pins.

## Tests

`tests/representation/test_sprint18_task14_lock_v7.py` is superseded by `tests/representation/test_sprint18_task14_lock_v8.py`, which validates the v8 lock, the full v8 → v7 → v6 → v5 → v4 → v3 → v2 → v1 chain, the refreshed pins, the superseded v7 provenance, the certified envelope root derivation from the real certification evidence, and the unchanged recipe/capacity/identity contract. Stale-lock refusal covers v7, v6, v5, v4, v3, v2 and v1.

`tests/representation/test_sprint18_task30_runner.py` adds real filesystem regressions:

- `test_certified_envelope_root_uses_the_recorded_output_directory` builds the nested `run/full/checkpoints` layout, proves the derived root is the recorded output directory, and proves the old run-root derivation fails closed with the exact missing path;
- `test_preflight_authenticates_all_nine_certified_envelopes` authenticates all nine real envelopes, then refuses a substituted file, a coherent-content swap without the certified payload digest, and a missing envelope before any fitting.

The existing capacity identity acceptance/refusal tests and the A08 bank/q95 regressions are retained.

## Historical source binding

- The v7 lock bytes `69ac1218…` / core `c4d33157…`, the v6 lock bytes `b4d1f435…` / core `5f65f907…` and all earlier locks remain exactly as reviewed and authored.
- The certified Task13B A02 capacity evidence `6169fa12…`, its nine fold container/payload digests, the v3 mechanism amendment, the locked Task 23 A04 ledger `3540ef09…`, the B0 epoch-100 checkpoints, the raw cache identities and the A07 cleanup/custody records are unchanged. **No B0 rerun and no DATA change.**
- The retained historical envelopes (nine × 22,501 B) live only under the recorded server output directory `evidence.remote_full`; their raw digests are never rewritten.

## What does not change

- The frozen scientific recipe, the certified N4 scope, the v3 mechanism amendment, the v1/v2 specifications, the final-B0/cache identities, the role rosters and permissions.
- The capacity recertification identity comparison (closed 22-field set plus canonical payload digest).
- Every arm other than N4-CRN; the historical v1 lock continues to bind B0/K3/K7/K9/N1/N2/N3.

## Required gates before any normative run

1. Independent source-evidence review of this lock, the six pinned paths, the amendment, the correction record and the local proof.
2. Separate Bronze exact-snapshot checkpoint of the reviewed source.
3. Non-force source push and isolated fast-forward sync.
4. Fresh runtime worker: real certified-DATA CUDA preflight (the nine-file custody authentication executes there) before any fitting, then the bounded smoke and the full campaign. The historical envelope output directory must exist on the runtime host; if it does not, the run fails closed with the exact prerequisite path instead of scanning or inventing digests.

The local proof is a tiny CPU-only mechanics proof (real model, real fold fits, real envelope save/strict restore, real nested `full/checkpoints` layout, canonical bank/q95 and public N4 scoring); it is not certified-DATA, CUDA, capacity or full-runtime evidence.
