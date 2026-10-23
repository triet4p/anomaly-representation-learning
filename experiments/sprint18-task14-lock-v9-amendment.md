# Sprint 18 Task 14 lock v9 — Task 30 N4 compute-probe record contract

- **Status:** prospective candidate; **no self-approval**. Independent source-evidence review, a separate Bronze exact-snapshot checkpoint and non-force source/sync are required before any normative runtime.
- **Task / attempt:** `S18-T30` / `S18-T30-A13`
- **Base:** `1d9a646e9f57051fdc582e950919905f1dd70326` (`refs/heads/master`), parent `39b51d5475f14794caa98842162fb587491d7289`.
- **New lock:** [`experiments/sprint18-task14-config-lock-v9.json`](./sprint18-task14-config-lock-v9.json), schema `s18-task14-config-lock-v9`, lock version 9, file `a9990bd189706a0fdadfeaee422ef307599e3f4d9c5a72102dfa1f015d5cd57a`, core `b171e50014a9bfad0edc865fc42a52a49d073b8b0a2e47788d5020b741266bbf`.
- **Preserved history:** the v8 prospective lock (`sprint18-task14-config-lock-v8.json`, raw `8258a439…`, core `e4482efb…`), the v7 prospective lock (raw `69ac1218…`, core `c4d33157…`), the v6 prospective lock (raw `b4d1f435…`, core `5f65f907…`), the v5 prospective lock (raw `feeff6ab…`, core `e48455b0…`), the v4 prospective lock (raw `edbd65f3…`, core `3151a4ea…`), the v3 prospective lock (raw `5ce105b4…`, core `c6c3fe84…`), the v2 prospective lock (raw `6dd815c5…`, core `4b399874…`) and the v1 lock (raw `c4c1a8bd…`, core `6eb108b2…`) are byte-identical and are chained by digest. No historical lock file was edited, re-sealed or moved.

## Why a new lock version

The A12 bounded certified-DATA CUDA smoke reached the Task 30 compute block after the B0 strict restore, the three bounded capacity folds, the bounded final fit, the scale-envelope save/restore, the canonical Fit-bank gate, Calibration q95 and the Development slice had all passed, and then failed closed at the N4 training-FLOPs probe:

```
runner.py:2498 _run_seed -> :1958 _compute_block -> :1959 <lambda> fit_n4_scale_scope
-> crn.py:746 ValueError: N4 fit records must be N4ScaleRecord entries
```

`_compute_block` built its bounded probe as `probe_records = list(fit_rows)[:512]`, but `_run_seed` passes `fit_rows=support["fit_rows"]` — a tuple of certified Fit **file** rows (`CertifiedSample`, which carry `.sample`), not `N4ScaleRecord` entries (`history_id/file_id/patch_index/context/residual`). The same argument is legitimately used for the file-level probes (`probe_row.sample` forward and the 128-file paired B0 step), so one input cannot serve both roles. The already-present correct helper `_probe_records(pools, fit_history_ids, limit=512)` (already used for the scale-envelope probe) was not passed to the compute block.

The identical line exists in the A07 source (`7735669a…`), so this is a **pre-existing latent defect** first reached by A12 because the A07 smoke stopped earlier at the bank gate. It is deterministic and unconditional in `_run_seed`, and therefore would also have failed the full campaign at the same point. The corrected runner bytes and the cutover lock-consumer bytes are both pinned by the v8 lock, so the established pattern applies: a new version that preserves the previous bytes and supersedes exactly the corrected scope. Hence v9.

## Exact correction

`experiments/sprint18_task30_n4_runner.py`:

- `_compute_block` takes a new **required** `scale_records` argument and fits its bounded 512-record probe on those rows (`probe_records = list(scale_records)[:512]`); it no longer reads the N4 probe from `fit_rows`;
- `fit_rows` remains the certified Fit file-row input for `probe_row`/`masking_seed` and for the 128-file paired B0 training step — the two inputs are now separate and type-correct;
- `_run_seed` passes `scale_records=_probe_records(pools, fit_history_ids)`, the same cache-derived `N4ScaleRecord` sequence already consumed by the final fit, the fold fits and the scale-envelope probe.

Nothing else in the compute block changed: `profile_inference_flops` still executes the fit under the profiler on the disposable `probe_scale` head, `probe_steps = 50 · ⌈512/512⌉ = 50`, `n4_scale_fit_flops_total = probe_fit_flops / 50 × final_steps`, `n4_training_flops_per_fit_file = total / 5,446`, the paired B0 reference is the same real 128-file `train_sprint18_epoch` step scaled by `100 · ⌈5,446/128⌉`, and `validate_compute_envelope` still applies the frozen N4 limits (`n4_scale_parameters ≤ 100,000`, `S_pop` parameters equal to B0, `S_pred ≤ B0 + 4,417`, combined inference FLOPs ≤ ⌈1.10 × B0⌉, `training_flops_ratio ≤ 0.25`). No guard, tolerance, digest, budget, scope, recipe or recipe constant was relaxed.

The same block was inspected for correlated input-type/accounting errors: `probe_row` is (and remains) a `CertifiedSample`; `fit_history_ids` is a `list[str]`; `bounds` carries the frozen float edges; `final_steps` is the completed final-fit `optimizer_steps`; `prof_b0_model` is moved to the run device by `RepresentationTrainer.__init__`, so no device-mismatch class remains in this block. No other caller passes file rows to an N4 scale consumer.

## What changes

`experiments/sprint18_task30_n4_runner.py`: the compute-probe record contract above (required `scale_records`, call-site wiring).

`experiments/sprint18_task13b_n4_capacity.py`: the versioned lock consumer moves to v9 only — it loads `sprint18-task14-config-lock-v9.json` (no fallback), validates the complete **v9 → v8 → v7 → v6 → v5 → v4 → v3 → v2 → v1** digest chain with every historical lock's bytes and core digest pinned, returns the superseded v8 identity plus `historical_v7_lock` … `historical_v1_lock`, and refuses every stale schema/version. The certified-envelope-root derivation, the capacity identity contract and the scientific constants are unchanged.

`tests/representation/test_sprint18_task14_lock_v9.py` supersedes `tests/representation/test_sprint18_task14_lock_v8.py` (clean cutover; the v8 lock bytes remain pinned and unchanged). It validates the v9 lock, the full v8 → v7 → v6 → v5 → v4 → v3 → v2 → v1 chain, the refreshed pins, the superseded v8/v7 provenance, the certified envelope root derivation, stale-lock refusal for v8 … v1, and the unchanged recipe/capacity/identity contract.

`tests/representation/test_sprint18_task30_runner.py` adds the compute-block regressions (below).

`CHANGELOG.md`: one `[Unreleased] / Fixed` entry naming the compute-probe contract, the observed error and the unchanged accounting.

## Reviewed implementation pins

The pin set remains the same six required paths; only the Task 30 runner pin is refreshed:

| Path | SHA-256 |
|---|---|
| `experiments/sprint18-context-residual-normalization-spec-v3.md` | `e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2` (unchanged) |
| `experiments/sprint18_task30_n4_runner.py` | `3af994ba002b5eb79599858ba90ceff18e099accdd79b28c0f1d63a31caaff80` (corrected) |
| `src/representation/__init__.py` | `9c3c857a76acebea08f4d6eebede4af7799024cdac2f7193c2d0601fdbcbf09b` (unchanged) |
| `src/representation/crn.py` | `7902dd9b7209b2ad9f7ef0f0e6f7316eb13c7db20237bc04974336b863bb9251` (unchanged) |
| `src/representation/sprint18_checkpoint.py` | `fce263df093bc622ba0a69bfac0824941fd6a0d1c4c9ab02bcbcff62432aaef2` (unchanged) |
| `tests/representation/test_sprint18_crn.py` | `2203d3f0b74be7ddb9f1f693b1321834b9078e86d990b6764996187218809186` (unchanged) |

`runner_identity` binds `experiments/sprint18_task13b_n4_capacity.py` at `02e0d1096ed1115a95710b3152b14937e2d78419781e53b6fbeb02ad742864a8` (its bytes at the reviewed source commit `1d9a646e…` are `471de1d5…`). The superseded v8 runner pin `a1110f77…` is retained under `review.superseded_v8_implementation_pins`, alongside the preserved v7 (`ad7fd861…`), v6 (`4bb560e1…`), v5 (`19331f57…`), v4 (`f24e825e…`) and v3 (`e9237327…`) pins.

## Tests

Real CPU execution of the actual `_compute_block` (the function the runtime failed in), on a real `V1RepresentationModel` (public collator + forward), a real `ContextResidualScaleModel`, real `CertifiedSample`/`FileSample` rows and real `N4ScaleRecord` pools; CUDA is parametrized and honestly skipped when unavailable.

- `test_compute_block_probes_certified_scale_records_end_to_end[cpu]` — the actual 512-record probe, paired B0 step, profiler and envelope gate run end to end; assertions cover `envelope_pass`, `n4_scale_fit_steps == 6550`, `n4_scale_parameters == 4417`, positive finite FLOPs, the exact accounting identity `total == per_file × 5,446`, the returned training ratio, and the parameter/FLOP envelope limits.
- `test_compute_block_refuses_fit_file_rows_in_the_scale_probe` — the A12 failure mode: passing the certified Fit file rows as `scale_records` still fails closed with `N4 fit records must be N4ScaleRecord entries`.

Observed local proof (CPU, deliberately small `V1Config`: `d_model=128`, `sequence_layers=1`, frozen arm geometry; 512-record probe; `final_steps=6550`; captured run `artifacts/sprint-18/s18-t30-a13/compute-proof.json`): `S_pred` parameters 603,203 vs paired B0 598,786 (difference exactly the 4,417 scale parameters), `S_pop` 234,240 in both branches, inference FLOPs per Fit file 14,445,694 (`S_pred`) and 16,174,624 (`S_pop`/B0), combined inference ratio 0.9466, N4 scale-fit total 84,643,852,680 FLOPs (512-record probe, 50 probe steps, scaled to the 6,550 completed final-fit steps), per-Fit-file 15,542,389.4 vs B0 8,231,209,643.3, `training_flops_ratio_vs_B0 = 1.89e-03`, `envelope_pass = true`. With the pre-fix source the same call reproduced the exact A12 `ValueError` in 2.35 s.

Related observation (pre-existing, not changed here): `n4_score_files` executes its forward under `torch.no_grad()`, so the profiler reports 14,319,904 FLOPs for that forward versus 16,174,624 for the grad-enabled B0 forward on identical input (11.5 % lower). The measured N4 `S_pred` probe is therefore 10.7 % below paired B0; the frozen combined limit (≤ ⌈1.10 × B0⌉) still holds at 0.9466, and inference FLOPs are profile measurements in both arms. This asymmetry predates the lock chain, is unchanged by this correction and is recorded for the independent review rather than silently normalized.

## Historical source binding

- The v8 lock bytes `8258a439…` / core `e4482efb…`, the v7 lock bytes `69ac1218…` / core `c4d33157…` and all earlier locks remain exactly as reviewed and authored.
- The certified Task13B A02 capacity evidence `6169fa12…`, its nine fold container/payload digests, the v3 mechanism amendment, the locked Task 23 A04 ledger `3540ef09…`, the B0 epoch-100 checkpoints, the raw cache identities and the A07 cleanup/custody records are unchanged. **No B0 rerun and no DATA change.**
- The retained historical envelopes (nine × 22,501 B) live only under the recorded server output directory `evidence.remote_full`; their raw digests are never rewritten.

## What does not change

- The frozen scientific recipe, the certified N4 scope (66,989 records; 6,550 final / 4,400-4,300-4,450 fold steps), the v3 mechanism amendment, the v1/v2 specifications, the final-B0/cache identities, the role rosters and permissions.
- The capacity recertification identity comparison (closed 22-field set plus canonical payload digest) and the certified envelope custody derivation.
- Every arm other than N4-CRN; the historical v1 lock continues to bind B0/K3/K7/K9/N1/N2/N3.

## Required gates before any normative run

1. Independent source-evidence review of this lock, the six pinned paths, the amendment, the correction record and the local proof.
2. Separate Bronze exact-snapshot checkpoint of the reviewed source.
3. Non-force source push and isolated fast-forward sync.
4. Fresh runtime worker: real certified-DATA CUDA preflight, then the bounded smoke that must pass **through `compute`** with `envelope_pass == true`, `n4_scale_fit_steps == 6550`, a finite positive `training_flops_ratio_vs_B0 ≤ 0.25` and an emitted record, before the normative three-seed campaign.

## Local proof limits

The local proof is a tiny CPU-only mechanics proof: it drives the real `_compute_block`, the real public collators/forward/fit/profiler and the real envelope gate, and it reproduces the exact pre-fix failure, but it is **not** certified-DATA, CUDA, capacity or full-runtime evidence. The actual `_run_seed` bounded smoke cannot be run from this source-only attempt: the certified DATA root, the accepted B0 epoch-100 checkpoints/caches and the Task 23 A04 ledger are server-only paths, and this attempt is not authorized to upload source, commit, push or execute on the runtime host. The bounded smoke and the final envelope values therefore remain runtime gates for the next attempt.

- **Residual runtime risk, stated honestly:** the probe's initialized-interiority guard depends on the certified probe contexts and on the frozen bounds; it cannot be exercised with certified data locally. The 512-record prefix was already fitted successfully inside the A12 smoke's 768-record bounded final fit under the same frozen bounds, but the prefix is re-anchored by its own residual mean, so the next runtime attempt must confirm the probe guard (`N4ScaleGuardError` would fail closed with diagnostics, as designed).
