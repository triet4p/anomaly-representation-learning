# Sprint 18 certified data handoff — candidate `S18-ITER-0005` (assessment roster)

**Task / attempt / agent:** `S18-T71` / `S18-T71-A01` / `S18Task71A01`
**Role:** explicit `gold-task` — handoff publication and bounded read-only consumer smoke only. No generation, no training, no refit, no checkpoint, no DATA-batch verdict.
**Published:** 2026-10-09 (UTC) on the isolated runtime and the local evidence root `F:/ai-ml/anomaly-representation-learning`.
**Machine-readable companion:** `experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json` (schema `s18-t71-certified-data-handoff-v1`).

This handoff publishes the **already-certified** whole-16 Sprint 18 data roster. It does not re-run science: whole-16 preflight (16/16), 12/12 non-Confirmation materialization/qualification, 4/4 Confirmation materialization under the single user-authorized zero-output recovery, the fixed observable EG4 probe and the 18/18 whole-16 certification are established evidence from `S18-T69-A09` / `S18-T70-A03`, independently reviewed PASS (`agent://S18Task69R10`, `agent://S18Task70R03`) and checkpointed (`7be574d9…` CP03, `b032ef8b…` CP04 + DEP04 transport).

## 1. Where the data is

| Item | Value |
|---|---|
| Candidate | `S18-ITER-0005` (amended-policy assessment; seeds `32064–32079`) |
| Qualified DATA root — absolute (isolated runtime) | `/home/trietlm/anomaly-representation-learning-s18t68-a02-worktree/data/generated/sprint18-iterative-v1/S18-ITER-0005-ASSESS-P-ALLOWANCE-V1` |
| Qualified DATA root — relative to isolated worktree | `data/generated/sprint18-iterative-v1/S18-ITER-0005-ASSESS-P-ALLOWANCE-V1` |
| Root entries | `_assessment-001/`, `DESIGN/` (4), `FIT/` (3), `CALIBRATION/` (1), `DEVELOPMENT/` (4), `CONFIRMATION/` (4) |
| Root resolution note | The path was resolved against the actual isolated worktree root. The system path `/data` does **not** exist on `di-server` (verified) and no system mount scan was performed; `/data/generated/...` in the tasking is this same worktree-relative root expressed absolutely. |
| Immutable old failed root | `_attempt-001/` (original `PREFLIGHT-FAIL 15/16`, `candidate_rejected`) — preserved byte-immutable and distinct from the certified `_assessment-001/` attempt; marker `40e2cf3b…`, ledger `1579d9b6…`, raw `9f28471e…` |
| On-disk size | 1.8 GiB total (DESIGN 442 MiB, FIT 332 MiB, CALIBRATION 110 MiB, DEVELOPMENT 444 MiB, CONFIRMATION 442 MiB, `_assessment-001` 148 KiB); 1,628 shards; 1,648 files; 16 `manifest.json` |
| Dataset custody | Full waveforms stay server-side under the isolated worktree; nothing is committed or bulk-transferred. This handoff publishes source/outcome/manifest identities. |

## 2. Source, config and lineage identity

| Item | Value |
|---|---|
| Deliverable working base (local == origin == isolated HEAD) | `b032ef8b68fcf0cd6cfd48b2d4784cdcc8a5002c` (tree `80c21ec5e1bad0f89f324e9f09ea53abb059e62a`), ref `refs/heads/master` |
| Execution source (unchanged) | `8485344a2ee3143858db68fdbad3a6ed3def8890` (E) |
| Ancestry | A qualified-DATA/science `003b94bc…` → B Task69 outcome `7be574d9…` → C corrected execution `9e9e2227…` → D reviewed execution base `42df4249…` → E execution source `8485344a…` → F outcome checkpoint `b032ef8b…` (current deliverable base) |
| Binding (file, raw SHA-256) | `experiments/sprint18-iterative-assessment-c5-allowance-v1.json` — `b8448292b023c9268f64886e5ca68a393c36ff5f2c41573ff160d50ce8ee6c58` |
| Binding (canonical SHA-256) | `8e2705140d07d9c603989ec709de5ebb766e580a8619b37c7462f6dd2751dd3b` |
| Source closure SHA-256 | `7528315eb72efaf42e3bcddb77e69bc3b66cbacd2c420ed205321a7d59f388a2` |
| Generator profile / protocol / contract | `sprint18-iterative-v3` / `sprint15-benchmark-protocol-v7` / `sprint18-iterative-data-contract-v3` (`544de84b…`) |
| Generator version (in every manifest) | `2.0.0` |
| Chronicle manifest schema | `format = 1` (`synth.chronicle.CHRONICLE_FORMAT`) |
| Qualification record | `c5a8e8ccbc6be35a58fe7363ee4e8af489d3fafb93aaf6411c3936e5d349dc4b` — `experiments/sprint18-task69-assessment-S18-ITER-0005-A09-qualification-record.json` |
| Gates | Task69 `agent://S18Task69R10` PASS (snapshot `3458fba4…`) + CP03 `7be574d9…`; Task70 `agent://S18Task70R03` PASS (snapshot `fb5f38b2…`) + CP04 `b032ef8b…` + DEP04 transport (ordinary non-force push, exact-SHA runtime fast-forward) |
| Whole-16 certification | `1345d1c8e04c35f3fbb18bddec2b5d474f91ec24c1f1c5081040b342debe2897` — 18/18 checks, 0 failures |
| Role reconciliation | `13a1a18b276314a065d149703170985771adc8eeceae2f4fad27fe2752b49b95` — 13/13 conditions, 0 failures |
| Frozen statistics | `d81925f0442040b00f4180eed8c8dbef5acd5f3a13e8aaf28cbda6486bcf3b79` |

## 3. Role roster — actual per-role metadata from the materialized files

Role identity is published on two layers and kept distinct: the **assigned scientific role + permitted_use** from the frozen binding, and the **actual loader manifest value**, which carries the full `history_id` (observed, not rewritten). Shard rosters live inside each per-role `manifest.json` (`path/start/end/count/file_ids/sha256`) and are linked, not copied.

| Assigned role | `history_id` (= manifest `role` value) | Seed | `config_hash` | Samples | Shards | `manifest.json` SHA-256 |
|---|---|---:|---|---:|---:|---|
| DESIGN | `S18I-ITER-0005-DESIGN-01` | 32064 | `aae5e52747a7` | 6,484 | 102 | `6efabe14e35a429221e61e4c1f27440d3e535e34ff59882abb3b12d33976643c` |
| DESIGN | `S18I-ITER-0005-DESIGN-02` | 32065 | `184e601edc70` | 6,484 | 102 | `c0b3386d4c9e21e539a4ff65f7e5fec61a775e5460d21b2bb311173b20c1103e` |
| DESIGN | `S18I-ITER-0005-DESIGN-03` | 32066 | `4eae0187d333` | 6,477 | 102 | `b21ab241abd7349c2f58d9e5b9390be0183f2ee004ad54dc436682192fe84e3d` |
| DESIGN | `S18I-ITER-0005-DESIGN-04` | 32067 | `eba370f48dc3` | 6,483 | 102 | `080dc31254976a9704c7d8705fa496f2ee436747aebbad123dc17dd6ffe00066` |
| FIT | `S18I-ITER-0005-FIT-01` | 32068 | `2d7524242fd4` | 6,490 | 102 | `fb0d533c10f1a650647b95d06c87559f244f09a31973c75a35329fd8a04d19dc` |
| FIT | `S18I-ITER-0005-FIT-02` | 32069 | `8d51d85bb920` | 6,480 | 102 | `60cf8b5dff31eb2e6ff7e47242ae323a7f8e61e1a556890292fc311ba93bc305` |
| FIT | `S18I-ITER-0005-FIT-03` | 32070 | `8c19eaeb8712` | 6,478 | 102 | `366a07d07d5e6360ac6de6f4a59c98ef72b070120c81b325c965d8b7b47e801f` |
| CALIBRATION | `S18I-ITER-0005-CALIBRATION-01` | 32071 | `a33f48db784f` | 6,463 | 101 | `9b83fe7be68c0f1975883c915c5e90db14a854395c3f8d8709fab78d7aa84b56` |
| DEVELOPMENT | `S18I-ITER-0005-DEVELOPMENT-01` | 32072 | `ea6d2254d6d3` | 6,513 | 102 | `ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480747e37b48fe1e08` |
| DEVELOPMENT | `S18I-ITER-0005-DEVELOPMENT-02` | 32073 | `cfe48c2960db` | 6,460 | 101 | `cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310` |
| DEVELOPMENT | `S18I-ITER-0005-DEVELOPMENT-03` | 32074 | `efdcfa19b506` | 6,483 | 102 | `b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e` |
| DEVELOPMENT | `S18I-ITER-0005-DEVELOPMENT-04` | 32075 | `e33293cdf1cd` | 6,505 | 102 | `3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af` |
| CONFIRMATION | `S18I-ITER-0005-CONFIRMATION-01` | 32076 | `b2315511312e` | 6,457 | 101 | `db2b50444cb54c8a4bce700771efcef691e147b96ceb68985997e1115d74bbd2` |
| CONFIRMATION | `S18I-ITER-0005-CONFIRMATION-02` | 32077 | `925d262fcc01` | 6,468 | 102 | `54d787eb424066c3e8c18acb528f6ec19e2e9f4211b46e669390a3a330c8b762` |
| CONFIRMATION | `S18I-ITER-0005-CONFIRMATION-03` | 32078 | `928cddace62f` | 6,505 | 102 | `5c06a6613bcb7bc83b5646423f44f0440ecff8a276935b630c220ceff247ad3f` |
| CONFIRMATION | `S18I-ITER-0005-CONFIRMATION-04` | 32079 | `f5ffb05669f0` | 6,456 | 101 | `50a506a4670e5da45dbfa6b29db79f74a84e947b93de6f2d47edd65613c8e840` |

All 16 `manifest.json` digests were re-computed on the isolated runtime during this publication and match the reviewed A03 certification byte-for-byte (aggregate digest `378baf290aa8c17d3d8125942e65def735fefd556ee8c8c2cddad289cb71a309`). Per-role `waveform_stream_sha256` and `event_ledger_sha256` values are recorded in the certification carrier and the JSON handoff.

### Frozen role permissions (from the binding; no training on Cal/Dev/Conf)

- **DESIGN (4):** integrity, causal, construction and unchanged structural qualification; diagnostics only. No Fit/Calibration use or history selection.
- **FIT (3):** verified-healthy eligibility; fit only the fixed probe's feature mean/std and healthy centroid. No representation training, bank, alternate scorer, or non-probe fitting.
- **CALIBRATION (1):** verified-healthy eligibility; score with frozen Fit probe values and choose only the fixed probe q95. No fitting or retuning.
- **DEVELOPMENT (4):** structural qualification and one fixed-probe evaluation using frozen Fit/Calibration values. No selection, promotion, pruning, or tuning.
- **CONFIRMATION (4):** only after Task69 preceding-role PASS and Main release — structural certification and one fixed-probe evaluation using frozen Fit/Calibration values. No fitting, recalibration, retuning, rescue, or model score. No Confirmation rerun.

## 4. Custody state at publication (verified before and after the smoke)

| Item | Value |
|---|---|
| Live ledger | `_assessment-001/ledger.jsonl` — **38 events**, SHA-256 `ac10a975ae9c161633e5dac8ef05b3d1bfa2e33487f0d84841a2439d6594f3a6` |
| Historical ledger prefixes | first 30 events (interrupted A02 state) `05cbc0fb27dabba07805e2a85892f64b01d2c90227dca1f01f0fdbecf71ca509`; first 28 events (qualified) `625dcc1c6e14b43ac031afb590565a1af8461146579c8aff212d60bf3f061919` |
| Event census | 16 `role_materialization_started` / 16 `role_materialized`; 1 each `attempt_started`, `preflight_started`, `preflight_recorded`, `nonconfirmation_qualification_pass`, `confirmation_materialization_authorized`, `confirmation_recovery_authorized`; 0 fail/reject/retire |
| Records | attempt marker `875a3d03…`; preflight raw `3d1d774f…`; qualification record `c5a8e8cc…` |
| Root mtime | `1791516672` (unchanged by this publication) |
| Receipts (immutable) | execution v3 `696cbf7b…`, recovery v1 `9a420327…`, Task69 v2 `217fb15f…`, ASSESS v1 `45632676…`; historical `314140ee…`/`4cf2ee68…` unchanged |

## 5. Consumer invocation — the actual existing public loader

The public consumer path for this roster is **`synth.load_chronological`** (implemented in `src/synth/chronicle.py`, exported from `synth`). It verifies every shard's SHA-256 against the manifest, decodes each member with the public NPZ decoder, validates every `FileSample`, and checks manifest/order identity. There is **no** invented checkpoint/training CLI here: no `V1` model checkpoint, no training entry point, and nothing Chronological-v7-training-compatible is claimed or required for a DATA-only read. `synth.dataset.iter_materialized` is a different public API that targets the legacy per-split manifest layout and raises `TypeError` on a chronicle manifest (observed in this work); it is not the consumer path for this roster.

Exact invocation from the isolated runtime worktree with the locked environment (no sync, explicit root, fail-fast — no mount scans, no random fallback):

```sh
cd /home/trietlm/anomaly-representation-learning-s18t68-a02-worktree
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src:$PWD/experiments:$PWD   uv run --no-sync python -B - <<'PY'
from synth import load_chronological
root = "/home/trietlm/anomaly-representation-learning-s18t68-a02-worktree/data/generated/sprint18-iterative-v1/S18-ITER-0005-ASSESS-P-ALLOWANCE-V1"
samples, manifest = load_chronological(f"{root}/DESIGN/S18I-ITER-0005-DESIGN-01")
PY
```

Equivalent public API form for any role root path `P`:

```python
from synth import load_chronological
samples, manifest = load_chronological(P)   # (list[FileSample], manifest dict)
```

### What a consumer gets per `FileSample`

- `x`: `float32` `[C, T]` waveform — `C = 6` channels fixed; `T` variable, observed `200–800` timesteps across all 103,686 samples of the roster.
- Identity/provenance: `file_id`, `seed` (the sample's own RNG seed), `generator_version` (`2.0.0`), `config_hash`, `factory_provenance`.
- Conditioning (encoder-visible): `regime_sequence`, `robot_idx`/`program_idx`/`robot_code`/`program_number`, `operation` (with `arrival_time`/`start_time`/`end_time`/`duration`), `operating_context` (shift/load/ambient temperature).
- Diagnostic labels only (never model input): `file_label`, `health` (latent robot health), `episode`, `anomaly_labels`, `future_targets`, `anomaly_meta`/`anomaly_mask`, `split_provenance`. `FileSample.encoder_inputs()` returns exactly the `MODEL_INPUT_FIELD_NAMES` set and asserts disjointness from `DIAGNOSTIC_ONLY_FIELD_NAMES`.
- Window/patch metadata (public patchify contract): `Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True, pad_value=0.0)).patchify(sample)` → `PatchBatch` with `patches` `float32 [N, C, 32]`, `starts` `int64 [N]`, `valid_len` `int64 [N]`, `pad_mask` `bool [N, 32]` (`True` = padded). Invariants: `starts[0] == 0`; merged valid spans cover `[0, T)` exactly; `pad_mask` marks exactly `N*32 - sum(valid_len)` positions; the last span ends at `T`.

### Current fit/calibration status (frozen; immutable arrays)

- Fit (from FIT roles, verified-healthy only): **5,446 files / 167,419 valid patches**, feature dim **59**, healthy centroid norm **2.290221527135249e-11**.
- Calibration: **1,877 healthy files / 56,889 valid patches**, frozen q95 threshold **195.7544682761532** — frozen BEFORE Development; `S_pred` and `S_pop` remain independent and no private hard scores are mixed into this handoff.
- These values are the immutable arrays in the qualification record `c5a8e8cc…`, re-verified byte-exactly by the Task70 A03 frozen-statistics carrier `d81925f0…`. This handoff performs no refit, no recalibration, no bank fit, no learned model.

## 6. Actual read-only smoke (executed during this publication)

Bounded read-only smoke on the real 16-role DATA through the public loader only — no probe re-run, no integrity-gate re-adjudication, no training. Full bounded output: `artifacts/sprint-18/s18-t71-a01-readonly-consumer-smoke.json`.

| Measurement | Value |
|---|---|
| Roles loaded via `load_chronological` | 16/16 (one per role group and every individual role) |
| Samples read | 103,686 (every sample of every role root) |
| Contract checks | all pass (manifest format/role/protocol/config_hash per role; loader count and sample order per role; `FileSample.validate`; `encoder_inputs()` key-set; patch invariants) |
| Observed channels | `C = 6` |
| Observed lengths | `T ∈ [200, 800]` timesteps (variable) |
| Waveform bytes touched | 1,242,506,520 B (~1.16 GiB) cumulative across the 16 sequential role loads |
| Smoke wall time | 102.117 s in-process (104.48 s wall, 05:54:37Z → 05:56:20Z) |
| Process peak RSS | 391,260 KiB (~382.2 MiB), one role resident at a time |
| GPU | RTX 4060 Ti, 0% utilization, 1.6 GiB used (desktop) — CPU-bound read; no training |
| Comparison | observed generation of the 4 Confirmation roots: 166 s (A03 detached run); whole-16 certification: 139.3 s (A03) |

Sample identity actually observed (first sample of `S18I-ITER-0005-DESIGN-01`): `file_id=S-op-000000-bdccbdbf1e`, `file_label=normal`, `C=6`, `T=372`, `robot_code=robot-04`, `program_number=program-05`, operation times `[6234.806, 6234.806, 6834.806, 600.0]` s, `n_regimes=3`, patch batch `N=23` of `[6, 32]`, `valid_len` sum 724, `pad_true=12`. Per-role first-sample identities are recorded in the bounded smoke carrier.

## 7. Final public state check

- Isolated runtime: HEAD `b032ef8b…` (F), branch `sprint18-task68-a02-runtime`, porcelain 0; canonical checkout `23642b47…` + 2 geometry dirs untouched.
- Final public command, run once at F with source E unchanged:

```sh
cd /home/trietlm/anomaly-representation-learning-s18t68-a02-worktree
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src:$PWD/experiments:$PWD   uv run --no-sync python experiments/sprint18_iterative_candidate_v1.py --assessment   --binding experiments/sprint18-iterative-assessment-c5-allowance-v1.json --stage validate
```

  → **exit 0**, `STATIC_BINDING_PASS` (718 B stdout, 0 B stderr). This stage is the static binding/source identity gate: its printed `contacted:false` / `assessment_status:NOT_YET_RUN` fields are the binding carrier's freeze-time declarations (identical output in every prior run, including A03's post-run smoke); the live DATA state is separately evidenced by the digests in §4 (38-event ledger, 16 identical manifests, unchanged root mtime).
- Post-smoke custody: ledger 38 `ac10a975…`; first-30 `05cbc0fb…`; first-28 `625dcc1c…`; records `875a3d03…`/`3d1d774f…`/`c5a8e8cc…`; 16 manifests unchanged (aggregate `378baf29…`); root mtime unchanged. No orphan compute; throwaway drivers removed; persistent A03 recovery logs retained.

## 8. Limitations and honest disclosures

1. The historical frozen-statistics digest `181b83d1…` could not be re-derived (its derivation input is not stored); the frozen **values** were verified byte-exactly from the immutable arrays. Reviewed nonblocking by `S18-T70-R03`.
2. No pre-overwrite SHA-256 exists for the restored `task-70.md` prefix (self-inflicted overwrite, deterministically repaired; reviewed nonblocking by `S18-T70-R03`).
3. Exactly one nominal exceedance of the 15 d nominal P ceiling exists and is **reported, not waived**: seed 32076 (`S18I-ITER-0005-CONFIRMATION-01`), `15.545084957564008 d` under the user-accepted `[2, 16)` allowance (`sprint18-p-duration-allowance-v1`); all 16 roles pass the hard predicate `2.0 <= duration_d < 16.0`.
4. This publication is DATA-only: no training, no model checkpoint, no GPU use, no DATA-batch differential verdict. The Tasks 66–71 data-batch differential deep review is Main-assigned after this handoff's own review/checkpoint; nothing here marks the data prerequisite complete.
5. No future checkpoint SHA of this handoff is embedded: the deliverable identifies the current working base F only; any later checkpoint is recorded in the worker evidence artifact, not guessed inside the product carrier.

## 9. Pointers

- Worker evidence: `artifacts/sprint-18/task-71.md#A01` (runtime commands, identities, resources, gaps) and `artifacts/sprint-18/s18-t71-a01-handoff-proof.json`.
- Bounded smoke output: `artifacts/sprint-18/s18-t71-a01-readonly-consumer-smoke.json`.
- Data-batch review index (Tasks 66–71): `artifacts/sprint-18/s18-t71-data-batch-review-index.md`.
- Frozen inputs: `experiments/sprint18-iterative-assessment-c5-allowance-v1.json`, `experiments/sprint18-iterative-data-contract-v3.md`, `experiments/sprint15-benchmark-protocol-v7.md`, `experiments/sprint15-observable-probe-v7.md`, `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` (EG4).
