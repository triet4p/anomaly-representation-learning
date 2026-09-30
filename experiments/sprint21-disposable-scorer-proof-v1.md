# Sprint 21 Task 2 — Disposable-only scorer proof contract v1

**Status: DESIGN CONTRACT ONLY — NOT EXECUTABLE AND NOT AUTHORIZED YET.** No model has been loaded or run, no sample has been scored, and no history, root, or pilot coordinate has been generated or contacted. This contract freezes only the bounded Task 3 proof below. It is neither the scientific pilot protocol nor a pilot authorization.

**Review gate:** Task 3 may load the frozen checkpoints for this toy proof only after an independent Task 2 evidence review returns PASS. That review authorizes no real-history scoring, generation, preflight, role/seed binding, or other data contact.

## 1. Scope and narrow Fit/bank exception

Sprint 21 asks whether separately scored `S_pred` and `S_pop` can support a research-only, simulator-conditional description on genuinely fresh whole Evaluation histories. For this candidate fixed scorer, a new Fit/Calibration/Evaluation topology would require another fitted representation and bank. The proposed narrow, representation-specific exception is to restore the already-trained Sprint 17 `B0` parameters and its embedded healthy Fit-only reference bank unchanged, solely for this disposable proof and, only after the separate Task 4 protocol and review, a future disjoint research Evaluation roster.

This is defensible for a threshold-free diagnostic because the model and `S_pop` bank are frozen before any future Evaluation contact, no Evaluation sample contributes to representation, normalization, preprocessing, or bank fitting, and the primary `S_pred`/`S_pop` event-AUROC endpoints do not use an operating threshold. It is **not** an untouched prospective Fit/Calibration pipeline. It does not authorize reuse of historical Fit, Calibration, Development, Design, or research-sandbox roots as Evaluation; reuse of historical Evaluation outcomes; thresholded diagnostics; calibrated-risk claims; or generalization claims beyond a later declared synthetic mechanism and support.

Stored Sprint 17 Calibration thresholds are expressly excluded. The proof runner MUST NOT read, load, compare against, or apply those thresholds, and MUST NOT compute a new threshold. There is no Calibration role in this proof. Historical Calibration data and already-scored Development histories are not proof inputs. Historical Fit rows appear only as provenance for the embedded bank, not as test samples.

Any future Evaluation roster must be genuinely fresh and disjoint at whole-history/root/seed level from all previously materialized or contacted Sprint 17 and research-sandbox histories, including historical Fit, Calibration, Development, and Design roles. It may be bound only in Task 4 from authorized pre-existing provenance, after Task 3 passes. Task 2 binds no future history count, ID, root, data seed, or construction quota.

This exception changes none of Sprint 20's `NOT_READY` conclusion, Sprint 18's existing boundary or quota-48 contract, or any Gate V2 decision. It authorizes no Sprint 18 Cycle 4, eight-arm training, gate amendment, training/refitting, Confirmation, or Sealed access.

## 2. Frozen B0 assets, bank, and runtime

All three historical model seeds are mandatory in the fixed order below. They are scorer variants, not independent samples. Report each seed separately; do not choose a favorable seed, omit a failing seed, average seeds as histories, or replace a checkpoint.

| B0 model seed | Historical step-300 checkpoint | Bytes | Required SHA-256 |
|---:|---|---:|---|
| `171701` | `/tmp/sprint17-task7-out/checkpoints/b0_seed171701_step300.pt` | `18,239,321` | `45f9e151c5d3946804ea531eb2bcace26b1f62e634671f51b8741ac6b411a619` |
| `171702` | `/tmp/sprint17-task7-out/checkpoints/b0_seed171702_step300.pt` | `18,239,321` | `64ed2cedec210e4692f60bb4dd3430a57cf94abfcd50551abd97c9265ea785f8` |
| `171703` | `/tmp/sprint17-task7-out/checkpoints/b0_seed171703_step300.pt` | `18,239,321` | `9edb2122e357376a9ff1e065d73d5ab7704f8d2005991552332f1ced01b0e906` |

Verify every exact byte count and full SHA-256 before deserialization and again after scoring. Read directly from these original paths as read-only; do not copy, rewrite, or select by outcome. Any missing file, byte/hash drift, or checkpoint-identity mismatch stops the entire proof before scoring. No retry, CPU fallback, checkpoint substitution, or partial-seed success is permitted.

| Frozen provenance | Binding |
|---|---|
| B0 evidence/source commit | `8c15f0204a3e495569b7f143dc109943e8b808de` |
| Full B0 configuration SHA-256 | `1ff67f95428ef29aab05d9f6394a305b75c8c2a9e12431f3cda09b6958596144` |
| Frozen metric-code digest | `b2d6af7505abe459a84f76f5279a6a5c4298e7c142901edd4596fcfe3ddeb88f` |
| Historical source profile/protocol | `sprint15-v7` / `sprint15-benchmark-protocol-v7` |
| Historical role-binding digest (provenance only) | `075868b22c028a981cc63ffe37b8d29720cd324c3e2c7254ce688678758230ba` |
| Python / PyTorch | Python `3.12.13`; PyTorch `2.14.0+cu130` |
| Device | CUDA available; NVIDIA GeForce RTX 4060 Ti, `16,380 MiB` total memory |
| Historical interpreter | `/home/trietlm/anomaly-representation-learning/.venv/bin/python` |

The historical Task 7 record states that its committed `uv.lock` did not change across that execution. Task 3 MUST record the full `uv.lock` SHA-256 from the pinned source tree and verify the live environment against that tree before scoring. Do not install or upgrade dependencies during the proof.

The exact project source is the full Git tree at commit `8c15f0204a3e495569b7f143dc109943e8b808de`, not the current checkout. The historical task audit records the following direct scorer-dependency file hashes at that commit:

| Source file | Historical SHA-256 |
|---|---|
| `src/representation/model.py` | `7051f5f76be0bcc8e50d6d5ee2941a57c12a3285406d0ef27883c596aa6600f0` |
| `src/representation/data.py` | `88ea3becbc8328c5b21518b1f957fe710096cb96ed0cd2416fe6f139439d5eb4` |
| `src/representation/masking.py` | `c0efd33b24b8d01c067198f0bc04c0c90c1e99bad9a6a1ec006b37ce27c0465f` |
| `src/representation/contracts.py` | `0b23080fc49f218484a6f7939bb9288f07b607686312e91c3b14d9d6b84464ed` |
| `src/representation/layers/normalization.py` | `bfb5f68f5352115c2e044c27999cdf3fecab7aa3ce2cd05ce021204f1f9edd8d` |
| `src/synth/chronicle.py` | `b5992d6af3c4387d18df746b43be17b4bfc1fac6938e8b44be35b6fb7c65dafd` |

The full commit pins all other source files, including checkpoint, inference, patchification, schema, and scheduled-signal code. The metric digest's member-file hashes are in the [Task 1 scorer audit](../docs/exploratory-pilot-scorer-audit.md). Task 1 found current local scorer dependencies differ from this historical tree; hash differences are not semantic-equivalence evidence. Do not mix local/current modules with historical metrics or import a copied/exported source tree.

The frozen B0 model is six-channel V1 with patch/stride `32/16`, end padding enabled, `d_model=128`, four sequence layers/four heads, conditional normalization, `n_robots=9`, `n_programs=8`, minimum conditional bucket `32`, and `knn_k=5`. Its frozen masking ratio is `0.40` with random/information/block composition `0.34/0.33/0.33`. Each checkpoint represents step `300`; no training state may be advanced. Task 3 restores one seed at a time, uses `model.eval()` and `torch.inference_mode()`, and releases it before restoring the next seed.

Each checkpoint embeds its own bank. Historical provenance is `5,040` verified-healthy Fit files from `H-S17-V3-FIT-01..03` (historical data seeds `2804..2806`), `154,129` valid Fit patches, and `k=5`. The checkpoint payload pins the stored bank `k` and embedding tensor; there is no separate historical bank file or bank SHA-256. The expected restored bank is exactly `5,040 x 128` finite file embeddings, `k=5`, Fit-only. Task 3 MUST verify those dimensions and provenance, record a canonical digest of the restored CPU float32 bank tensor, and verify `k`, shape, and tensor digest remain unchanged after scoring. A missing bank, any mismatch, or any attempted `.fit()`/refit stops the proof. Do not claim a separate historical bank digest that was never recorded.

## 3. Frozen conditioning vocabulary

The authorized Sprint 17 binding records profile `sprint15-v7`. In the pinned source, that static profile's route configuration contains `robot-01` through `robot-09` and `program-01` through `program-08`; the scheduled generator assigns indices by sorting those identifier strings and enumerating from zero. Freeze the complete source-derived map here rather than deriving a new map from the toy subset or a future sample:

| Robot identifier | `robot_idx` | Program identifier | `program_idx` |
|---|---:|---|---:|
| `robot-01` | 0 | `program-01` | 0 |
| `robot-02` | 1 | `program-02` | 1 |
| `robot-03` | 2 | `program-03` | 2 |
| `robot-04` | 3 | `program-04` | 3 |
| `robot-05` | 4 | `program-05` | 4 |
| `robot-06` | 5 | `program-06` | 5 |
| `robot-07` | 6 | `program-07` | 6 |
| `robot-08` | 7 | `program-08` | 7 |
| `robot-09` | 8 | — | — |

The frozen map object is `{"robot_id_to_idx":{"robot-01":0,"robot-02":1,"robot-03":2,"robot-04":3,"robot-05":4,"robot-06":5,"robot-07":6,"robot-08":7,"robot-09":8},"program_id_to_idx":{"program-01":0,"program-02":1,"program-03":2,"program-04":3,"program-05":4,"program-06":5,"program-07":6,"program-08":7}}`. Its mapping digest is SHA-256 `4f769ccc64aac4c6a91fdd356eaa6963afe976ab847a22df23e43f9331418897`, computed from canonical UTF-8 JSON (sorted keys, compact separators).

The mapping is recoverable only from the authorized pinned source/config: `src/synth/chronicle.py`, `src/synth/scheduled.py`, and the `sprint15-v7` profile record in `experiments/sprint17-role-binding-v3.json`. Task 3 MUST rederive the same full identifier set from that pinned static config and compare its canonical map digest before loading a checkpoint. If source/config is unavailable, differs, or cannot reproduce the map, fail closed; do not infer a mapping from numeric ranges or silently reindex a partial vocabulary. Every sample must carry explicit robot/program strings and explicit integer fields that match this map. Missing fields MUST NOT fall through to the reader's default index `0`; unknown IDs, out-of-range values, or disagreement between strings and indices are errors. Future Task 4 must bind its actual generator and show the same mapping before any contact; a changed vocabulary requires a new reviewed contract.

The six signal channels are fixed in this order: `feed`, `current`, `temperature`, `arc-voltage`, `arc-power`, `torch-pressure`. Input tensors are finite `float32` arrays with shape `[6,T]`; do not transpose or silently adapt channel order.

## 4. Exact in-memory toy fixture and mask

Use exactly the two files below, in this order, as a fully enumerated disposable file roster for each checkpoint. They are mathematical arrays constructed in memory, not simulator outputs or generated histories. Both have `T=64`; with the frozen `32/16` patchifier they yield three full valid patches at starts `0,16,32`, each of length `32`.

For channel index `c=0..5` and time index `t=0..63`, calculate integer modulo/division values below and cast each value to little-endian IEEE float32. Store the resulting matrix as C-contiguous `[channel,time]` rows. No RNG, generator, data seed, file read, root, history, label-derived value, or model output participates in constructing either array.

| `file_id` | `robot_id -> robot_idx` | `program_id -> program_idx` | `x[c,t]` formula | SHA-256 of canonical `<f4`, C-order tensor bytes |
|---|---|---|---|---|
| `S21T2-TOY-A` | `robot-01 -> 0` | `program-01 -> 0` | `float32((((7*t + 3*c) mod 17) - 8) / 8)` | `236eed5f43d5a407fbc3598476a4e761859b658c168c90cdc0acb93ebbae7f0b` |
| `S21T2-TOY-B` | `robot-09 -> 8` | `program-08 -> 7` | `float32((((5*t + 11*c + 3) mod 19) - 9) / 9)` | `55afb88796385f9aaeeafe6b553b5166412c995103323bc115083981b14b9f38` |

The canonical per-file masking rule is the exact Task 7 `file_seed` rule: encode the exact UTF-8 `file_id`, compute SHA-256, interpret the first four digest bytes as unsigned big-endian, then take modulo `2**31`. The full identifier hashes and resulting explicit masking seeds are:

| `file_id` | SHA-256 of UTF-8 ID | Frozen masking seed |
|---|---|---:|
| `S21T2-TOY-A` | `3baf60cf45cf2ccf6955f1ed3c94a216e0a43926c93199a58dc5c303968a0b67` | `1001349327` |
| `S21T2-TOY-B` | `8ff378cdeffbb6002b68f3c89319ec6b49e9dedf26bc23d88152a1f7bc6add3b` | `267614413` |

With three valid patches and frozen mask ratio `0.40`, exactly one valid patch is a prediction target (`round(0.40 * 3) = 1`). Its position is determined solely by the pinned historical masking implementation and the explicit seed above. Task 3 MUST record the three-bit mask for each file/seed, prove it equals that implementation's output, and reproduce it exactly on a second call and when file order is reversed. A count other than one, any masked padding/invalid patch, or any mask/order/replay mismatch is failure. Do not use `score_file()` without an explicit seed.

When the pinned interface requires `FileSample` metadata, use `file_label=NORMAL`, `seed=0` only as an inert required metadata sentinel (never pass it to an RNG), `generator_version="S21-T2-MATH-FIXTURE-v1"`, `config_hash="s21-t2-math-fixture-v1"`, and an empty `regime_sequence`; set both numeric condition fields explicitly from the table. Keep the robot/program strings in the fixture record used for mapping validation. No event, health, anomaly, future-target, or history metadata is used as model input. The expected roster is exactly `["S21T2-TOY-A", "S21T2-TOY-B"]`; each ID occurs once, globally unique within the fixture, and no file is persisted.

## 5. Restore-only score contract and measurable criteria

For each of the three checkpoint seeds, independently restore the checkpoint's model and embedded bank, then score exactly the two fixture files one at a time using explicit per-file mask seeds. Return two independent file-level values per ID; never fuse, compare scales, threshold, or select on them:

- `S_pred` MUST equal the mean masked prediction MSE over the true prediction-mask patches.
- `S_pop` MUST equal the mean Euclidean distance to the five nearest embeddings in the restored historical Fit bank (`k=5`).

The proof has **no absolute golden score constants**: they require loading the frozen weights, which this design-only task is forbidden to do. Task 3 MUST report the observed values by `(model_seed, file_id, branch)` without changing this contract. Independently recompute each branch from the same exact inference tensors and restored bank; require `rel_tol=1e-6`, `abs_tol=1e-7`, finite, nonnegative scalar outputs. A second call and reversed file order MUST reproduce both scores within the same tolerance and produce bit-identical masks. Both branches MUST have exactly the same two-file roster, no missing/non-finite member, and remain separately named and separately validated. No AUROC or whole-history result is defined for this toy.

Run every restore in evaluation/inference mode only. The restored normalization state, model parameters/buffers, and embedded bank are immutable: snapshot their canonical digests or tensor contents before scoring and verify exact equality afterward. Do not instantiate or call a trainer, optimizer, scheduler, bank fitter, preprocessing/statistics update, Calibration code, threshold computation, or Task 7 training `main()`. Any parameter, buffer, running-statistic, bank `k`, bank-row, or bank-digest change is a failure. Do not treat a score or score-branch result as authorization to tune anything.

The required fail-closed cases are measurable and must produce no successful/partial score result:

| Invalid condition | Required disposition |
|---|---|
| Missing or hash/size-mismatched checkpoint, wrong config/seed/step, absent bank, bank `k != 5`, bank rows other than `5,040`, dimension other than `128`, non-finite bank | `PROVENANCE_MISMATCH`; stop before scoring |
| Missing/extra/duplicate toy ID, unequal roster lengths, a missing branch value, branch key-set mismatch, or non-finite required score | `ROSTER_OR_SCORE_INCOMPLETE`; emit no macro/partial success |
| Missing robot/program string or integer field, an unknown string, range failure, or a string/index disagreement | `CONDITIONING_MISMATCH`; do not default to index `0` |
| Input has channel count other than six, wrong shape/dtype, empty time axis, or any non-finite value | `INPUT_CONTRACT_FAILURE` |
| Wrong mask seed, wrong number of selected patches, mask on invalid padding, or changed mask on replay/order reversal | `MASK_CONTRACT_FAILURE` |
| Attempted training/refit/recalibration, changed model/statistics/bank state, source/runtime drift, or imported project module outside the pinned tree | `RESTORE_ONLY_OR_ISOLATION_FAILURE`; stop with no score result |

Exercise roster and validation errors using only in-memory mutated copies of the two toy records; do not alter checkpoints or create history-shaped fixtures. In addition to the exact two-file roster check, implement the reusable full-roster invariant for future Task 4 inputs: require a predeclared expected roster; every declared history exactly once; equal input/output roster lengths; one finite value per endpoint for every declared history; identical score-independent support keys across branches; all structural/hard-support predicates satisfied; and no numeric macro, bootstrap, or subset average if any declared history is structural-fail or uncomputable. Task 2 does not choose future `H` or history identities.

A proof succeeds only if all 3 checkpoints pass every positive criterion and all required negative cases fail closed. Any failure stops Task 3; it does not permit a seed swap, code change followed by an unreviewed rerun, new toy/seed, real-root investigation, or Task 4 binding.

## 6. Task 3 isolation and proposed smoke surface

The Task 1 audit reports the historical source tree at `/tmp/sprint17-task7` on the exact clean Task 7 commit and the original checkpoint paths above. The current checkout is not an acceptable substitute because scorer dependencies differ. After the Task 2 independent PASS, Task 3 SHOULD create a separate detached Git worktree from the historical Git object database at that exact base commit; it MUST NOT copy/export source files, import a bundle of source files, use the current checkout, or allow editable-install imports from another project tree. Add only the inference-only proof wrapper to that worktree. Before the first model load, commit/bind the final wrapper commit and record its parent, full tree/commit hash, wrapper-file hash, pinned-tree `uv.lock` hash, checkpoint hashes, live interpreter/Torch/CUDA/GPU identity, and clean worktree status. Any later code or environment change invalidates that proof binding.

Proposed Task 3 entry point: `experiments/sprint21_disposable_scorer_proof.py`. Proposed smoke surface (after review only):

```sh
cd /tmp/sprint21-disposable-proof-v1
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/sprint21-disposable-proof-v1/src \
/home/trietlm/anomaly-representation-learning/.venv/bin/python \
  /tmp/sprint21-disposable-proof-v1/experiments/sprint21_disposable_scorer_proof.py \
  --checkpoint-root /tmp/sprint17-task7-out/checkpoints \
  --output-dir /tmp/sprint21-task3-proof-out
```

The wrapper path is a proposed Task 3 deliverable, not an implemented file in Task 2. The wrapper MUST assert all project-module `__file__` locations resolve within `/tmp/sprint21-disposable-proof-v1`; the process MUST use only the pinned source tree and verified historical interpreter. Checkpoints stay read-only at their original paths; output goes only to the separate disposable proof output directory. Do not pass or discover any `data/` or history directory; do not import a dataset loader, simulator/generator, role-binding code that opens roots, or Task 7 `main()`. If any provenance, import-origin, runtime, or resource issue arises, fail closed and record it; no CPU fallback or retry. Task 3 records observed per-file scores, masks, bank/model immutability evidence, complete error-path results, wall time, and peak GPU memory. No wall-time/VRAM number is asserted here because the inference-only path has not been run.

## 7. Task split and future authorization boundary

The Task 1 audit predates the Sprint 21 split and contains references to old Task 2 duties for an actual Evaluation roster and executable pilot path. Under the current plan, those actual-pilot choices belong to Task 4. This contract freezes Task 2's topology exception and disposable proof only; it does not waive any Task 4 pre-contact requirement. Task 4 may proceed only if Task 3 passes, then must version and independently review the actual generator/config, complete pilot scorer/runner/source/checkpoint/bank/metric hashes, fresh disjoint whole-history roster and seed binding, support/attempt rules, and all other pilot parameters before contact.

No user has authorized the scientific pilot by approving this toy contract. Only after Task 2's independent evidence PASS may Task 3 implement and execute this bounded disposable proof. Even a successful proof authorizes no actual history, no generation/preflight, no role/seed binding, and no pilot scoring. Only a later separate Task 4 independent pre-contact PASS could authorize a one-shot research pilot; that pilot would remain descriptive research evidence and would not amend Sprint 20, Sprint 18, Gate V2, or authorize Confirmation/Sealed use.

## References

- [Sprint 21 plan](../docs/sprint-plans/sprint-21.md), especially current Tasks 2–4.
- [Task 1 frozen B0 scorer audit](../docs/exploratory-pilot-scorer-audit.md).
- [Task 1 evidence artifact](../artifacts/sprint-21/task-1.md).
- [Sprint 17 Task 7 evidence](../artifacts/sprint-17/task-7.md) and [committed B0 summary](sprint17-task7-b0-summary.json).
- [Sprint 19 estimands](../docs/exploratory-benchmark-estimands.md) and [six-channel order](../docs/SYNTH.md).
