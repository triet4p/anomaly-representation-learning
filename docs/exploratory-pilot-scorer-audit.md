# Sprint 21 Task 1 — Frozen B0 scorer feasibility audit

**Status: design-only audit; no data contact or scoring authorized.** This audit does not change the Sprint 20 `NOT_READY` decision, Gate V2, Sprint 18, or any role/seed binding.

## Verdict

**Conditional GO to Task 2 protocol design with Sprint 17 `B0` as a candidate frozen diagnostic scorer. NO-GO to score, generate, preflight, bind roles/seeds, or contact a new history now.** The three checkpoint files are still accessible at their recorded remote paths and their complete SHA-256 digests and byte counts match the frozen Task 7 records. The exact Task 7 source commit and score/metric digests are recoverable, and its checkpoint-save path embeds the Fit-only reference bank. The inference primitives can produce both independent score branches without training or fitting when given a compatible future `FileSample` and a restored checkpoint.

This is not yet an executable scorer authorization or a proof against actual future roots: none has been bound or opened. A pilot-only exception for the historical Fit/bank topology is REQUIRED and must be justified and independently reviewed before any contact. Task 2 must also bind the exact conditioning vocabulary, a genuinely fresh Evaluation roster, an inference-only path that does not invoke Task 7's train/refit main loop, strict full-roster metric guards, and one exact runtime/source commit. If these cannot be frozen and reviewed, the outcome is `UNAVAILABLE` and no new roots are contacted.

| Question | Audit result |
|---|---|
| Are the three checkpoint bytes accessible and hash-verifiable? | **Yes.** 3/3 exact byte counts and SHA-256 digests match the frozen record. |
| Is the `S_pop` bank reusable without refitting? | **Yes, as a checkpoint-embedded, Fit-only bank.** The checkpoint hash pins its bytes; no separate bank file or standalone bank digest is recorded. |
| Can B0 produce the two threshold-free primary scores without retraining? | **Technically yes, conditionally.** The restore/inference primitives support it; Task 7's existing CLI itself always trains/refits and is not a safe pilot entry point. |
| Are future roots' conditioning IDs, schema, support, and identity proven now? | **No.** They do not exist in a bound roster. Task 2 must freeze and validate these before contact. |
| Is the old Fit/Calibration → fresh Evaluation topology permitted without an exception? | **No.** A representation-specific, pilot-only exception is required. |
| May scoring/contact begin now? | **No.** No protocol, binding, or independent pre-contact review has passed. |

## Frozen assets and provenance

The remote asset inventory was read-only. All three paths exist and match both size and full digest in `experiments/sprint17-task7-b0-summary.json` and `artifacts/sprint-17/task-7.md`:

| Model seed | Exact remote checkpoint path | Bytes | SHA-256 |
|---:|---|---:|---|
| 171701 | `/tmp/sprint17-task7-out/checkpoints/b0_seed171701_step300.pt` | 18,239,321 | `45f9e151c5d3946804ea531eb2bcace26b1f62e634671f51b8741ac6b411a619` |
| 171702 | `/tmp/sprint17-task7-out/checkpoints/b0_seed171702_step300.pt` | 18,239,321 | `64ed2cedec210e4692f60bb4dd3430a57cf94abfcd50551abd97c9265ea785f8` |
| 171703 | `/tmp/sprint17-task7-out/checkpoints/b0_seed171703_step300.pt` | 18,239,321 | `9edb2122e357376a9ff1e065d73d5ab7704f8d2005991552332f1ced01b0e906` |

Combined checkpoint size is 54,717,963 bytes. No checkpoint was loaded, deserialized, copied, modified, or scored in this audit.

Task 7 records that each step-300 checkpoint was saved with its `NormalReferenceBank` (`save_checkpoint(..., reference_bank=bank)`). The payload stores `k` and the embedding tensor; `load_checkpoint(..., reference_bank=bank)` restores that bank. The historical provenance is **5,040 verified-healthy Fit rows and 154,129 valid patches**, from only `H-S17-V3-FIT-01..03` (health seeds 2804–2806); `k=5`. There is no separately recorded bank-file path or bank SHA. The checkpoint's exact SHA-256 therefore pins the model, configuration, and embedded bank together, while the Task 7 record supplies the Fit-only row/source provenance. Do not call `NormalReferenceBank.fit`, set `refit=True`, or rebuild the bank for a pilot.

Historical Calibration was `H-S17-V3-CAL-01` (seed 2807), 1,621 healthy-eligible rows. The recorded per-seed q95 thresholds are:

| Model seed | `S_pred` q95 | `S_pop` q95 |
|---:|---:|---:|
| 171701 | 0.09334097802639008 | 2.2176554203033447 |
| 171702 | 0.11303385347127914 | 2.0839240550994873 |
| 171703 | 0.09313471615314484 | 2.583014488220215 |

These thresholds are not inputs to the primary AUROC estimands. Do not recalibrate or use them for primary history selection. If a future protocol retains thresholded diagnostics, it must explicitly include the old Calibration topology and freeze which thresholds apply; otherwise omit those diagnostics.

The historical data tree recorded by Task 7 was `/tmp/sprint17-task7/data/generated/sprint17-ablation-v3/{FIT,CALIBRATION,DEVELOPMENT}`. The old Development histories were `H-S17-V3-DEV-01..04` (seeds 2808–2811) and have already been scored; **they are not pilot Evaluation**. No Confirmation or Sealed data file/root was opened, listed, hashed, imported, or used for this audit.

## Exact scoring and estimand compatibility

Task 7 froze a six-channel V1 model with 32/16 patch/stride, end padding, `d_model=128`, four sequence layers/four heads, conditional normalization, `n_robots=9`, `n_programs=8`, minimum conditional-bucket count 32, mask ratio 0.40 (composition 0.34/0.33/0.33), and `knn_k=5`. Its full configuration digest is `1ff67f95428ef29aab05d9f6394a305b75c8c2a9e12431f3cda09b6958596144`. It ran exactly 300 optimizer steps; the frozen absolute 980-step contrastive warmup left effective contrastive weight at 0.0 at step 300. This is a deliberately short diagnostic scorer, not a full-training baseline or a pilot-fitted representation.

At inference, `RepresentationInference.score_batch` computes the branches separately:

- `S_pred`: mean masked prediction MSE over prediction-mask patches.
- `S_pop`: mean Euclidean distance to the five nearest embeddings in the restored, healthy Fit bank.

The Task 7 scorer uses 32/16 patchification and the fixed file masking rule `sha256(file_id) mod 2**31`; `score_items` scores one file at a time and passes that seed explicitly. A future path must preserve this rule and stable, globally unique file IDs. Do not use the convenience `score_file()` path without an explicit deterministic masking seed: its default seed is `None`.

The frozen metric primitives match the Sprint 19 primary definitions for **computable, valid, common support**: event-window score is the maximum member-file score; `synth.events.roc_auc_tie_aware` uses average-rank ties; `synth.probe15.evaluate_histories` computes per-history P+W AUROC and an unweighted history macro. Its history-block lower bound uses 2,000 bootstrap replicates, RNG seed `20260202`, and the 2.5th percentile. The two branches remain separate; the same score-independent support must be supplied to both.

These primitives do **not** constitute the full Sprint 19 fail-closed contract by themselves. The existing evaluator zips role lists and can emit missing/`None` per-history values; it does not prove the declared roster is complete, enforce every hard support predicate, or prohibit averaging over a computable subset. A future wrapper must check the full predeclared `H`, equal input-list lengths, finite branch scores, identical branch support keys, every history's structural predicates, and complete endpoints before emitting either macro/bootstrap. Any missing/non-finite score, support mismatch, structural failure, or uncomputable history must make the full-roster endpoint `UNCOMPUTABLE`; no history may be dropped or pooled away.

### Conditioning and future-root requirements

The B0 config records cardinalities 9 robots and 8 programs, and the Task 7 record reports observed `robot_idx` 0–8 and `program_idx` 0–7. `ConditionalBatchNorm` consumes those integer IDs directly and raises when an ID is outside `[0,9)` or `[0,8)`. The materialized-data reader defaults a missing robot/program index to zero, so absence is not safe evidence of a match. In the generator, `ScheduledSignalGenerator.generate()` assigns indices from sorted robot/program identifiers present in each schedule; a changed or incomplete identifier set can reindex categories even while all indices remain in range.

The Task 7 record pins cardinalities/ranges but does not supply a separately frozen string-to-index vocabulary digest. Before contact, Task 2 must freeze the generator and the exact robot/program identifier-to-index mapping, require the explicit per-sample index fields and six-channel order, and fail closed on any mismatch. A fresh root that only happens to have in-range IDs is not enough. The scorer must use `model.eval()` and the restored checkpoint's normalization state; no per-pilot statistics, bank rows, preprocessing, thresholds, or weights may be fit or updated.

## Topology exception and scientific limits

A representation-specific topology exception is required because this path deliberately reuses the historical Fit-derived representation and Fit-only bank rather than fitting on a new pilot Fit role. The exception is narrowly defensible only for a research-only diagnostic on **genuinely fresh, disjoint whole-history Evaluation roots**. The new Evaluation roster must exclude all previously materialized/contacted roots and seeds, including the old Fit, Calibration, Development, and research-sandbox histories, using only authorized provenance; no old role or root may be relabeled. Historical Development is already observed and cannot be used again. The pilot roster can never become later Confirmation or Sealed evidence.

The primary `S_pred` and `S_pop` AUROCs are threshold-free; old Calibration thresholds are unnecessary for them. The exception does not make this scorer an untouched prospective Fit/Calibration pipeline, grant permission to reuse old Evaluation outcomes, or authorize broader use. It requires prospective justification and independent review before any new history contact.

Any result would be conditional on the declared synthetic mechanism, exact score implementation, supported conditioning vocabulary, and observed whole-history roster. The three model checkpoints are scorer variants, not three additional histories; Task 2 must predeclare all three or a subset and report model-seed results separately, never count repeated scorer-seed outputs as independent histories or select a favorable seed. This is not evidence of fleet/general-population performance, calibrated risk, causal effect, Gate V2 `MEASURABLE`, deployment readiness, or Sprint 18 recovery. Sprint 18 remains blocked before Cycle 4 with its existing 48-control construction reserve unchanged. Sprint 20 still rejects any tolerance-based acceptance rule; no soft pass or numerical gate is approved here.

## Source identity, runtime, and resource boundary

Task 7's evidence commit is `8c15f0204a3e495569b7f143dc109943e8b808de`. At audit time, the remote worktree `/tmp/sprint17-task7` was at that exact commit and `git status --porcelain` returned no output. Remote `git show` hashes of the tracked scorer dependencies were checked against local bytes. The five files in Task 7's frozen `metric_code_sha256` produce the exact recorded digest `b2d6af7505abe459a84f76f5279a6a5c4298e7c142901edd4596fcfe3ddeb88f` (canonical SHA-256 of ordered `[path, file_sha256]` pairs serialized as sorted-key compact UTF-8 JSON):

| Metric source file | SHA-256 |
|---|---|
| `src/synth/probe15.py` | `a08b3d5fb83001e1f5c43f4c56ff536bae85e41d494db289304aeb33a339242b` |
| `src/synth/events.py` | `85100f5e0aca47dd2e8b01a08c58f39d32be4f1eab469cdc38e4a4b57768169d` |
| `src/representation/attribution_metrics.py` | `c14f815c19d8e5791bce09cc0f07986e3a0f4c90c5c70cbf20f1c29f5a8d0488` |
| `src/representation/sprint17_ablation.py` | `52162995189985a30372e75c12538cfb15f93a2d36fe460551838c1c11e994af` |
| `experiments/sprint17_task7_b0.py` | `7969f3cacb77b4e9d0476250ba49ffe12f74a50f62157afc861179d9c141e825` |

The metric digest does not pin the entire inference stack. Several scoring dependencies differ between the historical execution commit and the current local working bytes; no semantic equivalence is inferred from a hash difference:

| Source file | SHA-256 at Task 7 commit | Current local SHA-256 |
|---|---|---|
| `src/representation/model.py` | `7051f5f76be0bcc8e50d6d5ee2941a57c12a3285406d0ef27883c596aa6600f0` | `cfe7578d2ca7d56d1aac332c83f2df6c4aa296f1c176473b73dddc8e522b95e1` |
| `src/representation/data.py` | `88ea3becbc8328c5b21518b1f957fe710096cb96ed0cd2416fe6f139439d5eb4` | `25c7090d64e6281b0d42a0427b67bfa0815049e7d2f22d9d30d7f65ef9538b22` |
| `src/representation/masking.py` | `c0efd33b24b8d01c067198f0bc04c0c90c1e99bad9a6a1ec006b37ce27c0465f` | `93b5453bbc605db0360d62df9fe868e9a7b8706989b8f304f98669a02059262e` |
| `src/representation/contracts.py` | `0b23080fc49f218484a6f7939bb9288f07b607686312e91c3b14d9d6b84464ed` | `8e4665954fae70023438a9c1dbc6f3be410d43896b3a3d62f5f69756e49049e2` |
| `src/representation/layers/normalization.py` | `bfb5f68f5352115c2e044c27999cdf3fecab7aa3ce2cd05ce021204f1f9edd8d` | `76fe6dde837d50adf6a70ad01568816c6aa96f10c9297e6711f597d3e1b75c6b` |
| `src/synth/chronicle.py` | `b5992d6af3c4387d18df746b43be17b4bfc1fac6938e8b44be35b6fb7c65dafd` | `f932e1a3c3bbc4bf7c8123e0ae9d3125200fedf0ce586d6d7eaf637d335f2cb6` |

The exact Task 7 commit remains available remotely and clean, but Task 7's `main()` always trains all three seeds, refits the bank, and recalibrates thresholds. It is therefore **not** an inference-only pilot entry point. Its `score_items` helper is usable on supplied `FileSample`s, and the checkpoint/inference APIs support restore-only scoring, but a future protocol must supply and hash-bind a role-neutral loader/scorer wrapper; it must not invoke `main()` or train/refit/calibrate.

The live remote runtime inventory was Python 3.12.13, PyTorch 2.14.0+cu130 with CUDA available, NVIDIA GeForce RTX 4060 Ti with 16,380 MiB total memory. The historical Task 7 run used this environment and reports the `uv.lock` unchanged across its execution commit. The record gives 1,821,698 parameters and 182,016,709 reference FLOPs at `T=512`; this FLOP helper intentionally constructs a batch without `file_samples`. The actual `score_items` collate includes `file_samples`, and the frozen model forward computes two contrastive-view embeddings from `sample.x` even in evaluation, although those embeddings do not feed `S_pred` or `S_pop`. Thus the reference FLOPs are not a complete inference-workload measurement. Task 7's 1,043.4-second run included training (about 330 seconds per seed) and does not isolate inference wall time or peak VRAM. No future-history throughput/memory envelope is established; Task 2 must bound input lengths/history count and verify a disposable inference-only path before any bound history is evaluated.

Any later scorer run must bind one clean, exact commit containing the generator, wrapper, score/metric code, and configuration; record the B0 training commit separately; verify the same remote runtime and all three or predeclared-subset checkpoint hashes immediately before use; and stop on any hash/source/config/vocabulary drift. Do not mix the current local inference stack with the historical metric hash and claim Task 7 parity.

## Audit boundary and references

This was a read-only source, record, asset-byte, and runtime audit. No checkpoint deserialization, project-module import, score, training, bank fit/refit, Calibration, generator, preflight, raw data/history inspection, role/seed binding, or remote mutation/checkout/pull occurred. The plan remains unchanged; Task 1 remains `[~]` pending independent review.

- [Sprint 21 plan and authorization](sprint-plans/sprint-21.md)
- [Sprint 19 estimands](exploratory-benchmark-estimands.md)
- [Sprint 20 `NOT_READY` decision](exploratory-tolerance-decision.md)
- [Sprint 17 Task 7 evidence](../artifacts/sprint-17/task-7.md)
- [Committed B0 summary](../experiments/sprint17-task7-b0-summary.json)
- [Task 7 driver](../experiments/sprint17_task7_b0.py)
- [V1 checkpoint loader](../src/representation/checkpoint.py)
- [V1 inference](../src/representation/inference.py)
- [Conditional normalization](../src/representation/layers/normalization.py)
- [Event and history metrics](../src/synth/events.py) and [probe aggregation](../src/synth/probe15.py)
