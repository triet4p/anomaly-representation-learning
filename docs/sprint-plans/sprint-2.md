# Sprint 2 — V1 Anomaly Representation Model

**Goal:** Implement and verify a variable-length, full-file representation model that learns normal context consistency with EMA latent prediction and normal population geometry with progressively weighted file-level contrastive learning.

**Status:** Active

**Source of truth:** [Project plan](../PLAN.md) · [Concept](../CONCEPT.md) · [Sprint 1](sprint-1.md)

## Scope and Contracts

Sprint 1 already supplies the synthetic data boundary. The model must consume complete `synth.schema.FileSample` objects and must not turn them into independent fixed semantic windows.

### Current synth contracts to preserve

| Path | Symbols / contract | V1 use |
|---|---|---|
| `src/synth/schema.py` | `FileSample.x [C,T]`, `SampleLabel`, `AnomalyMeta`, `PatchBatch`, `MaskResult` | Full-file identity/provenance; patch and anomaly mapping |
| `src/synth/config.py` | `SynthConfig`, `PatchConfig`, `MaskingConfig`, `ContrastiveConfig`, `SUPPORTED_CHANNEL_COUNTS=(3,6)` | Reuse generation settings and validate supported channels |
| `src/synth/patchify.py` | `Patchifier.patchify()`, `which_patches()`, `timestep_mask_to_patch_mask()`, `patch_to_timestep_scores()` | Variable `T`, `[N,C,W]` patches, valid lengths, padding, localization |
| `src/synth/masking.py` | `apply_masking()`, `compute_all_patch_stats()` | Fixed total ratio and random/info/block composition; never mask fully padded patches |
| `src/synth/contrastive.py` | `make_contrastive_views()` | Two views of one file; gain/offset/noise/edge-preserving shift only |
| `src/synth/dataset.py` | `DatasetBuilder`, `iter_materialized()`, atomic NPZ/ZIP shards | Lazy full-file loading, deterministic splits, no ragged-file padding on disk |
| `src/synth/generator.py` | `SessionGenerator.generate_normal()` / `generate_anomaly()` | Deterministic smoke fixtures and anomaly labels/masks |

The model-side batch boundary will pad only within a minibatch and will carry explicit `file_valid_mask [B,T_max]`, `patch_valid_mask [B,N_max]`, `patch_pad_mask [B,N_max,W]`, starts, and valid lengths. All pooling, attention, and losses must ignore invalid positions.

### Legacy inventory and reuse boundary

The legacy tree at `F:/DataImpact/TED-Hackathon-Agent/anomaly-detection/src/anomaly_detection` is read-only. The following decisions are deliberate:

| Legacy path and symbol | Decision | Reason |
|---|---|---|
| `layers/bottleneck.py:ResidualFactorizedVectorQuantizerEMA` | **Reuse narrowly, optional baseline only** | The EMA residual quantizer is a self-contained `[B,C,D]` codebook component. It may be wrapped for a comparison model, but it must not enter the primary V1 graph or dictate its loss. Preserve its per-level indices, EMA update, and commitment diagnostic only in the baseline. |
| `models/_archived/v6_rvq.py:V6RVQAnomalyDetectionModel` and `models/v6_rvq.py:V6RVQAnomalyDetectionModel` | **Reject model; no port** | The archived implementation is a sensor-factorized, fixed-window dual-branch autoencoder with shape/stat decoders; the active file is a production-license stub. Neither represents full-file latent prediction. |
| `models/glc_patch_rvq_mixer.py:GLCPatchRVQMixer` | **Reject graph** | It assumes macro `seq_len=240`, fixed patch count, context IDs, RVQ before a Tri-Mixer, and Gaussian overlap-fold raw reconstruction. These are reconstruction-centric and incompatible with variable `T` and no decoder. |
| `models/glc_mae_v9.py:GLCMaeV9` | **Reject graph; retain only conceptual comparison** | It uses fixed `seq_len`, `num_tokens`, random-only masking, and a CNN raw-patch decoder scored by reconstruction error. V1 predicts target latents and uses all three masking components. |
| `layers/patchify.py:Patchify` / `FlattenPatches` | **Reject implementation** | `tensor.unfold` requires a fixed-length input and emits sensor×patch tokens without the current `valid_len`/padding contract. Use `synth.Patchifier`. |
| `layers/masking.py:MaskToken` / `random_masking` | **Reject implementation** | It drops tokens with random-only shuffle/restore semantics and has no information-aware or block composition. A learned mask token is not required for the target-latent objective. |
| `criterions/base.py:BaseAnomalyDetectionLoss` | **Adapt convention only** | Keep the useful dict return with required `loss` plus diagnostics and serializable metadata. Rewrite all concrete losses for latent prediction and contrastive objectives. |
| `criterions/masked_reconstruction.py:MaskedReconstructionLoss` and `criterions/patch_rvq_reconstruction.py:PatchRVQReconstructionLoss` | **Reject** | Both optimize raw/normalized patch reconstruction; their errors must not become V1 anomaly scores. |
| `criterions/dual_head.py:DualHeadLoss`, `orthogonality.py:OrthogonalityLoss`, `flow_matching.py:FlowMatchingLoss` | **Reject for V1** | They supervise statistics/subspaces or flow matching, not the specified stop-gradient latent target and file-view contrastive pair. |
| `training/single_stage_trainer.py:SingleStageTrainer` | **Adapt orchestration pattern** | Reuse the model/criterion loop shape, optimizer setup, gradient clipping, and optional checkpoint hook. Remove MLflow coupling and fixed-window assumptions; add EMA update and λ schedule timing. |
| `training/glc_mae_trainer.py:GLCMaeTrainer` / `GLCMaeEvaluator` | **Adapt selected utilities** | Warmup/cosine schedule, history, best-state tracking, and dict-loss logging are useful. Masked reconstruction validation and reconstruction AUROC are rejected. |
| `models/base.py:BaseAnomalyDetectionModel`, `filter_model_kwargs`, `save_manual_checkpoint()` / `load_from_checkpoint()`; `models/factory.py:create_model()` | **Pattern only, not inheritance cut-and-paste** | Metadata-driven construction and checkpoint naming are useful patterns, but V1 needs ragged-file contracts, EMA state, optimizer state, and a reference bank. A small V1 registry/base may be introduced without importing the legacy package. |
| `training/datasets/raw.py:RawWindowDataset`, `RawWindowMosaicDataset`, `windows_from_signal`; `training/datasets/stat.py:StatWindowDataset`, `StatWindowMosaicDataset`, `write_sdf_to_mds` | **Reject data path** | These materialize fixed windows, optional robot/program metadata, Spark rows, and MDS shards. V1 reads `FileSample`/NPZ ZIP files and pads only a minibatch. |
| `layers/scorer.py:MADThreshold` | **Optional algorithm pattern** | Robust median/MAD thresholding may be applied independently to `S_pred` or `S_pop`, never to hide their separate values or replace reference-bank distances. |
| `notebooks/Training-GLCReconV9*.ipynb`, `Inference-GLCReconV9*.ipynb` | **Adapt workflow shape only** | Keep the readable configure → load → train/save → score/report flow and CPU-friendly checks. Reject Spark/Bronze/Delta/Mosaic wiring, context-ID codebooks, fixed windowing, and reconstruction verdicts. |

No writes, imports, or runtime dependencies on the legacy repository are permitted.

## Ordered Atomic Tasks

Status legend: `[ ]` pending · `[~]` in progress · `[x]` done. Only the first runnable task is marked `[~]`; all subsequent tasks remain pending until their prerequisites are complete.

### 1. [x] Define V1 configuration and public contracts

**Targets:** `pyproject.toml`; new `src/representation/__init__.py`; `src/representation/config.py`; `src/representation/contracts.py`; `tests/representation/test_contracts.py`.

**Change:** Add the internal ML dependencies required by the planned package (`torch`, `pydantic>=2`, and `einops`) without changing the existing `synth` API. Define validated V1 configuration (channel count, patch geometry, model dimensions, mask ratio, EMA decay, prediction/contrastive weights, ramp steps, kNN `K`) and typed batch/output contracts. Record exact tensor axes and invariants for padded files, masked patches, visible context, stop-gradient targets, file embeddings, and independent scores.

**Acceptance evidence:** A small contract fixture built from C=3 and C=6 `FileSample` objects validates; unsupported channel counts, invalid ratios, and inconsistent masks fail with actionable errors; imports work through the new package entry point; no contract requires a fixed file length.

**Focused test expectation:** `uv run pytest tests/representation/test_contracts.py -q` covering valid/invalid configuration, shape metadata, and mask/padding invariants.

**Artifact:** Implementer writes `artifacts/task_1_summary.md` with changed symbols, contract examples, command, and focused test result.

### 2. [x] Build the full-file dataset adapter and minibatch collation

**Targets:** `src/representation/data.py`; `tests/representation/test_data.py`.

**Change:** Implement lazy iteration over `FileSample`/`iter_materialized()` and a collator that invokes `synth.Patchifier` per file, pads only to the batch maxima, and returns `[B,C,T_max]` plus file/patch valid masks, starts, valid lengths, and patch padding masks. Preserve file IDs, labels, anomaly metadata/masks for evaluation but never use anomaly labels in training. Keep deterministic ordering/seeding and support both C=3 and C=6.

**Acceptance evidence:** Two files with different `T` collate without truncation; every real timestep and patch is recoverable by its metadata; padded positions never enter pooling or attention; a round trip through a Sprint 1 materialized shard yields the same file IDs and signal values.

**Focused test expectation:** `uv run pytest tests/representation/test_data.py -q` covering variable lengths, C=3/C=6, batch padding, lazy iteration, and metadata preservation.

**Artifact:** Implementer writes `artifacts/task_2_summary.md`.

### 3. [x] Integrate fixed-ratio random, information-aware, and block masking

**Targets:** `src/representation/masking.py`; `src/representation/data.py` masking hook; `tests/representation/test_masking_bridge.py`.

**Change:** Adapt `synth.masking.apply_masking()` to batched torch inputs without changing its NumPy source contract. Produce boolean `mask [B,N]` over valid patches only, retain per-sample composition counts, and enforce `round(total_mask_ratio * n_valid)` exactly whenever possible. Keep composition fixed for ablations and ensure block selections remain contiguous in patch order. Do not add a learned masking network.

**Acceptance evidence:** Random-only, random+info, random+block, and random+info+block runs have equal masked counts for the same valid patch counts; fully padded patches are never masked; same seed is reproducible; information strata and block spans are auditable in output diagnostics.

**Focused test expectation:** `uv run pytest tests/representation/test_masking_bridge.py -q` covering ratio, valid-only masking, composition, reproducibility, and contiguous blocks.

**Artifact:** Implementer writes `artifacts/task_3_summary.md`.

### 4. [x] Implement the local patch encoder

**Targets:** `src/representation/layers/patch_encoder.py`; `tests/representation/test_patch_encoder.py`.

**Change:** Add a channel-aware local encoder mapping `[B,N,C,W]` float patches to `[B,N,D]`, with `patch_pad_mask`-aware pooling over real timesteps and no sensor×patch flattening. Make it accept variable `N` after batch padding and expose finite outputs for valid input.

**Acceptance evidence:** C=3 and C=6 patches produce the configured `D`; changing only padded values cannot change a valid token; all-invalid patches are marked invalid and produce no usable embedding; no decoder or reconstruction output exists.

**Focused test expectation:** `uv run pytest tests/representation/test_patch_encoder.py -q` covering axes, channel counts, padding invariance, and finite outputs.

**Artifact:** Implementer writes `artifacts/task_4_summary.md`.

### 5. [x] Implement the variable-length sequence/context encoder

**Targets:** `src/representation/layers/sequence_encoder.py`; `tests/representation/test_sequence_encoder.py`.

**Change:** Add the sequence encoder over patch tokens with positional information that works for runtime `N`, accepts `patch_valid_mask` as key-padding information, and returns contextual patch latents `[B,N,D]`. Invalid padded patches must not affect valid outputs or file pooling.

**Acceptance evidence:** Files with different patch counts share one module; permutation/position behavior is explicit; padded-token perturbation leaves valid-token outputs unchanged under eval; attention never treats invalid patches as context.

**Focused test expectation:** `uv run pytest tests/representation/test_sequence_encoder.py -q` covering runtime lengths, key-padding masks, and finite `[B,N,D]` output.

**Artifact:** Implementer writes `artifacts/task_5_summary.md`.

### 6. [x] Implement the EMA target encoder

**Targets:** `src/representation/layers/ema.py`; `tests/representation/test_ema.py`.

**Change:** Add an `EMATargetEncoder` that mirrors the context encoder, starts with identical weights, is excluded from the optimizer, runs without gradients, and updates only after an optimizer step using `theta_target = m*theta_target + (1-m)*theta_context`. Expose a state-safe update and train/eval behavior.

**Acceptance evidence:** Target parameters have no gradients and are not optimizer parameters; initial outputs match; one update moves target weights toward context by the configured decay; repeated updates are deterministic and serialized in state dicts.

**Focused test expectation:** `uv run pytest tests/representation/test_ema.py -q` covering initialization, stop-gradient, update arithmetic, and optimizer exclusion.

**Artifact:** Implementer writes `artifacts/task_6_summary.md`.

### 7. [x] Implement the masked latent predictor head

**Targets:** `src/representation/layers/predictor.py`; `tests/representation/test_predictor.py`.

**Change:** Add a predictor mapping contextual visible-region states and positional/mask context to target latent estimates at masked patch positions. Preserve `[B,N,D]` alignment and provide a prediction mask that intersects requested masks with valid patches; do not reconstruct raw patches.

**Acceptance evidence:** Predictor output has target latent dimensionality; visible and invalid positions cannot contribute to the prediction loss; masked positions remain aligned to their original starts; gradients flow to the context path only.

**Focused test expectation:** `uv run pytest tests/representation/test_predictor.py -q` covering alignment, masked/valid selection, dimensions, and gradient routing.

**Artifact:** Implementer writes `artifacts/task_7_summary.md`.

### 8. [x] Wire the V1 representation model

**Targets:** `src/representation/model.py`; `src/representation/__init__.py`; `tests/representation/test_model.py`.

**Change:** Compose patch encoder, sequence/context encoder, EMA target encoder, predictor, mask bridge, and mask-aware file pooling into `V1RepresentationModel`. Accept the model batch contract and return context latents, stop-gradient target latents, masked predictions, masks, and per-view/file embeddings. Use `make_contrastive_views()` for same-file views; keep train/eval behavior explicit and omit every raw reconstruction head.

**Acceptance evidence:** A mixed-length C=3/C=6 batch completes forward on CPU; output keys and shapes match contracts; target branch has no gradient; file embedding is computed over valid patches only; anomaly labels are not consumed by forward.

**Focused test expectation:** `uv run pytest tests/representation/test_model.py -q` covering forward shapes, mixed lengths, target stop-gradient, valid pooling, and absence of reconstruction outputs.

**Artifact:** Implementer writes `artifacts/task_8_summary.md`.

### 9. [x] Implement masked latent-prediction criterion

**Targets:** `src/representation/criterion.py:LatentPredictionCriterion`; `tests/representation/test_prediction_criterion.py`.

**Change:** Adapt the legacy dict-loss convention into a criterion that computes distance between predicted masked latents and `stop_gradient(target_latents)` only where `mask & patch_valid_mask` is true. Return aggregate loss, per-patch prediction error, and auditable masked counts. Define an explicit empty-mask behavior without silently training on visible or padded tokens.

**Acceptance evidence:** Perturbing visible or padded targets cannot change loss; target tensors remain gradient-free; loss is finite and correctly normalized for one or many masked patches; empty valid mask follows the documented error/zero policy.

**Focused test expectation:** `uv run pytest tests/representation/test_prediction_criterion.py -q` covering masked reduction, stop-gradient, invalid-mask exclusion, and empty-mask behavior.

**Artifact:** Implementer writes `artifacts/task_9_summary.md`.

### 10. [x] Implement file-level contrastive criterion and progressive λ schedule

**Targets:** `src/representation/criterion.py:FileContrastiveCriterion, ProgressiveLambda`; `tests/representation/test_contrastive_criterion.py`.

**Change:** Implement same-file two-view contrastive learning over pooled file embeddings, with in-batch negatives and stable behavior for batch size one. Add `lambda_at(step)` with `lambda(0)=0` and a monotonic ramp to `lambda_max` over `ramp_steps`. Keep metadata-based positives out of the objective.

**Acceptance evidence:** Identical same-file views score as positives; unrelated in-batch files act as negatives; normalized embeddings and finite loss are produced; schedule endpoints and monotonicity are exact; batch-size-one behavior is explicit and tested.

**Focused test expectation:** `uv run pytest tests/representation/test_contrastive_criterion.py -q` covering positives/negatives, degenerate batches, temperature validation, and ramp endpoints.

**Artifact:** Implementer writes `artifacts/task_10_summary.md`.

### 11. [x] Implement joint criterion and representation-aware trainer

**Targets:** `src/representation/criterion.py:JointRepresentationCriterion`; `src/representation/trainer.py`; `tests/representation/test_trainer.py`.

**Change:** Combine `L_pred + lambda_at(step)*L_con`, expose each term and λ in logs, and adapt the legacy warmup/cosine/best-state loop to variable-length model batches. The trainer must perform optimizer step, then EMA target update, gradient clipping, deterministic CPU operation, and optional validation without changing target weights. Avoid MLflow and reconstruction-specific validation modes.

**Acceptance evidence:** One tiny train/validation epoch changes context weights, updates EMA only after the optimizer step, records prediction/contrastive/joint losses and λ, tracks/restores best state, and keeps validation target parameters unchanged.

**Focused test expectation:** `uv run pytest tests/representation/test_trainer.py -q` using a tiny deterministic model/batch and asserting update order, ramped λ, clipping path, history, and restore-best behavior.

**Artifact:** Implementer writes `artifacts/task_11_summary.md`.

### 12. [x] Implement independent context and population inference

**Targets:** `src/representation/inference.py`; `tests/representation/test_inference.py`.

**Change:** Add an inference API that scores one full file or a batch using `S_pred` from masked latent prediction/context mismatch and `S_pop` from a normal-only file/patch reference bank (kNN/prototype distance). Return both scores, patch/timestep localization using `Patchifier.patch_to_timestep_scores()`, and optional independent MAD thresholds. Do not fuse scores in V1; do not use anomaly labels to build the bank.

**Acceptance evidence:** Context and population scores are independently present and finite; changing the reference bank changes only `S_pop`; changing local masking/prediction changes only `S_pred`; reference-bank fitting rejects abnormal-labeled samples; variable-length localization aligns to original `T`.

**Focused test expectation:** `uv run pytest tests/representation/test_inference.py -q` covering score independence, normal-only bank, kNN `K` boundaries, patch-to-timestep mapping, and threshold behavior.

**Artifact:** Implementer writes `artifacts/task_12_summary.md`.

### 13. [x] Add V1 checkpoint and reference-bank serialization

**Targets:** `src/representation/checkpoint.py`; `tests/representation/test_checkpoint.py`.

**Change:** Implement atomic save/load for context encoder, EMA target encoder, predictor, optimizer/scheduler state, validated config, training step, and optional normal reference bank. Follow the useful legacy metadata pattern but do not depend on legacy `BaseAnomalyDetectionModel`; reject incompatible config/channel/schema versions before loading.

**Acceptance evidence:** Save/load round trip reproduces CPU outputs and both independent scores; EMA and optimizer states are restored; interrupted temporary writes do not replace a valid checkpoint; incompatible configuration and missing required keys fail clearly.

**Focused test expectation:** `uv run pytest tests/representation/test_checkpoint.py -q` covering round-trip state, metadata validation, atomic replacement, and optional bank persistence.

**Artifact:** Implementer writes `artifacts/task_13_summary.md`.

### 14. [x] Isolate the optional legacy RVQ baseline

**Targets:** `src/representation/baselines/rvq.py`; `tests/representation/test_rvq_baseline.py`.

**Change:** Provide an explicit opt-in baseline wrapper that reuses only the algorithm from `layers/bottleneck.py:ResidualFactorizedVectorQuantizerEMA` (or a narrowly vendored equivalent with provenance), maps its documented `[B,C,D]` interface separately, and reports its commitment/indices diagnostics. Keep it out of `V1RepresentationModel`, joint criterion, default trainer, and default inference path; do not port V6/GLC model graphs or decoders.

**Acceptance evidence:** Baseline can be constructed and smoke-forwarded only when requested; primary V1 imports and checkpoints do not instantiate or depend on RVQ; tests demonstrate the baseline is a comparison path and has no raw reconstruction verdict.

**Focused test expectation:** `uv run pytest tests/representation/test_rvq_baseline.py -q` covering opt-in isolation, shape contract, and registry/default-path exclusion.

**Artifact:** Implementer writes `artifacts/task_14_summary.md`.

### 15. [x] Add local training and inference notebooks

**Targets:** new `notebooks/train_v1_representation.ipynb`; new `notebooks/infer_v1_representation.ipynb`; `tests/representation/test_notebooks.py`.

**Change:** Mirror the readable legacy notebook sequence—configure, load Sprint 1 full-file shards, inspect batch contracts, run a tiny CPU training pass, save/load a checkpoint, fit a normal reference bank, and report separate context/population scores. Use local `uv` execution and generated small-profile data; omit Spark, Databricks, MDS, robot/program codebooks, and reconstruction heatmaps.

**Acceptance evidence:** Both notebooks have valid JSON and import only current package modules; each executes from a clean CPU environment against bounded synthetic data; displayed diagnostics include variable lengths, mask composition, λ, both scores, and patch/timestep localization.

**Focused test expectation:** `uv run pytest tests/representation/test_notebooks.py -q` performs notebook syntax/structure checks; the task also records a bounded `uv run jupyter nbconvert --to notebook --execute ...` result in its artifact.

**Artifact:** Implementer writes `artifacts/task_15_summary.md`.

### 16. [x] Run final CPU smoke, notebook execution, and design review gates

**Targets:** No new source target; review all Sprint 2 files, tests, notebooks, and task artifacts.

**Change:** Exercise the complete path with a tiny C=3/C=6 variable-length dataset: collate → mask → forward → joint loss → one trainer step/EMA update → checkpoint round trip → independent inference scores. Execute both notebooks on CPU and inspect the resulting artifacts. Review that no reconstruction decoder/score, fixed `T`, metadata positive, learned mask network, or default RVQ dependency slipped into V1.

**Acceptance evidence:** The end-to-end smoke and both notebook executions complete successfully on CPU; focused representation tests pass; task artifacts 1–15 exist and identify commands/results; review checklist explicitly confirms tensor/mask contracts, stop-gradient, EMA order, fixed ratio, progressive λ, normal-only reference bank, score separation, and checkpoint compatibility.

**Focused test expectation:** Run the bounded smoke command plus `uv run pytest tests/representation -q` only after all task-local tests are complete; run notebook execution with `uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3` for both notebooks. This is the sprint gate, not a substitute for task-local tests.

**Artifact:** Implementer writes `artifacts/task_16_summary.md` containing the commands, observed outputs, review checklist, and any explicit blocker. Mark this task `[x]` only after the owner reviews the evidence.

### 17. [x] Remediate masked prediction context usage

**Targets:** `src/representation/layers/predictor.py`; `tests/representation/test_predictor.py`.

**Change:** Replace the point-wise masked predictor path with cross-attention from mask/position queries to visible contextual patch states. Preserve valid-mask exclusion, target-leakage prevention, aligned `[B,N,D]` outputs, and no reconstruction head.

**Acceptance evidence:** A masked query responds to a nearby visible context perturbation while remaining invariant to padded/invalid-token perturbations; visible and invalid positions remain excluded from prediction loss.

**Focused test expectation:** `uv run pytest tests/representation/test_predictor.py tests/representation/test_model.py -q`.

**Artifact:** Implementer writes `artifacts/task_17_summary.md`.

### 18. [x] Remediate controllable batched contrastive view generation

**Targets:** `src/representation/model.py::_view_embeddings`; `tests/representation/test_model.py`.

**Change:** Persist the NumPy augmentation generator through model extra state, advance it across training forwards, restore it through state dicts, batch both augmented views, and place all view tensors on the input device/dtype.

**Acceptance evidence:** Training forwards generate changing views; identically seeded fresh models and serialized RNG state reproduce views; homogeneous view batches preserve shape/device/dtype; CUDA coverage is conditional.

**Focused test expectation:** `uv run pytest tests/representation/test_model.py -q`.

**Artifact:** Implementer writes `artifacts/task_18_summary.md`.

### 19. [x] Repair project-mode lock and package test reproducibility

**Targets:** `uv.lock`; `pyproject.toml`; `tests/representation/test_checkpoint.py`.

**Change:** Regenerate the lockfile from declared PyTorch/Pydantic/Einops dependencies, configure project-mode pytest to discover `src`, and update checkpoint state assertions for serialized model extra state.

**Acceptance evidence:** `uv lock --check` passes and project-mode `uv run pytest tests/representation -q` installs/runs the complete representation suite.

**Focused test expectation:** `uv lock`; `uv lock --check`; `uv run pytest tests/representation -q`.

**Artifact:** Implementer writes `artifacts/task_19_summary.md`.

### 20. [x] Stabilize notebook cell identifiers

**Targets:** `notebooks/train_v1_representation.ipynb`; `notebooks/infer_v1_representation.ipynb`; `tests/representation/test_notebooks.py`.

**Change:** Added stable unique `id` fields to every notebook cell and strengthened structure validation to require non-empty unique IDs without changing notebook logic.

**Acceptance evidence:** Focused structure checks pass; both notebooks execute through nbconvert with no nbformat MissingIDFieldWarning.

**Focused test expectation:** `uv run pytest tests/representation/test_notebooks.py -q`; execute both notebooks with `uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3`.

**Artifact:** Implementer writes `artifacts/task_20_summary.md`.

### 21. [x] Add explicit contrastive warmup configuration

**Targets:** `src/representation/config.py:V1Config`; `src/representation/criterion.py:ProgressiveLambda`; `src/representation/trainer.py`; `src/representation/checkpoint.py`; `tests/representation/test_contrastive_criterion.py`; `tests/representation/test_trainer.py`; `tests/representation/test_checkpoint.py`.

**Change:** Add validated `contrastive_warmup_steps` with a zero-step default. Make `ProgressiveLambda` return exactly zero through step `T_warmup`, then linearly ramp for the configured duration to `lambda_max` and remain monotonic. Build the trainer's default criterion from the model configuration so warmup, ramp, maximum weight, temperature, and prediction weight propagate through the notebook-facing `V1Config`. Preserve checkpoint compatibility by serializing the new field through `V1Config.to_dict()` and rejecting mismatched configurations.

**Acceptance evidence:** Boundary tests cover `T_warmup-1`, `T_warmup`, `T_warmup+1`, ramp completion, monotonicity, and invalid warmup values. Trainer metrics report the configured delayed λ schedule, and checkpoint round trips preserve the warmup metadata.

**Focused test expectation:** `uv run pytest tests/representation/test_contrastive_criterion.py tests/representation/test_trainer.py tests/representation/test_checkpoint.py -q`.

**Artifact:** Implementer writes `artifacts/task_21_summary.md` with schedule semantics, changed symbols, checkpoint/default compatibility, command, and focused test result.

### 22. [x] Make notebooks consume persisted generated datasets

**Targets:** `docs/SYNTH.md`; `notebooks/train_v1_representation.ipynb`; `notebooks/infer_v1_representation.ipynb`; `tests/representation/test_notebooks.py`.

**Change:** Document the production and bounded `uv` generation commands, resume behavior, ignored output layout, manifest/split semantics, and the stable `V1_DATA_ROOT` default (`data/generated/production`). Replace in-notebook sample generation with `FileDataset` reads from verified persisted `train`, `val`, and `test` shards. Train reads persisted train/validation examples; inference fits its normal reference bank from persisted train embeddings and scores persisted validation/test examples without passing anomaly labels to bank fitting. Both notebooks fail early with actionable manifest, split, incomplete, and empty-dataset errors.

**Acceptance evidence:** Notebook source contains no `SessionGenerator`, generator calls, or `DatasetBuilder`; structure tests require stable IDs, configurable root resolution, manifest checks, and required split readers. A real bounded dataset generated by the CLI was consumed successfully by both notebooks with `V1_DATA_ROOT`, then removed.

**Focused test expectation:** `uv run pytest tests/representation/test_notebooks.py -q`; generate a temporary bounded dataset with `uv run python -m synth.cli --output data/generated/task22-smoke --small --shard-size 4 --seed 7 --overwrite`, then execute both notebooks with `V1_DATA_ROOT=data/generated/task22-smoke` and remove all generated outputs.

**Artifact:** Implementer writes `artifacts/task_22_summary.md` with exact generation/execution commands, default path, split semantics, and cleanup evidence.

## Notes / Blockers

- Sprint 2 completed with the primary V1 path independent of the optional RVQ baseline.
- Final gate evidence is recorded in `artifacts/task_16_summary.md`; legacy files and generated notebook outputs remain untouched.
- Real-data deployment, Spark/Databricks integration, and reconstruction-centric baselines remain out of scope.
