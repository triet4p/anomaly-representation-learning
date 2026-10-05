# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Planned the prospective Sprint 18 iterative-v3 data contract correction: fixed-phase failure-ordinal P/W subtype alternation (ordinal excluded from the hash preimage) repairing the v2 variable-phase contradiction against the unchanged per-group imbalance bound, with its executable bound proof and disposable public-proof contract frozen for review before any implementation.
- Added a restore-only Sprint 21 disposable scorer proof entry point with pinned lock, full-seed configuration, source-derived conditioning, metric and role-binding provenance gates, and production mask-seed rejection.
- Added a versioned research-only Sprint 18 control/eligibility mechanism diagnostic, result/attempt schema, and review-gated one-shot runner for the fixed 918100–918107 block; Task 61 verifies instrumentation only with disposable fixture seed 93061.
- Added per-checkpoint immutable Fit-bank digests to Sprint 21 pilot provenance and bound the GPU-memory check to Torch's measured device property.
- Added exact restored-bank digest checks before scoring and full-roster-uncomputable summaries for operational aborts.
- Added the separately versioned Sprint 22 pilot runner and disposable minimum-48/scorer proof harness; the authorized v1 remote invocation failed closed before a proof record was written, so no pilot authorization is implied.
- Added the research-only Sprint 22 H=32 pilot protocol and frozen pending-review binding; neither authorizes history contact or execution.
- Stabilized joint prediction and contrastive training: enforced input normalization parity across prediction and contrastive branches via `ConditionalBatchNorm`, bounded contextual and EMA-target latent scales via top-level LayerNorm, and isolated whole-file contrastive optimization via a dedicated 2-layer MLP projection head (`contrastive_projector`).
- Calibrated contrastive task parameters: increased augmentation jitter/cutout bounds and raised temperature to $\tau=0.2$, eliminating trivial whole-file contrastive saturation.
- Implemented stationary model selection policy: added `stationary_joint_loss` with fixed target contrastive weighting ($\lambda_{\max}$) and pre-warmup overrides in `JointRepresentationCriterion` and `RepresentationTrainer`, preventing warmup states from outranking joint-trained checkpoints.
- Implemented coherent checkpoint synchronization: ensured saved model weights, optimizer state, scheduler state, global step counter, and normal reference bank always describe one coherent training state, eliminating Frankenstein checkpoint states.
- Added rich diagnostic tracking: exposed raw/weighted losses, stationary joint loss, positive-pair similarity, empirical negative-pair similarity, contrastive margin, learning rate, unclipped gradient norm, and context/target/predictor/file latent norms across training and validation history.
- Exposed user-configurable `lambda_max` with conservative rerun default `0.1` across `V1Config`, training notebooks, schedule ramps, stationary selection, and progress bars.
- Corrected notebook resume ergonomics: derived `start_epoch = start_step // num_batches_per_epoch`, bounded training loop ranges, and added clean skip guards when configured total epochs are already completed.
- Produced versioned Kaggle export packages (`kaggle-20260904-01`, `kaggle-20260904-02`, and replacement `kaggle-20260904-03`) with complete file catalogs and SHA-256 checksum manifests.

- Added `TrainingParams` and `InferenceParams` dataclasses to training and inference notebooks for direct in-notebook parameterization in VSCode and JupyterLab.
- Added `StreamingBatchDataset` for streamingly collating minibatches from sharded dataset archives in constant $\mathcal{O}(1)$ host RAM.
- Added automatic CUDA device detection and memory diagnostic reporting across both representation notebooks.
- Added comprehensive anomaly detection evaluation metrics (MAD thresholds, precision, recall, F1, AUROC, per-family breakdown) to the production inference notebook.
- Added transparent batch device placement helper `_move_batch` to `RepresentationInference`.
- Added `in_memory` caching option to `StreamingBatchDataset` and parameter dataclasses, enabling fast in-memory loading and global per-epoch random shuffling for datasets $\le 25\text{K}$ samples.
- Added integrated `tqdm` progress tracking across RAM preloading, batch training steps, validation passes, reference bank fitting, and test inference scoring.
- Added `tqdm>=4.70.0` project dependency.
- Unified training and inference workflows into single canonical notebooks (`train_v1_representation.ipynb` and `infer_v1_representation.ipynb`) with automatic Kaggle vs. Local environment path detection.
- Added `ConditionalBatchNorm` layer (`src/representation/layers/normalization.py`) using `dict[str, torch.Tensor]` inputs and outputs with hierarchical fleet fallback `(robot, program) -> robot -> fleet` and variable-length sequence mask-awareness.
- Extended synthetic data generation (`src/synth/`) with `FleetConfig` to model distinct robot calibration offsets and program physical operating envelopes, saving `robot_idx`, `program_idx`, `robot_code`, and `program_number` on `FileSample` and NPZ shards.
- Wired `robot_idx` and `program_idx` through `RepresentationBatch`, collation, and `V1RepresentationModel` forward pass.
- Added `scripts/package_kaggle_dataset.py`, `scripts/upload_to_kaggle.sh`, and `scripts/upload_to_kaggle.ps1` to automate dataset packaging, metadata generation, and Kaggle cloud kernel launches.
- Added `_resolve_shard_path` and automatic Kaggle mount detection in `synth.dataset` to bypass container-level SHA-256 checks on read-only cloud mounts where archive re-compression changes outer hash signatures.
- Added support in `synth.dataset.iter_materialized` and notebook Cell 1 for Kaggle's automatic extraction layout, transparently reading `.npz` files directly from unzipped `shard-XXXXX/` directories without zip overhead.

### Changed
- Vectorized `compute_all_patch_stats` in `synth.masking` to compute patch variance, peak-to-peak range, and derivative energy in a single 2D NumPy operation, accelerating information-aware patch stratification by over $13\times$.

- Updated `train_v1_representation.ipynb` from toy fixed-sample slice to production training with AdamW, cosine annealing learning rate scheduler, and gradient clipping.
- Updated `infer_v1_representation.ipynb` to restore trained model weights and normal reference bank directly from production checkpoints.
- Configured `.gitignore` to ignore `checkpoints/` and `*.pt` binary artifacts.

### Fixed

- Corrected Sprint 21 pilot program-share calculation and scorer conditioning: IDs now resolve from the pinned robot/program map, and missing serialized indices are rejected without changing the manifest schema.
- Fixed the Sprint 22 pilot runner's binding checkpoint-size guard by defining its frozen 18,239,321-byte checkpoint size; no pilot contact was authorized.
- Fixed Sprint 18 Task 68 source checks to accept only declared UTF-8 text LF/CRLF checkout conversion while binding canonical hashes, tracked Git blobs, and the corrected Main checkpoint receipt.
- Bound Sprint 18 Task 68 to the observed PyTorch `2.14.0+cu130` runtime and the exact locked-venv/source-path entrypoint.
- Fixed the Sprint 18 bound entrypoint's experiment imports by binding the repository root alongside `src` and `experiments` while retaining exact-path enforcement.
