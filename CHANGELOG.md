# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
