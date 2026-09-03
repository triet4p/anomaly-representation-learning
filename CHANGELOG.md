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

### Changed

- Updated `train_v1_representation.ipynb` from toy fixed-sample slice to production training with AdamW, cosine annealing learning rate scheduler, and gradient clipping.
- Updated `infer_v1_representation.ipynb` to restore trained model weights and normal reference bank directly from production checkpoints.
- Configured `.gitignore` to ignore `checkpoints/` and `*.pt` binary artifacts.
