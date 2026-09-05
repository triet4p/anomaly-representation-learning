# Sprint 6 Task 7 — Collect and interpret server results

## Local review set (`experiments/20260905/server-geometry-run-1/`)

Copied back via `scp` (bounded): input + executed notebooks for both stages, both stdout logs, cache `geometry-manifest.json` + `metrics.json`, analysis `metrics.json`, representative `figures/pca.png` (~0.5 MB total). Deliberately not copied: `embeddings.npz` (2.4 MB), `records.csv` (734 KB), `neighbors.csv` (25.0 MB), `geometry-diagnostics.zip` (13.2 MB) — sizes recorded here, content verified on server via manifest checksums and summary scripts. No checkpoint/data committed.

## Concrete results

- Final local commit `69d665e`; remote repo `/home/trietlm/anomaly-representation-learning` executed exactly `69d665e`; GPU RTX 4060 Ti, torch 2.14+cu130, device `cuda`.
- Checkpoint `checkpoints/v1_representation_20260904_01.pt` (step 7840); dataset `data/generated/production` (test head-5000, all normal).
- Extraction exit 0 / 12.4 s → 5000 records, manifest-linked cache. Analysis exit 0 / 29.6 s → metrics + 75k neighbor rows + 8 figures + diagnostics bundle.

## Geometry interpretation (5000 normal test files, d=128)

- Representation health: rank 127/128, effective rank 17.84, anisotropy 26.48, **not collapsed**, zero dead dims; embedding norm 9.28 ± 0.58 — a healthy anisotropic but full-rank space.
- Population geometry: mean distance to fitted normal references 3.57; same-class cosine 0.45 (test normals vs train normals); PCA variance 0.207 + 0.150 on the first two axes.
- Neighborhoods (k=15): same-fleet/same-label/same-split rates 1.0 (single-population run: all R01/normal/test, so these are vacuous); same-regime rate 0.18 — neighbors mix across regime summaries, i.e. local geometry is not regime-dominated.
- Separation block is null-valued as designed (no abnormal files in this head sample; supervised margin needs both classes).

## Limitations

- Head-5000 sampling covers only normal files from the top of production `test`; anomaly-family geometry and supervised separation are untested in this run — rerun with a mixed/offset sample for those.
- Single seed (`20260904`), single device (CUDA), remote torch 2.14 vs local 2.10 (checkpoint loaded cleanly; no version-sensitive behavior observed).
- `fleet_precheck` skipped (legacy pre-fleet dataset); all files default to bucket 0 with per-batch range guards enforced — recorded in the cache manifest.
