# Sprint 6 Task 7 — Collect and interpret server results (corrected rerun)

## Local review set (`experiments/20260905/server-geometry-run-1/`)

Copied back via `scp` (bounded): input + executed notebooks for both stages, both stdout logs, cache `geometry-manifest.json` + `metrics.json`, analysis `metrics.json`, representative `figures/pca.png` (~0.5 MB total). Deliberately not copied: `embeddings.npz` (2.4 MB), `records.csv` (734 KB), `neighbors.csv` (25.0 MB), `geometry-diagnostics.zip` (~13.2 MB) — sizes recorded here, content verified on server via manifest checksums and summary scripts. No checkpoint/data committed. Refreshed from the corrected `a326e48` rerun (prior `69d665e` run preserved on the server as `server-geometry-run-1-prev/`).

## Concrete results

- Executed source commit `a326e48`; remote repo `/home/trietlm/anomaly-representation-learning`; GPU RTX 4060 Ti, torch 2.14+cu130, device `cuda`.
- Checkpoint `checkpoints/v1_representation_20260904_01.pt` (step 7840); dataset `data/generated/production` (test head-5000, all normal).
- Extraction exit 0 / 12.4 s → 5000 records, manifest-linked cache. Analysis exit 0 / 20.2 s → metrics + 75k neighbor rows + **12 figures** + diagnostics bundle.

## Geometry interpretation (5000 normal test files, d=128)

- Representation health: rank **127/128 (near-full, not full)**, effective rank 17.84, anisotropy 26.48, **not collapsed**, zero dead dims; embedding norm 9.28 ± 0.58 — a healthy anisotropic near-full-rank space.
- Population geometry: mean distance to fitted normal references **within the extracted test cache (self-excluded)** 3.5745; PCA variance 0.207 + 0.150 on the first two axes.
- Checkpoint-bank distance (separate provenance): cache `S_pop` mean 7.3446 ± 0.7053 over the 5000 rows (distance to the checkpoint's fitted normal bank); `S_pred` mean 0.0735 ± 0.0306.
- Within-cache similarities (capped budgets): same-class count 4999, mean 0.4162; different-class count 0 in this all-normal head sample, margin null.
- Neighborhoods (k=15): same-fleet/same-label/same-split rates 1.0 (single-population run: all R01/normal/test, so these are vacuous); same-regime rate 0.18 — neighbors mix across regime summaries, i.e. local geometry is not regime-dominated.

## Limitations

- Head-5000 sampling covers only normal files from the top of production `test`; anomaly-family geometry and supervised separation are untested in this run — rerun with a mixed/offset sample for those. All-normal limitation explicit.
- Single seed (`20260904`), single device (CUDA), remote torch 2.14 vs local 2.10 (checkpoint loaded cleanly; no version-sensitive behavior observed).
- `fleet_precheck` skipped (legacy pre-fleet dataset); all files default to bucket 0 with per-batch range guards enforced — recorded in the cache manifest.
