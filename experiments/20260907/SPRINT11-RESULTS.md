# Sprint 11 results — Chronological Factory Geometry and Early-Warning Validation (all real executions)

Consolidated final report for Sprint 11 (Tasks 1–36 evidence, Batch A1–F2 gates).
No model was trained, re-run, or re-analyzed for this summary; every number below
cites its exact accepted task artifact (`artifacts/sprint-11/task-<N>.md`) and the
Batch F3 evidence gate plus the sprint-wide deep review remain before sprint
completion. Training, analysis, and plan commits are recorded separately and never
conflated (see §2).

Layout mapping:

- `assets/` — the only bounded representative figures committed for this report
  (7 PNGs, 264 KB, sha256-verified byte-identical to the ignored review copies).
- Review-sized JSON/CSV aggregates, executed notebooks, logs, manifests, and run
  scripts live **only** in the ignored local mirrors
  `../../artifacts/sprint-11/batch-D|E1|E2|E3|F1|F2/` (never committed; digests
  verified against server copies on transfer).
- Checkpoints, references, datasets, embeddings, row-level scores, and bulk traces
  remain **server-only** at the recorded paths (§3); they were never transferred
  and are not committed.

## Contents

- `assets/task33_effrank_robot.png`, `assets/task33_effrank_family.png` — geometry
  effective-rank bars (Batch F1).
- `assets/task34_recall_by_family.png`, `assets/task34_fp_by_robot.png` — static
  family recall / false-positive slices (Batch F1).
- `assets/task35_calibration_7d_control.png`,
  `assets/task35_calibration_7d_hybrid.png`,
  `assets/task35_false_alerts_by_robot.png` — 7-day calibration and false-alert
  slices (Batch F2).

## 1. Goal and methodology

Sprint 11 replaces independently timestamped synthetic files and the near-chance V1
geometry with (i) a deterministic 3–6 month shared-unit factory simulation with
causal scheduling and robot health, (ii) a robot-program-conditioned V2 latent
geometry with localized synthetic boundary learning, and (iii) server-verified
static detection plus calibrated one-day/seven-day early-warning evidence.
Methodology: `../../docs/LATENT_GEOMETRY_METHODOLOGY.md` (normal-manifold core
plus localized synthetic boundary learning; regularized hierarchical Mahalanobis
geometry; empirical/conformal confidence kept separate from the censored
survival/risk layer). Sprint plan:
`../../docs/sprint-plans/sprint-11.md`.
The predeclared Sprint 11 hypothesis is tested by four gates G1–G4 (§9); the
integrated verdict is **0/4 pass** (G1 FAIL/FAIL, G2 UNAVAILABLE, G3 FAIL/FAIL,
G4 FAIL/FAIL), an honest, reproducible negative result (§9–§11).

## 2. Commit provenance (training vs analysis vs plan kept distinct)

| role | commit | scope |
|---|---|---|
| client build (Tasks 1–23 source/docs/tests/notebooks/configs) | `a2fb8f9` | `src/synth/{chronicle,health,scheduled,scheduler,splits,temporal}.py`, 12 `src/representation/v2_*.py`, 4 canonical notebooks, `experiments/v2_staged/*.yaml`, 21 focused test files, methodology + sprint plan (`task-24.md`) |
| dataset/audit fix (arrival jitter) | `d31d172` | `src/synth/config.py`, `scheduler.py`, `chronicle.py`, scheduler tests, methodology §20.3; **execution commit for Tasks 25–26** (`task-24.md`, `task-26.md`) |
| 2-epoch contract device fixes | `91f5a5c`, `69ed4d5` | CUDA placement fixes + regression test; **execution commit for Task 27 is `69ed4d5`** (`task-27.md`) |
| 5-epoch stage-schedule + selection source | `86812d3` | `STAGE_BOUNDARY_SCHEDULE`, raw variance/covariance reporting, `apply_balance_selection`; **execution commit for Tasks 28–29** (`task-28.md`, `task-29.md`) |
| full-stage schedule freeze | `1f88da0d793efc452bc33a4a31d3a9fd73b3f990` | explicit `boundary_warmup_steps: 50, boundary_ramp_steps: 100` in `full_hybrid.yaml`; **frozen training commit for Tasks 30–32** (`task-29.md`, `task-30.md`, `task-31.md`, `task-32.md`) |
| Batch F1 analysis (geometry + static) | `0c89be839311d8a33cc7925f9f5a0f046d2fd09c` | adds only `src/representation/v2_batch_f1.py`, `experiments/v2_staged/batch_f1_analyze.py`, `tests/representation/test_v2_batch_f1.py`; no model/inference behavior change (`task-33.md`, `task-34.md`) |
| Batch F2 analysis (early warning) | `e669e2ab635b91b6afbc6e4d9262688b0025d0e8` | adds only `src/representation/v2_batch_f2.py`, `experiments/v2_staged/batch_f2_analyze.py`, `tests/representation/test_v2_batch_f2.py`; one grad-scope fix, no behavior change (`task-35.md`) |
| plan closeouts | `a832bbc` (Batch D), `43f78e6` (Tasks 28–29), `b783ee0` (Tasks 33–34), `e4cd87c` (Tasks 35–36) | docs-only task-status updates, each after its evidence gate |

Local == remote (`git rev-parse HEAD` equal both sides) was verified before every
server run; remote pulls were fast-forward only; unrelated remote work (two
pre-existing untracked experiment dirs) was preserved and never touched
(`task-24.md`, `task-30.md`).

Reference/config/data hashes:

- Dataset manifest sha256
  `cf788f92d1b5947d40cb40c03bc7fb05676b7a304adacb0dc1bfb29e7c0e7135`,
  `config_hash ece4183ef68a` (Tasks 25–36, both run manifests).
- Control checkpoint sha256
  `f91030ec787bb81068949899470959119c3999adda0baf29587e326b46a1f0fb`
  (984 841 B); hybrid checkpoint sha256
  `83d94b9daf8a6c873ea8af84a1001885021e27a0a9e39a3f136fc97366917aca`
  (984 841 B) (Tasks 30–32, both run manifests).
- `experiments/v2_staged/full_control.yaml` sha256
  `79266fd295aaa5996c88bee8925a5ae6c5c5e6f8ea6b83fdc30ff31971b3dc59`
  (64 hex, matches `task-30.md` and the committed YAML at
  `../../experiments/v2_staged/full_control.yaml`, reverified by local
  `sha256sum`).
- `experiments/v2_staged/full_hybrid.yaml` sha256
  `0dafe893d960a47051a0d091d1fb963da284f325c22aae00366a455738cdbceb`
  (`task-29.md`, `task-31.md`). A superseded first freeze `cdf73751…`
  (coefficients only, inherited warmup 500 + ramp 2000 that would hold alpha at
  0 over the ~300-step run) was caught by re-gate and correctively frozen; it
  never ran (`task-29.md`).

## 3. Server roots (server-only; never transferred, never committed)

Remote host `trietlm@192.168.30.244`, repo
`/home/trietlm/anomaly-representation-learning`, GPU
`NVIDIA GeForce RTX 4060 Ti`, torch 2.14.0+cu130, seed 0 throughout
(`task-24.md`, Tasks 27–32):

- Dataset: `data/generated/sprint11-server/` — `files/shard-00000.zip`
  (`ddad869b…`), `files/shard-00001.zip` (`b0a024b9…`), 11 MB total
  (`task-25.md`, `task-26.md`).
- Control training: `/tmp/sprint11-task30-control/full/control-normal-only/default/`
  (`v2_checkpoint.pt` + `provenance.json` + `full/matrix_summary.json`).
- Hybrid training: `/tmp/sprint11-task31-hybrid/full/hybrid-boundary/default/`
  (same layout).
- Extraction: `/tmp/sprint11-task32-extract/infer-{control,hybrid}/` and
  `/tmp/sprint11-task32-extract/traj-{control,hybrid}/` (executed notebooks,
  metrics, provenances, figures).
- Analysis reruns: `/tmp/sprint11-task3334-f1/` (+ `-rerun`),
  `/tmp/sprint11-task3536-f2/` (+ `-rerun`).
- In-memory-only artifacts (never persisted to disk anywhere): patch embeddings
  and row-level score tensors (`task-32.md`, `task-33.md`, `task-35.md`).

## 4. Generated calendar, data, and split counts (Tasks 25–26)

Locked server profile at `d31d172` via the public entry point
(`python -m synth.cli --chronological --profile server`, seed 0): 3 robots over
routes-A/B line topology, 300 units, 6 h mean arrival spacing with one-slot
deterministic arrival jitter (21600 s; the jitter fix for zero cross-robot
overlap is recorded in `task-26.md`, superseded grid-locked manifest `0f363ea0…`
overwritten in place, never consumed), 90-day calendar, 45-day cutoff, 7-day
precursor quarantine (`task-25.md`).

Persisted counts (manifest `cf788f92…`; audit 49/49 PASS through the public
reader, `task-26.md`): **600 files — 381 normal / 219 abnormal; dev-train 44,
dev-val 11, static test 364, temporal view 523, quarantined 470**
(`precursor_horizon` 389, `failure_event` 23, `maintenance_transition` 58);
episodes — degradation 24, failure 23, maintenance 23; precursors —
`failure_within_1d` 82 files, `failure_within_7d` 423 files.
Split discipline verified: dev (55) all NORMAL, pre-cutoff, quarantine-free;
dev∩static, dev∩temporal, train∩val all empty; train/val is the chronological
80/20 split of verified-healthy pre-cutoff files; static holds every abnormal
from any time plus every post-cutoff normal; temporal view exactly
chronological (`task-26.md`).
Audit also verified: no same-robot overlap, 36 cross-robot async overlapping
pairs, causal route chaining (demo `unit-0000`: route-B
`robot-02/program-03 → robot-03/program-01`), health `G == γH` on 600/600 files
with program-sensitive manifestation, failure/maintenance causality, 600/600
`encoder_inputs()` leakage-free, and 600/600 decoded byte-exact determinism
(`task-26.md`).
Client smoke (Task 23, tiny profile, ignored
`artifacts/sprint-11/task-23-smoke/`): 120 files (28 dev-train / 7 dev-val /
73 static / 83 temporal / 43 quarantined); found and fixed 3 notebook defects
before any server work; 2-epoch smoke AUROC 0.33 correctly claimed as
untrained-geometry artifact, never as quality evidence (`task-23.md`).

## 5. Staged execution: 2-epoch contracts, 5-epoch balance, 50-epoch full runs

All stages ran the committed staged runner (`src/representation/v2_staged.py`)
with chronological dev views, seeded-shuffled in-epoch order, and sealed test
views asserted disjoint and never scored or fitted (887 forbidden ids at
contract/balance scale; `task-27.md`, `task-28.md`).

- **Task 27 — 2-epoch contracts** at `69ed4d5` (exit 0, 3.3 s stage wall):
  both cells finite (loss 0.5432→0.3958, grad norms 0.82/0.91, cond median
  476.2 / worst 6938.8, bit-identical restore, 984 841 B checkpoints);
  variants honestly coincide (alpha 0.0 — warmup 500 exceeds the 12-step run);
  two genuine CUDA placement defects fixed at source with regression tests
  (15 focused tests green) after 3 failed runs whose outputs were all discarded
  (`task-27.md`).
- **Task 28 — 5-epoch coefficient matrix** at `86812d3` (4/4 exit 0, 6.87 s):
  stage-specific schedule warmup 5 + ramp 10 (alpha at max by step 15) so every
  varied coefficient is active and observable; all cells finite, cond < 1e4,
  rank 5.6–7.8/32, background leak ~1e-6 ≪ latent scale ~1.0:

  | cell | variant | final α | best stationary (val final) | worst cond | eff. rank | gap@2.0 | monotone | bg leak / scale |
  |---|---|---|---|---|---|---|---|---|
  | balance-a | hybrid (α_max 0.5) | 0.5 | 329.33 (329.33) | 5.64e3 | 6.20 | −0.0004 | no | 8.5e-7 / 1.0011 |
  | balance-b | hybrid (α_max 1.0) | 1.0 | 332.71 (332.71) | 6.03e3 | 5.59 | +0.0029 | yes | 5.5e-7 / 0.9982 |
  | balance-c | hybrid (reduced bg wt) | 1.0 | 344.45 (344.45) | 3.48e3 | 6.86 | −0.0002 | no | 8.0e-7 / 1.0058 |
  | balance-control | control (α_max 0.0) | 0.0 | 554.56 (585.69) | 5.42e3 | 7.75 | −0.0112 | no | 1.2e-6 / 1.0154 |

  (`task-28.md`; selection in `task-29.md`.)
- **Task 29 — coefficient selection gate** (mechanical, predeclared
  `apply_balance_selection`, test views never opened): **balance-b WINS**
  (only cell passing all of finite / schedule-active / geometry / no-collapse /
  separation / ordering / localization); balance-a and balance-c REJECTED on
  separation + ordering; control is a valid reference (its negative gap
  −0.0112 confirms the boundary signal comes from hybrid training, not the
  probe). The winning gap is honestly small (+0.0029 at 5-epoch scale — not a
  detection claim); validation stationary favored hybrids but did not decide
  the winner (stationary-best balance-a was rejected on boundary function).
  Full-run config frozen complete with checksum before Tasks 30+
  (`task-29.md`).
- **Task 30 — 50-epoch normal-only control** at `1f88da0` (exit 0, 300 steps,
  shared 17.51 s stage wall, 34.6 MB peak CUDA): alpha held 0.0 all 50 epochs;
  stationary-best **554.56 at epoch 3** (coherent best-state restore; training
  ran all 300 steps, full history saved); train loss 0.5432→0.1108;
  checkpoint `f91030ec…`, bit-identical restore, frozen dev-train hierarchy
  (median cond 381.3, worst 5418.5, all finite); training-time probes rank
  7.75/32, anisotropy 7.39, scale 1.0154 (`task-30.md`).
- **Task 31 — 50-epoch hybrid boundary** at `1f88da0`, identical
  commit/data/seed/compute contract (exit 0): alpha 0.0 epochs 1–8, first
  nonzero epoch 9 (steps cross warmup 50), saturated 1.0 from epoch 26
  (step 156 ≥ max_step 150) through epoch 50; stationary-best **344.43 at
  epoch 25**; train loss 0.5432→1.1766 (includes the engaged boundary term —
  recorded, not compared); checkpoint `83d94b9d…`, bit-identical restore,
  frozen hierarchy (median cond 318.5, worst 4574.1); probes rank 8.95/32,
  anisotropy 5.92, scale 1.0215. No outcome-driven retuning (`task-31.md`).
- **Task 32 — bounded extraction** at `1f88da0` (4/4 `nbconvert --execute`
  exit 0, 0 failures): canonical `infer_v2_static` + `trajectory_v2_early_warning`
  against both checkpoints with restored-checkpoint references (never refit),
  dev-val-only / allowed-fit-only calibration; only digest-verified
  review-sized notebooks, logs, manifests, metrics, summaries, and figures
  transferred (ignored `artifacts/sprint-11/batch-E3/`, 706 KB, `ALL DIGESTS
  MATCH`); no interpretation made at extraction — values carried verbatim for
  Tasks 33+ (`task-32.md`).

## 6. Control vs hybrid geometry (Task 33, analysis `0c89be8`, training `1f88da0` frozen)

Pool: 12 964 valid patches per variant (identical masks/counts); temporal view
untouched; latents/scores/masks stayed in server memory
(`../../artifacts/sprint-11/task-33.md`; aggregates in ignored
`artifacts/sprint-11/batch-F1/task33_geometry.json` `8a1560b5…`,
`task33_slices.csv` `16cb51fc…`, `run_manifest.json` `4da8aa57…`):

| measure | control | hybrid | Δ (hybrid − control) |
|---|---|---|---|
| effective rank (d=32) | 16.84 | 15.39 | −1.45 (per-robot uniform, no outlier) |
| anisotropy (smax/smin) | 2.18e7 | 2.35e7 | +1.7e6 (near-singular directions retained both) |
| covariance cond median / worst (frozen, float64, all levels) | 600.7 / 5418.6 | 415.1 / 4574.1 | −185.6 / −844.6; all finite (level detail in artifact) |
| mixture/fallback coverage | 84.8% finest / 15.2% pair / 0% robot/fleet | identical | — (same dev-fitted hierarchy; 6 mixture comps per pair) |
| within/between ratios (robot / pair / regime / label) | 0.979 / 0.986 / 1.120 / 0.965 | 0.983 / 0.987 / 1.077 / 0.969 | ≈1.0 throughout — no conditional clustering |
| clean/corrupt ordering (64 files, 193 patches) | 0.523, gap −0.034 | 0.420, gap −0.024 | hybrid below chance; background MSE ~5e-7/1e-7 stable |
| sparse-evidence tail-hit (dev-95%) | 84/219 (0.384) | 81/219 (0.370) | −0.014 |
| severity tail means low(<0.5, n=27) / mid[0.5,1.0) (no high files) | 30.86 / 25.57 | 25.33 / 18.50 | lower-severity bins carry *higher* energy — no monotone ordering |
| trajectory continuity (successive/random, 3 robots) | 1.029 / 0.972 / 1.048 | 0.944 / 0.972 / 1.039 | successive steps indistinguishable from random pairs |

![Effective rank by robot](assets/task33_effrank_robot.png)
![Effective rank by family](assets/task33_effrank_family.png)

Rerun to a second output root produced byte-identical
`task33_geometry.json`/`task33_slices.csv`; 7 focused tests green
(`task-33.md`). Effective rank, anisotropy, and conditioning are reported
strictly as representation health — never equated with separation (§9 G1).

## 7. Static detection and localization (Task 34, sealed 364-file static test)

Restored references, operating point at the 95th percentile of **dev-val
confidences only (n=11)** — fragile by construction (hybrid threshold differs
~3e-8 from the extraction by GPU cross-process noise at ~1e-7 against a coarse
n=11 quantile, flipping 4 boundary calls; documented as a limitation, not
silently fixed). Control operating point reproduces the Task 32 extraction
exactly; hybrid AUROC/AUPRC reproduce exactly
(`../../artifacts/sprint-11/task-34.md`; ignored `batch-F1/task34_static.json`
`0519a917…`, `task34_slices.csv` `8431f49d…`):

| metric (control → hybrid; Δ = hybrid − control) | control | hybrid | Δ |
|---|---|---|---|
| AUROC (confidence) | 0.4823 | 0.4709 | −0.0114 |
| AUPRC (confidence) | 0.5817 | 0.5862 | +0.0045 |
| AUROC (tail_energy) | 0.4788 | 0.4738 | −0.0050 |
| F1 @ validation point | 0.2132 | 0.1502 | −0.0630 |
| precision / recall | 0.5472 / 0.1324 | 0.5588 / 0.0868 | — |
| tp / fp / fn / tn | 29 / 24 / 190 / 121 | 19 / 15 / 200 / 130 | −10 / −9 / +10 / +9 |
| operating threshold | 0.9107 | 0.9286 | — |
| **argmax localization hit rate (219 masked)** | **0.237 (52/219)** | **0.315 (69/219)** | **+0.078 — the single hybrid improvement in Sprint 11** |
| top-3 overlap mean [median; p75] | 0.181 [0.000; 0.245] | 0.258 [0.021; 0.401] | localization without separation |

- **Signals are inverted:** median tail_energy abnormal 27.14 vs normal 32.95
  (control); 17.58 vs 34.48 (hybrid). Elevated fraction 0.857 vs 0.950;
  0.227 vs 0.938. Confidence 0.607 vs 0.679; 0.625 vs 0.643. Static
  abnormalities sit *closer* to the frozen references than normals on average.
- **Matched per-file deltas (n=364):** confidence mean −0.026 (median −0.0357,
  std 0.174); tail_energy mean −6.43 (median −6.23, std 7.88). Hybrid lowers
  energies without improving ranking.
- **False positives:** robot-02 dominates both (control 5/30, 14/76, 5/39;
  hybrid 4/30, 7/76, 4/39); worst pair robot-02/program-03 8/46 → 2/46
  (fewer calls, fewer hits overall); by regime active 13/57 → 9/57.
- **Family recall — no family detected under either variant** (hybrid ≤
  control in 7/8, tie in 1): control — contextual_replacement 4/26,
  cross_channel 6/33, freq_phase 4/23, missing_event 3/18, over_regularity
  2/26, realistic_stuck 5/32, subtle_drift 3/37, wrong_transition 2/24;
  hybrid — 3/26, 3/33, 2/23, 1/18, 2/26, 5/32, 1/37, 2/24.
- Severity slices mirror family recall; no severity bin is recovered.

![Static recall by family](assets/task34_recall_by_family.png)
![Static false positives by robot](assets/task34_fp_by_robot.png)

## 8. Chronological early warning and calibration (Task 35, untouched 523-file temporal view)

Analysis `e669e2a`, training `1f88da0` frozen; commissioned + guarded
short-term baselines (restored, never refit); suspect guard (abnormal/
quarantined files update neither baseline); survival fit on allowed pre-cutoff
non-test cohort only (**n=77, quarantined precursors incl. n=22**, sealed views
asserted disjoint); operating point at the 95th percentile of allowed-fit 7d
negatives (control 0.5751, hybrid 0.6854); row-level timelines/embeddings/
scores stayed in server memory
(`../../artifacts/sprint-11/task-35.md`; ignored `batch-F2/task35_temporal.json`
`4b79b75b…`, `task35_slices.csv` `7d233a88…`, `run_manifest.json`
`07b20656…`). Core metrics reproduce the independent Task 32 extraction
exactly; F2 rerun byte-identity confirmed bit-for-bit. (Hybrid 7d median lead
time differs trivially between the Task 32 notebook, 6.485 d via
torch non-interpolating median, and the F2 runner, 6.525 d via numpy
interpolating median on n=16 — identical raw lead times, Batch F2 evidence
gate INFO.)

| metric (control → hybrid; Δ) | control | hybrid | Δ |
|---|---|---|---|
| event recall 1d (23 episodes) | 0.043 (1/23) | 0.000 (0/23) | −1 hit |
| event recall 7d | 0.826 (19/23) | 0.696 (16/23) | −3 hits |
| lead-time 7d median [p25, p75], n | 6.49 [6.15, 6.84] d, 19 | 6.53 [6.15, 6.86] d, 16 | — |
| lead-time 1d | 0.79 d (n=1) | undefined (n=0) | — |
| warning persistence mean [median] | 0.535 [0.644] | 0.487 [0.618] | −0.048 |
| false-warning runs (218.6 robot-days) | 20 | 12 | −8 |
| false runs / robot-day [/ 30 robot-days] | 0.0915 [2.744] | 0.0549 [1.647] | −0.037 [−1.098] |
| AUROC 1d / 7d | 0.226 / 0.433 | 0.259 / 0.433 | +0.034 / −0.001 |
| AUPRC 1d / 7d (baselines 0.163 / 0.837) | 0.103 / 0.792 | 0.107 / 0.793 | +0.004 / +0.001 (below prevalence both horizons, both variants — worse than constant) |
| concordance C_7d (114 211 pairs) | 0.414 | 0.408 | −0.007 (higher risk → *later* failure) |
| Brier 1d / 7d (constant ref 0.136 both) | 0.227 / 0.386 | 0.204 / 0.400 | −0.023 / +0.014 (7d ≈ 2.9× reference) |
| ECE 1d / 7d | 0.252 / 0.418 | 0.222 / 0.446 | −0.030 / +0.028 |
| conformal healthy coverage | 0.982 | 0.982 | 0.000 |

- 7d recall is prevalence-driven, not evidence-driven: 401/479 usable files sit
  inside some 7d window, so a flat high-risk predictor recalls nearly
  everything; the 6.5-day median lead reflects early firing across saturated
  windows with persistence ~0.5. The 1d view (78/479 events) exposes the
  failure: 1/23 → 0/23.
- Hybrid "fewer false alerts" is threshold arithmetic (operating point 0.685
  vs 0.575 warns less often overall: fewer false runs *and* fewer hits), not
  improvement.
- Risk is miscalibrated, not just weak: observed 7d failure rates 0.72–1.00
  across every predicted-risk decile against an 84% event rate.
- Slices: known-robot 7d recall robot-01 0.67→0.33, robot-02 1.00→1.00,
  robot-03 0.71→0.57 (per-robot 7d AUROC 0.11–0.42, ECE 0.29–0.72; false runs
  concentrate on robot-02/03); program sensitivity — warned fraction inside 7d
  windows program-01 0.167→0.040 vs program-02 0.702→0.625, program-03
  0.764→0.748 (program-01, largest cohort at 247 files, nearly blind; per-program
  7d AUROC 0.09–0.34 throughout); censoring — 523 = 479 observed + 44 censored
  (all NaN-time; per-horizon excluded 44; usable 479 with 78/401 events).
- Maintenance handling is mechanically correct (58 scored breaks, no bridging,
  suspect guard) but maintenance *contrast* is unmeasurable: the predeclared
  ±3d window saturates (all 523 files within 3 days of 23 episodes over ~219
  robot-days), so within-window displacement trivially equals the overall
  median (control 3.94, hybrid 3.25).

![7d calibration, control](assets/task35_calibration_7d_control.png)
![7d calibration, hybrid](assets/task35_calibration_7d_hybrid.png)
![False-warning runs by robot](assets/task35_false_alerts_by_robot.png)

## 9. Hypothesis gates and integrated verdict (Task 36)

Gates restate methodology §7/§17 and sprint acceptance criteria as conventional
effectiveness bars — not fit to the numbers, no threshold moved after opening
any test view. Every result sits at or below chance, so any reasonable bar
fails (`../../artifacts/sprint-11/task-36.md`):

| gate | bar | control | hybrid | verdict |
|---|---|---|---|---|
| G1 geometry separates normal/abnormal | AUROC > 0.6, abnormal energies above normal medians | 0.4823; medians inverted (tail 27.1 vs 33.0, conf 0.61 vs 0.68) | 0.4709; deeper inversion (17.6 vs 34.5) | **FAIL / FAIL** |
| G2 boundary improves held-out families | held-out recall Δ > 0 | gate UNAVAILABLE (family-agnostic synthesis; proxy: hybrid ≤ control 7/8, tie 1) | same | **UNAVAILABLE** |
| G3 aggregation preserves sparse/local evidence | tail-hit clearly above normal + monotone severity | tail-hit 0.384; severity inverted; aggregation preserves tail and mean equally poorly | 0.370; same | **FAIL / FAIL** (aggregation exonerated — patch energies themselves inverted) |
| G4 trajectories give calibrated warning | 7d AUROC > 0.6 AND ECE_7d < 0.15 AND Brier_7d < 0.136 AND 1d recall > 0.5 | 0.433 / 0.418 / 0.386 / 0.043 | 0.433 / 0.446 / 0.400 / 0.000 | **FAIL / FAIL** (all four sub-conditions fail both variants) |

**Integrated verdict: 0 of 4 gates pass.** The hybrid improves on the control on
exactly one reported measure — argmax localization 0.237→0.315. No gate is
rescued by threshold choice: discrimination is at/below chance, so no
separating operating point exists, and **no threshold retuning was attempted or
is proposed**.

Root causes R1–R5 (`task-36.md`): **R1** real families map inside the frozen
normal references (primary; the exact failure mode methodology §3 warns
normal-only training permits; hybrid probe 0.42 < control 0.52 — the boundary
objective distorts geometry around non-resembling synthetic perturbations while
leaving real anomalies inside the manifold); **R2** the survival layer fits
noise and prevalence masks it at 7d (continuity ≈1.0, concordance 0.41, 7d
recall an 84%-event-rate artifact); **R3** program-01 blind spot (data/model
ambiguity, recorded not assigned); **R4** structural evaluation gaps
(no cold-start units, no held-out families, fragile n=11 static point,
saturated maintenance window); **R5** calibration collapse is consequence, not
cause — representation failure cannot be tuned around.

## 10. Availability ledger (what exists vs what is unavailable — no substitution)

- **Known-robot evidence: EXISTS.** All 3 temporal robots appear in dev views;
  known-robot slices are the only robot evidence (recall without
  discrimination, §8).
- **Genuine cold-start (held-out robot/program): UNAVAILABLE.** All 3 robots
  and all 3 programs appear in dev views (`cold_start_robots=[]`,
  `cold_start_programs=[]`); no learning-curve / cold-start evidence exists in
  this materialization. Stated, not proxied (`task-35.md`, `task-36.md`).
- **Genuine held-out-family: UNAVAILABLE.** Training synthesis is
  family-agnostic random patch perturbation and never reserves a family; all
  eight test families are equally unseen by both variants. Every family slice
  is known-family description, never held-out evidence — the critical
  all-unseen-versus-held-out distinction (`task-33.md`, `task-34.md`,
  `task-36.md`).
- **Operational vs scientific status kept distinct:** operational SUCCESS (all
  staged runs exit 0 at verified commits; provenance exact; F1 aggregates
  byte-identical across reruns; F2 core metrics reproduce the Task 32
  extraction exactly) vs scientific NEGATIVE (the implemented
  normal-manifold-plus-boundary design, as selected through the frozen staged
  gate, does not separate, warn, or improve) (`task-36.md`).

## 11. Limitations (negative results not softened)

Single seed 0; one server-scale materialization; dev-train references from 44
files; allowed survival fit n=77; static operating point on fragile n=11
dev-val (documented ~3e-8 boundary ties flipping 4 calls); 600-file / 44-train
scale; 3 robots / 3 programs (program/robot coverage limits); anisotropy
~2.2e7 both variants (near-singular directions retained); inverted energies
across patch/file/confidence signals; ±3d maintenance-contrast window
saturated; unavailable genuine held-out-family and cold-start gates (§10). The
negative verdict is strong within these bounds (effects at/below chance leave
no room for a favorable re-read) but does not generalize beyond them. Any
redesign is outside Sprint 11: this report records, it does not redesign.

## 12. Evidence-review standing and closeout state

- Batches A1–F2 each passed a fresh `evidence-reviewer` PASS with zero
  actionable findings, most recently Batch F2 (`Sprint11BatchF2Evidence`:
  verdict PASS, provenance hashes match Tasks 30/31/32/35/36 exactly, F2 rerun
  bit-for-bit, no unverified claims).
- Task 37 is marked done **for its evidence gate only**. Sprint status remains
  **Active**; Batch F3 evidence review and the sprint-wide differential deep
  review are still required before any completion claim, and `docs/PLAN.md`
  keeps Sprint 11 active accordingly.
