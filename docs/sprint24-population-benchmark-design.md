# Sprint 24 population-movement benchmark — design and frozen contract (Task 7-M20)

**Status: readiness design and executable contract. No official corpus was generated, no encoder was trained or refit, and no official outcome exists. The tiny real smoke and the full benchmark are separate gated steps (M20 smoke now; the official run only after M20 evidence PASS, a completed worker record and a fresh Bronze checkpoint at the M20 boundary).**

- Task / attempt / contract: `S24-7-M20` / `S24-7-M20-A01` / `S24-POPULATION-BENCHMARK-CONTRACT-A01`.
- Criterion: [sprint plan](sprint-plans/sprint-24.md) Task 7-M20 (lines 292–300).
- Machine-readable contract: [sprint24-population-benchmark-contract-a01.json](sprint24-population-benchmark-contract-a01.json), SHA-256 `509705131c4775987db26f194ac4952ff8ee220e4a368b8a79d99e42df924b09`.
- Frozen enumeration digest: `727ba7835e6ffa32e49f6f0b7660faabb76ebe2c7fa5046eb3188203c69abc6f`.
- Runner: `experiments/sprint24_task7m20_population_benchmark.py` over `src/population_benchmark/`.
- Historical M17/M18/M19 contracts, caches and reports are immutable and are not superseded by this benchmark.

## 1. Question and one mandatory outcome

The benchmark asks one finite engineering question: **on the same source-controlled synthetic windows, does a frozen-encoder readout separate a predominantly normal population carrying a planned 5% fault burden from the same population after nested replacement to 15% and 27.5%, at a controlled false-alarm rate, at weekly and monthly horizons, across all nine families and three severities?**

It compares three fixed readouts over the identical windows and actual waveforms:

1. `POOLED` — the existing 128-d frozen-encoder pooled file embedding;
2. `PREPOOL` — a pre-pooling patch/context distribution statistic from the same frozen encoder;
3. `RAW` — deterministic raw-signal descriptors with no learned representation.

A `LENGTH` diagnostic (timesteps, patch count, valid patch count) is computed beside them and can never qualify a readout.

Exactly one action is mandatory after the official run. The verdict is an engineering decision on this benchmark, never a business SLA, never evidence that the population lacks signal, and never a causal mechanism claim.

| Condition | Action |
|---|---|
| Coverage invalid, a readout cannot be computed, or the run is incomplete | `INVALID-IMPLEMENTATION` (fix and rerun; never a model negative) |
| Process/model/runtime failure | `RUNTIME-FAIL` (never a model negative) |
| `POOLED` passes both gates | `KEEP_POOLED` |
| `POOLED` fails, `PREPOOL` passes both gates | `CHANGE_READOUT` |
| Both learned readouts fail, `RAW` passes both gates | `TARGET_REPRESENTATION` |
| No readout passes | `STOP_SCALE_CURRENT_CONFIGURATION` |

Tie-break: `POOLED` is preferred because it is already deployed; `PREPOOL` next because it changes only the aggregation layer; `RAW` is never a keep outcome because it bypasses the learned representation. A learned readout passing while `RAW` fails is still a valid learned route.

## 2. Frozen namespaces (no same-source leakage)

Six independent history/context conditions, fresh seeds `64301–64306` (checked against the certified old block `32064–32079` and the executed M8/M18 blocks `42401–42406`, `53101–53106`), `S24PB-ITER-0001-BENCH-01…06`.

Per history, four **disjoint source-index bands** derive their seeds through the accepted `s24p-paired-v1` pair-seed rule:

| Role | Index band | Size per history | Purpose |
|---|---|---:|---|
| Reference population | `[0, 400)` | 400 normal files | the fixed population `R` in every statistic and the only source of standardization |
| Bandwidth pool | `[400, 620)` | 220 normal files | kernel bandwidth only |
| Baseline–baseline calibration | `[10 000, 10 000 + 300·160)` | 300 five-percent windows | thresholds only |
| Held-out test | `[1 000 000, 1 000 000 + 1350·160)` | 1350 episodes | controls and nested burden windows |

Independence is at the **source/window** level: no file, pair seed or window is shared across roles, and test windows across episodes use disjoint sources. Histories remain shared *context conditions* (each window is scored against its own history reference), and six histories are six clusters, not six independent populations; this is disclosed everywhere.

Design constraint on the reference: every statistic conditions on one fixed finite reference population per history. Results do not generalize to a different reference draw, and this is stated with every gate.

## 3. Window geometry: the 40/60/160/250 choice

The inherited candidate sizes are week `{40, 60}` and month `{160, 250}` (M15/M18). This benchmark freezes:

- **week = 40 files = one calendar block**;
- **month = 160 files = four calendar blocks**.

Rationale: the monthly window must be the exact union of four weekly blocks so that the nested composition is auditable (8 = 4×2, 24 = 4×6, 44 = 4×11); 4×40 = 160 preserves that identity, while week-60 maps to neither inherited monthly size (4×60 = 240).

Implications, accepted explicitly:

- The smaller pair carries fewer absolute faults per window (2/6/11 weekly; 8/24/44 monthly) and wider per-cell finite-sample intervals than the 60/250 candidates.
- The gates are macro over 27 cells and 1350 windows per horizon, so macro uncertainty is much tighter than per-cell uncertainty.
- No window size, severity, family or history selection happens after outcomes; the 60/250 alternatives are not executed as a rescue path.

## 4. Admission then nested membership

**Admission (common, before membership).** A source pair is admitted only when all of the following hold, and the policy never reads labels or encoder outputs:

- generation disposition is `PAIRED`;
- both the twin and the variant are finite, have six channels and at least 64 timesteps;
- lineage config hash and generator version match the frozen generator.

A rejected pair is recorded with a closed reason and is **not redrawn**. Admission removes sources; it never substitutes new outcomes.

**Membership.** Within each block, admitted sources are ordered by `sha256("2407|history|episode|block|slot")` with the inherited null seed. The first `ceil(burden × admitted_block_size)` ranks are rendered as abnormal at that burden. With all 40 sources admitted this is exactly the inherited 2/6/11 table; with admission loss the fault count scales with the admitted block so the requested *fraction* is preserved (one-fault rounding). Each window persists its synthetic calendar identity (horizon unit, episode, history seed, block indices) as a deterministic block index; no wall-clock time base is claimed. The same source roster is used at 5%, 15% and 27.5%, so:

- 5% faults ⊂ 15% faults ⊂ 27.5% faults (same sources, more replacements);
- the monthly set is the union of the four weekly sets;
- requested and realized fault counts and burdens are both published for every window;
- a window is `COVERED` only when every block keeps at least 90% of its planned files; otherwise it is an explicit `ABSTAIN-COVERAGE-…` and is excluded from alarm denominators.

Controls are the 5% compositions of the same episodes. They are legitimate varied normal populations (different compositions and sources), not identical vectors, so the calibration null is not degenerate.

## 5. The three readouts and the one movement statistic

All readouts consume the **same actual waveforms and windows** of each episode; labels never enter any feature, encoder, reference, bandwidth or threshold fit. One item per file per readout:

| Readout | Item | Source | Weight |
|---|---|---|---|
| `RAW` | 85-d whole-file descriptor: per-channel mean/std/RMS/slope/min/max, pairwise correlations and mean differences, low/high spectral energy, max step, valid fraction | waveform only (`representation.handcrafted.patch_features`) | one row per file |
| `POOLED` | 128-d frozen `file_embedding` (mean of valid patch context latents) | accepted checkpoint/bank as-is, deterministic Sprint18 mask seed | one row per file |
| `PREPOOL` | 128-d first two moments of the pre-pooling `context_latents` in a frozen Gaussian RFF space (64 RFF means + 64 RFF stds) | same frozen encoder, `patch_valid_mask` only | one row per file, independent of patch count |
| `LENGTH` (diagnostic) | `[T, n_patches, n_valid_patches]` | waveform metadata | one row per file |

**Pseudo-N is structurally excluded:** long files with many patches contribute exactly one item and one equal weight. Patch counts are diagnostics, never sample sizes.

**Statistic (identical functional form for all readouts).** For a window population `W` and the history reference population `R`:

$$U=\frac{2}{nm}\sum_{i,j}k(w_i,r_j)-\frac{1}{n(n-1)}\sum_{i\ne j}k(w_i,w_j)-\frac{1}{m(m-1)}\sum_{i\ne j}k(r_i,r_j),\qquad k(x,y)=e^{-\lVert x-y\rVert^2/(2\sigma^2)}.$$

- Standardization (center/scale) and bandwidth `σ` (median pairwise distance) are fitted **only** on the reference population and the independent bandwidth pool, per history × readout × encoder.
- `PREPOOL`'s RFF bandwidth is fitted on a deterministic subsample of 1500 bandwidth-pool patch latents; the RFF draw seed is frozen.
- The statistic is label-blind; there is no labelswap assumption and no paired-U dependence on same-source couplings.
- The reference population is **normal-only**: a window statistic measures movement away from the normal population, and the 5% control windows calibrate the small-burden offset rather than a pure-normal null.

**Calibration.** The calibration estimand is the 5%-burden window-versus-reference statistic: a control-distribution calibration, not a paired baseline-minus-baseline contrast. Thresholds are order statistics of **300 five-percent calibration windows** per horizon and encoder, from the calibration namespace, at the `0.975` quantile (nominal 2.5% false-alarm target, leaving finite-sample headroom under the 5% gate). Each calibration episode yields one weekly and one monthly 5% window sharing its sources; thresholds are per horizon and never pooled across horizons. The calibration tail's own uncertainty is reported from the order-statistic Beta law. No permutation cell matrix is run; 0 CPU permutations are forecast.

**Encoder combination.** Each encoder has its own calibrated threshold; a window alarms when at least 2 of the 3 frozen encoders exceed theirs. The decision additionally requires at least 2 of 3 encoders to pass the same FPR/detection/non-duration gates individually. The encoders are run replicates, never independent data.

## 6. Gates, aggregation and the duration guard

Unit: one window. Primary aggregation: unweighted macro over the 27 family × severity cells, separately per horizon and burden, with exact Clopper–Pearson intervals.

A readout qualifies only if **all** of the following hold at **both horizons**:

| Gate | Threshold |
|---|---|
| Held-out false-alarm rate (macro, 5% controls) | ≤ 0.05 |
| Detection (macro, 15% and 27.5%) | ≥ 0.80 each |
| Family floor (every one of nine families, 15% and 27.5%) | ≥ 0.60 |
| Non-duration macro (eight families, 15% and 27.5%) | ≥ 0.80 |
| Nested monotonicity, point estimates | `d(27.5%) ≥ d(15%) ≥ FPR` |
| Encoder robustness | ≥ 2 of 3 encoders pass FPR, macro detection and non-duration |
| Coverage | ≥ 45 realized windows in every cell × horizon × burden |

The duration guard forbids a duration-only pass: the duration family must clear the family floor like every other family, but the eight non-duration families must clear 0.80, so a high duration score cannot carry the readout. The `LENGTH` diagnostic is published beside the gates and is never used as a pass criterion or a tie-breaker.

The gates are fixed **decision thresholds on macro rates**, not per-cell hypothesis tests; no Holm or other multiplicity family is implied, and there is no interim look or optional stopping.

**Finite-sample honesty.** At 50 episodes per cell, a per-cell detection of 40/50 has an exact 95% interval of roughly `[0.664, 0.897]`; macro rates over 1350 windows are much tighter, but no power guarantee is claimed. The gate is an engineering target on this finite benchmark. A miss is reported as a miss and never as absence of signal.

## 7. Planned counts, resource and storage forecast

| Quantity | Planned value |
|---|---:|
| Test episodes (27 cells × 50) | 1 350 |
| Test windows per horizon and burden | 1 350 |
| Calibration episodes (windows per horizon) | 300 |
| Reference / bandwidth files per history | 400 / 220 |
| Captured files | 316 320 |
| Encoder forwards (per-file, 3 encoders) | 948 960 |
| Model loads | 3 |
| CPU permutations | 0 |
| Estimated wall time at the measured M18 rate (87.8 files/s) | 3.0 h |
| Stored bytes (items + waveforms + metadata) | 4.86 GB |
| Declared free-space budget at the explicit `--out` | 6.5 GB |

Storage layout is bounded and explicit, never a mount scan:

- `events.jsonl` — append-only custody events with `fsync` per record;
- `chunks/<history>/<kind>/<chunk>.items.npz` — the **exact sufficient statistics** for every readout (raw, pooled, pre-pooled, length);
- `chunks/<history>/<kind>/<chunk>.waveforms.npz` — actual waveforms, compressed into one member per file;
- `chunks/<history>/<kind>/<chunk>.json` — file order, source/admission/attempt lineage, requested/realized membership, coverage, prepool spec and member SHA-256;
- `design.json`, `contract.json`, `run.json`, `statistics.json`, `decision.json`, `replay.json`.

Items and waveforms are separate so a fresh replay can load only items and skip waveforms. The pre-pooling readout stores its exact per-file sufficient statistic; raw patch-latent tensors are **not** stored, so alternative patch kernels are not recomputable and no full-patch mechanism claim is made. This is a declared limitation, not an inferred capability.

A run is resumable chunk-by-chunk (`--resume`, implemented with per-chunk hash verification; the tiny smoke did not exercise a resumed run), but a partial run is never complete: `decision.json` is written only after all chunks, and a failure records a terminal event plus `RUNTIME-FAIL`/`INVALID-IMPLEMENTATION` where appropriate.

## 8. Label evidence versus causal hypothesis

- **Evidence produced:** label-blind movement between defined populations, planned vs realized burden, exact denominators, calibrated finite-sample gates and a replay that recomputes every statistic without an encoder call.
- **Not tested:** any causal mechanism, any business-density operating characteristic, any claim that a fault type is or is not represented, and any generalization to a different reference draw or real fleet.
- **Labels** are construction and post-hoc evaluation metadata only. They select no membership, enter no feature, and never decide an alarm.

## 9. Readiness evidence and commands

Readiness (this task) covers:

- deterministic design and contract builders; `--mode design` and `--mode contract` produce the manifests and verify the frozen digest;
- the real tiny smoke through three accepted frozen checkpoints/banks, all three readouts, durable chunks, a fresh-process encoder-free replay and the decision table;
- behavior fixtures (throwaway inputs only) for all four action branches plus coverage-invalid and readout-invalid, abstention denominators, length weighting/no-pseudo-N, corruption hash failure and readout-row identity;
- focused consumer-contract tests under `tests/population_benchmark/`.

Exact commands:

```bash
# design manifest + forecast (no model)
python experiments/sprint24_task7m20_population_benchmark.py --mode design --out <fresh dir>

# frozen contract (also writes docs/sprint24-population-benchmark-contract-a01.json)
python experiments/sprint24_task7m20_population_benchmark.py --mode contract --out <fresh dir>

# real tiny smoke on accepted bytes
python experiments/sprint24_task7m20_population_benchmark.py --mode smoke --out <fresh dir> \
  --b0-root <accepted b0 root> --bank-root <accepted bank root> \
  --preflight <verified preflight.json> --device cpu --attempt-id <attempt>
```

Official run (M21 only, after M20 PASS + completed worker + Bronze checkpoint):

```bash
nohup python experiments/sprint24_task7m20_population_benchmark.py \
  --mode official --out /tmp/<run>/official \
  --permit-official --release-manifest /tmp/<run>/release.json \
  --b0-root /home/trietlm/s18-t23-a04-runs/full \
  --bank-root /home/trietlm/s18-t32-a02-runs/scores \
  --preflight /tmp/<run>/preflight/preflight.json \
  --device cuda:0 --attempt-id <attempt> \
  > /tmp/<run>/official.log 2>&1 &
```

`release.json` must contain `{"task":"S24-7-M20","m20_evidence_pass":true,"m20_checkpoint":true,"m21_release":true,"contract_sha256":"509705…"}`. Exit code 0 means `OFFICIAL-COMPLETE`; any nonzero exit prints a closed refusal reason and writes no verdict. Resume with `--resume` only against the same frozen design.

Fresh offline replay (no encoder import is possible):

```bash
python experiments/sprint24_task7m20_population_benchmark.py --mode replay \
  --run /tmp/<run>/official --out /tmp/<run>/replay
```

`replay.json` must report `REPLAY-MATCH`, `encoder_calls: 0` and `torch_imported: false`; the replay installs an import guard that refuses `torch` and `representation`.

## 10. Limitations and non-claims

- Finite synthetic benchmark; engineering targets are not a business SLA.
- One fixed finite reference population per history conditions every statistic.
- Six histories are clusters, and weekly/monthly windows of one episode share sources and calendar context; dependence is disclosed and handled descriptively, not as an independence claim.
- No encoder training, refit, fine-tuning, checkpoint change, Phase C action or official capture occurred in this task.
- A miss is never presented as absence of signal, and an invalid run is never presented as a model negative.
