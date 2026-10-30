# Sprint 24 paired-experiment execution readiness (Task 7-M17)

- Task / attempt / agent: `S24-7-M17` / `S24-7-M17-A04` / `S24M17GoldA04` (role Gold).
- Criterion: `docs/sprint-plans/sprint-24.md:266-271`.
- Review cycle / round: `S24-7-M17-C1` / R1 (candidate frozen below; worker stays active for reviewer corrections).
- Contract: `docs/sprint24-paired-execution-contract-a01.json` — payload digest `b9e2305130e692f336bc8dab23d2e167ee1be2625dc251f52fcce4009daf9cc3`, file bytes sha256 `d14662f8280d75f0097aa518b41aaabf39365586fd2fc26200e1abf2deff1bd2`.
- What this task does NOT do: no official paired corpus, no official full inference, no training, no bank refit, no Phase C, no M12 helper repair, no edits to accepted M11–M16 products. Cumulative observed forwards are **77,845** (77,653 before M17 + 192 exercised capture forwards: 96 run + 96 deterministic replay; 6 model loads: 3 + 3).
- Supersedes: A03 candidate (`S24-7-M17-A03`) — same contract payload, but the A03 manifest no longer covered the post-handoff `--device` CLI edit and A03 never executed the real-asset smoke. This A04 record re-freezes with the actual smoke.

## 1. Owner-approved scientific revision (option A, frozen)

M15 Stage-1 per-pair E1 is UNDEFINED (accepted unbiased MMD^2 needs >= 2 rows per arm; a 1-vs-1 twin/abnormal pair yields one embedding per side). The owner-approved replacement, frozen here:

- Score unbiased E1 on INDEPENDENT GROUPS of K=8 paired sources per arm (twin arm = 8 twin embeddings, abnormal arm = 8 abnormal embeddings, same family x severity, disjoint seeds).
- N=24 then N=48 INDEPENDENT SCORED GROUPS per cell (192/384 source pairs per cell), never files or repeated variants.
- NO 96 branch anywhere in the contract/runner/gray-zone logic; any G>48 needs a separate explicit owner authorization.
- Matched-size normal-normal null calibration: same-K (K=8 twin embeddings per arm) normal-normal twin groups give the frozen q95 per family x severity cell per encoder seed; encoders are run-replicate medians, never independent data.
- Paired cross-arm dependence: twin/abnormal arms share the pre-injection source, so the two MMD arms are NOT independent samples; the cross-term couples them. The independent unit across groups is the binary response (above/below the frozen q95), never the MMD value. No transfer of M15 per-pair sign/t power — power is recomputed at the group level.
- 0.5 is a PROPOSED bar on the response fraction (share of scored groups above the frozen null q95), with the exact Clopper-Pearson lower bound required above the measured false-alarm anchor (0.06). Misses are INCONCLUSIVE; no absence claim exists without a separately owner-approved relevant-effect margin (none exists).
- Week 40/60, month 160/250, burdens 0.05/0.15/0.275 with M13 rounding; 3 fixed encoders (181801/181802/181803); duration within-family scope only.

## 2. Executable freeze

Frozen before any official outcome (contract `S24-PAIRED-EXECUTION-CONTRACT-A01`):

- Exact scored-group grid (24/48), K=8, pairs-per-cell map (192/384 pairs), 27 cells (9 families x 3 severities), 6 histories (`S24P-ITER-0001-PAIRED-01…06`, seeds 53101–53106, lineage-separate from certified/M8 blocks; 2,304 derived pair slots unique, zero collision).
- Group-level operating characteristics (exact sign power: G=24 0.65/0.75/0.85 = 0.2106/0.6074/0.9428; G=48 = 0.4699/0.9296/0.9996), exact Clopper-Pearson rows, sequential plan (24→48 predeclared; no 96), Holm multiplicity over 9 families, dilution ladder (M13 rounding), calibration spec, evaluation spec, dispositions, denominators.
- Runtime pins (verified read-only against the accepted preflight AND the live server digests): checkpoints `9292d7c5…` / `8ae58da9…` / `79edbdf3…`, bank files `546e515d…` / `59b95f3c…` / `c234c0b6…`, bank tensors `2d6e0a11…` / `8764b246…` / `adaf2bd1…`, main-checkout venv interpreter (torch 2.14.0+cu130, CUDA True, RTX 4060 Ti). Isolated smoke ran on CPU (`device: cpu` recorded) with FP32-value-identical frozen-state digests; cache identity distinguishes cpu/cuda so no official-cache confusion.
- Seed/history/calendar/checkpoint/bank/denominator/disposition/decision-parameter freeze: all identities above; no outcome-driven tuning or silent redraw (contract verifier fails closed on drift).

## 3. Reusable runner (public entry points)

- Generator: M16 `SessionGenerator.generate_paired_twin` (+ family/severity sweeps) under domain `s24p-paired-v1`; grouping key = pair seed; labels never enter encoder inputs (`FileSample.encoder_inputs`).
- Capture: `restore_s24_b0_bundle` + `capture_s24_file_features` on accepted bytes (cuda path preserved; `--device cpu` default for the isolated smoke), frozen-state digests before/after (recorded in `capture.json`), order/shape/finiteness admission; no random weights, no fallback, no refit.
- Evaluation: `score_scored_group` (K=8 vs K=8 unbiased E1, fail-closed on singleton/non-finite), `summarize_cell` (response fraction + exact interval + inconclusive reading), `holm_over_families`.
- CLI: `experiments/sprint24_task7m17_paired_execution.py` (`--mode contract|smoke|capture`, `--device cpu|cuda` for capture).

## 4. Verification (actual, exercised)

- Contract build: `CONTRACT-COMPLETE`, payload digest `b9e23051…` (above); independent recomputation (separate binomial-tail code, no shared math): **PASS**, 6/6 group entries match, 3/3 CP spots match, dilution ladder exact, grid is exactly [24, 48], zero 96 entries.
- Tiny local smoke through the public entry (real M16 pairs on throwaway synthetic inputs, fixed-width summary rows as Kx8 group rows): **SMOKE-COMPLETE**, 16/16 PAIRED, pair seeds unique, deterministic replay byte-identical, singleton/non-finite refused, 0 forwards, 0 model loads.
- **Actual real-asset capture smoke (criterion :269, this attempt):** `--mode capture --device cpu` against the 6 verified accepted bytes (3 checkpoints + 3 banks, 6/6 sha256 match, staged outside the repo at `F:/tmp/s24-m17-a04-run/`): **CAPTURE-COMPLETE** — 16 PAIRED pairs (subtle_drift/0.6, history 53101) → 32 files × 3 seeds = **96 forwards, 3 model loads**; order/shape/finiteness admission passed; frozen-state digests identical before/after per seed (`frozen_state_unchanged: true`); grouped E1 + twin-vs-twin null + cell summary per seed with run-replicate medians; **deterministic replay in a fresh process byte-identical** (excl. `elapsed_s`). Second identical run adds 96 forwards + 3 loads (replay custody).
- Focused tests: `tests/representation/test_sprint24_paired_execution.py` **12 passed**; combined with twin + design suites **33 passed**.
- Server runtime re-verified read-only (A02/A03): isolated checkout `/tmp/s24-7-m11-a03-run/repo` @ `b7890d5` clean; main checkout @ `23642b4` preserved; 3/3 checkpoint + 3/3 bank-file digests match; main-venv torch CUDA True. Server checkouts predate the pushed M16/M17 code, so the real-asset smoke ran locally on CPU against the verified bytes instead — identical public entry, identical frozen-state digests.
- Restore fail-closed probes: missing-file and digest-mismatch refusals verified (`S24 checkpoint file is missing`, `S24 checkpoint bytes differ`); local fixture banks never accepted.

## 5. Forward/model-load custody

- Cumulative observed before M17: 77,653.
- This attempt: contract 0 + smoke 0 + capture run1 96 (3 loads) + capture replay 96 (3 loads) = **192 forwards, 6 model loads**.
- Cumulative observed after M17: **77,845**.
- Official corpus / full inference: NONE (needs M17 C1 PASS + Bronze checkpoint + M18 release).

## 6. What remains / non-claims

- C1/R1 evidence assessment of this candidate, then the separate Bronze exact-snapshot checkpoint before M18.
- No representation verdict, no scale claim, no absence claim, no business-value claim follows from this readiness. The tiny smoke cell reads INCONCLUSIVE by construction (2 groups, 0/2 responses, exact CI [0.0, 0.8419]) — it exercises the full public path, not a decision.
- `duration_anomaly` pairs are within-family evidence only; normal-return pairs are construction abstentions, never normal evidence.

## 7. Reproduction

```bash
python experiments/sprint24_task7m17_paired_execution.py --mode contract --out <fresh dir>
python experiments/sprint24_task7m17_paired_execution.py --mode smoke
python experiments/sprint24_task7m17_paired_execution.py --mode smoke --out <fresh dir>
python -m pytest tests/representation/test_sprint24_paired_execution.py -q
python -m pytest tests/synth/test_paired_twin.py tests/representation/test_sprint24_paired_decision_experiment.py tests/representation/test_sprint24_paired_execution.py -q
# Actual real-asset capture (needs the 6 verified accepted bytes + preflight):
python experiments/sprint24_task7m17_paired_execution.py --mode capture --device cpu --b0-root <b0-root> --bank-root <bank-root> --preflight artifacts/sprint-24/evidence/s24-7-m4-a01/g2-accepted/preflight.json --out <fresh dir>
```

## 8. Candidate manifest (frozen bytes)

- `docs/sprint24-paired-execution-contract-a01.json` (file `d14662f8…` — same contract bytes; payload `b9e23051…af9cc3`)
- `docs/sprint24-paired-execution-readiness.md` (this file, A04)
- `src/representation/sprint24_paired_execution.py`, `experiments/sprint24_task7m17_paired_execution.py` (A04: `--device` threading + frozen-state recording + attempt labels), `tests/representation/test_sprint24_paired_execution.py`, `CHANGELOG.md` (Unreleased M17 entry)
- `artifacts/sprint-24/task-7-m17-a04.md` (worker record), `artifacts/sprint-24/evidence/s24-7-m17-a04/` (evidence: `contract/`, `smoke/`, `capture/run1/`, `capture/replay/`, `tests.log`, `manifest.json`)
- Source commit: `c67cbe49b68b10c89f8f1455388f414e2c66d8e4` (plus M17 working-tree files above; worker no commit/push).
