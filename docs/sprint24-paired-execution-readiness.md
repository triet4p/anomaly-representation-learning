# Sprint 24 paired-experiment execution readiness (Task 7-M17, C2 correction)

- Task / attempt / agent: `S24-7-M17` / `S24-7-M17-A05` / `S24M17GoldA05` (role Gold).
- Criterion: `docs/sprint-plans/sprint-24.md:266-271` (correction cycle `S24-7-M17-C2`, linked to preserved C1/R1 `changes_requested` and C1/R2 PASS reopened for the missing official driver; C2/R1 + C2/R2 `changes_requested` consumed, R3 candidate below under Main's conditional R3 authorization).
- Contracts:
  - Historical A01 (immutable record): `docs/sprint24-paired-execution-contract-a01.json` — payload digest `b9e2305130e692f336bc8dab23d2e167ee1be2625dc251f52fcce4009daf9cc3`, file bytes sha256 `d14662f8280d75f0097aa518b41aaabf39365586fd2fc2625dc251f52fcce4009daf9cc3`. Preserved byte-identical; it authorizes only the tiny smoke and explicitly forbids official runs.
  - Corrected C2 (this attempt): `docs/sprint24-paired-execution-contract-c02.json` — payload digest `c3bd1862153700a8ef036f75fc0e2810995d5d97e2076acf4823f222a9442c10`, file bytes sha256 `5f633020963ff977bc68b7ccdfb608eccb8d1b2207da4a863db4a37b90e4255c`. Supersedes A01 for the official path only; A01 remains the historical record. C2 freezes the complete official matrix/generation/capture/two-stage evaluation; it does NOT itself authorize the full-matrix inference run (needs C2 PASS + checkpoint + M18 release).
- What this task does NOT do: no official full-matrix inference run, no training, no bank refit, no Phase C, no M12 helper repair, no edits to accepted M11–M16 products or to A01 bytes. Physical cumulative observed forwards are **83,173**: 77,845 baseline + R1 816 (408 run + 408 replay, 6 loads) + R2 2,136 (1,068 run + 1,068 replay, 6 loads) + R3 2,376 (1,188 run + 1,188 replay, 6 loads). Candidate evidence dirs hold only the LATEST R3 pair (2,376); the full ledger below accounts every round monotonically since physical forwards cannot be reset on resubmission.

## 1. Owner-approved science (unchanged option A)

- Unchanged from A01: unbiased E1 on independent K=8 scored groups; 24→48 groups/cell (192/384 pairs); NO 96 branch; matched-size normal–normal null calibration; paired cross-arm dependence absorbed in the group contrast; group-level power; Clopper-Pearson exact intervals; Holm over 9 families; 0.5 = response fraction; misses inconclusive; dilution ladder (M13 rounding) week 40/60 month 160/250 burdens 0.05/0.15/0.275; 3 fixed encoders as run-replicate medians; duration within-family scope.
- Seed plan mapping: 2,304 unique pair seeds = 6 histories × 384 slots per history. Same (history_seed, pair_index) re-derives the same pair seed in every cell (cross-cell seed sharing is structural); within-cell disjointness across groups comes from disjoint pair-index ranges (G24 = indices [0,192), G48 adds [192,384)). No silent 384-total shrinking; no pseudo-N.

## 2. Complete official path (C2, R3)

- Public orchestrator: `run_official_matrix(bundles, permitted=False, ...)` + CLI `--mode official-run` (gated by `_require_official_release`; default refuses; full-matrix inference requires C2 PASS + checkpoint + M18 release). M18 has a runnable entry.
- Shared engine: `collect_stage1_cell` (generation → capture → frozen-bandwidth scoring → null responses → sensitivity → abstention denominators) and `evaluate_stage2_arm` are called by BOTH the orchestrator and the tiny fixture — one call graph, no shadow path.
- Stage 1: frozen causal-reference sigma per seed (median pairwise distance on the captured reference pool, never test cells); matched null q95 per seed from disjoint null-band twin-vs-twin groups with the frozen sigma; G48 extends G24; per-cell response fractions + exact intervals + futility look; family Holm summaries (`summarize_stage1_family` → `holm_over_families`, imported and wired); E2_K8 occupancy + D_LOC within-prototype diagnostics per cell.
- Stage 2 (full-N mixture, R3): P_elevated = ALL n_abnormal abnormal files + ALL n_ordinary ordinary files from disjoint slices (e.g. week-40 ELEVATED: 11 + 29 = 40 files); P_control = N ordinary files from a THIRD disjoint slice (40 files); full-N unbiased E1(P_elevated, reference) vs E1(P_control, reference) with the frozen reference sigma via `mmd2_unbiased` (never `score_scored_group`, never [:8], never abnormal-only/ordinary-only); LOW/MID/ELEVATED dilution + history-cluster bootstrap (B=2000, seed 2407, descriptive).
- Stage-2 stride: every arm's slice footprint is abnormal + ordinary + control (n_abnormal + n_ordinary + N), so all 12 arm bands are disjoint even under burden subsets.
- Reference lineage: reference twins from the reserved reference band, disjoint from pair/null/stage-2 bands; bandwidth/prototypes never see test, null, or population files; labels select FileSample arms only (`encoder_inputs` exclusion covered by the schema suite); group key = pair seed.
- Abstentions: `NORMAL_RETURN`/`ABSTAIN-*` tallied in denominators; slices without usable pairs abstain, never crash.
- Matrix: 27 cells (9 families × 3 severities) × G∈{24,48} × 6 histories (seeds 53101–53106); reference pools 220/history; 12 stage-2 population arms with frozen M13 counts.

## 3. Reusable runner (public entry points)

- Module: `src/representation/sprint24_paired_official.py` (orchestrator + shared stage-1/stage-2 engine + contract builder; imports — never copies — frozen A01 math).
- CLI: `experiments/sprint24_task7m17_paired_execution.py` (`--mode official-contract` 0 forwards; `--mode official-smoke` tiny same-engine both-stage fixture; `--mode official-run` gated full-matrix execution; existing `contract|smoke|capture` modes unchanged).
- Evaluation: frozen `score_scored_group` (stage 1 K=8 groups) / full-N `mmd2_unbiased` (stage 2 populations) / `summarize_cell` / `holm_over_families` (wired, not dead code).
- Lineage/no-leakage: group key = pair seed; reference/control seeds from disjoint bands; labels select FileSample arms only.

## 4. Verification (actual, exercised)

- Official contract build: `OFFICIAL-CONTRACT-COMPLETE`, payload `c3bd1862…`; verifier `PASS`; independent recomputation (separate binomial-tail code): **PASS**, grid exactly [24,48], zero 96 entries, dilution ladder exact; 27 cells × 192/384 pairs, 6 histories, 12 stage-2 arms enumerated.
- Orchestrator gate + call graph: `--mode official-run` without `--permit-official` refuses `REFUSED-OFFICIAL-RELEASE` (verified live); `run_official_matrix` and `tiny_official_smoke` share `collect_stage1_cell`/`evaluate_stage2_arm`/`capture_reference_pool`/`frozen_reference_bandwidth`/`fit_null_q95` (test-enforced); stage-2 builds P_elevated (abnormal + ordinary) and P_control (disjoint ordinaries) and scores full-N MMD vs reference (no truncation — test asserts `score_scored_group(twin[` absent from the stage-2 path).
- Tiny official smoke (`--mode official-smoke --device cpu`, 6/6 sha256-verified accepted bytes staged outside the repo): **TINY-OFFICIAL-SMOKE-COMPLETE** — subtle_drift/0.6, history 53101; stage 1: 2 groups (16 pairs → 32 files) + reference pool (220 twins → 220 files) + null fit (64 twins → 64 files); stage 2: weekly-40/ELEVATED P_elevated (11 abnormal + 29 ordinary = 40 files) + P_control (40 disjoint ordinaries) → 80 files; total 396 files × 3 seeds = **1,188 forwards, 3 model loads**; cell INCONCLUSIVE by construction (0/2, exact CI [0.0, 0.8419]); stage-2 transition + control per seed (e.g. 0.0017/0.0001, 0.0017/-0.0081, -0.0030/-0.0028).
- Deterministic replay: fresh-process second run byte-identical excl. elapsed/cumulative (+1,188 forwards, +3 loads).
- Focused tests: `tests/representation/test_sprint24_paired_official.py` **10 passed** (matrix/seed/disjointness/contract/numerics+Holm+sensitivity/error-boundaries/orchestrator-callgraph incl. mixture/control assertions/counts/CLI); combined **43 passed** (12 A01 execution + 10 official + 21 twin/design).
- A01 bytes byte-identical (`d14662f8…`); A01 CLI modes re-verified (`contract`/`smoke` pass).
- Forward ledger (monotonic, physical): baseline 77,845 + R1 816 + R2 2,136 + R3 2,376 (1,188 run1 + 1,188 replay, 6 loads) = **83,173 cumulative**. No official corpus/full inference.

## 5. What remains / non-claims

- C2/R3 evidence assessment of this candidate, then the separate fresh Bronze exact-snapshot checkpoint before M18.
- No representation verdict, no scale claim, no absence claim follows from this readiness. The tiny smoke exercises both official stages on real bytes; it is not a decision.
- Full-matrix inference is NOT authorized by this contract: it requires C2 evidence PASS + Bronze checkpoint + M18 release.

## 6. Reproduction

```bash
python experiments/sprint24_task7m17_paired_execution.py --mode official-contract --out <fresh dir>
python experiments/sprint24_task7m17_paired_execution.py --mode official-smoke --device cpu --b0-root <b0-root> --bank-root <bank-root> --preflight artifacts/sprint-24/evidence/s24-7-m4-a01/g2-accepted/preflight.json --out <fresh dir>
python experiments/sprint24_task7m17_paired_execution.py --mode official-run --help
python -m pytest tests/representation/test_sprint24_paired_official.py tests/representation/test_sprint24_paired_execution.py -q
```

## 7. Candidate manifest (frozen bytes)

- `docs/sprint24-paired-execution-contract-c02.json` (versioned contract; payload `c3bd1862…`)
- `docs/sprint24-paired-execution-contract-a01.json` (unchanged historical record; `d14662f8…`)
- `docs/sprint24-paired-execution-readiness.md` (this file, A05 R3 update)
- `src/representation/sprint24_paired_official.py`, `experiments/sprint24_task7m17_paired_execution.py` (official orchestrator + shared engine + gated run + full-N stage-2 mixture; A01 modes unchanged), `tests/representation/test_sprint24_paired_official.py`, `CHANGELOG.md` (Unreleased M17-C2 entry)
- `artifacts/sprint-24/task-7-m17-a05.md` (worker record), `artifacts/sprint-24/evidence/s24-7-m17-a05/` (evidence: `contract/`, `smoke/`, `official-smoke/run1/`, `official-smoke/replay/`, `tests.log`, `manifest.json`)
- Source commit: `9e0a6ab56c7536bde12f0bc104d22e016050cfe2` (M17 checkpoint; worker no commit/push).
