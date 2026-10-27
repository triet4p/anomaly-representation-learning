# Sprint 24 controlled-data generator readiness report (Task 7-M8)

- Protocol: `S24-CONTROLLED-DATA-v2` (execution readiness), base design `S24-CONTROLLED-DATA-v1` digest `a5762fc572eaa861190b486d372724fad0fd4c48f8d5ea599a2db8c6803b1493` (preserved as historical).
- Execution freeze: `91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd` (pinned before any synthesis; candidate source freeze for review is separate).
- Evidence: `artifacts/sprint-24/evidence/s24-7-m8-r2stage/` — `execution-freeze.json`, `planned-slots.json` (6 histories × 1380 slots = 8280, 4 with owner-authorized reassignment and originals retained), `cost.json`, `smoke-manifest.json` (96 throwaway waveforms, 0 refused), `run.json`.
- Worker artifact: `artifacts/sprint-24/task-7-m8.md`; review cycle `S24-7-M8-C1/R2`.
## 1. What was built (and what was not)

The M7 gap rows are now implemented against real synthesis machinery (`SessionGenerator`, 9 hard `AnomalyFamily` injectors, `FileSample`): a burden-schedule planner with deterministic per-slot abnormal quotas and severity bands, a multi-history calendar builder on the real `PeriodGrid`, duration/roster controls with predicted-vs-realized verification, and a single generation path serving both dryplan and smoke. M7 v1 bytes are untouched.

Not built and not executed: no official corpus, no embedding/forward/inference, no training, no reference-bank build, no business-density scale, no Confirmation, no Phase C.

## 2. Machine-checkable freeze (before synthesis)

The readiness CLI writes `execution-freeze.json` first, then validates, then (in `readiness` mode) synthesizes the smoke. Rerun determinism verified (identical freeze digests across runs). The freeze pins the remapped 30-week/7-month calendar, per-slot quotas, roster controls, cost (8280 files; est. 24,840 forwards + 96 smoke envelope; `READY-NO-FORWARDS-EXECUTED`, 0 executed) and the refusal policy.

## 3. Calendar validation (real grids, file-level nesting)

At `c1 = 0` the span enumerates exactly 30 weekly / 7 monthly periods on the real calendar; every arm file nests file-level inside a same-band month (`validate_week_month_nesting` passes on all 6 histories). The v1 26-week layout could not co-nest (0/800 strict passes over a 12-hour c1 sweep; BASELINE-A landed in `WARMUP-PARTIAL`, single-month M04 could not hold two elevated blocks); the v2 remap (BASELINE-A in M02, ELEVATED-A in M04, BASELINE-B across M05–M06, ELEVATED-B in M07, ≥4-week arm gaps) resolves this with no threshold/estimand/lineage change.

## 4. Tiny waveform smoke (same path, throwaway-marked)

96 files across W05 (BASELINE-A) and W13 (ELEVATED-A), 16/history (4 abnormal + 4 normal per week per history), generated through the identical slot-plan and `generate_planned_file` path — no separate small-mode semantics. Every file: realized injector annotation recomputed from the realized mask (extent/channels asserted exact, never copied from the request); `file_id` carries no arm/band/timestamp/normalization marker (inherited N-/A- scheme only); encoder inputs audited label-free per file; `SMOKE-THROWAWAY` + `not_official_corpus` markers on every record.

Burden contrast is observable (planned W05 2/40 = 0.05 vs W13 11/40 = 0.275; smoke sampling 4+4/week/history preserves per-history quotas). Duration/severity/channel truth varies per file (e.g. durations 15–285 steps, severities 0.39–0.76, single-channel `cross_channel` through full-mask families); nuisance stays in-envelope (T range 189–1148 across 87 unique lengths, same generator/nuisance in every arm).

## 5. Actual annotations match realized injectors

`trace_annotation` derives family/severity/`[start, end)`/channels/abnormal-step fraction from the realized mask/meta and asserts exact agreement; requested-vs-realized labels agree 96/96 in smoke. One planned seed required the deterministic same-seed fallback (recorded as realized provenance); `missing_event`/`freq_phase_mismatch` were excluded from the planned rotation after verified legitimate strength-gate rejection on planned seeds — disclosed, not rescued.

## 6. Schema support and implementation limitations (honest)

- Supported: 30-week/7-month shared calendar (other `c1` values may enumerate 30/8 or 31/7 — generation release must use the frozen `c1 = 0` projection); 40/week + 220 reference quotas; all-9 deterministic rotation with the 4-slot owner-authorized reassignment (`missing_event`→`subtle_drift` on edge-only sessions, originals retained); per-file severity band `[0.35, 0.85]`; single-shot generation with fail-fast `REFUSED-GENERATION`; non-overwrite and seed/lineage/refusal gates.
- Limitations: smoke weekly fractions oversample abnormal slots by design (full-plan bands verified separately at plan time); planned inference is an estimate needing the execution release; injector repairs (`missing_event` level/scale match + severity blend, `freq_phase` RMS match + full-severity blend) preserve semantics/gates and are proven by before/after discrimination with no synth regressions; no specificity/deployability/business claim follows.
## 7. Reproduction

```bash
python experiments/sprint24_task7m8_controlled_data_readiness.py \
  --mode readiness \
  --expected-base-protocol-sha256 a5762fc572eaa861190b486d372724fad0fd4c48f8d5ea599a2db8c6803b1493 \
  --out artifacts/sprint-24/evidence/s24-7-m8-r2stage
```

`--mode dryplan` runs metadata only. Tests: `python -m pytest tests/representation/test_sprint24_controlled_data_execution.py -q` (12 passed); regressions `test_sprint24_controlled_data_protocol.py` (9), `tests/synth/test_anomalies.py` (48), `tests/synth/test_normal.py` (12), `tests/synth/test_six_channel_shards.py` (17) pass. Typecheck: `python -m pyright` on all touched files reports 0 errors. Readiness executes 0 forwards, 0 model loads, 0 waveform reads, 0 embedding reads; it writes 96 throwaway waveforms, 0 official corpus files.

## 8. Readiness verdict

Generator readiness is demonstrated at local metadata + tiny-smoke scale: the full 8280-slot plan generates single-shot with zero refusals (846/846 abnormal slots accept on exact planned seeds), the 96-file smoke verifies mask-exact with no markers, and the 4 reassigned slots carry explicit owner-authorized provenance. This report distinguishes readiness from an observed full corpus and authorizes nothing further: official corpus generation, inference, training, refit, scale, Confirmation and Phase C each need their separate release. Ready for independent evidence review `S24-7-M8-C1/R2`.
