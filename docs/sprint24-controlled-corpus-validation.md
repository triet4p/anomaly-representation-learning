# Sprint 24 official controlled-corpus validation report (Task 7-M9)

- Protocol: `S24-CONTROLLED-DATA-v2` (official corpus execution), base design `S24-CONTROLLED-DATA-v1` digest `a5762fc572eaa861190b486d372724fad0fd4c48f8d5ea599a2db8c6803b1493` (preserved as historical).
- Execution freeze: `91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd` (persisted to `data/generated/sprint24_controlled_v2/execution-freeze.json` BEFORE any synthesis; re-verified byte-identical at audit; digest-drift refuses).
- Official corpus: `data/generated/sprint24_controlled_v2/` (fresh, separate; gitignored) — `execution-freeze.json`, `corpus-manifest.json` (`9606648db17d64496e7268d81417e25b1806ca0ce2acdd99498a071d7a991ea3`), `realized-slots.json`, `run.json`, `waves/` (6 histories × 1,380 files).
- Evidence: `artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json` (`5fddabc1059412a589e8811d84b67e1c41856f84201f506eb48f51cedfd8eaac`).
- Worker artifact: `artifacts/sprint-24/task-7-m9.md`; review cycle `S24-7-M9-C1/R2`.
- CLIs: `experiments/sprint24_task7m9_controlled_data_generation.py` (freeze-first generation), `experiments/sprint24_task7m9_controlled_data_audit.py` (exhaustive reload/audit, R2 reload-list accounting fix).
- Tests: `tests/representation/test_sprint24_controlled_data_corpus.py` (6 passed).

## 1. What was executed (and what was not)

The M8 frozen v2 plan was executed locally into a fresh supplemental corpus: six histories (`S24C-ITER-0001-CONTROLLED-01…06`, seeds `42401–42406`), 8,280 files (220 reference + 29 × 40 weekly per history), 60 planned weekly/monthly pairs. Every planned slot was synthesized once through the real M8 generation path (`generate_planned_file`) on its exact planned seed/family/severity with zero runtime fallback, reseed, threshold rescue or family substitution: 8,280 realized, 0 refused. The four owner-authorized prospective reassignments (`CONTROLLED-02|W15|038`, `CONTROLLED-04|REFERENCE|163`, `CONTROLLED-05|W15|013`, `CONTROLLED-06|W29|009`, `missing_event` → `subtle_drift` at identical seed/severity/week/arm/quota with originals retained) were realized exactly as planned.

Not executed and not claimed: no embedding, no forward (0 forwards; historical 52,789 preserved), no model/embedding read, no refit, no training, no business-density scaling, no Confirmation, no Phase C, no model-performance claim of any kind.

## 2. Reload: every persisted waveform and annotation through the supported interface

All 8,280 persisted `.npz` files under `waves/` were reloaded through the supported `synth.dataset.load_sample` interface and re-validated (`FileSample.validate()`): 8,280/8,280 reload, 0 unreadable, 0 checksum drift (per-file SHA-256 re-verified against the corpus manifest), 0 unmanifested files, 0 duplicate file/slot identities, 0 seed mismatches, 0 non-finite arrays, 0 dtype/shape violations (all `[6, T]` float32, T ∈ [179, 1404]). Every reloaded annotation was recomputed from the realized injector (`trace_annotation`: mask extent/channels asserted exact, never copied) and matched the persisted manifest annotation on all 10 compared fields; reloaded labels matched planned labels 8,280/8,280 (846 abnormal / 7,434 normal). Encoder inputs were re-audited label-free per file (`verify_encoder_label_free`); file IDs keep the inherited `N-`/`A-` scheme with no arm/band/timestamp markers. Planned-versus-realized accounting closes exactly: 8,280 planned slots = 8,280 realized files + 0 refused + 0 unexpected-missing.

## 3. Observed burden, provenance and calendar/identity custody

- Weekly realized burden is exact: BASELINE-A 36/720 = 0.050, BASELINE-B 36/720 = 0.050, ELEVATED-A 132/480 = 0.275, ELEVATED-B 132/480 = 0.275 (all inside the planned `[0.02, 0.10]` / `[0.20, 0.35]` bands).
- Monthly realized burden is exact: LOW M02 51/990 = 0.052, M03 53/1068 = 0.050, M05 51/1062 = 0.048, M06 54/1032 = 0.052; ELEVATED M04 267/1026 = 0.260, M07 264/960 = 0.275 (all inside the planned bands).
- Stable baseline controls hold: both weekly baselines and both monthly LOW runs are low and flat (monthly LOW range ≤ 0.004 per history; control-pair realized burden gaps ≤ 0.013 in absolute value).
- All-9 families realized (846 abnormal): realistic_stuck 116, cross_channel_inconsistency 107, subtle_drift 97, duration_anomaly 95, missing_event 95, contextual_replacement 93, freq_phase_mismatch 83, wrong_transition 82, over_regularity 78. Severity spans 0.351–0.850 (mean 0.608) inside the planned `[0.35, 0.85]` band; abnormal-step duration spans 9–1,071 steps.
- Calendar: every arm file nests file-level inside a same-band month on the real `c1 = 0` grid (`validate_week_month_nesting` PASS, 400 arm files/history, 0 escapes); reference files (220/history) all satisfy `end_time <= c1` (causal separation, plan-check PASS).
- Roster/context: `validate_roster_support` PASS on all histories (29–40 common observable `(robot_id, program_id)` cells per planned pair against quota ≥ 4; retained depth 52–289 files/arm against quota ≥ 12).
- Disjointness: weekly arm file sets are pairwise disjoint within every history (120/120/80/80, 0 shared files); monthly RUN vs elevated months share no files (same end_time-keyed membership); pair chronology is `NONADJACENT-RESEARCH-PAIR` throughout (audit records `NONADJACENT-SHARED-SUPPORT` only where the capped-n rule itself shares no files but pairs share arms — the declared dependent-pairs class).
- Lineage: new `S24C-ITER-0001` namespace only; no old-corpus identity read, reused, extended or mutated; slot seeds unique and outside the certified old block (plan-check PASS).

## 4. Observed support for all 60 pairs (separate from planned support and independence)

Each of the 60 planned pairs (5 weekly + 5 monthly per history) was balanced from the *realized* files with the frozen M6 capped-n rule (hash order `sha256('2407:{history}:{region}:{file_id}')`, common `(robot_id, program_id)` strata, cap = per-stratum minimum, retention counted after selection; labels joined only afterwards). Observed dispositions use the M6 vocabulary with the planned v2 minima (common strata ≥ 4, retained ≥ 12/arm, retained gap ≥ 0.10):

- 48 baseline-vs-elevated TRANSITION pairs: all FEASIBLE. Retained burden gaps 0.121–0.322 (all ≥ 0.10); common strata 29–40; retained depth 52–167/arm; post-balancing mix drift 0.000 throughout (hash-ordered retention zeroes the observable mix gap on this corpus); exposure ratios 0.439–0.731 (reported, no floor applied here).
- 12 baseline-vs-baseline CONTROL pairs: all LIMITED (support minima met, retained contrast below the minimum, as designed for controls). Weekly controls retain 75–86 files/arm with gaps 0.000–0.013; monthly controls retain 270–289 files/arm with gaps ≤ 0.011 in absolute value; exposure ratios 0.957–0.995.
- 0 UNSUPPORTED pairs; 0 pairs refused; no unsupported cell was hidden or rescued — the table above reports every observed gap, depth, stratum count, exposure and mix value.

Pair counts are support accounting, not independent samples: pairs sharing an arm inside a history are dependent (declared `NONADJACENT-SHARED-SUPPORT` class where applicable). Observed corpus support is reported separately from planned support; neither implies statistical independence or any movement-endpoint outcome, which is not executed in this task.

## 5. Repeatability, non-overwrite/refusal and downstream loading

- Repeatability (bounded real subset): 8 abnormal planned slots regenerated in a fresh plan+generator pass produce byte-identical NPZ files (8/8 hashes match).
- Non-overwrite: rerunning the generation CLI against the existing corpus refuses (`REFUSED-OVERWRITE`, no bytes changed); rerunning the audit CLI against the existing evidence dir refuses identically. A dedicated refusal test also covers the pre-generation output guard.
- Digest drift: generation with a wrong `--expected-execution-sha256` fails closed before any synthesis; the audit refuses a persisted freeze or manifest digest that differs from the frozen `91d4ce26…9a1d98a3bd`.
- Downstream loading: reloaded files batch through the supported `collate_variable_files` + `Patchifier(PatchConfig())` path (8-file batch → signals `(8, 6, 741)`, patches `(8, 46, 6, 32)`; 4-file second-history batch OK), proving the corpus is usable downstream without any model claim.

## 6. Reconciliation of stale M8 readiness prose (no old record rewritten)

The authoritative M8 evidence is the C1/R2 PASS (`artifacts/sprint-24/reviews/task-7-m8-evidence.md`, frozen manifest `artifacts/sprint-24/reviews/task-7-m8-frozen.json`, checkpoint `artifacts/sprint-24/checkpoints/task-7-m8-cp01.md`): all-9 rotation restored, single-shot `REFUSED-GENERATION` with root injector fixes, 4 prospective reassignments, 12 focused tests passing at R2 (the earlier review text's "11 passed" was superseded by the worker's 12-test R2 record; this report follows the worker record and the executed test file).

Two stale sentences remain in `docs/sprint24-controlled-data-generator-readiness.md` §5/§7 and are NOT edited (historical evidence prose is immutable): §5's "One planned seed required the deterministic same-seed fallback" and "`missing_event`/`freq_phase_mismatch` were excluded from the planned rotation after verified legitimate strength-gate rejection" describe the failed R1 candidate, which C1/R1 recorded as `changes_requested` and which R2 removed (fallback loop deleted; all-9 rotation restored; root fixes proven by before/after discrimination, 846/846 single-shot). §7's "(12 passed)" test count is the authoritative R2 count (the review report's "11 passed" line is the stale one). This section reconciles them explicitly: read §5's fallback/exclusion sentences as R1-failed-evidence retained in `artifacts/sprint-24/evidence/s24-7-m8-a01/`, and the R2 PASS + this M9 official corpus (8,280/8,280 single-shot, 0 refusals) as the superseding observed fact.

## 7. Reproduction

```bash
python experiments/sprint24_task7m9_controlled_data_generation.py \
  --out data/generated/sprint24_controlled_v2
python experiments/sprint24_task7m9_controlled_data_audit.py \
  --corpus data/generated/sprint24_controlled_v2 \
  --out artifacts/sprint-24/evidence/s24-7-m9-a01
```

`--expected-execution-sha256` pins the freeze (default `91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd`); any drift fails closed before synthesis. Both CLIs refuse non-empty outputs (`REFUSED-OVERWRITE`). Tests: `python -m pytest tests/representation/test_sprint24_controlled_data_corpus.py -q` (6 passed, incl. the persisted audit-accounting check); regressions `test_sprint24_controlled_data_execution.py` + `test_sprint24_controlled_data_protocol.py` (20 passed). Typecheck: `python -m pyright` on the touched files reports 0 errors. R2 re-ran the audit after the reload-accounting fix (stale `corpus-audit.json` removed first; corpus untouched) → `realized_files 8280`, `reloaded_files 8280`, `audit_digest 5fddabc1…f51cedfd8eaac`. Generation executes 0 forwards, 0 model loads, 0 waveform/embedding reads.

## 8. Storage/time costs and limits

- Storage (observed, not estimated): corpus total 131,148,699 bytes (~125 MiB); waveforms 119,955,680 bytes (~114 MiB); `corpus-manifest.json` 9,171,113 bytes; `realized-slots.json` 2,010,056 bytes; audit `corpus-audit.json` 1,520,978 bytes. Mean waveform file ≈ 14.5 kB; mean T ≈ 400-step assumption from the v2 cost model remains a stated assumption, now accompanied by the observed T range 179–1,404.
- Time (observed, Windows-11/Python-3.13.17): generation 308 s (~5.1 min, ≈ 27 files/s); audit 60 s (≈ 138 files/s reload+validate).
- Forward accounting: M9 adds 0 forwards; cumulative observed stays 52,789.
- Limits: data validation only — no endpoint, disposition, specificity, calibration, delay, false-alarm, per-file accuracy, generalization, deployment-readiness or business-threshold claim follows from this corpus. Monthly horizons use the same capped-n rule as weekly (the M6 monthly rule was UNSUPPORTED on the old corpus; here it is applied as observed support accounting on the new corpus, not as a validated monthly inference).

## 9. Validation verdict

The official controlled corpus is generated and validated as observed data: exact freeze before synthesis, real single-shot path on all planned seeds/slots, 8,280/8,280 files reloaded and mask-exact through the supported interface, full planned-vs-realized accounting closed, 60/60 pairs with realized support honestly reported (48 FEASIBLE transitions + 12 LIMITED controls, 0 UNSUPPORTED), repeatability/non-overwrite/downstream loading demonstrated, versioned manifest + audit + instructions + costs published, and stale M8 prose reconciled without altering historical evidence. No scientific design change was made; no freeze-incompatible finding arose, so no owner resolution was needed. This report authorizes nothing further: inference, training, refit, scale, Confirmation and Phase C each need their own release.
