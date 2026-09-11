# Sprint 15 Observable Probe v4 — Frozen Pre-Outcome Specification (candidate 4)

**Status:** Frozen prospectively before any Fit, Calibration, or Confirmation outcome exists.
NOT APPROVED for retuning after outcomes: any change requires a new probe version, Main approval,
and a fresh evidence review before downstream execution. No probe, threshold, or score has touched
any Sealed root; none may until EG6.

**Normative contract:** `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` §EG4 (thresholds below are
transcribed verbatim; on conflict the gate document governs).
**Probe identifier:** `sprint15-observable-probe-v4`.
**Code:** `src/synth/probe15.py` (SHA256
`a08b3d5fb83001e1f5c43f4c56ff536bae85e41d494db289304aeb33a339242b`) — behavior identical to
the reviewed v3 freeze; no feature, scoring, threshold, or metric line changed for candidate 4.
No binding wrapper was needed: the probe binds roles only at Task 14 execution.
**Tests:** `tests/synth/test_probe15.py` (SHA256
`394573913f597a564eec356ff4a3c95fbf1f8238fa5bebcbd2d31762a7c7ec67`; 11 tests, all passing —
re-executed 2026-09-10 in Task 13 cycle-4).
**Candidate binding:** candidate 4 (`sprint15-benchmark-protocol-v4`, Fit H-FIT-19..21,
Calibration H-CAL-7, Confirmation H-CONF-22..25 — none materialized yet).

## 1. Causal observable feature set (frozen, 59 features, C=6)

Per file, from the `[C, T]` waveform plus two model-visible timing scalars only
(`duration_s`, `time_since_reset_s` from the manifest row). Per channel: RMS, standard deviation,
least-squares slope over the time index, zero-crossing rate of the centered signal, and power in 3
spectral bands (low/mid/high thirds of one-sided FFT bins, normalized by channel power, 0 when
unp-powered). Cross-channel Pearson correlations (15 upper-triangle pairs; 0.0 when either channel
is constant). Plus `duration_s` and `max(time_since_reset_s, 0)`. Fixed order
(`feature_names`), pure NumPy, RNG-free, deterministic; non-finite outputs map to `0.0`.

Forbidden as inputs: labels, cohorts, subtypes, failure times, health states, episodes,
quarantine/censoring flags, member views, simulator state, allocation metadata. The constructor
signature `(x, duration_s, time_since_reset_s)` makes leakage structurally impossible (pinned by
`test_feature_signatureadmits_no_labels_or_state`).

Feature families are exactly the Sprint 12-justified causal telemetry set carried by V2 §EG4:
channel level/RMS/variance/slope, differences and cross-channel relationships, spectral bands,
phase/timing (via slope + zero-crossings), transition counts, duration, usage (duration +
time-since-maintenance), and time since maintenance.

## 2. Fit-only standardization and model fitting (frozen)

- Input rows: Fit-role verified-healthy files only (`file_label` NORMAL, not quarantined).
- Standardization: per-feature mean/std over those rows; std floored at `1e-6` (`STD_FLOOR`).
- Model: healthy centroid = mean of standardized healthy rows; file score = squared Euclidean
  distance from the centroid (`score_files`). No failure data, no labels, no thresholds enter
  fitting. Fit statistics and centroid are persisted by Task 14 with the Fit role roster.

## 3. Calibration-only threshold selection (frozen)

Single operating threshold = 95th percentile (`THRESHOLD_QUANTILE = 0.95`, NumPy linear
interpolation) of Calibration-role verified-healthy file scores. Selected once, on healthy rows
only; frozen before any Confirmation outcome. Rationale: conventional operating point consistent
with the `≤ 0.05` false-alert regime; not tuned (no downstream outcome existed at freeze time).

## 4. Per-event aggregation (frozen, existing conventions)

Event score = max file score over POS-eligible window files (`synth.events.window_score`).
Control score = max over control-window members. History event AUROCs are tie-aware
(`roc_auc_tie_aware`) over P+W / P / W positives vs controls; abrupt companion AUROC separately
(no minimum, no silent exclusion).

## 5. History-block bootstrap and macro rules (frozen)

- Macro P+W / P / W AUROCs = unweighted means of the four per-history AUROCs.
- History-block bootstrap: resample the four histories with replacement, `B = 2000` replicates at
  frozen seed `BOOTSTRAP_SEED = 20260202`; 95% LCB = 2.5th percentile of replicate macros.
- Directional rule: count of histories with P+W AUROC `> 0.55`.
- All deterministic given inputs (repeat-identical, pinned by fixtures).

## 6. Recall, lead, false-alert, abrupt rules (frozen, EG4 verbatim)

At the frozen Calibration threshold, per history: P event recall `≥ 0.50` with median first-alert
lead `≥ 1.0` day (`lead_days` over flagged pos-file ends); W event recall `≥ 0.25` with median lead
`≥ 0.5` day; false-alert episodes `≤ 0.05` per evaluated robot-day on every history
(`false_alert_episodes` over eligible rows, ≤2-day grouping, reset-split). Abrupt results reported
separately (counts, AUROC companion) and overall with no predictive minimum.

## 7. Pre-outcome deterministic metric fixtures (executed, no roots)

Executed 2026-09-10 via `tests/synth/test_probe15.py` (no data roots, synthetic inputs only):

- constant scores → AUROC exactly `0.5`; perfect separation → exactly `1.0`; reversed → exactly `0.0`;
- onset-only alert → first-alert lead exactly `0.0` d;
- bootstrap twice-identical; LCB ≤ macro; replicate count exact;
- empty flags → `(0, 0.0)` false episodes/rate; hand-grouped two-episode case → `(2, 0.1)`;
- threshold rule exact (`94.05` on `0..99`); feature determinism/shape/finiteness incl. constant and
  zero inputs; standardization floor exact; end-to-end synthetic evaluation (`macro 1.0`,
  directional 2, recalls 1.0, leads 7.0/7.0 d, FAR 0.0, JSON repeat-identical).

## 8. Task 14 execution command (canonical, not yet run)

```bash
uv run python -m synth.cli --chronological --profile sprint15-v4 --seed <S> --role <R> \
  --protocol sprint15-benchmark-protocol-v4 --output data/generated/sprint15-v4/<ROLE>/
```

for Fit seeds 1304–1306 (`H-FIT-19..21`), Calibration seed 1307 (`H-CAL-7`), Confirmation seeds
1308–1311 (`H-CONF-22..25`) — each exactly once, first-attempt, after Gate C2 cycle-4 PASS. Sealed
seeds 1312–1315 receive structural audit and seals only; no probe, score, or threshold may touch
them. Fitting uses Fit only, threshold uses Calibration healthy only, evaluation runs once on all
four Confirmation histories via `probe15.evaluate_histories`.
