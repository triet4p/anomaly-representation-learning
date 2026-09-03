# Sprint 1 — Synthetic Data Generation Subsystem

**Goal:** Implement the complete synthetic data-generation pipeline specified in `docs/DATA.md`.

**Status:** ✅ Complete

---

## Tasks

### Phase 1: Core Contract
- [x] `src/synth/schema.py` — `FileSample`, `AnomalyMeta`, `RegimeMeta`, `PatchBatch`, `MaskResult`
- [x] `src/synth/config.py` — `SynthConfig` with all sub-configs; `config.hash()` for provenance

### Phase 2: Physics & Regime Generation
- [x] `src/synth/physics/causal.py` — explicit C=3 legacy and coherent C=6 process-derived channels
- [x] `src/synth/physics/noise.py` — natural per-channel noise for all supported channels
- [x] `src/synth/regimes.py` — Regime graph, variable durations, smooth cosine transitions
- [x] `src/synth/normal.py` — variable-T, multi-regime, causal cross-channel structure

### Phase 3: Anomaly Injection Framework
- [x] `src/synth/anomalies/base.py` — `InjectionContext`, `AnomalyResult`, shared utilities (blend, local stats, dynamic region)
- [x] `src/synth/anomalies/contextual.py` — Contextual Replacement
- [x] `src/synth/anomalies/transition.py` — Wrong Transition (too fast / too slow / too early / too late)
- [x] `src/synth/anomalies/stuck.py` — Realistic Stuck (σ_abn ≈ 0.3–0.6 × σ_expected, residual dynamics)
- [x] `src/synth/anomalies/regularity.py` — Over-Regularity (low-pass suppress jitter)
- [x] `src/synth/anomalies/drift.py` — Subtle Drift (bounded trajectory drift)
- [x] `src/synth/anomalies/freq_phase.py` — Frequency/Phase Mismatch (FFT-based, partial channel)
- [x] `src/synth/anomalies/cross_channel.py` — Cross-Channel Inconsistency (lag/gain/phase, dynamic region only)
- [x] `src/synth/anomalies/duration.py` — Duration Anomaly (true time-stretch/compress via interpolation)
- [x] `src/synth/anomalies/missing_event.py` — Missing Event (replaced with valid-but-wrong behavior)
- [x] `src/synth/anomalies/easy_sanity.py` — Easy Sanity (spike, flatline — opt-in only)
- [x] `src/synth/anomalies/registry.py` — `ANOMALY_REGISTRY`, `inject_anomaly` dispatch, `HARD_FAMILIES`, `EASY_FAMILIES`
- [x] `src/synth/strength.py` — `StrengthGate`, `generate_with_strength_gate` (reject too-weak/too-strong)

### Phase 4: Patch, Mask, Contrastive
- [x] `src/synth/patchify.py` — `Patchifier`: variable-length patches, starts, valid mask, timestep↔patch mapping
- [x] `src/synth/masking.py` — `apply_masking`: random + info-aware (stratified) + block, fixed total ratio
- [x] `src/synth/contrastive.py` — `make_contrastive_views`: gain/offset/noise/shift, semantics-preserving

### Phase 5: Dataset & Diagnostics
- [x] `src/synth/dataset.py` — exact split counts, bounded-memory sharded writer/reader, atomic manifest, resume integrity checks
- [x] `src/synth/diagnostics.py` — normal/anomaly plots and statistics for every supported channel
- [x] `src/synth/generator.py` — top-level orchestrator, strength-gate retry loop

### Phase 6: Tests
- [x] Existing focused tests cover per-family masks, metadata, variable T, patching, masking, contrastive views, and split hygiene
- [x] `tests/synth/test_six_channel_shards.py` — C6 family validity, exact counts, integrity, clean/resume determinism, incompatible resume

### Phase 7: Experiment
- [x] `experiments/quality_check.py` — Runnable end-to-end local review experiment
- [x] `synth-generate` / `python -m synth.cli` — bounded-memory production/small-profile materialization

### Phase 8: Docs & Infra
- [x] `docs/PLAN.md`, `docs/SYNTH.md`, and this sprint status updated for C6 and production defaults
- [x] `.gitignore` — Ignore generated artifacts and datasets
