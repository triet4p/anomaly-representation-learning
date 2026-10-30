# Sprint 24 Task 7-M16 — Public paired-twin constructor

- Task / attempt / agent: `S24-7-M16` / `S24-7-M16-A01` / `S24M16GoldA01`.
- Criterion: `docs/sprint-plans/sprint-24.md:259-264` (owner releases constructor only).
- Review cycle: `S24-7-M16-C1`; worker artifact `artifacts/sprint-24/task-7-m16-a01.md`; evidence `artifacts/sprint-24/evidence/s24-7-m16-a01/`.
- What this task does NOT do: no official paired corpus, no embedding capture, no model loads, no forward passes, no training, no bank refit, no remote execution, no scale, no Phase C release, no M15 numerical/business decisions. Cumulative observed forwards stay **77,653**.

## 1. Public entry point

```python
from synth.config import SynthConfig
from synth.generator import SessionGenerator
from synth.schema import AnomalyFamily

gen = SessionGenerator(SynthConfig())
pair = gen.generate_paired_twin(42401, AnomalyFamily.SUBTLE_DRIFT, 0.6)
pair.twin      # FileSample: unmodified twin, NORMAL label, no anomaly_meta/mask
pair.abnormal  # FileSample: requested variant, ABNORMAL label on PAIRED
pair.disposition  # "PAIRED" or "NORMAL_RETURN"
pair.group_id     # "s24p-paired-twin-v1:<pair_seed:010d>"
pair.source_id    # 16-hex pre-injection source binding
```

Sweeps reuse the same source:

```python
severity = gen.generate_paired_severity_sweep(42401, AnomalyFamily.SUBTLE_DRIFT, [0.35, 0.60, 0.85])
family = gen.generate_paired_family_sweep(42401, [AnomalyFamily.SUBTLE_DRIFT, AnomalyFamily.REALISTIC_STUCK], 0.6)
```

`derive_pair_seed(history_seed, pair_index)` derives deterministic pair seeds
under the `s24p-paired-v1` domain inputs used by the M15 design. Smoke:

```bash
python experiments/sprint24_task7m16_paired_twin.py --mode smoke
python -m pytest tests/synth/test_paired_twin.py -q
```

## 2. What "paired" means here

Each call rebuilds the exact common pre-injection source of the legacy
`generate_anomaly` path from the same pair seed: base RNG draws, roster,
gains/offsets, regime sequence, feed envelope, causal draw, and the noise-seed
position. The twin applies the anomaly-path noise stream to the clean signal;
the abnormal applies that same starting noise stream to the realized injected
signal of the actual strength-gate selected attempt. The twin is never an
independent `generate_normal(seed)` draw: at equal seeds the normal path uses
its own base draw and diverges (seed 42401: normal T=386 roster (4, 2) versus
paired T=450 roster (3, 1)).

## 3. Noise, retries, duration

- **Signal-dependent noise:** post-injection S1 local-variance scaling and S2
  local-std-scaled walk read the modified signal, so twin and abnormal differ
  outside the realized mask by construction. The pairing controls the
  pre-injection base and the noise starting position, not the realized noise.
- **Retries:** the abnormal always pairs the actual strength-gate selected
  attempt (`n_attempts`/`strength` match the legacy path exactly, including
  multi-attempt seeds such as 42447 with `n_attempts=2`).
- **Duration family:** `duration_anomaly` pairs within family only. It shares
  the source but changes T/regimes by interpolation (seed 42401: twin T=450,
  abnormal T=810), so it is never a length-matched cross-family comparison.
- **No fake success:** declined injections return `disposition="NORMAL_RETURN"`
  with a NORMAL label, no `anomaly_meta`, and the injector's `rejection_meta`
  (missing-event edge sessions such as 42471; exhausted strength gates).
  Unsupported families, severities outside [0, 1], non-integer or negative
  pair seeds, and empty sweeps raise `ValueError`; easy-sanity families stay
  opt-in behind `anomaly.include_easy_sanity`.

## 4. Grouped identity

Every variant of one pair seed carries the same `group_id` and `source_id` in
`anomaly_meta.extra`/`rejection_meta.extra`. Correlated variants and resampled
repeats must never be counted as independent evidence; grouping key = pair
seed. Labels select variants only and never enter encoder, reference,
prototype, bandwidth, or detector fits. `FileSample.encoder_inputs()` on both
sides stays label/mask free.

## 5. Compatibility

Legacy deterministic behavior is preserved: the abnormal side is
byte-identical to `generate_anomaly` (waveform, mask, file id, strength, and
attempt count) at the same seed/family/severity, and all existing generator
tests still pass. No accepted M11–M15 products were edited. `PairedTwin` and
`derive_pair_seed` are exported from `synth` alongside `SessionGenerator`.

## 6. Limitations and non-claims

- Constructor only; no official corpus, embedding, forward, training, refit,
  remote, scale, or Phase C claim follows.
- Duration pairs are explicitly not cross-family evidence.
- Normal-return dispositions are construction abstentions, never normal
  evidence; no silent substitution is performed.
- Business thresholds remain M15 proposals awaiting owner approval.
