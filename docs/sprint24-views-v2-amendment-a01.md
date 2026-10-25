# S24 Protocol Amendment A01 — Group-aware sampling with per-history feasible abnormal quotas + parent-custody reset splits

- Protocol: `S24-PROTO-v1` (SHA-256 `80647c63f6ec49112bf79c18961f2a357d904add72ffece282c75976a210b6e1`).
  This amendment changes sampling/grouping mechanics only. No Sprint 18 data, record, gate, or
  verdict is altered; no waveform regenerated or substituted; source hashes unchanged (below).
- Owner decisions (auto-selected = owner answers, standing instruction): S24-T4-D1 (Q1 healthy-outside
  singleton; Q2 group-exclusive role ownership with exact file picks inside claimed groups), S24-T4-D2
  (per-history feasible abnormal quotas at fixed file totals), S24-T4-D3 (split-at-reset with parent custody).
  Main records D1/D2/D3 in the sprint plan; this doc is the worker-owned exact rule record.

## Old rule (frozen S24-PROTO-v1 Sec. 2-3)

- File-level stratified sampling without replacement from `V-ADAPT` with exact per-history quotas
  B1/B2/B3 = 40/120/320 files (8/24/64 abnormal), V-VAL 60 (12ab), V-TRIG 100 (20ab), V-R0 50 (10ab).
- Groups must never split across adaptation/validation/evaluation frames (frames named, sampling
  mechanics unnamed).
- Nearest-interval attachment specified for abnormal files outside any interval; healthy-outside
  handling unspecified.
- No reset-edge split rule specified.

## New rule (S24-VIEWS-v2, implemented in `representation.sprint24_views`)

- Eligibility first (unchanged frozen predicate, extended to abnormal rows): not quarantined/censored,
  no reserve identity, no maintenance overlap. Only eligible rows enter any fixed slice or budget.
- Groups: failure windows + same-robot overlapping degradation episodes merged (open episodes skipped);
  each merged span split at manifest maintenance reset edges; pieces keep parent linkage
  (`parent_of`: piece group -> parent merged-span id; D3). Files assigned to the piece holding
  `end_time`; abnormal-outside attaches to nearest same-robot interval (gap > 7d -> singleton, D1-Q1);
  non-abnormal-outside forms a singleton (D1-Q1). Every outside file is audited.
- Time frames by frozen Task 2 decile bin edges; groups spanning an E-bin edge snap to the frame
  holding their latest member. E1/E2/E3 groups claimed first (excluded from all sampling).
- Sampling (D1-Q2 + D2): whole PARENT episodes claimed exclusively per role in order V-R0 (seed 2403,
  early time half), V-VAL (2401), V-TRIG (2410), B3 (2402); unused members stay unused in-role, never
  leak. Exact file quotas picked inside claimed parents (seeded, no replacement). B2/B1 are nested
  file subsets of B3 (seeds 2402+1/+2). Total file counts frozen (40/120/320 + 50/60/100); abnormal
  quota per role = source-derived feasible count (D2), eligible-normal fill, both classes kept where
  supported, methods paired on identical ID sets. No performance-based selection.
- Label release unchanged (period close + 86,400 s; Task 7 `c1+`; Task 9 `end(Ek)+`); evaluation and
  probe labels never released to the learner; past-only replay eligibility unchanged.

## All-4 direct accepted-source feasibility (commit `3780b718`, metadata-only audit)

- Script `s24-t4-accepted-audit.py` + report `s24-t4-accepted-audit.json` (7,633 bytes,
  SHA-256 `585010dba45b0f01b1b91df634dfa9480bb7df012eac7202b5bfeebcec72977f`), exit 0, 0 errors.
  Extended-T passing pools and post-withholding `(ab, nm)` frames:
  DEV-01 ADAPT (90, 750), DEV-02 ADAPT (136, 821), DEV-03 ADAPT (133, 792), DEV-04 ADAPT (110, 798).
- Group-aware allocator outcome under frozen 106ab quotas: DEV-01 infeasible (fails closed at B3,
  claimed parents hold 46ab/410nm, need 64/256); DEV-02..04 feasible.
- Deterministic D2 rule (`feasible_abnormal=True`): each role claims parents in seed-shuffled
  order until its abnormal quota is met, then stops (unused parents stay for later roles); the role
  takes ALL available abnormal up to its quota and fills normals to the exact file total. File
  totals always exact; abnormal counts are the source-derived feasible values. Direct accepted-source
  proof (`s24-t4-newrule-allocator-proof.json`, 3,365 bytes, SHA-256
  `c586fa24e4ec0b75b340ca88807251fd7ba33e5e507494bb47a6377d92800d4a`, exit 0, 0 errors, all-4
  feasible, parent-custody clash None everywhere): DEV-01 V-R0 (8, 42), V-VAL (12, 48), V-TRIG (16, 84),
  B3 (51, 269), B2 (24, 96), B1 (8, 32); DEV-02/03 frozen (10/12/20/64ab); DEV-04 V-R0 (10, 40),
  V-VAL (12, 48), V-TRIG (20, 80), B3 (63, 257) — one abnormal under the frozen 64, exact 320 files kept.
- Reset-edge crossings (accepted metadata): 6/5/4/4 (DEV-01..04); each is a failure+degradation
  (or single degradation) span straddling scheduled maintenance; 7 spanning files total on DEV-01
  (all normal, 0 abnormal); per-file serialization holds (0 same-robot overlaps); D3 keeps parent
  custody so no parent episode spans two roles.
- Accepted manifest hashes unchanged: DEV-01 `ce7c004b…`, DEV-02 `cee368b9…`,
  DEV-03 `b1da9571…`, DEV-04 `3af08113…` (full hashes in Task 2 inventory).
