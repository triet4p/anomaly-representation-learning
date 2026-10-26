# Sprint 24 weekly/monthly collective-movement study — execution record

- Study identity: `S24-BIZ-WM-v1`, freeze digest `6f96c2750c15552bcdf25678a51ed7b24a1f7aecdb5914bf5985d3012d1daefd`.
- Frozen protocol: [`sprint24-weekly-monthly-protocol-v1.md`](sprint24-weekly-monthly-protocol-v1.md), digest `8da4afda89dd303099843eecda546cdc337696525ece0eed59f400d1031d8b63`.
- Frozen at Task 7-M3 checkpoint `0a17ffa7927f0e2187cd262f7770e3dfbd7dbc48`; execution attempt `S24-7-M4-A01`.
- Criterion: `docs/sprint-plans/sprint-24.md:166-167`.

**Status: execution not performed — the cache preflight failed closed.** The frozen study requires 8,754 reusable embedding rows per representation seed and **0** exist. No movement value, contrast, uncertainty interval or disposition was computed. Per the frozen protocol, a blocked cache preflight is a **data-availability blocker, not a negative scientific result**.

Execution is paused pending an owner decision on a bounded recapture and forward-cap allowance. The assigned evidence review (`S24-7-M4-C1` R1) is not consumed, the criteria are unchanged, and no forward, remote write, protocol amendment or disposition is proposed by this record.

**Update — owner decision received.** The owner authorized a recapture and raised this study's cumulative forward cap to 54,000. The resulting bounded amendment ([`sprint24-weekly-monthly-cache-budget-amendment-a01.md`](sprint24-weekly-monthly-cache-budget-amendment-a01.md)) and the capture runner are prepared for the readiness review `S24-7-M4-PRE-C1R1`; nothing has been deployed or executed. The refusal below remains the current preflight result and is not superseded by the amendment.

## 1. Release prerequisites and their verification

The frozen protocol (§15) lists five release prerequisites. Four hold; the cache-coverage prerequisite fails.

| # | Prerequisite | Verification | Result |
|---|---|---|---|
| 1 | Owner decisions recorded | M4 release text acknowledges `D1–D3`/`D10` as research defaults and the `OWNER-REQUIRED` rows | satisfied by release |
| 2 | Frozen study, protocol and manifest digests unchanged | Study/protocol digests re-verified; the M3 metadata CLI reproduced all 14 data outputs byte-identically; label-blind manifest digests `e86157b0…` (periods), `6b848cf6…` (pairs), `1057570f…` (references) match | PASS |
| 3 | Every required row reusable under an identical cache identity | Full inventory of the accepted lineage and the declared cache stores; 0 of 26,262 rows available | **FAIL** |
| 4 | Confirmation custody unchanged, no retraining/corpus/Phase C | Nothing read, written or trained | PASS (unchanged) |
| 5 | Written execution record accounting for every cell and contrast | This document plus `execution-ledger.json` | PASS (192/192 cells, 8/8 contrasts accounted) |

## 2. Required row plan (compiled without computing movement)

The plan is derived from the accepted Development metadata, the frozen decile edges and the inherited eligibility predicate — the same construction path the accepted M2/M3 passes used. Reference rows are the causal pool ending at or before each history's `c1`; span rows are the eligible files in `(c1, tmax]`.

| History | Reference rows | Span rows | Required rows | Reference state |
|---|---:|---:|---:|---|
| `S18I-ITER-0005-DEVELOPMENT-01` | 850 | 1,226 | 2,076 | `REFERENCE-OK` |
| `S18I-ITER-0005-DEVELOPMENT-02` | 974 | 1,353 | 2,327 | `REFERENCE-OK` |
| `S18I-ITER-0005-DEVELOPMENT-03` | 935 | 1,118 | 2,053 | `REFERENCE-OK` |
| `S18I-ITER-0005-DEVELOPMENT-04` | 919 | 1,379 | 2,298 | `REFERENCE-OK` |
| **Per seed** | **3,678** | **5,076** | **8,754** | |
| **Three seeds** | 11,034 | 15,228 | 26,262 | |

Reference digests reproduce the frozen protocol's recorded values (`dd9c9f98…`, `cd05f55f…`, `ee8f0ec2…`, `be814983…`). Every row identity and per-history union digest is in `row-plan.json`.

## 3. Accepted capture identity, re-derived

The accepted G2 run recorded one `s24_feature_cache_identity` value per `(history, representation seed)` in its window-statistics table. Re-deriving that value from the recorded preflight provenance (checkpoint, bank tensor, arm config, data binding, preprocessing contract, model seed, scale state `none`, device `cuda`) and the reconstructed captured-file union reproduced **all 12 recorded identities exactly**. This proves the identity derivation end to end without opening any model, checkpoint or waveform.

The study-required identity applies the same function to the study-required union (reference rows plus all 5,076 span rows):

| History | Seeds equal to the accepted identity | Required vs accepted |
|---|---|---|
| `DEVELOPMENT-01` | 0 of 3 | **differs** — the accepted capture omitted 13 span files |
| `DEVELOPMENT-02` | 3 of 3 | equal |
| `DEVELOPMENT-03` | 3 of 3 | equal |
| `DEVELOPMENT-04` | 3 of 3 | equal |

The three `DEVELOPMENT-01` identities differ for one precise reason: the accepted G2 run never captured the members of its first window (`n = 28`, below the inherited 30-file floor). Those 13 files are required by the frozen study and were excluded from the accepted capture, so even a *persisted* copy of the G2 capture would not satisfy the required identity for that history. Full identity values are in `required-identities.json` and the comparison in `accepted-g2-identities.json`.

The identity binds the capture device (`cuda`). A CPU-produced copy is not interchangeable, per the frozen `s24_feature_cache_identity` contract.

## 4. Why the cache preflight fails closed

No persisted per-file embedding store exists anywhere in the accepted lineage.

- The accepted G2 runner held per-file embeddings **in memory** (`embeddings[(history, seed)]`) and persisted only aggregate tables: window statistics, controls, event mapping, false alarms, bootstrap. Its own tracked header states that stored waveform shards are source bytes, not embedding caches, and that no hash-compatible embedding cache is reused.
- A read-only inventory of every `/home/trietlm/s24-*` tree found **no `.npz`, `.npy`, `.pkl`, `.parquet` or `.arrow`** artifact. The only `.npz` files in the wider home tree are Sprint 18 RCA census arrays over **FIT** histories (context-patch embeddings), not Development file embeddings.
- The earlier Task 7 label-efficiency lineage wrote `feature-cache-records.jsonl` — identity/support **metadata**, not arrays — and `per-file-scores.jsonl`, which holds scores rather than 128-d embeddings.
- The accepted G2 output directory contains only `preflight.json`, `state.json`, `costs.json`, `calibration.json`, the four window/reference manifests and the aggregate tables.
- The retained G2 window statistics cannot substitute: they aggregate 14-day windows against a reference pool, whereas the study's `E1` is a period-to-period sample distance. Re-aggregating them would change the estimand, which the frozen protocol forbids.

Result: **0 of 8,754 required rows per seed**, and therefore 0 of 26,262 rows across the three accepted seeds, exist under the required identity. The declared cache-admission contract in `src/representation/sprint24_weekly_monthly_execution.py` describes the store a future accepted capture would have to write; no producer of that store exists today.

## 5. Consequence for endpoints, contrasts and dispositions

- No movement endpoint (`E1`–`E4`) was computed, no `C1`/`C2` association, no `C3`/`C4`/`N1`/`N2`/`N3`/`N3b` contrast, no history-cluster bootstrap and no Holm-adjusted auxiliary inference.
- All **192** declared cells are published as `NOT-EXECUTED-CACHE-PREFLIGHT-FAILED`; all **8** declared contrasts are published as not executed; both horizons are `NOT-COMPUTED` with all seven frozen decision criteria marked `NOT-EVALUATED`.
- No `PROMISING`, `NOT-PROMISING` or `INCONCLUSIVE` disposition is claimed. `INCONCLUSIVE` is a scientific disposition requiring an analysis; here there is no analysis to run, so manufacturing one would misreport a data-availability blocker as a result. Unknown is never treated as a pass.
- `specificity_claim` is `false` for both horizons. No specificity, recall, detection delay, false-alarm rate, per-file accuracy or PR-AUC claim is made. Scaling and Phase C remain unreleased.

## 6. Cost, cap and custody

| Quantity | Value |
|---|---:|
| Forwards executed by this attempt | 0 |
| Model / checkpoint / waveform loads | 0 / 0 / 0 |
| Required rows per seed | 8,754 |
| Available rows per seed | 0 |
| Missing rows per seed | 8,754 |
| Required rows, three seeds | 26,262 |
| Declared smoke (`8 files/history x 4 histories x 3 seeds`) | 96 |
| Minimal full recapture, no smoke | 26,262 |
| Minimal full recapture with the declared smoke | 26,358 |
| Cumulative after a full recapture | 52,693 (**11,693 above the cap**) |
| Cumulative after a full recapture with smoke | 52,789 (**11,789 above the cap**) |
| Span-only capture | 15,228 — **not sufficient on its own**; 0 reference rows are reusable |
| Inherited cumulative forwards | 26,431 of 41,000 |
| Inherited headroom | 14,569 |
| Cap reset | none |

Custody: the accepted G2 review artifacts were copied from the server with identical SHA-256 before and after transfer; the accepted Development manifests and frozen decile edges were digest-verified before use; no remote file was written, moved or modified. Custody digests are in `custody.json`.

## 7. Limitations

- The reuse negative is a property of the **accepted lineage**: the G2 run spent its forwards but discarded the per-file embeddings. It says nothing about the encoders, the calendar design or the study's scientific premise.
- The acceptance criterion's "actual-cache public-entry-point smoke" could not be run because no actual cache exists. The executable smoke instead proves (a) the admitted path with a complete identity-matched store on synthetic fixtures and (b) the refusal path on the accepted inputs; 12 focused tests cover identity strictness, device mismatch, missing sidecars, incomplete coverage, duplicate IDs and non-finite embeddings.
- The execution-side cache contract is declared by this task rather than inherited; a future accepted capture must write it (or an explicitly documented equivalent) at the study-required identity, including the 13 `DEVELOPMENT-01` window-1 files.
- Everything the frozen protocol already discloses about monthly coarseness, weekly composition confounding and the absence of healthy/quiet controls still holds, but none of it is exercised because no outcome exists.

## 8. What would unblock execution

These are owner decisions, not worker actions; none was taken or implied here. The minimal truthful recapture is the full per-seed row plan: 3,678 reference + 5,076 span = 8,754 rows per seed, 26,262 across the three seeds. A span-only capture does **not** unblock the study because zero reference rows are reusable either. Adding the declared 96-forward smoke (`8 files/history x 4 histories x 3 seeds`) gives 26,358 new forwards.

| Option | New forwards | Cumulative | Over the 41,000 cap by |
|---|---:|---:|---:|
| Full recapture, no smoke | 26,262 | 52,693 | 11,693 |
| Full recapture with the declared smoke | 26,358 | 52,789 | 11,789 |
| Span-only capture (insufficient on its own) | 15,228 | 41,659 | 659 |

A truthful minimal allowance is therefore either 11,693 additional forwards (no smoke) or 11,789 (with smoke), or an explicit cap extension to at least 52,693 / 52,789. No approval is invented here: the 41,000 cap, the 26,431 cumulative count and every criterion remain unchanged, and the 659 span-only figure is arithmetic only, never a viable plan.

- **No protocol or study-identity amendment is required.** The frozen period manifests already include the 13 `DEVELOPMENT-01` window-1 files in the required union; G2's omission was a capture-window property, not a study-membership property. A recapture must record its support digest over the study's full 8,754-ID union per seed rather than G2's window-support union, and that is the identity this preflight already requires.
- **A `cuda` cache identity cannot be produced on CPU.** The frozen identity binds the capture device, and a CPU capture is not interchangeable. Any recapture must run on a GPU host through a separately reviewed runner against the accepted assets, because the waveform shards and the accepted checkpoint/bank assets live on the server. Unreviewed local capture is not a substitute and would be refused by this gate.
- **Transferring a cache cannot unblock today** because no persisted cache exists to transfer. A producer run would itself need the allowances above, plus a reviewed runner and a persisted store written at the study-required identity.

No protocol change, cap reset, retraining, corpus generation, Confirmation contact or Phase C execution is proposed.

## 9. Reproduction

```bash
python experiments/sprint24_task7m4_weekly_monthly_execution.py \
  --data-root F:/ai-ml/_s24m1-local-metadata \
  --edges experiments/sprint24_task7m1_dev-edges.json \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-01=ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480747e37b48fe1e08 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-02=cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310 \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-03=b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e \
  --expected-manifest-sha256 S18I-ITER-0005-DEVELOPMENT-04=3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af \
  --g2-accepted-root artifacts/sprint-24/evidence/s24-7-m4-a01/g2-accepted \
  --out artifacts/sprint-24/evidence/s24-7-m4-a01
```

The command exits `3` with `"admission": "REFUSED"`. Passing one or more `--cache-root` directories inspects those stores instead; an empty or absent root contributes zero rows. Focused checks:

```bash
PYTHONPATH=src python -m pytest tests/representation/test_sprint24_weekly_monthly_execution.py -q
```

## 10. Evidence inventory

All outputs are under `artifacts/sprint-24/evidence/s24-7-m4-a01/`; two independent CLI processes produced byte-identical data outputs.

| Output | Content | SHA-256 |
|---|---|---|
| `preflight.json` | digest verification, row plan totals, identity comparison, inventory, admission | `0b9dac77c391e29ea17cc406be8cec5b259e31a439d3889e823e20ce45739da5` |
| `row-plan.json` | exact required file IDs, per-history digests and counts | `bb8747d7d87c4fa776896920048f3bf34287d77bd0e9bdb2ef8ccd9bae564e85` |
| `required-identities.json` | 12 study-required cache identities and support digests | `f7d67f5ff38309418ab42385293d596edc87083a4d5199c5e8a03cf7c6db203f` |
| `accepted-g2-identities.json` | re-derived accepted identities and the required comparison | `677776adec0b50571f222d2bb469c162daf89f32ccbad4ae62fb7ba169e83a55` |
| `cache-inventory.json` | inspected roots and artifacts | `b8b7b9c31c2835f1fe219e44c8e7ebb7f12fd8b51474253992fe95665c34cb44` |
| `coverage.json` | per-group required/available/missing rows under identity-strict matching | `79cb77f2211772330670eccdeaf2efadac733950b3ead6093870871eab2eb6b2` |
| `execution-ledger.json` | 192 cells, 8 contrasts and both dispositions with true states | `ac107e60c613c5322a2914e62baea97f6cc99a8103e56219e087c1a10247a0a5` |
| `cost.json` | zero-forward cost, envelope and cap accounting | `2885e5f05ba3ab0c6c5b3b4da351de730be60e9e71bd1a3402bd4fddedc5e971` |
| `custody.json` | input and accepted-artifact digests | `ea2df3612788d2f52039979e1aefa9f77525520f998d2bccd35d506f490d586a` |
| `blocked.json` | the fail-closed blocker record | `dcf59c1bfbf708e9b484685a953de9043f8405648e437f321eebfa89d6205ba7` |
| `capture-fixture/fixture-smoke.json` | local no-forward fixture/public-path smoke of the capture store writer and refusal gate | `a91f365eb06af893520c066b1d8f0cdc3975360e0480a8fe5ccf8f34cd9b71c2` |
| `capture-fixture/store-hashes.json` | hashes of the 12 fixture entries (inadmissible by provenance) | `29883751e927e73d049d4bfcb2c8c722a998425e606a0d2b8d70b808fd12dd25` |
| `run.json` | run identity, environment, timing, output digest map | (self-referential; digest map inside) |
| `g2-accepted/` | transferred accepted G2 records with `remote-sha256-before.txt` / `remote-sha256-after.txt` | see custody |
| `m3-reverify/` | reproduced frozen M3 outputs (14 data files byte-identical) | see `run.json` of M3 |

## 11. Owner decision and bounded recapture amendment

After the refusal was accepted as ground truth, the owner answered **"Capture lại, nâng cap lên 54.000"** (recapture; raise the cap to 54,000). The bounded amendment is [`sprint24-weekly-monthly-cache-budget-amendment-a01.md`](sprint24-weekly-monthly-cache-budget-amendment-a01.md); it is prepared for readiness review `S24-7-M4-PRE-C1R1` and no deployment, remote execution or movement has occurred.

- Cap accounting: 54,000 cumulative, 26,431 spent, 27,569 available; the declared 96-forward smoke plus the 26,262-row full capture totals 26,358 new forwards and a cumulative 52,789, leaving 1,211. Without the separate smoke the cumulative is 52,693, leaving 1,307. This is an accounting amendment, not a retry budget, and the frozen protocol document and study digest are unchanged.
- Identity: the study-required identity remains the existing `s24_feature_cache_identity` over the study's full required union. For `DEVELOPMENT-01` it is a **new cache-input identity** because G2 omitted the 13 window-1 files; that is declared as exact study coverage, never as a G2 match.
- Capture route: `experiments/sprint24_task7m4_capture_runner.py` with `fixture` (local, no model), `smoke` (96 forwards) and `full` (26,262 forwards). Only `gpu-forward` provenance bound to the execution commit is admissible; `gpu-forward-smoke` and `fixture-no-forward` are refused by the admission gate, so a smoke or local fixture can never satisfy the study.
- Local public-path smoke: `capture-fixture/fixture-smoke.json` — 0 forwards, 0 model loads, 12 fixture entries written and read back, all rejected for their provenance, coverage `0/26,262`, admission `REFUSED`. This proves the writer/sidecar/identity/coverage plumbing and the provenance guard without a local forward; the local host has a CPU-only build, so no local `cuda` identity was claimed.
- Sequence after review PASS and a fresh Bronze checkpoint: exact reviewed commit, empty remote checkout, bounded `smoke` then `full`, store identity/coverage proof (`26,262/26,262`), then the study admission CLI, and only then endpoint computation. Final evidence remains `S24-7-M4-C1R1` plus a separate Bronze final checkpoint.

This section records preparation only. It is not an execution result, and it changes no endpoint, criterion or disposition.
