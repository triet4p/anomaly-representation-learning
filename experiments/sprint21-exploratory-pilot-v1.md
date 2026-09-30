# Sprint 21 exploratory pilot protocol v1

**Protocol ID:** `sprint21-exploratory-pilot-v1`  
**Status:** FROZEN FOR INDEPENDENT PRE-CONTACT REVIEW — **NOT AUTHORIZED FOR BOUND-SEED EVALUATION**.  
**Claim:** research-only, descriptive, simulator-conditional.

Task 3's actual all-three-checkpoint disposable proof has passed and was independently reviewed PASS. This protocol and binding now freeze Task 4's choices; they do not authorize generation or scoring. An independent reviewer must inspect this document, `experiments/sprint21-pilot-binding-v1.json`, the exact Task4 runner/source commit, and the safe dry-run evidence, and return zero actionable findings before any bound data seed is evaluated. No Confirmation/Sealed path or data was inspected or authorized. No tolerance gate, miss allowance, Gate V2 decision, Sprint 18 quota/Cycle 4 boundary, or Sprint 20 `NOT_READY` decision changes.

## 1. Prospective unit, fixed block, and freshness boundary

The independent sampling unit is one **whole generated history**. The fixed roster has 32 unique history IDs, roots, and data seeds: `H-S21-PILOT-01..32`, seeds `921000..921031`, and roots `histories/H-S21-PILOT-01..32` beneath the one-shot output root. The complete coordinate-by-coordinate values and 32 per-seed config fingerprints are in the canonical-hash-bound binding.

`H=32` is a prospective sample-size choice only. It is not an `H=32,m=2` decision, a permitted-miss count, a precision guarantee, or a tolerance-selection rule. There is no outcome-driven extension, replacement, seed search, favorable-seed selection, post-outcome budget change, or reuse of these research-only coordinates for future Evaluation, Confirmation, or a gate.

Freshness is a static collision check against the cited, hash-bound repository role/evidence catalog only. The binding enumerates prior history/data ranges `742–797`, `910–933`, `1000–1017`, `1100–1117`, `1200–1217`, `1300–1317`, `1400–1417`, `1500–1517`, `1600–1617`, `2700–2715`, `2800–2815`, `31800–31847`, `918000–918015`, and `918100–918107`; single seeds `0`, `9000`, `9001`, `93058`, and `93061`; and distinct model-only ranges `171701–171703` and `181801–181803`. The fixed block is disjoint from every enumerated value. Citations and SHA-256 digests are bound in the JSON. This is not a claim that every possible server namespace was searched: there is no history-root scan and no Confirmation/Sealed inspection.

## 2. Exact construction contract

Use only the historical pinned generator/source tree at base commit `8c15f0204a3e495569b7f143dc109943e8b808de`, with `synth.chronicle.sprint15_v7_history_config(data_seed)`, profile `sprint15-v7`, generator version `2.0.0`, profile protocol `sprint15-benchmark-protocol-v7`, and manifest protocol tag `sprint21-exploratory-pilot-v1`. The exact generator protocol SHA-256 is `a1fc09db2e246ed79d0595aec953a7788fd1b47c4981fbe1d92017d944b8d7b6`; normalized template and each full resolved config hash/config hash are frozen in the binding.

For a coordinate that is reached, call exactly:

```python
materialize_chronological(
    cfg, root, shard_size=64, overwrite=False,
    role=history_id, protocol="sprint21-exploratory-pilot-v1", sprint15=None,
)
```

Then call the existing `synth.balanced.allocate_quotas(rows, ledger, maintenance_windows, data_seed, QuotaConfig(), method="exact")`. This is the pre-existing `DEFAULT_QUOTA`/`QuotaConfig()` construction method and quota; no new construction quota or custom selector is introduced. The exact quota is P=24 (P1=12, P2=12), W=24 (W1=12, W2=12), A=16 (A1=8, A2=8), controls=48, two programs, 240 robot-days, at least six positive and six negative robots, positive-robot cap 0.35, negative-robot cap 0.40, program cap 0.60, and 14-day spacing. Full `dataclasses.asdict(DEFAULT_QUOTA)` and its digest are in the binding.

Do not run a preflight, quota audit, threshold/audit scorer, historical Task7 training `main()`, training, fitting, refitting, calibration, or any current-checkout fallback. Allocation infeasibility is retained as a hard scientific support failure; it is not retried, searched, or replaced.

## 3. Hard support floors and diagnostic margins

A reached history is qualified only if every frozen structural/hard flag passes. The numerical support floors are P≥10 eligible event windows, W≥10, A≥8, total positive windows≥30, eligible same-history control windows≥25, eligible robot-days≥150, at least six contributing positive robots and six contributing negative/control robots, at least two programs in each P and W, each P/W/A cohort share between 15% and 60% of positive windows, largest positive-robot share≤35%, largest negative-robot share≤40%, largest program share≤60% separately for P and W, and lead-support fraction≥80% separately for P and W. The frozen structural checks also require the source manifest/seed contract, valid row allowlist and chronology, nine-robot vocabulary including reserve robot, reserved program, valid cohorts/subtypes and degradation duration shapes, exact 64-positive allocation, exact existing 48-control construction quota, and allocator seed bound to that history. The runner records all counts, flags, allocation disposition, support keys, and lead-support details.

The design margins P≥13, W≥13, A≥10, total positive≥38, controls≥32, and robot-days≥188 are **diagnostic only**. A margin miss does not fail a history, stop execution, change the endpoint, or create a tolerated-miss rule. Support-margin strata for both margin-met and margin-missed hard-pass histories are descriptive only.

**Continuation/stop rule:** on the first history that fails any hard/structural support flag (including exact allocation failure), persist its complete manifest/support evidence, mark the remaining fixed coordinates unattempted, and stop immediately. Do not materialize another history, restore a scorer, or score earlier support-pass histories after this stop. Write a scientific-stop summary with all three scorer-seed endpoints, full-roster macros, variances, and bootstrap LCBs explicitly `UNCOMPUTABLE`/null; no subset is ever summarized. Design-margin misses do not activate this stop. A source/hash/runtime/asset failure stops before data generation; any unexpected operational/inference failure is recorded `ABORTED` with completed and unattempted score coordinates and is never resumed or retried.

## 4. Frozen scorer exception and input boundary

The only scorer is the immutable Sprint 17 B0 representation, restored one model seed at a time from the original read-only step-300 checkpoint paths. Exact model seeds, filenames, 18,239,321-byte sizes, and full hashes are in the binding:

- `171701`: `b0_seed171701_step300.pt`, `45f9e151c5d3946804ea531eb2bcace26b1f62e634671f51b8741ac6b411a619`
- `171702`: `b0_seed171702_step300.pt`, `64ed2cedec210e4692f60bb4dd3430a57cf94abfcd50551abd97c9265ea785f8`
- `171703`: `b0_seed171703_step300.pt`, `9edb2122e357376a9ff1e065d73d5ab7704f8d2005991552332f1ced01b0e906`

All three seeds are mandatory, remain separate scorer variants, and use the same independent whole-history roster. Their scores are never pooled as additional histories or used to select a favorable seed. The exact Task3 wrapper is commit `f8a8fa6dd3ae1534c8f8db8d552898ffe88314ba`, SHA-256 `d1d8239d34944ab54f371555f68d1431f7d4784ba0a67bd4fec49359ba8f7a32`. It restores the checkpoint-embedded historical Fit-only reference bank unchanged (5,040 rows × 128 dimensions, k=5); the Task3 proof recorded the real-checkpoint restore, scores, state immutability, and negative cases. Historical provenance records 5,040 healthy Fit files and 154,129 valid Fit patches. No Evaluation row enters model, preprocessing, normalization, or bank fitting. There is no standalone bank file/hash; each exact checkpoint carries its bank. Stored Calibration thresholds are not read or used.

The reviewed Task 3 restore proof recorded immutable embedding digests for these checkpoint-resident Fit-only banks, now carried in the binding: `171701` = `432852b7c8ba6d1845b26e3e8f4d34f2e986258d39da22b844c15def2bc00662`; `171702` = `6bae50edd1aee8ac4a00a6255e85db01b89eeaef7341405ac58c93e0a15b511a`; `171703` = `7dfe8c3e63b764f0a043c5c38ba63fced463ce78846d38153bc5588a2a6efff8`. The safe Task 4 dry-run binds these reviewed digests and exact checkpoint file hashes without deserializing checkpoints or recomputing the embedded banks.


The narrow Fit/bank reuse exception is defensible only for this threshold-free descriptive diagnostic: the model/bank predate all newly generated histories, the bank is historical Fit-only, and the endpoints are within-history event AUROCs with no operating threshold. This is not a newly fitted prospective Fit/Calibration/Evaluation topology, does not reuse historical Fit/Calibration/Development/Design roots as Evaluation, and grants no Calibration, training, refit, bank-fit, threshold, recalibration, causal, prevalence, or deployment claim.

The fixed, source-derived conditioning map is:

- `robot-01..robot-09` → `robot_idx` 0..8 in sorted identifier order.
- `program-01..program-08` → `program_idx` 0..7 in sorted identifier order.
- Canonical map digest: `4f769ccc64aac4c6a91fdd356eaa6963afe976ab847a22df23e43f9331418897`.

The input signal is finite float32 `[6,T]` in the pinned order `feed`, `current`, `temperature`, `arc-voltage`, `arc-power`, `torch-pressure`. Each source file is scored once per model seed. Global file IDs are `history_id::source_file_id`; the existing Task7 mask rule hashes the UTF-8 global ID with SHA-256 and interprets the first four bytes unsigned big-endian modulo `2**31`. The runner independently recomputes both scores. It constructs only signal `x` plus explicit, validated numeric robot/program IDs for model use; labels, anomaly/event/split/future metadata are removed from the batch passed to inference and are used only for support/diagnostic bookkeeping. Required `NORMAL`/`seed=0` `FileSample` sentinels are inert and never seed an RNG.

The source tree is exact historical base commit above, not the current checkout. Task3's frozen full B0 config digest is `1ff67f95428ef29aab05d9f6394a305b75c8c2a9e12431f3cda09b6958596144`; metric-code digest is `b2d6af7505abe459a84f76f5279a6a5c4298e7c142901edd4596fcfe3ddeb88f`; `uv.lock` SHA-256 is `4a7866878c81cdd8f5ba283cd8d0a772073bfc75941f73ec761ede5eeb2e239a`; Task3 role-binding canonical/raw digests are `075868b22c028a981cc63ffe37b8d29720cd324c3e2c7254ce688678758230ba` / `f212ca2fd5a0f60eeb206435b610fa6a8689774ea490db2aa94d44c15a01baf1`. Direct source hashes, all metric member hashes, runtime, mapping and checkpoint identities are repeated in the binding and verified by the runner. Runtime is Python 3.12.13, PyTorch 2.14.0+cu130, CUDA on NVIDIA GeForce RTX 4060 Ti (16,380 MiB); no dependency installation or CPU fallback.

## 5. Endpoints, missingness, and uncertainty

Canonical positive event windows come from the pinned Task7 `evaluable_support` and `synth.events` predicates; canonical same-history controls are from `events.anchor_rows`/`events.select_control_windows` and must exactly match Task7 control support. Window score is the maximum file score among its members. For each history and each branch separately, the primary is tie-aware AUROC of all P+W event-window scores versus that same history's eligible controls. Secondary within-history metrics are P-only, W-only, and A-only AUROCs versus the same controls. Support and lead-support counts/reasons accompany scores.

`S_pred` is the mean masked prediction MSE over valid prediction-mask patches. `S_pop` is mean Euclidean distance to the five nearest embeddings in that checkpoint's embedded Fit bank. The two branches remain separate. No threshold, calibration, score fusion, or outcome selection is performed.

Every primary and secondary macro requires all 32 declared histories to be present, hard-pass, metric-computable, finite, and on the frozen common support. Any hard failure, missing/non-finite metric, roster defect, or branch support mismatch makes the whole-roster endpoints/macro/variance/bootstrap uncomputable; never average a surviving subset. If all 32 qualify, each scorer seed gets its own unweighted history macro and unbiased sample variance across independent histories. A deterministic 2,000-replicate whole-history bootstrap (NumPy `default_rng` seed `20260202`, 2.5th percentile LCB) is descriptive uncertainty only, not a gate or precision claim. Secondary P/W/A history macros are also reported only for a complete computable roster. Both sides of each design-margin stratum are summarized descriptively; no margin is selected as a success rule.

## 6. One-shot evidence and operational commands

The sole output root is `/tmp/sprint21-exploratory-pilot-v1`. Each attempt is no-overwrite and records `attempt.json`, append-only/fsynced `pilot-ledger.jsonl`, one support/manifest report per materialized history, one score JSONL per history/model seed if scoring is reached, and `summary.json`. Ledger events record each materialization start/completion, each history/seed score, seed completion, scientific support stop, or operational abort. Hash drift, missing assets, source/runtime/module isolation failure, or unexpected scoring errors stop; a scientific support failure stops immediately under §3. No retry, resume, continuation, replacement, omission, or rerun in the same root. Any partial output remains evidence and is not overwritten.
The binding's `device_total_mib` is the rounded result of `torch.cuda.get_device_properties(0).total_memory` converted to MiB; the remote dry-run observed 15,948 MiB for that exact Torch property.

Safe pre-contact dry-run (allowed because it checks exact source/runtime/config constructors and checkpoint file hashes only; it performs no bound-history materialization, preflight, deserialization, or scoring):

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/sprint21-proof-hist-v1/src \
/home/trietlm/anomaly-representation-learning/.venv/bin/python \
  /tmp/sprint21-proof-hist-v1/experiments/sprint21_pilot_runner.py \
  --dry-run \
  --worktree-root /tmp/sprint21-proof-hist-v1 \
  --checkpoint-root /tmp/sprint17-task7-out/checkpoints \
  --binding /tmp/sprint21-pilot-binding-v1.json \
  --expected-binding-sha256 "$(/home/trietlm/anomaly-representation-learning/.venv/bin/python -c 'import json; print(json.load(open("/tmp/sprint21-pilot-binding-v1.json"))["binding_sha256"])')" \
  --output-dir /tmp/sprint21-exploratory-pilot-v1
```

Only after the independent Task4 pre-contact review returns zero actionable findings may an authorized operator use the same exact arguments with `--run --review-passed` instead of `--dry-run`. `--review-passed` is an explicit gate acknowledgement, not a substitute for the independent review. The binding remains `FROZEN_PENDING_PRECONTACT_REVIEW` and `contact_authorized:false` until the review; this artifact does not assert that review or any run has happened.

No result from this pilot can amend Sprint 20, Sprint 18, Gate V2, or any future acceptance rule. Any future threshold/tolerance decision requires a separate prospective user decision, protocol, and untouched independent histories.
