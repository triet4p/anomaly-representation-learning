## [2026-09-04] Scheduled joint loss hid prediction degradation and produced an incoherent checkpoint

**Symptom:** Validation joint loss rose after contrastive lambda activation, contrastive loss rapidly approached zero, and the final checkpoint reported step 7840 while containing epoch-2 model weights with no contrastive training.
**Root cause:** The contrastive branch fed raw physical-scale signals into an encoder shared with normalized prediction inputs; latent outputs could drift in scale; weak augmentations made InfoNCE saturate; best-state selection compared losses computed under different effective lambda values; and restoration replaced model weights without synchronizing optimizer, scheduler, step, and reference-bank construction.
**Fix / workaround:** Normalize both branches identically, bound latent outputs, route InfoNCE through a projection head, calibrate augmentations/temperature, select with a stationary objective, and restore/save model, optimizer, scheduler, step, lambda state, and reference bank coherently.
**Watch out for:** Any warmup or ramped multi-objective training where checkpoint ranking uses the scheduled total loss, especially when branches share encoders but preprocess inputs differently. Always log raw and weighted component losses plus latent norms.

## [2026-09-05] Hard-task agents repeatedly fail with upstream HTTP 500

**Symptom:** A `hard-task` invocation routed to `opencode-go/muse-spark-1.3-contributor` exhausted ten retries on HTTP 500 and returned no implementation output.
**Root cause:** OMP version 18.1.4 had the agent-routing/runtime defect reported by the user; the repository and packaging assignment were not the cause.
**Fix / workaround:** Upgrade OMP, then start a fresh `hard-task` invocation; the post-upgrade run completed the same assignment successfully.
**Watch out for:** Repeated provider-style HTTP 500 failures from a subagent before it emits any artifact while running OMP 18.1.4. Preserve task state, upgrade OMP, and retry rather than debugging project code.

## [2026-09-05] Static package review polluted the export with bytecode

**Symptom:** A read-only evidence review left 46 `.pyc` files in five `__pycache__` directories under `exports/kaggle-20260905-01/src/` even though `src.zip` and the outer ZIP remained clean.
**Root cause:** The reviewer imported modules from the on-disk export with normal Python bytecode writing enabled.
**Fix / workaround:** Remove all generated cache directories, set `PYTHONDONTWRITEBYTECODE=1`, and validate notebook imports with AST/text/archive inspection rather than importing from the export tree.
**Watch out for:** Any packaging verification that imports copied source before asserting a cache-free directory. Check the folder after every review, not only archive membership.

## [2026-09-05] Geometry extraction mixed CPU batches with CUDA normalization state

**Symptom:** Kaggle GPU extraction failed in `wrapper_CUDA_index_add` because the index tensor was on CPU while the reduction accumulator was on `cuda:0`.
**Root cause:** `BoundedEmbeddingExtractor` moved the model and conditional-normalization statistics to CUDA but sent its direct unmasked model call a CPU batch; it also left the loaded model in training mode, entering `ConditionalBatchNorm._update_statistics`, whose indices and sources assumed co-located devices.
**Fix / workaround:** Put extraction in evaluation mode, move the direct-call batch through the inference device transfer, and make conditional-normalization update/gather reductions explicitly co-locate indices, sources, accumulators, and returned statistics with no-op transfers on already aligned tensors.
**Watch out for:** Any code path that calls the model directly instead of `RepresentationInference.score_batch`, especially after `model.to(cuda)`, and any `index_add`/advanced-index operation whose index originated in a CPU dataloader.

## [2026-09-05] Static notebook checks missed an undefined finalization helper

**Symptom:** Geometry extraction completed its streaming work and then crashed while building the output manifest because `_artifact_info` was called four times but was never defined or imported.
**Root cause:** Notebook code-cell compilation validates syntax only, and prior focused tests did not drive the public `extract_embeddings` path through successful final artifact publication.
**Fix / workaround:** Define the typed `_artifact_info` helper beside checksum utilities, audit every finalization dependency, and retain a tiny real CPU synthetic test that materializes data, saves a coherent checkpoint/reference bank, calls public extraction, and verifies every artifact size/hash and manifest link.
**Watch out for:** Pipelines whose expensive processing is tested separately from their final manifest/archive return path. Require one bounded end-to-end smoke that reaches the public return value; syntax compilation cannot catch `NameError`.

## [2026-09-05] Class-pair diagnostics starved and distance matrices ballooned on ordered data

**Symptom:** Separation metrics returned null on mixed-class data despite anomalies present, and analysis memory spiked into multi-GiB territory on larger caches.
**Root cause:** The pair budget was consumed in lexicographic row order, so normal-first materialization exhausted it on same-class pairs before cross-class pairs were reached; reference distances were computed by broadcasting a query×reference×dimension temporary before reduction.
**Fix / workaround:** Budget same-class and different-class pairs separately with deterministic strided sampling, and compute squared distances from norms plus block matrix products with running top-k merge, self-exclusion, and a roundoff clamp.
**Watch out for:** Any capped-pair statistic applied to naturally ordered (normal-first) datasets, and any `(a[:,None,:] - b[None,:,:])**2` pattern — check both dimensions for a materialized 3-D temporary.

## [2026-09-05] Inference overwrites the restored checkpoint reference bank

**Symptom:** Executed inference reported `S_pop` from 64 train embeddings despite restoring an 8192-row checkpoint bank, and its AUROC sat near chance (0.538/0.509).
**Root cause:** Both branches of the inference notebook refit the reference bank on the 64-file reference batch, unconditionally discarding the restored checkpoint bank; the behavior persists in the canonical notebook at HEAD.
**Fix / workaround:** Treat every reported `S_pop` as bank-dependent and state the bank row count alongside the score; resolve the overwrite with an explicit decision before any rerun claims full-bank scoring.
**Watch out for:** Any `bank.fit` call after `load_checkpoint` with `reference bank restored: True` — restoration plus silent refit reads as full-bank scoring but measures a tiny refit bank.

## [2026-09-06] Near-chance detection survived full-bank scoring, tuning, and probes

**Symptom:** Full 8192-row restored-bank inference (F1 0.195), calibrated thresholds (best F1 0.515 on balanced data), score fusion (identical to S_pred alone), supervised linear probe ceiling (AUROC 0.541), production mixed eval (F1 0.233), and a six-cell λ/τ/augmentation grid (F1 0.28–0.33) all stayed near chance.
**Root cause:** The frozen 128-d embeddings carry a small stable prediction-loss normal/abnormal gap (+13–18%) but the contrastive loss never separates (≤0.5%); no scoring, threshold, fusion, probe, or cheap hyperparameter change creates separability that is not in the representation.
**Fix / workaround:** Retrain at the representation level (objective/architecture/signal) instead of further scoring or λ/τ/aug tuning; keep the per-cell ablation table as the baseline that a new run must beat.
**Watch out for:** Any proposal to fix detection with thresholds, fusion, or bank size on these embeddings — Tasks 3–7 closed those axes with numbers. Demand a probe-ceiling lift on frozen embeddings before accepting a scoring-side fix.

## [2026-09-13] Preflight assertion before diagnostic persistence lost a long failure

**Symptom:** A 16-binding rejection-only preflight ran for 1107.67 seconds, then an assertion expecting `PREFLIGHT-PASS` / `16/16` failed before the result was written, losing the per-binding failure reasons.
**Root cause:** The wrapper asserted the unanimous-pass contract before persisting the deterministic raw `run_preflight` result; a failed assertion discarded the in-memory result even though the expensive computation had completed.
**Fix / workaround:** For the one authorized diagnostic replay, write the complete raw result and per-binding gate reasons first, hash the persisted record, then perform any pass assertion. This captured the deterministic `PREFLIGHT-FAIL` / `15/16` result and sole failing seed.
**Watch out for:** Any expensive preflight, audit, or batch command whose evidence is printed or persisted only after a success assertion. Persist raw deterministic outputs before assertions so fail-closed decisions retain exact rejection evidence.

## [2026-09-23] Windows `:raw` selector can masquerade as a missing graph

**Symptom:** Reading an existing `graphify-out/graph.json:raw` returned `ENOENT` with the literal suffix in the Windows path, causing the graph to be reported absent.
**Root cause:** The Windows path resolver treated `:raw` as part of the filename instead of a read selector.
**Fix / workaround:** Check graph existence through `read("graphify-out/")` or `Path.is_file()`/`stat()` first; use a path without `:raw` for Windows graph existence checks.
**Watch out for:** A selector-suffixed file read in a Windows project returning `ENOENT`; verify the actual path before claiming absence.

## [2026-09-23] Module invocation stopped before cluster-only refreshed outputs

**Symptom:** `python -m graphify cluster-only .` printed only `Loading existing graph...` and returned without changing `GRAPH_REPORT.md` or `graph.html` after a targeted merge.
**Root cause:** The difference between the module invocation and the installed console entry point is not isolated; the module command did not complete the cluster-only workflow in this Windows environment.
**Fix / workaround:** Run the documented console command `graphify cluster-only .`; it completed re-clustering 3,430 nodes / 7,703 edges into 162 communities and refreshed `GRAPH_REPORT.md`, `graph.json`, and `graph.html`.
**Watch out for:** If the module form exits after the loading message, verify the report and HTML outputs; use the installed `graphify` console entry point rather than treating the partial output as success.
## [2026-09-23] Role-block diagnostics received summary records where support was expected

**Symptom:** A complete-looking measurability run stopped with `KeyError: 'support'` before producing a scientific verdict.
**Root cause:** Role-block evaluation synthesized manifest-like records from summary data; those records matched the expected histories but did not contain the nested structural `support` field.
**Fix / workaround:** Pass the explicit per-history structural-summary mapping to role-block evaluation and read support from that mapping instead of mixing manifest and summary schemas.
**Watch out for:** Diagnostics that join records by history ID across different schema layers; validate the source schema and pass its actual owner rather than constructing partial manifest stand-ins.

## [2026-09-27] Sealed row digest was missing from the attempt marker

**Symptom:** The one-shot study persisted its first sealed v7 row, then stopped with `KeyError: 'record_sha256'` while updating the attempt marker.
**Root cause:** `_seal_row()` returns a new shallow copy containing `record_sha256`; the persistence loop appended that copy but read the digest from the original, unsealed row. The earlier smoke exercised metric generation but not the shared study result/marker persistence loop.
**Fix / workaround:** Keep and use the sealed row object for result persistence and marker updates. Exercise the same persistence path with a disposable sealed fixture import, verify the final result/summary and marker hashes, and prove a duplicate attempt is blocked. For an already consumed one-shot output, preserve its bytes and continue only under a reviewed amendment into a distinct output directory; never regenerate the completed row.
**Watch out for:** Any row-sealing helper that copies rather than mutates, especially when result persistence and attempt-marker persistence are separate writes. End-to-end smoke must traverse the actual marker loop, not only the metrics path.

## [2026-09-28] Reference profile hashes differ from seed-resolved config hashes

**Symptom:** The Task 61 disposable smoke blocked before building seed `93061` because its resolved v7 config hash did not equal the frozen v7 reference hash.
**Root cause:** Task 58 freezes profile hashes at reference seed `0`, while each diagnostic coordinate resolves a seed-specific config. The observed fixture hashes (`6cfa66671c28` for v7 and `80233f567d28` for C2) therefore differed from the reference hashes (`7de530a28d23` and `720e31e94681`).
**Fix / workaround:** Compare frozen release hashes/rates with the frozen reference-config contract, but preserve each coordinate's actual resolved config hash in its row without requiring it to equal the reference hash.
**Watch out for:** Multi-seed diagnostics that validate per-coordinate config hashes; distinguish the profile reference binding from the row's seed-resolved hash.

## [2026-09-29] Verify complete evidence assembly before consuming a one-shot output root

**Symptom:** The authorized disposable proof exited fail-closed with `KeyError: 'runtime'` and wrote no proof JSON.
**Root cause:** Preflight measured the runtime but omitted it from its returned context, while final proof-record assembly read `context["runtime"]`.
**Fix / workaround:** Include the measured runtime in the preflight context. After correction, the focused consumer tests and both source AST parses passed; the original fixed output root is preserved and cannot be reused under the v1 contract.
**Watch out for:** A late result-assembly lookup can fail after a long one-shot workflow has consumed its output authorization. Validate the complete evidence path and its returned context on an independently authorized disposable run; never recycle a consumed output root.

## [2026-10-02] Raw source closure rejected the equivalent Linux checkout

**Symptom:** Task 69's clean Linux pre-contact guard exited 2 because the Sprint 15 probe document's raw SHA-256 differed from its frozen Windows digest even though both checkouts referenced the same Git blob.
**Root cause:** The binding hashed Windows CRLF working-tree bytes, while the Linux checkout produced the tracked LF bytes; a raw-byte source identity therefore treated a declared checkout conversion as a source mutation.
**Fix / workaround:** Bind canonical SHA-256 over strict UTF-8 text with only uniform LF/CRLF normalization, preserve all accepted raw hashes as provenance, bind tracked Git blob IDs, and reject any other content/EOL/policy/blob/receipt change.
**Watch out for:** Never generalize newline conversion to binary or invalid text, accept mixed/bare CR, change historical document bytes, or reuse the prior release receipt after the source scheme or checkpoint parent changes.

## [2026-10-02] CUDA build suffix and bare import path blocked the runtime guard

**Symptom:** Task 69's corrected Linux source/Git checks passed, but the runtime guard rejected observed `torch.__version__` `2.14.0+cu130` against the bound `2.14.0`; direct script execution also failed to resolve the bound `synth` source without explicit `PYTHONPATH`.
**Root cause:** The exact runtime pin omitted the observed CUDA wheel build suffix, and Python's direct-script import path did not include the bound `src` tree.
**Fix / workaround:** Bind the exact observed `2.14.0+cu130` string and require `PYTHONDONTWRITEBYTECODE=1`, exact bound-root `src:experiments` `PYTHONPATH`, and the existing locked `.venv` for static and candidate stages; pass that same environment to public CLI children.
**Watch out for:** Never strip or wildcard the build suffix or bypass source-origin guards. A bare or stale-root invocation must fail closed; a new Task68 checkpoint requires a fresh receipt before any candidate contact.

## [2026-10-07] Paused-attempt proof custody beats regeneration on resume

**Symptom:** A resumed continuation faced complete but unreviewed fixture roots from a user-paused attempt (four fresh-process reloads, two comparisons, allocator/probe results) plus unrun focused tests, changelog, and durable summary.
**Root cause:** The pause invalidated the prior attempt for gate advancement but left its on-disk proof records valid; regenerating them would risk overwriting append-only evidence and waste two full CLI generations.
**Fix / workaround:** Verify every owned file hash against the pause handoff first, preserve inherited roots byte-identical with zero regeneration, and run only the unrun discriminating work (focused pytest scope + read-only hash/ledger reconciliation + durable non-ignored summary).
**Watch out for:** Any resume/continuation assignment after a PAUSED-PARTIAL handoff: identity-check before touching, never rerun ground-truth failures or re-contact frozen seeds, and record provenance explicitly (inherited vs newly executed).

## [2026-10-07] Stage ledger Path/list unpack mismatch refuses before any role write

**Symptom:** `materialize-nonconfirmation` exited 2 with `AttributeError: 'list' object has no attribute 'is_symlink'` before writing any role root; the events list from `_load_stage_state` was passed where a ledger `Path` was required.
**Root cause:** `_load_stage_state` returns `(candidate_root, attempt_root, events)` while `_candidate_paths` returns `(candidate_root, attempt_root, ledger_path)`; two callers unpacked the former into a `ledger` name and fed the list into `read_ledger`/`append_event`/`_fail_candidate`.
**Fix / workaround:** Carry the ledger as a `Path` from `_candidate_paths` and the events from `_load_stage_state` at every stage caller (`materialize-nonconfirmation` dispatch, `run_confirmation`, `record-nonconfirmation-pass`), with distinct `*_ledger_path` names; prove with a genuine disposable ledger read/write plus a real single-role public CLI materialization, and a regression asserting Path-vs-list separation.
**Watch out for:** Same-named tuple unpacks across helpers with different third-element types. A green run on unrelated suites does not cover the stage boundary; require the disposable stage/ledger proof above.

## [2026-10-07] Eager CLI version probe refused every command without app metadata

**Symptom:** The first public CLI child on the deployed Linux worktree exited 1 with `importlib.metadata.PackageNotFoundError: No package metadata was found for anomaly-representation-learning` from `build_parser()`, retiring a 16/16-preflight-PASS candidate before any role write, while pre-contact `validate_runtime` had passed exit 0.
**Root cause:** `version(_DISTRIBUTION_NAME)` ran eagerly during parser construction for every invocation, but the isolated worktree `.venv` holds third-party `dist-info`s only; the guard checked `import synth` (source-tree origin) and never built the parser or probed the CLI entrypoint, and local proof ran where the editable install masked the gap. A second local trap: `src/*.egg-info` also satisfies metadata discovery, so whole-`src` on `sys.path` can never reproduce absence.
**Fix / workaround:** Resolve the version lazily in a dedicated `--version` action (stdout on success, honest exit-2 stderr otherwise, no fallback); run the real same-interpreter `python -m synth.cli --help` plus bound origin probe inside `validate_runtime` before contact; prove absence with a junction import-root (`synth` -> live `src/synth`, no `dist-info`/`egg-info` on the discovery path, resolved `__file__` equality asserted) driving real `dispatch_stage` writes plus public loader reloads on disjoint disposable seeds.
**Watch out for:** Any `importlib.metadata` call at import or parser-build time; validator coverage that stops at `import package`; local editable installs or `src/*.egg-info` masking a server-layout metadata gap; proof fixtures that copy source trees or mock metadata discovery instead of linking real package folders.

## [2026-10-08] New CLI branch orphaned a sibling dispatch while keeping its flag

**Symptom:** After inserting the `--assessment` branch, `--smoke-no-contact --stage validate` exited 2 at the Linux runtime gate instead of running the no-contact smoke; the flag still parsed and `run_no_contact_smoke()` still existed but nothing called it, and the worker handoff still claimed restoration.
**Root cause:** The insertion replaced the `if args.smoke_no_contact: ... return 0` block ahead of normal `validate` with the assessment branch, and the follow-up closure refresh rebound the edited runner bytes — so the fall-through looked like a bound, passing state while the dispatch was dead.
**Fix / workaround:** Restore the exact accepted-baseline 5-line block after the new branch (assessment-first refusal preserved), refresh ONLY the runner member in both binding carriers, and prove the real CLI both ways: pre-fix fall-through refusal vs post-fix exit-0 disposable PASS plus stage/release/assessment precedence refusals.
**Watch out for:** Any `main()` branch insertion ahead of normal dispatch: diff the baseline `main()` dispatch order line-for-line before refreshing closures, and never claim a dispatch restoration without invoking the changed CLI flag end-to-end.

## [2026-10-08] Threading a stale release checkpoint starved the legitimate Task69 checkpoint

**Symptom:** The exact public `--assessment materialize-confirmation` with the corrected v2 receipt (`assessment_checkpoint_commit=003b94bc`) still exited 2 with `Task69 assessment release does not authorize this assessment Confirmation stage`, while the superseded v1 receipt (`bf49a819`) passed the field gate but failed the parent gate — no receipt value could pass both through the real CLI.
**Root cause:** `main()` threaded `assessment_checkpoint` from the preflight `--release` receipt (`bf49a819`, frozen for the preflight/non-Confirmation contract) into `run_assessment_confirmation`, which used that same value for BOTH the Task69 release field check AND the `git rev-parse 7be574d9^` ancestry check (whose true parent is the reviewed assessment implementation `003b94bc`); a follow-up trap was "fixing" it by refreshing the frozen binding/closure, which orphans the qualified marker/record/ledger pins that the correction must preserve.
**Fix / workaround:** Resolve the reviewed execution identity from the Task69 release's own `assessment_checkpoint_commit` field (pinned to `003b94bc`, ancestry-checked against `7be574d9^`) and authenticate the presented binding as the immutable qualified DATA identity (binding/closure/record/28-event ledger pins), leaving all carriers byte-identical; prove with consumer custody tests plus real disposable public CLI writes/reloads and the real authorization helper on qualified-old fixture bytes.
**Watch out for:** Any single checkpoint variable feeding two gates with different legitimate values (release-field vs ancestry-parent); any binding/closure "refresh" inside a correction whose acceptance depends on the old frozen custody. Never mutate the qualified carrier to fit corrected code — authenticate old DATA by immutable pins and new code by released-source validation.

## [2026-10-08] Corrected code behind an old source-closure check refuses before and after deploy

**Symptom:** The C01 inner checkpoint fix passed all focused tests, yet the exact public `--assessment` CLI still exited 2 with `frozen source mismatch` on the corrected runner, and the handoff claim that a reviewed deploy would clear it was unfeasible: no checkpoint/deploy rewrites binding files, and rewriting them would orphan the qualified DATA pins.
**Root cause:** `main()` unconditionally validated the on-disk runner against the old live closure before any stage logic, so corrected bytes could never reach the corrected inner gate; the DATA closure authenticates old qualified bytes while execution bytes need a separate sealed identity anchored in the release plus Git ancestry.
**Fix / workaround:** Gate `--assessment` contact stages behind a sealed execution descriptor bound to the reviewed execution checkpoint, both Main releases, and the corrected runner blob/canonical bytes resolved via `rev-parse <exec>:path` and `HEAD:path` plus a working-tree byte check; keep bare validate on the exact old guard and all DATA carriers byte-identical.
**Watch out for:** Any late-entry correction whose fixed file is also a frozen closure member: never claim deploy clears a source-mismatch refusal unless the deploy path actually rewrites the binding, and never refresh the binding when acceptance depends on its old pins.

## [2026-10-08] A sealed-descriptor check that equates old and new bytes can never pass

**Symptom:** The C02 `_verify_assessment_execution_descriptor` forbade the old runner blob at line 702 yet required the descriptor blob to equal the `003b94bc`-committed blob (the same old bytes) at line 718, and additionally required `HEAD:` to equal it at line 720 — so no descriptor could ever authenticate, and the `good` test fixture itself errored.
**Root cause:** The A-side history identity (old committed bytes) and the C-side execution identity (new worktree/commit bytes) were conflated into one equality check, while the descriptor product was deferred to Bronze instead of authored with the correction.
**Fix / workaround:** Split the contract: pin the A blob for the history check only, authenticate C against the current worktree blob+canonical bytes pre-commit (HEAD still carries A by construction), enforce committed-C equality post-checkpoint in the separate Main execution-release guard, author the descriptor now, and carry a positive REAL-auth test (not negative-only).
**Watch out for:** Any guard that both forbids value X and requires equality with a commit that holds X; any `good`-named fixture that is only ever asserted to raise; any product file a worker claims Bronze will create.

## [2026-10-08] A release validator with no public CLI caller cannot gate contact

**Symptom:** Review found that `validate_assessment_execution_release()` correctly rejected absent receipts when called directly, but the normal Confirmation CLI never called it, so the separate Main authorization was not required.
**Root cause:** The assessment CLI parsed only the Task69 result receipt and dispatched Confirmation without a distinct corrected-execution receipt input; the standalone validator was dead code.
**Fix / workaround:** Add `--execution-release` for assessment Confirmation and call the existing validator with the descriptor already authenticated by the entry gate, before runtime checks or Confirmation dispatch; keep bare read-only assessment validation release-free.
**Watch out for:** An unused, well-tested authorization helper does not protect its consumer. Exercise the normal public contact command with missing and foreign receipts and assert the refusal precedes runtime/contact rather than testing only the helper.


## [2026-10-09] A pre-checkpoint HEAD assumption rejects every deployed entry

**Symptom:** After the reviewed C checkpoint deployed, every public `--assessment` entry exited 2 with `assessment execution descriptor anticipates an unreviewed commit`, even though the descriptor pinned the exact corrected bytes: the verifier demanded `HEAD:` differ from the descriptor blob and equal the old qualified blob, which can only hold while HEAD still equals B.
**Root cause:** The v1 static contract encoded the AUTHORED candidate state (HEAD predates the correction, still carries A) as a permanent rule instead of a transient state: it never accepted the DEPLOYED reviewed state where HEAD equals the corrected base C, and the release validator hardcoded parent(C) == B so a separate child-C correction could never authorize either.
**Fix / workaround:** Pin the stable reviewed base C in the v2 descriptor (base commit + committed descriptor blob + corrected runner identities that must differ from A), accept HEAD == base C and HEAD-descends-from-C in the static gate while keeping the authored pre-C branch strict, and move the D lineage (parent(D) == C, C child of B child of A) into the Main execution-release guard with the D SHA minted only after checkpoint; prove the authored-to-deployed transition with a genuine disposable git replica plus real-subprocess CLI negatives, never mocks.
**Watch out for:** Any guard that requires HEAD to differ from the sealed bytes after the sealed bytes are committed; any release validator that hardcodes a single parent so a child correction is unreachable; any test that asserts a committed-HEAD equality while HEAD still predates the commit.

## [2026-10-09] Detach long remote compute with setsid and </dev/null before disconnect

**Symptom:** The one-shot assessment Confirmation dispatch ran as `nohup … &` inside an SSH heredoc without `setsid` and without stdin decoupling; the SSH session close killed launcher pid 5994 ~4 minutes in, after the runner had legitimately appended ledger events 29–30 but before the first `synth.cli` child saved any output (0/4 materialized, stdout/stderr 0 B, no orphan process).
**Root cause:** POSIX OpenSSH non-interactive commands run in a session whose process group is signalled when the client disconnects; `nohup` alone ignores SIGHUP but does not detach the process group or close stdin, so the launcher still died with the session.
**Fix / workaround:** Launch detached as `setsid nohup <command> > log 2>&1 < /dev/null &`, then verify the PID survives, the log is non-empty, and the exit/log artifacts exist before disconnecting (poll on a 5–10 minute cadence, never spin). A correctly detached relaunch (`setsid nohup … & < /dev/null`, pid 7469) demonstrably survived the session close.
**Watch out for:** Any SSH heredoc background launch without `setsid` and `< /dev/null`; treating a killed launcher as a scientific failure (the candidate was NOT retired and the ledger bytes were legitimate append-only records); resuming an interrupted one-shot stage without explicit Main/user authorization (the fail-closed no-resume policy is intentional).


## [2026-10-12] Stable sampler seeds can still violate the frozen protocol

**Symptom:** Sprint18 experiment permutations were deterministic, but their seed digests did not match the frozen sampler contract.
**Root cause:** One shared helper applied an internal canonical-JSON identity hash to experiment representation and N4 orders instead of the exact protocol-defined seed strings; it also rejected the N4 two-history fold scope.
**Fix / workaround:** Use the exact inherited v1 representation seed string and the v2 N4 final/fold seed string for experiment scopes. Keep the explicitly separate one-history Task6 smoke ordering isolated from those experiment rules.
**Watch out for:** Determinism alone does not prove schedule compatibility. Compare seed derivation and fold identity with the normative protocol before relying on checkpoint sampler order.

## [2026-10-19] Healthy-only certified loaders refuse Development by design

**Symptom:** The T23 runner's Development scoring crashed unconditionally with `ValueError: Task6's model smoke loader is restricted to Fit and Calibration histories`, because it called `load_certified_sample_subset` on DEVELOPMENT histories without `role_action`; passing a healthy `role_action` would instead refuse with `non-normal-label rows` since Development mixes normal and abnormal files.
**Root cause:** The healthy-only subset loaders enforce both the Fit/Calibration role gate and the normal-label gate; Development descriptive scoring needs the full frozen support (normal + abnormal) under the separate DEVELOPMENT `score_development` permission with the same manifest/shard/member verification but no eligibility filtering.
**Fix / workaround:** Add a runner-local full-support Development loader that calls `validate_role_permission(..., "score_development", healthy=False)`, then mirrors the frozen manifest digest/identity, file/shard membership, exact shard coverage, per-shard sha256, member identity/label, and channel/length/cardinality checks, returning every manifest-bound row unfiltered.
**Watch out for:** Any scoring path over Development/Confirmation histories that reuses a healthy-role subset loader. Also: tiny `experiment`-scope checkpoint/bank/scoring fixtures must satisfy the frozen full-support geometry (complete Fit-healthy rows, frozen batch schedule) or use the smoke scope; coherent checkpoints require the optimizer LR/step, sampler order/seed, scheduler epoch, and coefficient epoch to match the saved position exactly.

## [2026-10-19] Next-epoch-ready checkpoints need the next-epoch optimizer LR before the save

**Symptom:** The B0-FULL full run trained epochs 1-10 genuinely, then failed deterministically at the epoch-10 diagnostic write with `ValueError: optimizer parameter group 0 does not match the epoch schedule` (runner `_save_coherent` -> `save_sprint18_checkpoint` -> `validate` -> `_validate_optimizer_state`).
**Root cause:** The diagnostic branch saved next-epoch-ready sampler/scheduler/coefficient state (`completed 10`, `active 11`, LR(11)) while the live AdamW still held LR(10): the trainer pins the LR only at each trained epoch start, but the frozen validator requires the saved optimizer LR to equal `learning_rate_for_epoch(active_epoch)`. The pre-existing tiny proof never hit this because it saved at a position where completed/active LRs already matched.
**Fix / workaround:** Advance every optimizer param group to `learning_rate_for_epoch(active_epoch)` immediately before building the diagnostic/primary save states in `run_seed`. The next trained epoch re-pins the identical value, so the LR/lambda trajectory is byte-identical.
**Watch out for:** Any coherent checkpoint whose sampler `current_epoch` (active) differs from the last trained epoch: the optimizer, scheduler, sampler, and coefficient states must all describe the active position, and the regression must save/strict-restore/resume across every real boundary (10->11, 30->31, 60->61, 100), not just a 0/1 fixture.

## [2026-10-20] Windows `Path.write_text` silently converts LF files to CRLF and inflates diff review

**Symptom:** After scripted multi-line edits, `git diff --stat` reported ~850/1,400 changed lines in files where only ~20/350 lines actually changed.
**Root cause:** `Path.write_text(text, encoding="utf-8")` uses Windows text-mode newline translation, so an LF file is rewritten with CRLF; the `edit` tool preserves the original style.
**Fix / workaround:** After any scripted write, compare CRLF counts with `git show HEAD:<path>` and rewrite with `newline="\n"` (or strip `\r\n`) to match HEAD. Verify every recorded hash with direct `sha256sum <file>` or Python: this shell's `sha256sum -c` hashes CRLF files differently from direct hashing and reports FAILED for a list generated by itself.
**Watch out for:** Large unexpected diff stats after Python-driven edits; line-ending-only churn can hide the real change and pollute a review/checkpoint snapshot.

## [2026-10-20] Accepted CUDA-generated Sprint18 checkpoints/caches are not bit-restorable on a local CPU host

**Symptom:** `restore_sprint18_checkpoint` raised "checkpoint has CUDA RNG state but restore runtime has no CUDA device"; separately, the runner's exact cache-residual recomputation raised a mismatch of ~1e-8 on real A04 records.
**Root cause:** The accepted A04 checkpoints/caches were produced on CUDA; local `torch 2.10.0+cpu` has no CUDA generators and its float32 layer-norm/mean rounding differs by ulps from the cache-producing runtime.
**Fix / workaround:** For CPU-only verification use `restore_rng=False` (model/optimizer/EMA bytes still strict-validate) and, when exact recomputation is a documented environment limit, consume the cache's recorded residual target while recording the observed drift; never loosen the product's fail-closed equality check.
**Watch out for:** Claiming bit-exact cache replay or a normative result from a CPU host; a normative run must use the accepted execution environment.
