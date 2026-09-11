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
