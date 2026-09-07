**Yes. My recommendation is a diagnosis-first reset: test whether the signals are detectable, remove a likely shortcut in the scoring objective, and postpone failure-risk modeling until anomaly evidence works.** Do not start with more epochs, a larger Transformer, or threshold tuning.

Sprint 11 gives us a useful experimental pipeline, but not a validated representation. We should preserve that pipeline and change one scientific assumption at a time.

## 1. First priority: investigate a concrete scoring shortcut

Reading the implementation reveals a stronger concern than “the model needs better tuning.”

In [`v2_patch.py`](src/representation/v2_patch.py:38), the Gaussian head receives the **same latent that its Gaussian density subsequently scores**:

\[
z=f_\theta(x),\qquad
(\mu,\log \sigma^2)=h_\phi(z,c),\qquad
E=\operatorname{NLL}(z;\mu,\sigma^2).
\]

The mean head is linear in \(z\), so it can represent \(\mu(z,c)=z\). The variance head can approach its lower clamp. In that configuration, the residual vanishes and energy becomes almost constant—even when the latent representation varies.

**[INFERENCE] This is a plausible explanation for improving normal loss alongside nearly identical normal/anomaly energies. It is an architectural shortcut, not yet a confirmed diagnosis of the saved checkpoints.** Variance/covariance regularization on \(z\) does not prevent the mean head from copying \(z\).

### Recommended action

Inspect the accepted checkpoints on healthy, abnormal, and paired-corruption inputs:

- Residual magnitude \( \|z-\mu\|^2 \).
- Fraction of predicted log variances at the lower clamp.
- Separate residual and log-variance contributions to energy.
- Whether corruptions change \(z\), but the predicted mean follows that change.

Then compare two controlled alternatives:

| Alternative | What changes | Purpose |
|---|---|---|
| **Frozen-feature conditional density** | Freeze features; fit normal-only density by robot/program/regime without letting each query determine its own reference mean | Simplest diagnostic control |
| **Target-hidden context prediction** | Predict a patch’s latent from context that cannot see that patch; score against a stop-gradient target | Removes direct self-copying while retaining contextual anomaly detection |

For the second option, hide the target **before contextual encoding**; masking after a Transformer has already seen it does not remove the shortcut. Account for overlapping patches too.

Keep independent population and context scores. Neither method alone guarantees both contextual and population anomaly detection.

## 2. Establish whether the synthetic problem is learnable without deep learning

Before another representation redesign, establish a credible baseline on observable telemetry.

Use a small, interpretable patch-feature set:

- Channel level, RMS, variance, and slope.
- Differences and cross-channel relationships.
- Spectral-band energy.
- Transition and duration statistics.

Fit a **regularized conditional distance model** on healthy training files, using the existing hierarchical fallback. Preserve patch localization and aggregate with a fixed upper-tail rule.

Compare:

1. Handcrafted features + conditional geometry.
2. Current frozen learned features + the same geometry.
3. Revised learned features + the same geometry.

This separates competing explanations:

| Outcome | Interpretation |
|---|---|
| Handcrafted features work; learned features fail | Representation/objective is the leading suspect |
| Neither works | Investigate signal visibility, preprocessing, labels, and feature adequacy; not proof that the data is impossible |
| Population distance works; context energy fails | Context-scoring mechanism is the leading suspect |
| Static detection works; early warning fails | Investigate precursor visibility and temporal modeling separately |

Also inspect paired signals **before and after preprocessing**. A physically meaningful amplitude or cross-channel anomaly may be removed by normalization.

**Success criterion:** learned representations must beat a simple baseline—not merely produce finite losses and noncollapsed embeddings.

## 3. Replace generic noise corruption with controlled mechanism-based corruption

The current [`synthesize_corrupted_patches`](src/representation/v2_objectives.py:57) adds Gaussian noise inside selected patches.

That teaches sensitivity to localized noise. It does not necessarily teach sensitivity to gradual drift, altered channel relationships, timing changes, or regime-inappropriate behavior.

### Recommended action

Create a limited development corruption set covering distinct mechanisms:

- Level/gain drift.
- Cross-channel coupling changes.
- Timing or phase shifts.
- Localized transients.
- Slow, persistent degradation.

Preserve physically valid context and retain nuisance-only transformations as negative controls. Otherwise the model may learn “any change is abnormal.”

Reserve **entire mechanisms** for final evaluation. New random seeds or amplitudes from the same mechanism are not sufficient evidence of mechanism generalization.

Before a long run, require a small-set learnability check:

- Can the model consistently rank corrupt patches above paired clean patches?
- Does ordering improve with severity on development examples?
- Does unaffected background remain stable?
- Does it avoid flagging benign nuisance changes?

**If it cannot learn its own training corruptions, stop there.** Diagnose gradients, competing objectives, or the scorer before running another full experiment.

## 4. Fix the evidence scale and development protocol

The nominal 600-file dataset yielded only **44 healthy training files and 11 validation files**. That is a weak basis for conditioned latent geometry and calibration.

For example, an ordinary conformal calibration set of 11 samples has probability resolution approximately:

\[
\frac{1}{11+1}\approx 0.083.
\]

That is too coarse to characterize a 5% tail reliably.

### Recommended action

Generate additional independent factory histories with:

- More verified-healthy exposure per relevant condition.
- Enough separate calibration data for the chosen false-alert target.
- Non-saturated maintenance comparison windows.
- Explicit held-out robots/programs where cold-start is a target.
- Multiple independent seeds and failure episodes.

**Do not weaken precursor quarantine to increase training counts.** Add healthy exposure instead.

Because Sprint 11 outcomes now inform redesign, treat its test data as a **historical diagnostic benchmark**, not an untouched final test. Freeze the next design using development data, then evaluate on newly sealed histories.

Report uncertainty across independent histories/robots/episodes—not by pretending correlated patches are independent samples.

## 5. Rebuild early warning only after checking precursor information

One-day recall was **1/23 → 0/23**. The next question should be:

> Do observable signals contain usable evidence before the failures we expect to predict?

The generator deliberately supports abrupt failures. Those should not silently become failures of an assumed predictable-degradation task.

### Recommended action

Use simulator state **only for diagnostic stratification**, never as model input:

- Progressive failures with observable precursors.
- Weak-precursor failures.
- Deliberately abrupt failures.

Evaluate simple causal trend/persistence scores before a flexible survival head. Compare against:

- Constant-risk baseline.
- Observable operating-history baseline, such as usage and time since maintenance.
- Geometry-trajectory model.

Use chronological out-of-sample evaluation, proper censoring treatment, and false alerts per robot-day alongside recall and lead time.

**Calibration deserves separate work once ranking is useful.** Recalibration can improve probability quality, but it cannot manufacture discriminative information or rescue a chance-level anomaly score.

## Recommended order

| Step | Deliverable | Continue only when |
|---|---|---|
| **1. Scorer diagnosis** | Residual/variance decomposition on accepted checkpoints | We know whether the self-copy shortcut is active |
| **2. Signal baseline** | Handcrafted and frozen-latent conditional-distance comparison | We identify where observable information is lost |
| **3. Bounded representation experiment** | Target-hidden or fixed-reference scorer; controlled corruption study | Development separation is clear and nuisance behavior remains acceptable |
| **4. Independent static evaluation** | New sealed histories, reserved mechanisms, uncertainty estimates | Improvement survives beyond development examples |
| **5. Early-warning evaluation** | Causal trajectories and calibrated risk versus simple baselines | Added value exists at an acceptable false-alert rate |

**My first choice is Steps 1–2, not a new large architecture.** The self-conditioned density head is a concrete, code-supported risk; the simple baseline tells us whether fixing it is enough or whether the data/representation also needs redesign. This gives the next experiment a falsifiable purpose instead of repeating Sprint 11 with different coefficients.
