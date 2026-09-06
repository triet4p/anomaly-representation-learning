# Latent Geometry Methodology

## Purpose and settled direction

This document records the specialized methodology for the next anomaly-representation-learning direction. It is derived from the authoritative methodology source discussion and is intended for technical review and eventual sprint planning. It preserves the distinctions, contracts, formulas, calibration rules, trajectory definitions, evaluation protocols, failure risks, and rationale that constrain future implementation.

The selected direction is a **normal-manifold core plus localized synthetic anomaly-aware boundary learning**. It is neither pure normal-only training nor a conventional binary anomaly classifier. The geometric core is **regularized hierarchical Mahalanobis geometry**, with mixture density for multimodal normal behavior, empirical/conformal anomaly calibration, and a separate longitudinal survival/risk layer.

## 1. Domain context and operational objective

The operating assumptions are:

1. Two normal files are most similar when both have the same `(robot, program)` pair.
2. Two normal files from the same robot but different programs have only weak similarity.
3. Two files for the same program but from different robots are different and must not be treated as strong positives.
4. The operational goal extends beyond detecting obvious faulty files. The system must provide calibrated confidence, capture subtle anomalies and transitional behavior, and track how a specific robot's files move away from healthy latent clusters before a failure within roughly one day or one week.
5. Reliable timestamps and failure events are available.
6. The selected training direction is a normal-manifold core plus localized synthetic anomaly-aware boundary learning—not pure normal-only training and not a conventional binary anomaly classifier.
7. The selected geometric distance is regularized hierarchical Mahalanobis geometry, with mixture density for multimodal normal behavior, empirical/conformal anomaly calibration, and a separate longitudinal survival/risk layer.

## 2. Architecture and data flow

The system has two connected but deliberately separated responsibilities: learn and score conditional normal geometry, then model chronological movement and failure risk.

```text
Full telemetry file x[C,T]
        │
        ├── robot/program/regime/timestamp context
        │
        ├── patchify into variable-length patch sequence
        │
        ▼
Context-conditioned patch encoder fθ
        │
        ├── patch latent z_t,i
        │       │
        │       ├── conditional normal energy E_t,i
        │       │       ├── localized boundary loss on synthetic corruption
        │       │       └── patch-energy sequence
        │       │                 │
        │       │                 └── upper-tail/distributional file aggregation
        │       │
        │       └── file state h_t
        │                     │
        │                     ├── hierarchical Mahalanobis displacement
        │                     ├── velocity, trend, and persistence/CUSUM
        │                     └── chronological trajectory features
        │
        ├── clean/corrupted paired views
        │       ├── background consistency outside anomaly mask
        │       └── local boundary energy margin inside anomaly mask
        │
        ▼
Calibrated anomaly confidence (empirical/conformal)
        │
        ▼
Separate failure-risk layer
(one-day / seven-day discrete-time survival probabilities)
```

The anomaly representation must preserve local and contextual evidence rather than compressing every file to a single undifferentiated mean. File-level decisions are built from patch energies and state summaries, then calibrated against verified healthy data. Anomaly confidence and failure probability are different outputs and must not be conflated.

## 3. Why a normal-manifold core needs a localized boundary objective

Synthetic anomaly-aware training can directly reduce the failure mode in which anomalies lie inside a normal cluster. Normal-only self-supervision cannot guarantee separation because its loss never says that an anomalous transformation must move outward.

A conventional anomaly classifier is intentionally rejected. It risks learning generator shortcuts and recognizing only known anomaly families.

Normal-only training remains valuable because it provides:

- a defensible model of normal behavior;
- potential open-set detection; and
- no dependence on known anomaly classes.

However, normal-only training constrains only normal samples. Every location outside—and even some locations inside—the normal support remains unconstrained. Consequently, abnormal instances may map directly into normal clusters.

Synthetic anomaly-aware training supplies the missing force using paired counterfactuals:

- a clean patch should be close to its context-conditioned normal manifold;
- the corresponding corrupted patch should be farther from that manifold; and
- unmodified patches should remain unchanged between clean and corrupted files.

The generator can produce counterfactual pairs with the same robot, program, regimes, base waveform, and nuisance conditions, differing only in the localized synthetic corruption. This controls shortcuts and nuisance variables.

## 4. Training contract

The encoder never receives anomaly class, anomaly mask, binary label, or severity as an input. The anomaly mask is used only to construct the training loss.

For a clean file $x$ and paired corrupted file $\tilde{x}$:

### 4.1 Normal geometry loss

Train clean patches with:

$$
\mathcal L_{\text{normal}}
:=
\mathcal L_{\text{conditional-density}}
+
\lambda_v \mathcal L_{\text{variance}}
+
\lambda_c \mathcal L_{\text{covariance}}
$$

This learns context-conditioned normal distributions, prevents dimensional collapse, reduces harmful anisotropy, and preserves meaningful variance. The context model should predict a distribution or prototype assignment rather than only a point latent.

### 4.2 Background consistency

For patches outside the injected anomaly:

$$
\mathcal L_{\text{background}}
:=
\frac{1}{|\bar M|}
\sum_{i\notin M}
\left\|
 z_i(x)-z_i(\tilde{x})
\right\|^2
$$

This prevents the complete corrupted file from moving arbitrarily. Normal regions in both versions should retain the same geometry.

### 4.3 Localized boundary loss

Let $E_i$ be context-conditioned normality energy. Require corrupted patches to have higher energy than paired clean patches:

$$
\mathcal L_{\text{boundary}}
:=
\frac{1}{|M|}
\sum_{i\in M}
\max\left(
0,
\delta + E_i(x)-E_i(\tilde{x})
\right)
$$

This relative margin is preferable to an absolute global margin because each corruption is compared with its own clean counterfactual, controlling robot, program, regime, and file identity.

### 4.4 Combined objective and schedule

$$
\mathcal L
:=
\mathcal L_{\text{normal}}
+
\alpha\,\mathcal L_{\text{boundary}}
+\beta\,\mathcal L_{\text{background}}
$$

Ramp $\alpha$ from zero after normal geometry begins stabilizing. The boundary objective must shape the encoder, not merely a head over a frozen representation. Once the encoder discards a feature, a downstream head cannot recover it.

## 5. Treatment of the existing whole-file contrastive objective

The whole-file contrastive objective is a prime suspect because it can encourage invariance to changes that anomaly detection must preserve. For the first redesign experiment, either remove it or restrict it to explicitly normal-preserving transformations and reduce its weight substantially.

Do not add the boundary loss while retaining every existing objective unchanged, because that makes the experiment uninterpretable. The first redesign must isolate whether localized counterfactual geometry supplies the missing separation rather than mixing an additional signal into an already competing objective stack.

## 6. Local population geometry and file aggregation

Averaging all patch embeddings into one file embedding can hide short anomalies. Use patch-level normal geometry:

```text
patch latent
    ↓
condition on regime/context
    ↓
distance or energy against normal patch manifold
    ↓
patch anomaly-energy sequence
    ↓
upper-tail aggregation
    ↓
file score
```

Use a robust upper-tail statistic such as:

$$
S_{\text{file}}
:=
\operatorname{mean}\left(
\operatorname{TopQ}\{E_1,\ldots,E_P\}
\right)
$$

Calibrate $q$ using normal validation files, never test anomalies. This changes which geometric information survives from patch to file level; it is not merely threshold tuning.

The aggregation should preserve, at minimum:

- energy quantiles;
- upper-tail energy;
- fraction of elevated patches;
- regime-conditioned latent summaries;
- transition statistics;
- cross-channel consistency; and
- duration/persistence of elevated regions.

Do not begin with plain mean pooling. An interpretable distributional summary should precede any learned attention/set encoder.

## 7. Synthetic shortcut risk and generalization gates

Synthetic-aware training may learn injector artifacts, mask boundaries, amplitude cues, duration distributions, or severity-specific shortcuts. Therefore, random train/test splitting over the same anomaly families is insufficient.

Evaluate all of the following:

1. Known-family generalization on disjoint files.
2. Complete held-out anomaly families excluded from boundary training.
3. Held-out severity, duration, progression speed, and location ranges.

Success criteria include:

- low energy for clean patches;
- increased energy for paired corruptions;
- consistency of unaffected patches;
- separation of held-out anomaly families;
- stable normal-regime coverage;
- survival of patch separation through file aggregation; and
- useful unsupervised energy independent of any frozen supervised probe.

The central representation contract is: learn the conditional manifold of normal patches, preserve normal counterfactual consistency, and explicitly force only synthetically corrupted regions outside that manifold.

## 8. Hierarchical longitudinal geometry and failure-risk system

Reliable timestamps and failure events change the system from independent anomaly detection into hierarchical latent-state monitoring and failure-risk estimation.

This requires three modifications:

1. Replace the global normal manifold with `(robot, program)`-conditioned geometry.
2. Model chronological movement explicitly rather than scoring files independently.
3. Separate anomaly confidence from predicted failure risk.

### 8.1 Hierarchy of normality

The primary reference distribution is:

$$
p(z \mid \text{robot}=r,\text{program}=p,\text{regime}=g)
$$

It is not a program-global distribution and not a fleet-global distribution.

Use the hierarchy:

```text
Fleet prior
    └── Robot prior
          └── Robot-program manifold
                └── Operating-regime components
```

The hierarchy exists for statistical fallback, not to declare different robots similar.

For a sparse robot-program pair:

$$
\mu_{r,p}
:=
w_{r,p}\hat{\mu}_{r,p}
+
(1-w_{r,p})\mu_r
$$

where $w_{r,p}$ grows with the number and quality of verified healthy baseline files.

Fallback order:

1. `(robot, program, regime)`
2. `(robot, program)`
3. Robot
4. Fleet, with explicitly low confidence

There is no program-only fallback across robots.

### 8.2 Multimodal normality

A robot-program pair can contain legitimate operating regimes, load levels, production phases, start/steady/stop transitions, environmental variation, and sensor states. Do not force these into one spherical cluster.

For patch latent $z_i$ use a conditional mixture or prototype energy:

$$
E_i
:=
-\log
\sum_{k=1}^{K}
\pi_k(c_i)\,
p(z_i\mid\mu_k(c_i),\Sigma_k(c_i))
$$

where $c_i$ includes operating context. Begin with regime-conditioned prototypes and shrinkage covariance before adopting a neural mixture-density network.

Low anisotropy by itself is not success. The target is conditional separability plus stable trajectories.

## 9. Patch and file states

### 9.1 Patch state

Patch latent:

$$
z_{t,i}
:=
f_\theta(x_{t,i},\text{context})
$$

Patch energy:

$$
E_{t,i}
:=
E(z_{t,i}\mid r,p,g_{t,i})
$$

The patch representation must preserve:

- cross-channel inconsistencies;
- transition violations;
- short-duration anomalies;
- local drift; and
- missing or duplicated events.

### 9.2 File state

$$
h_t
:=
A\left(
\{z_{t,i},E_{t,i},g_{t,i}\}_{i=1}^{P_t}
\right)
$$

The aggregation must retain energy quantiles, upper-tail energy, fraction of elevated patches, regime-conditioned latent summaries, transition statistics, cross-channel consistency, and duration/persistence of elevated regions. Do not begin with plain mean pooling. An interpretable distributional summary should precede any learned attention/set encoder.

## 10. Longitudinal movement

Absolute displacement from the fixed healthy baseline:

$$
D_{\mathrm{abs}}(t)
:=
(h_t-\mu_{r,p})^\top
\Sigma_{r,p}^{-1}
(h_t-\mu_{r,p})
$$

Velocity between comparable consecutive files:

$$
D_{\mathrm{vel}}(t)
:=
\left\|
h_t-h_{t-1}
\right\|_{\Sigma^{-1}}
$$

Trend over a recent window:

$$
D_{\mathrm{trend}}(t)
:=
\operatorname{slope}\left(
D_{\mathrm{abs}}(t-w+1),\ldots,D_{\mathrm{abs}}(t)
\right)
$$

One-sided persistence/CUSUM:

$$
C_t
:=
\max\left(
0,
C_{t-1}+D_{\mathrm{abs}}(t)-\kappa
\right)
$$

These distinguish stable health, reversible benign transitions, sudden defects, early persistent degradation, established degraded states, and maintenance/sensor-reset boundaries.

## 11. Fixed and short-term baselines

Never allow an automatically adapting reference bank to absorb slow deterioration.

Maintain both:

- a fixed commissioning baseline, built from verified healthy history, used for long-term absolute drift and reset only after documented maintenance/re-commissioning; and
- a conservative short-term operational baseline, used to suppress harmless variation, never updated from warning/uncertain files, and never allowed to replace the fixed baseline.

Being close to the short-term baseline but far from the fixed baseline is evidence of gradual migration.

## 12. Episodes and required metadata

Segment each robot timeline into episodes:

```text
healthy commissioning
    → operation
    → degradation
    → failure
    → maintenance
    → new commissioning baseline
```

Required per-file data include:

- robot ID;
- program ID;
- timestamp;
- stable chronological sequence identifier;
- maintenance/reset events;
- failure time and category when available;
- intervention side; and
- exposure/file duration.

Do not bridge trajectories across maintenance without an explicit episode boundary.

## 13. Hierarchical synthetic boundary learning

Generate clean/corrupt pairs from the same robot, program, regime sequence, base signal, sensor configuration, and timestamp context.

Strong positives are:

- verified-healthy files from the same pair and nearby stable periods; and
- unmodified patches of clean/corrupt pairs.

Different programs on one robot receive only weak regularization. Same program across robots is not a positive. Files near failure are not temporal positives merely because they are adjacent; pulling all neighboring files together would erase gradual degradation.

## 14. Progressive synthetic degradation

Generate ordered trajectories:

```text
clean
 → barely detectable deviation
 → subtle persistent deviation
 → moderate degradation
 → obvious fault
```

For increasing severity $s_1<s_2<\cdots<s_n$, enforce:

$$
E(\tilde{x}_{s_{j+1}})
\ge
E(\tilde{x}_{s_j})+\delta_j
$$

Apply this locally to affected regions and test unseen severity, progression speed, onset location, and held-out families.

## 15. Anomaly confidence

Anomaly confidence answers how unusual a state is relative to verified healthy behavior for the same pair. Use empirical or conformal tail calibration:

$$
p_{\text{normal}}(t)
:=
\frac{
1+\sum_{j\in\mathcal V_{r,p}}
\mathbf 1[D_j\ge D_t]
}{
|\mathcal V_{r,p}|+1
}
$$

Report anomaly evidence such as $1-p_{\text{normal}}$. This controls false alerts under the healthy reference distribution; it is **not** failure probability.

## 16. Failure-horizon risk

Failure risk asks for:

$$
P(T_{\text{failure}}-t\le 1\text{ day}\mid\mathcal H_t)
$$

and:

$$
P(T_{\text{failure}}-t\le 7\text{ days}\mid\mathcal H_t)
$$

$\mathcal H_t$ contains:

- absolute displacement;
- patch-energy tail statistics;
- velocity;
- trend;
- persistence;
- robot-program context;
- time since maintenance; and
- a recent trajectory window.

Initially keep this risk layer separate from the representation encoder so it is possible to determine whether prediction comes from real latent movement rather than label leakage. Start with a calibrated discrete-time survival model that handles censored files, unequal intervals, multiple horizons, and calibration. Consider an end-to-end temporal Transformer only if the interpretable trajectory model is insufficient.

Anomaly confidence and failure-horizon risk therefore have different semantics:

- anomaly confidence measures unusualness relative to verified healthy behavior for the same robot-program context; and
- failure risk estimates the probability of a future failure within a specified horizon conditional on the history.

## 17. Chronological evaluation protocol

Random file splitting is invalid because adjacent files from the same pair may be near duplicates. Use earlier time for training, later time for calibration/validation, and the latest untouched episodes for testing.

Evaluate separately:

- **Known-robot future prediction:** use earlier history to evaluate later failures; this is the primary operational target.
- **New-robot cold start:** hold complete robots out and measure how confidence improves as healthy commissioning data accumulate.

### Detection metrics

- patch/file AUROC and AUPRC;
- localization overlap;
- top-tail mass;
- false positives per robot-day; and
- event-level recall.

### Early-warning metrics

- median warning lead time;
- failures warned at least one day/week ahead;
- warning persistence;
- false warning episodes per robot-month;
- time-dependent AUROC/AUPRC; and
- concordance index.

### Confidence metrics

- one-day/seven-day Brier score;
- calibration curves;
- expected calibration error;
- conformal healthy coverage;
- calibration by robot/program; and
- confidence degradation under sparse baselines.

The root objective is: for each robot-program operating context, learn a stable healthy manifold, preserve local deviations, and measure whether consecutive file states remain within, suddenly leave, or progressively drift away from that manifold.

## 18. Regularized hierarchical Mahalanobis geometry

Use regularized, condition-specific Mahalanobis distance as the core geometric distance, but not as the entire decision system.

For latent state $z$ under context $(r,p,g)$:

$$
D_M^2(z)
:=
(z-\mu_{r,p,g})^\top
\Sigma_{r,p,g}^{-1}
(z-\mu_{r,p,g})
$$

This is preferable to Euclidean distance because dimensions have different variances and correlations.

Do not use naïve full empirical covariance. The observed representation is anisotropic and has low effective rank, making direct inversion unstable. Estimate covariance only from verified healthy data and regularize it:

$$
\tilde{\Sigma}
:=
(1-\lambda)\hat{\Sigma}
+
\lambda\Sigma_{\text{parent}}
+
\epsilon I
$$

Shrink pair/regime covariance toward the robot-level covariance. Use diagonal or low-rank-plus-diagonal covariance when healthy sample size is insufficient. Freeze healthy estimates during monitoring and exclude uncertain/anomalous files from updates.

For multimodal normal behavior, use mixture density energy:

$$
E(z)
:=
-\log
\sum_k
\pi_k\,
\mathcal N(z;\mu_k,\Sigma_k)
$$

This is better than distance to one centroid.

Use the geometry for:

- patch anomaly energy;
- localized synthetic boundary loss;
- file-state displacement; and
- trajectory velocity.

Mahalanobis distance alone does not yield calibrated anomaly probability, persistence/trend detection, failure-horizon probability, or guaranteed handling of non-Gaussian structure.

Convert distance to anomaly confidence using healthy validation data and empirical/conformal calibration, not a theoretical $\chi^2$ threshold. Feed distance, velocity, trend, persistence, and local-energy summaries into the separate failure-risk model.

## 19. Settled architectural decision

The precise architectural decision is:

> Use regularized hierarchical Mahalanobis geometry as the core deviation measure, mixture density for multimodal normal behavior, conformal calibration for anomaly confidence, and a longitudinal survival model for failure risk. Train the encoder with a normal-manifold objective plus localized synthetic counterfactual boundary and background-consistency losses; do not replace this with a conventional binary anomaly classifier.

This decision preserves conditional normality, localizes the synthetic separation force, protects unaffected regions, and keeps anomaly confidence distinct from failure probability. It also constrains future work to use chronological, robot-program-aware evaluation rather than random file splits or fleet-global positives.

## 20. Chronological factory data-generation methodology

### 20.1 User requirements and settled choices

The data generator must replace independently randomized file timestamps with a causal factory event simulator. The simulator must:

- span a global calendar of 3–6 months;
- represent a physical car/unit entering the production line and traversing a shared defined route through multiple robots;
- ensure every robot processes at most one program/operation at a time, while consecutive operations can touch or have a short idle gap;
- allow asynchronous robot timelines, so different robots can operate at overlapping global times;
- use a primary robot-wide calendar/usage health process with program-dependent sensitivity, rather than independent health per `(robot, program)` pair;
- generate isolated anomalies and temporal degradation episodes so the benchmark measures ordinary file detection and early warning one day/week before failure;
- retain reliable timestamps, maintenance events, and failure events;
- use persisted training data that is verified healthy before an initial chronological cutoff of two months, split chronologically 80/20 into train/validation;
- include in static test all normal files after the cutoff plus all abnormal files across the whole calendar, including pre-cutoff abnormalities;
- quarantine pre-failure precursor files from healthy train/validation even when they lack an obvious binary anomaly label;
- create synthetic corrupted counterfactuals in memory from healthy training files for localized boundary loss, without turning them into historical training records or healthy reference-bank rows;
- maintain separate static-detection and chronological early-warning evaluation views; and
- distinguish physical unit/car identity from program/recipe identity.

### 20.2 Causal factory event simulator

The generator is a factory event simulator, not a collection of independent files with randomly sampled timestamps.

Each file represents one scheduled operation event and must carry, at minimum:

```text
OperationEvent
- operation_id
- unit_id
- product_type
- route_id
- route_position
- robot_id
- program_id
- start_time
- end_time
- duration
- arrival_time
- queue_delay
- idle_before
- operating regime
- latent robot health state
- program-specific health manifestation
- anomaly provenance
- degradation/failure episode
- maintenance boundary
```

The generation order is:

```text
Factory calendar
    ↓
Unit/car arrivals and assigned production programs
    ↓
Shared unit routes, robot queues, operation intervals, and idle gaps
    ↓
Robot health trajectories, degradation episodes, failures, and maintenance
    ↓
File-level normal signals conditioned on the scheduled operation
    ↓
Localized, intermittent, progressive, or acute anomaly manifestation
    ↓
Chronological labels, quarantine assignment, and split indices
```

Timestamps are outputs of scheduling. They must never be sampled independently after file generation.

### 20.3 Factory scheduling model

For physical unit $u$ visiting route stage $k$ on robot $r_k$, define:

- $a_{u,k}$: arrival time at the robot;
- $v_{r_k}$: time when the robot becomes available;
- $d_{u,k}$: operation duration; and
- $\tau_{u,k}$: travel time to the next robot.

Operation start and end are:

$$
s_{u,k}
:=
\max(a_{u,k},v_{r_k})
$$

$$
e_{u,k}
:=
s_{u,k}+d_{u,k}
$$

After scheduling the operation:

$$
v_{r_k}
:=
e_{u,k}
$$

The unit reaches its next route stage at:

$$
a_{u,k+1}
:=
e_{u,k}+\tau_{u,k}
$$

This construction guarantees no overlap on one robot, natural queues, natural idle intervals, asynchronous robot operation, causal movement of each car through the line, and globally sortable events with locally ordered robot streams.

Idle time before event $(u,k)$ is:

$$
I_{u,k}
:=
s_{u,k}-v_{r_k}^{\text{previous}}
$$

The idle interval can be zero for consecutive operations or several seconds/minutes when the robot is waiting. Same-robot event intervals must never overlap. Different robots may process different units simultaneously.
First-stage factory arrivals are periodic by default; profiles may add a deterministic seed-drawn uniform jitter (clamped at zero) so sparse server-scale calendars keep asynchronous cross-robot utilization instead of a perfectly periodic grid in which robots never work simultaneously.

### 20.4 Identity semantics

Do not overload `program_id` to mean both a physical car and an operation recipe.

- `unit_id`: unique physical car/unit moving through the route.
- `product_type`: car model or production family.
- `program_id`: repeatable robot operation recipe.
- `robot_id`: physical robot.
- `route_id`: defined production route.
- `route_position`: operation position in the route.
- `operation_id`: unique robot-unit interaction.

This supports repeating one program across many units, assigning different robot-specific programs to one car, tracing one physical car across robots, maintaining robot-program-conditioned normal references, and separating unit defects from robot degradation.

### 20.5 Robot-wide health with program sensitivity

Let $H_r(t)$ represent latent physical health of robot $r$ at calendar time $t$. It evolves through calendar aging and operation-dependent wear:

$$
H_r(t_{n+1})
:=
H_r(t_n)
+
\Delta_{\text{calendar}}
+
\Delta_{\text{wear}}(p_n,d_n,\ell_n)
+
\epsilon_n
$$

where $p_n$ is the processed program, $d_n$ is operation duration, $\ell_n$ is operating load, and $\epsilon_n$ is small stochastic variation.

Program-specific observable degradation is:

$$
G_{r,p}(t)
:=
\psi_{r,p}\!\left(H_r(t)\right)
$$

A simpler initial form is:

$$
G_{r,p}(t)
:=
\gamma_{r,p}H_r(t)
$$

where $\gamma_{r,p}$ is program sensitivity.

Consequences:

- A robot has one underlying physical health process.
- Programs do not have independent physical health.
- Different programs reveal the same robot degradation with different strength.
- A sensitive program may expose an early warning first.
- A less sensitive program may remain close to its healthy distribution longer.
- Evidence from all programs can contribute to robot-level risk.
- Observed normal geometry remains `(robot, program)`-conditioned because their distributions differ.

The monitoring hierarchy is:

```text
Robot-program latent state
    ↓
Program-specific deviation evidence
    ↓
Robot-level evidence aggregation
    ↓
Robot failure risk
```

### 20.6 Robot, unit, and interaction effects

A generated signal conceptually depends on:

$$
x_{u,r,p,t}
:=
F\left(
B_{r,p},
U_u,
H_r(t),
\psi_{r,p},
C_t,
\eta
\right)
$$

where:

- $B_{r,p}$ is healthy robot-program behavior;
- $U_u$ is physical unit/car variation;
- $H_r(t)$ is robot health;
- $\psi_{r,p}$ is program-specific sensitivity;
- $C_t$ is operating context; and
- $\eta$ is sensor/process noise.

An abnormal file can arise from robot degradation, a defective incoming unit, a robot-program interaction, a transient sensor artifact, an isolated synthetic anomaly, or legitimate unit-to-unit variation. These causes must remain independently represented in provenance. Otherwise, a defective unit could make every robot on its route appear to be degrading.

### 20.7 Temporal anomaly and degradation policies

Anomaly injection is not IID per file. Support at least the following modes.

#### Isolated local anomaly

An isolated local anomaly affects one operation or a short sequence without changing persistent robot health. This supports ordinary file-level detection and localization.

#### Robot degradation episode

A robot degradation episode changes $H_r(t)$ gradually and affects subsequent operations until failure or maintenance.

#### Program-sensitive precursor

One globally degrading robot can manifest different evidence under different programs:

$$
E_{r,p_1}(t)
>
E_{r,p_2}(t)
\quad\text{when}\quad
\gamma_{r,p_1}>\gamma_{r,p_2}
$$

#### Intermittent degradation

Early symptoms appear probabilistically:

$$
P(A_t=1\mid H_r(t))
:=
\sigma\!\left(
a_{r,p}H_r(t)+b_{r,p}
\right)
$$

As health worsens, manifestations become more frequent, persistent, and strong.

#### Progressive degradation

An episode follows a causal progression:

```text
healthy
 → weak latent drift
 → intermittent subtle symptoms
 → persistent symptoms
 → obvious abnormal files
 → failure
 → maintenance
 → recommissioned healthy state
```

#### Abrupt failure

Some failures have little or no observable precursor. This prevents the benchmark from assuming that all failures are predictable and supplies an honest ceiling for one-day/week warning performance.

#### Maintenance and reset

Maintenance is an explicit episode boundary:

$$
H_r(t_{\text{after maintenance}})
\sim
P_{\text{recommissioned}}
$$

A recommissioned state can be healthy without exactly matching the original commissioning state. Pre- and post-maintenance trajectories must not be connected without the boundary.

### 20.8 Failure generation

Failure should arise from evolving robot health, not an independently randomized timestamp.

A hazard policy can use:

$$
P(\text{failure in }[t,t+\Delta t)\mid H_r(t),p_t)
:=
1-\exp\!\left(
-\lambda_r(t)\Delta t
\right)
$$

with:

$$
\lambda_r(t)
:=
\lambda_{0,r}
\exp\!\left(
\alpha H_r(t)
+
\beta W_r(t)
\right)
$$

where $W_r(t)$ can include accumulated workload, recent overload, or persistent degradation. The generator may deliberately control counts and placement of failure episodes, but observations, robot health, and failure times must remain causally coherent.

### 20.9 Label contract

Keep separate labels for distinct scientific questions.

#### Observable file anomaly labels

```text
is_file_anomalous
anomaly_family
anomaly_mask
anomaly_severity
```

These support static file detection and localization.

#### Latent simulation state

```text
robot_health_state
degradation_episode_id
degradation_stage
degradation_severity
```

These are simulation ground truth used for diagnostics and must never be model inputs.

#### Future-event targets

```text
time_to_next_failure
failure_within_1d
failure_within_7d
is_censored
```

These support survival/risk supervision and evaluation only.

A critical early-warning example is:

```text
is_file_anomalous = false
failure_within_7d = true
```

This is not an obvious faulty file but belongs to a trajectory approaching failure. Do not relabel every pre-failure file as a conventional binary anomaly; anomaly detection and failure prediction must remain distinct targets.

Future-failure fields, latent health state, and maintenance outcomes must not leak into encoder inputs or anomaly scoring features unless explicitly represented as allowed operational history.

### 20.10 Chronological split policy

Let $t_c$ be the development cutoff, initially two months after the simulation starts. Let the total simulated calendar span 3–6 months.

The user-requested base split is:

$$
\mathcal D_{\text{dev-base}}
:=
\{x_t:y_t=\text{normal},\,t<t_c\}
$$

$$
\mathcal D_{\text{train}}
:=
\text{earliest }80\%\text{ of }\mathcal D_{\text{dev}}
$$

$$
\mathcal D_{\text{val}}
:=
\text{latest }20\%\text{ of }\mathcal D_{\text{dev}}
$$

$$
\mathcal D_{\text{test-static}}
:=
\{x_t:y_t=\text{normal},\,t\ge t_c\}
\cup
\{x_t:y_t=\text{abnormal},\,\forall t\}
$$

The 80/20 split must be chronological, not random. Random splitting can put adjacent, nearly duplicate files from one robot-program stream into train and validation, producing temporal leakage.

### 20.11 Precursor quarantine

A file can lack an obvious injected anomaly while still occurring during degradation before failure. Such files must not be accepted as verified healthy development data.

Choose quarantine horizon $H_q$ at least as long as the largest warning horizon:

$$
H_q\ge 7\text{ days}
$$

Define:

$$
\mathcal Q
:=
\left\{
x_t:
0<T_{\text{failure}}-t\le H_q
\right\}
$$

Then verified healthy development data is:

$$
\mathcal D_{\text{dev}}
:=
\left\{
x_t:
t<t_c,
y_t=\text{normal},
T_{\text{failure}}-t>H_q,
\text{not post-failure},
\text{not maintenance-transition}
\right\}
$$

Without quarantine, the normal manifold would learn to absorb the transitional states the system is expected to detect later. Quarantined pre-failure files belong to chronological early-warning evaluation, including subtle files with no conventional binary anomaly label.

### 20.12 Static and chronological test views

Maintain two explicit indices over the generated calendar.

#### Static file-detection view

$$
\mathcal D_{\text{test-static}}
:=
\{x_t:y_t=\text{normal},\,t\ge t_c\}
\cup
\{x_t:y_t=\text{abnormal},\,\forall t\}
$$

This supports file/patch AUROC and AUPRC, F1 using validation-calibrated thresholds, localization, and family-level detection.

#### Chronological early-warning view

$$
\mathcal D_{\text{test-temporal}}
:=
\left\{
x_t:t\ge t_c
\right\}
\cup
\left\{
\text{held-out pre-cutoff failure episodes}
\right\}
$$

It preserves complete eligible robot timelines, timestamps, operation order, and episode boundaries. It supports warning lead time, failure recall at one day/week, false alerts per robot-day, warning persistence, time-dependent discrimination, and calibrated horizon risk.

The two indices can reference overlapping files, but they answer different questions and must never be conflated. The temporal view must never be shuffled.

### 20.13 Compatibility with localized synthetic boundary training

The persisted historical training split contains verified healthy operation events only. During training:

1. Select a verified healthy file.
2. Generate an in-memory corrupted counterfactual.
3. Keep unit, robot, program, regime sequence, base signal, sensor configuration, timestamp context, and nuisance variables fixed.
4. Use the synthetic anomaly mask only for localized boundary and background-consistency losses.
5. Discard the corrupted counterpart after the update.
6. Never add the corrupted counterpart to the empirical healthy reference bank.
7. Never treat it as an independently sampled historical abnormal event.

This preserves a normal-only historical split while providing an explicit geometric force against anomalies inside the normal manifold.

### 20.14 Determinism and generator provenance

A fixed seed must deterministically reproduce:

- unit arrival calendar;
- unit routes;
- program assignments;
- robot queues and operation intervals;
- idle and travel times;
- robot health paths;
- failure and maintenance episodes;
- program-specific anomaly manifestation;
- file signals and local anomaly masks; and
- chronological split indices.

Every generated file must retain sufficient provenance to reconstruct why it exists at that timestamp and which latent processes contributed to its signal. Generator randomness should be partitioned into stable independent streams so changing one concern, such as sensor noise, does not unnecessarily reorder the factory calendar or move failure episodes.

### 20.15 Required invariants

The documented simulator must enforce all of the following:

- No overlapping operations on one robot.
- Stable unit identity across its route.
- Causal route ordering.
- Nonnegative queue, travel, and idle durations.
- Every timestamp lies within the configured 3–6 month calendar.
- One shared calendar/usage health state per robot.
- Program-dependent manifestation of robot health.
- Unit effects are distinct from robot health and robot-program interactions.
- Failure times are causally coupled to health trajectories.
- Explicit maintenance and recommissioning boundaries.
- Future-failure labels and latent health are never leaked into encoder inputs.
- No precursor-quarantine files enter healthy train/validation.
- The 80/20 development split is chronological.
- All post-cutoff normal files enter static test.
- All abnormal files across the entire calendar enter static test, including pre-cutoff abnormal files.
- Static and temporal evaluation views remain separate.
- No healthy reference-bank row comes from an in-memory synthetic corruption.
- Schedules, health paths, anomaly episodes, labels, and split indices are deterministic under a fixed seed.

## 21. Chronological generation architectural decision

Replace random independent timestamps with a deterministic, causal 3–6 month shared-unit factory scheduler; model one robot-wide health process with program-specific sensitivity; generate causal progressive/acute anomaly and failure episodes; quarantine pre-failure states from healthy chronological train/validation; and preserve separate static-detection and chronological early-warning test views.

Alternatives considered were independently timestamped IID files, independent robot streams without shared unit routing, independent `(robot, program)` health states, random 80/20 splits, and one combined anomaly/failure label.

The reasons are the real production-line constraints, prevention of temporal and precursor leakage, and the need to measure latent drift and calibrated one-day/week warning.

Consequences include new unit/route/operation/episode metadata, deterministic scheduling and health state, maintenance segmentation, program-sensitive manifestations, dual evaluation indices, and no implementation until sprint planning.
