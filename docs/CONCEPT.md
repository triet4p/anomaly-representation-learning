# Concept — Representation Learning for Hard Anomaly Detection

## 1. Core Problem

Hướng anomaly detection dựa chủ yếu vào reconstruction thường ngầm giả định:

\[
\text{abnormal} \Rightarrow \text{high reconstruction error}
\]

Nhưng giả định này không đúng một cách tổng quát.

Một số abnormal như:

- stuck / constant;
- pattern quá đều;
- periodic pattern đơn giản;
- smooth drift;
- repeated state;
- một cấu trúc sai nhưng đơn giản;

có thể **dễ reconstruct hơn normal**.

Vì vậy, mục tiêu không nên tiếp tục là:

> Làm normal dễ reconstruct và abnormal khó reconstruct.

Mà nên chuyển sang:

> **Học structure và geometry của normal behavior để abnormal trở nên không phù hợp với representation space của normal.**

Đây là pivot chính của hướng nghiên cứu hiện tại.

---

## 2. Lessons from V6 / Reconstruction-Centric Direction

Các diagnosis từ V6 cho thấy một số điểm quan trọng.

### 2.1 Codebook usage không phải vấn đề chính

Với codebook hiện tại, usage có thể vẫn rất rộng, entropy và perplexity cao. Vì vậy:

> High reconstruction performance hoặc anomaly failure không nhất thiết đến từ codebook collapse.

Chỉ tăng codebook capacity không xử lý được vấn đề nền.

### 2.2 Code repetition không tự động là anomaly

Một code có thể có self-transition rất cao trong normal data.

Do đó:

\[
\text{long repeated code sequence}
\]

không thể được coi là anomaly nếu không so với run-length distribution bình thường của chính state đó.

### 2.3 Priors vẫn hữu ích nhưng chỉ là evidence bổ sung

Có thể dùng:

\[
S_{\mathrm{unigram}}(k)=-\log P(k)
\]

và:

\[
S_{\mathrm{transition}}(k_t,k_{t-1})
=
-\log P(k_t\mid k_{t-1})
\]

Ngoài ra, quantization distance cũng là một evidence độc lập:

> Một latent có thể bị ép vào code gần nhất dù thực tế nằm xa prototype.

Tuy nhiên các score này vẫn hoạt động trên representation cũ. Chúng cải thiện inference nhưng chưa giải quyết tận gốc vấn đề representation.

---

## 3. New Research Framing

Model mới tập trung vào:

\[
X \rightarrow Z
\]

thay vì:

\[
X \rightarrow \hat X
\]

Anomaly detection chủ yếu diễn ra trong latent space.

Ta muốn representation của normal có hai tính chất:

1. **Context consistency**  
   Một local behavior phải phù hợp với context xung quanh trong cùng file/session.

2. **Population consistency**  
   Toàn bộ file/session phải nằm gần manifold hoặc population của normal behavior.

Hai tính chất này tương ứng với hai learning objectives chính:

- Latent Prediction;
- Contrastive Learning.

---

## 4. Semantic Unit: Variable-Length File

Input semantic unit không còn là một fixed window độc lập.

Ta coi một file/session là:

\[
X \in \mathbb{R}^{C\times T}
\]

với \(T\) variable.

File được chia thành patches:

\[
X \rightarrow (p_1,p_2,\ldots,p_N)
\]

Patch chỉ là **computational unit**.

Semantic unit chính vẫn là toàn bộ file.

Encoder sinh:

\[
(z_1,z_2,\ldots,z_N)
\]

Từ đó tồn tại hai mức representation.

### Patch-level representation

Dùng cho:

- latent prediction;
- local context reasoning;
- patch anomaly score;
- anomaly localization.

### File-level representation

Aggregate:

\[
z_F=A(z_1,z_2,\ldots,z_N)
\]

Dùng cho:

- contrastive learning;
- kNN / reference-bank scoring;
- prototype comparison;
- file-level anomaly detection.

---

## 5. Objective A — Latent Prediction

Mindset gần với MAE:

> mask → predict

nhưng không reconstruct raw waveform.

Thay vì dự đoán:

\[
\hat x_M
\]

ta dự đoán latent target:

\[
\hat z_M=P(z_C)
\]

với objective:

\[
\mathcal L_{\mathrm{pred}}
=
d\left(
\hat z_M,
\operatorname{sg}(z_M)
\right)
\]

Trong đó:

- \(z_C\): representation của visible context;
- \(z_M\): target representation của masked region;
- \(P\): predictor;
- \(\operatorname{sg}\): stop-gradient.

Target encoder có thể dùng EMA:

\[
\theta_T
\leftarrow
m\theta_T+(1-m)\theta_C
\]

Ý nghĩa của objective:

> Model phải hiểu masked behavior nên có representation như thế nào khi nhìn context còn lại.

Do đó anomaly có thể bị phát hiện vì **context mismatch**, kể cả khi waveform của nó rất dễ reconstruct.

---

## 6. Objective B — Contrastive Learning

Contrastive objective chịu trách nhiệm tổ chức geometry của latent space.

Từ cùng một file \(X\), tạo hai augmented views:

\[
X^{(1)},X^{(2)}
\]

và khuyến khích:

\[
E(X^{(1)}) \approx E(X^{(2)})
\]

Positive pair ban đầu nên đến từ **hai views của cùng một file**, thay vì dùng metadata như:

- same robot;
- same machine;
- same sensor;
- same source.

Lý do:

> Metadata similarity không đảm bảo behavioral similarity.

Sau này có thể thử metadata như weak supervision nếu cần.

File-level contrastive learning giúp representation trả lời:

> File này có thuộc cùng population behavior với normal training files không?

---

## 7. Joint Training Strategy

Hai task nên được joint train, nhưng không dùng contrastive weight cố định ngay từ đầu.

Objective:

\[
\mathcal L(t)
=
\mathcal L_{\mathrm{pred}}
+
\lambda(t)\mathcal L_{\mathrm{con}}
\]

với:

\[
\lambda(0)=0
\]

sau đó tăng dần tới:

\[
\lambda_{\max}
\]

Baseline đơn giản:

\[
\lambda(t)
=
\lambda_{\max}
\min
\left(
1,\frac{t}{T_{\mathrm{ramp}}}
\right)
\]

Mindset:

> **Prediction shapes the representation; contrastive progressively organizes its geometry.**

Early training:

- latent prediction học local/contextual structure;
- tránh để contrastive áp geometry quá sớm khi representation còn chưa ổn định.

Later training:

- contrastive tăng dần;
- latent space bắt đầu có population structure rõ hơn.

---

## 8. Masking Policy

Random masking thuần chưa chắc đủ tốt.

Masking chính là một phần của learning objective vì nó quyết định model buộc phải học loại context nào.

V1 nên kết hợp ba thành phần:

\[
M
=
M_{\mathrm{random}}
\cup
M_{\mathrm{info}}
\cup
M_{\mathrm{block}}
\]

### 8.1 Random masking

Vai trò:

- coverage rộng;
- tránh bias quá mạnh;
- tạo baseline ổn định.

### 8.2 Information-aware masking

Dựa vào patch-level statistics như:

- variance / standard deviation;
- range;
- slope;
- derivative energy;
- signal energy;
- entropy nếu cần.

Không nên luôn mask patch có information score lớn nhất.

Thay vào đó nên stratify để model gặp đủ:

- stable patches;
- transition patches;
- dynamic patches;
- extreme patches.

### 8.3 Block masking

Mask một vùng contiguous:

```text
visible visible [ MASK MASK MASK MASK ] visible
```

Mục tiêu:

- giảm khả năng nội suy trivial từ hai patch lân cận;
- ép model sử dụng long-range context;
- phù hợp với contextual anomalies kéo dài.

### 8.4 Design rule cho ablation

Khi so các masking policy:

> Giữ total mask ratio cố định, chỉ thay composition.

Learned masking network chưa cần ở V1.

Rule-based masking dễ:

- kiểm soát;
- debug;
- visualize;
- ablate;
- giải thích.

---

## 9. Baseline Architecture

```text
Variable-length File
        ↓
Patchify
        ↓
Local Patch Encoder
        ↓
Sequence Encoder
        ↓
Patch Representations
       / \
      /   \
Latent     File Pooling
Predictor       ↓
      ↓      Projection Head
Prediction       ↓
Loss        Contrastive Loss
```

V1 không cần ngay:

- waveform decoder;
- RVQ;
- hierarchical codebook;
- discrete priors;
- learned masking network.

Mục tiêu đầu tiên là chứng minh:

> Representation learning mới thực sự giải quyết các anomaly mà reconstruction-centric model thất bại.

---

## 10. Anomaly Scoring

Inference nên giữ các evidence tách biệt trước khi fusion.

### 10.1 Context mismatch

Từ latent prediction:

\[
S_{\mathrm{pred}}
=
d(\hat z,z)
\]

Score cao khi local behavior không predictable từ context.

### 10.2 Population mismatch

Từ file/patch embedding so với normal reference bank:

\[
S_{\mathrm{kNN}}
=
\frac{1}{K}
\sum_{j\in\mathcal N_K(z)}
d(z,z_j)
\]

Hoặc dùng:

- prototype distance;
- local density;
- Mahalanobis distance;
- conformal score về sau.

### 10.3 Fusion

Chưa nên fuse ngay từ đầu.

Trước hết phải kiểm tra độc lập:

\[
S_{\mathrm{pred}}
\]

và:

\[
S_{\mathrm{kNN}}
\]

để biết mỗi objective đang thực sự đóng góp gì.

Sau đó mới xét:

\[
S
=
\alpha S_{\mathrm{pred}}
+
\beta S_{\mathrm{pop}}
\]

---

## 11. Synthetic Data Philosophy

Synthetic benchmark không nên tạo anomaly quá hiển nhiên.

Mục tiêu là tạo:

> **Abnormal khó bắt nhưng vẫn có semantic distinction với normal.**

Không nên biến abnormal thành:

- noise cực lớn;
- spike vô lý;
- waveform hoàn toàn out-of-distribution;
- lỗi mà chỉ nhìn amplitude cũng phân biệt được ngay.

Nếu anomaly quá dễ, benchmark không kiểm tra được hypothesis của model.

Ngược lại, abnormal cũng không được chỉ là một phiên bản noisy khác của normal.

### Principle

Normal và abnormal nên có:

- local statistics có thể khá giống nhau;
- amplitude range có thể overlap;
- frequency có thể gần nhau;
- noise level có thể tương đương;

nhưng khác nhau ở:

- temporal rule;
- transition logic;
- cross-channel dependency;
- duration;
- order;
- context;
- regime consistency.

### Các nhóm anomaly ưu tiên

#### A. Easy-to-reconstruct anomalies

- stuck;
- overly smooth segment;
- repeated periodic block;
- low-complexity wrong pattern.

Đây là nhóm trực tiếp kiểm tra failure mode của reconstruction.

#### B. Contextual anomalies

Một patch riêng lẻ có vẻ normal nhưng sai trong context:

- wrong regime transition;
- transition quá sớm / quá muộn;
- normal state xuất hiện sai vị trí;
- valid pattern nhưng duration bất thường.

#### C. Cross-channel anomalies

Từng channel riêng lẻ hợp lệ nhưng relationship giữa channels sai:

- phase relation sai;
- correlation bị phá;
- response channel không follow driving channel;
- dependency lag sai.

#### D. Subtle dynamics anomalies

- frequency shift nhỏ;
- slope drift nhỏ;
- damping thay đổi;
- transition shape thay đổi;
- repeated behavior kéo dài hơn normal.

---

## 12. Evaluation Strategy

Evaluation nên đi theo ba tầng:

\[
\text{Controlled Synthetic}
\rightarrow
\text{Public Benchmark}
\rightarrow
\text{Real Data}
\]

### 12.1 Controlled Synthetic

Cho phép kiểm soát rõ:

- anomaly type;
- anomaly strength;
- anomaly duration;
- contamination;
- difficulty.

Đây là nơi tốt nhất để xác nhận hypothesis.

### 12.2 Baselines bắt buộc

So ít nhất:

1. Reconstruction / MAE;
2. Latent Prediction only;
3. Contrastive only;
4. Joint Prediction + Contrastive.

Mục tiêu không chỉ là đạt F1 cao nhất mà còn xác định:

> Improvement đến từ objective nào và trên loại anomaly nào.

### 12.3 Metrics

Không chỉ đo F1.

Cần đo cả:

- AUROC;
- AUPRC;
- file-level F1;
- patch-level localization;
- prediction-score separation;
- kNN separation;
- nearest-neighbor retrieval;
- embedding variance;
- cosine similarity distribution;
- PCA / UMAP visualization khi cần.

---

## 13. Contaminated Training

Train set thực tế có thể không sạch hoàn toàn.

### Contrastive risk

Nếu một abnormal file lọt vào training:

\[
X_{\mathrm{abn}}^{(1)}
\leftrightarrow
X_{\mathrm{abn}}^{(2)}
\]

contrastive vẫn học invariance cho behavior đó.

### Prediction risk

Nếu abnormal pattern lặp đủ nhiều và đủ consistent, predictor cũng có thể học nó.

Do đó contamination robustness phải là một evaluation dimension.

Synthetic benchmark nên thử:

\[
0\%,1\%,5\%,10\%,20\%
\]

training contamination.

V1 nên log:

- per-file prediction loss;
- per-patch prediction loss;
- embedding norm;
- embedding variance;
- nearest-neighbor density.

Không nên mặc định:

> high-loss sample = anomaly

vì hard normal cũng có thể có loss cao.

---

## 14. Role of VQ / Codebook Later

VQ chưa cần trong baseline mới, nhưng không bị loại bỏ vĩnh viễn.

Hướng cũ:

\[
X
\rightarrow
z_{\mathrm{reconstruction}}
\rightarrow
VQ
\]

Hướng tiềm năng sau này:

\[
X
\rightarrow
z_{\mathrm{normality-aware}}
\rightarrow
VQ
\]

Nếu representation mới đủ tốt, codebook có thể trở lại để cung cấp:

- discrete states;
- state frequency;
- transition priors;
- run-length statistics;
- interpretable prototypes;
- discrete sequence modeling.

Khi đó VQ trở thành **modeling layer trên một representation tốt hơn**, thay vì cố cứu một reconstruction-centric representation.

---

## 15. Implementation Roadmap

### Milestone A — Data Pipeline

- variable-length file loader;
- patchification;
- padding / attention mask;
- patch statistics;
- controlled synthetic generator;
- anomaly injection;
- data visualization notebook.

### Milestone B — Prediction Only

- context encoder;
- EMA target encoder;
- masking policy;
- latent predictor;
- prediction loss.

Goal:

> Prediction học được và representation không collapse.

### Milestone C — Representation Diagnosis

Kiểm tra:

- embedding variance;
- pairwise cosine distribution;
- nearest neighbors;
- retrieval quality;
- normal/abnormal separation trên synthetic test.

### Milestone D — Contrastive

- two-view augmentation;
- file pooling;
- projection head;
- contrastive loss;
- progressive \(\lambda(t)\).

### Milestone E — Anomaly Inference

Đánh giá riêng:

\[
S_{\mathrm{pred}}
\]

và:

\[
S_{\mathrm{kNN}}
\]

sau đó mới thử fusion.

### Milestone F — Ablation

Masking:

- random only;
- random + info;
- random + block;
- random + info + block.

Objectives:

- prediction only;
- contrastive only;
- joint constant weight;
- joint progressive weight.

### Milestone G — Robustness

- anomaly difficulty;
- contamination ratio;
- unseen anomaly types;
- cross-domain / cross-device generalization nếu có.

### Milestone H — Real Data

Chỉ sau khi synthetic benchmark chứng minh được:

1. reconstruction baseline thực sự thất bại trên easy-to-reconstruct anomalies;
2. representation model bắt được failure mode đó;
3. improvement không đến từ trivial dataset artifacts.

---

## 16. Main Research Hypothesis

Hypothesis trung tâm:

> **Anomaly detection không nên phụ thuộc vào việc abnormal có khó reconstruct hay không. Thay vào đó, model nên học contextual structure và population geometry của normal behavior, để abnormal bị phát hiện bởi sự không phù hợp trong latent space.**

Có thể viết gọn thành:

\[
\boxed{
\text{Normality}
=
\text{Context Consistency}
+
\text{Population Consistency}
}
\]

và:

\[
\boxed{
\text{Anomaly}
=
\text{Context Mismatch}
\;\lor\;
\text{Population Mismatch}
}
\]

---

## 17. Final Mindset

Hướng nghiên cứu hiện tại là:

\[
\boxed{
\text{Variable-Length File}
\rightarrow
\text{Patch Representation}
\rightarrow
\text{Latent Prediction}
+
\text{Progressive Contrastive Learning}
}
\]

với masking:

\[
\boxed{
\text{Random}
+
\text{Information-Aware}
+
\text{Block}
}
\]

và anomaly inference dựa trên hai complementary signals:

1. **Context mismatch**  
   Behavior này có phù hợp với context của chính file không?

2. **Population mismatch**  
   Behavior/file này có thuộc normal representation space không?

Điểm cốt lõi không còn là:

> **Can the model reconstruct this signal?**

mà là:

> **Does this behavior belong here?**
