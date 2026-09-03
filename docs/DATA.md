# Data Requirements

## 1. Mục tiêu

Bộ dữ liệu phải phục vụ hướng anomaly detection dựa trên **representation learning**, trong đó abnormal không được định nghĩa đơn giản là tín hiệu có reconstruction error lớn.

Mục tiêu chính:

- học structure của normal behavior;
- hỗ trợ latent prediction ở patch-level;
- hỗ trợ contrastive learning ở file-level;
- đánh giá anomaly theo context, temporal structure và cross-channel relation;
- đặc biệt kiểm tra được các anomaly **dễ reconstruct nhưng vẫn bất thường**.

Nguyên tắc cốt lõi:

> **Hard anomaly != ambiguous anomaly**

Abnormal phải khó phát hiện, nhưng vẫn phải có nguyên nhân rõ ràng khiến nó vi phạm normal generative process.

---

## 2. Semantic Unit

Đơn vị semantic chính là **một file/session hoàn chỉnh**, không phải một fixed window.

Mỗi sample có dạng:

\[
X \in \mathbb{R}^{C \times T}
\]

trong đó:

- \(C\): số channel/sensor;
- \(T\): variable-length.

File được chia thành patch để xử lý:

\[
X \rightarrow (p_1, p_2, \ldots, p_N)
\]

Patch chỉ là **computational unit**. File/session vẫn là semantic unit chính.

---

## 3. Yêu cầu đối với Normal Data

Normal data phải có đủ diversity để model không học shortcut quá đơn giản.

### 3.1 Regime diversity

Một file normal nên có thể chứa nhiều regime hợp lệ, ví dụ:

- idle;
- periodic;
- active;
- recovery;
- transition giữa các regime.

Các sequence hợp lệ có thể như:

```text
idle -> active -> recovery
idle -> periodic -> active -> recovery
periodic -> active -> recovery
```

### 3.2 Natural variation

Normal phải có variation thực tế về:

- amplitude;
- offset;
- frequency;
- phase;
- trend;
- regime duration;
- transition duration;
- channel gain;
- channel offset;
- sensor noise;
- low-frequency drift nhỏ.

Normal noise phải là noise tự nhiên của sensor, không được trùng với cách sinh anomaly.

### 3.3 Cross-channel structure

Các channel không nên độc lập hoàn toàn.

Normal cần có các relation như:

- correlation;
- phase relation;
- lag nhỏ;
- shared trend;
- shared regime transitions.

Điều này cho phép tạo anomaly mà từng channel riêng lẻ vẫn plausible nhưng relation giữa các channel bị sai.

---

## 4. Yêu cầu đối với Abnormal Data

### 4.1 Không tạo abnormal bằng shortcut

Không nên coi các trường hợp sau là benchmark chính:

- amplitude cực lớn;
- noise cực mạnh;
- NaN;
- constant tuyệt đối;
- spike quá rõ;
- giá trị ngoài physical range;
- distribution khác hoàn toàn normal.

Các loại này chỉ nên giữ làm **easy sanity anomalies**.

### 4.2 Hard anomaly phải local-plausible

Một anomaly tốt nên thỏa gần đúng:

\[
P(x_{\text{patch}} \mid \text{normal})
\approx
P(x_{\text{patch}} \mid \text{abnormal})
\]

nhưng:

\[
P(x_t \mid \text{context}, \text{normal})
\neq
P(x_t \mid \text{context}, \text{abnormal})
\]

Tức là anomaly region có thể tự nó nhìn hoàn toàn normal, nhưng sai khi đặt trong context.

### 4.3 Các anomaly family chính

#### Contextual Replacement

Thay một vùng bằng một patch normal khác nhưng không phù hợp với context hiện tại.

Yêu cầu:

- donor phải cùng hoặc gần regime;
- match local mean/std;
- boundary phải smooth;
- tránh tạo noise/texture khác biệt rõ.

#### Wrong Transition

Transition giữa hai regime vẫn plausible nhưng timing sai.

Ví dụ:

- quá nhanh;
- quá chậm;
- xảy ra quá sớm;
- xảy ra quá muộn;
- sai thứ tự regime.

Không nên tạo plateau hoặc step quá rõ.

#### Realistic Stuck

Không dùng constant tuyệt đối.

Hard version nên chỉ làm giảm dynamics:

\[
\sigma_{\text{abnormal}}
\approx
0.3 - 0.6
\cdot
\sigma_{\text{expected}}
\]

và vẫn giữ:

- sensor noise nhỏ;
- slow drift;
- residual dynamics.

#### Over-Regularity

Signal vẫn đúng range và frequency regime nhưng trở nên quá đều.

Ví dụ:

- phase jitter biến mất;
- cycle-to-cycle variation giảm mạnh;
- waveform quá sạch.

Đây là anomaly quan trọng vì có thể rất dễ reconstruct.

#### Subtle Drift

Từng thời điểm vẫn nằm trong normal range nhưng trajectory tổng thể sai.

Drift phải đủ nhỏ để không bị detect chỉ bằng amplitude threshold.

#### Frequency / Phase Mismatch

Frequency hoặc phase sai nhẹ so với normal expectation.

Hard mode nên:

- chỉ thay một phần channel;
- severity vừa phải;
- vẫn giữ amplitude và local statistics gần normal.

#### Cross-Channel Inconsistency

Từng channel riêng lẻ vẫn normal nhưng relation giữa chúng bị phá.

Ví dụ:

- lag một channel;
- phase shift;
- gain relation sai;
- channel lấy trajectory plausible từ thời điểm khác.

Chỉ inject vào vùng có đủ dynamics và correlation.

Không inject vào đoạn idle/flat vì khi đó intervention có thể gần như không tạo khác biệt thực sự.

#### Duration Anomaly

Một regime normal kéo dài quá lâu hoặc quá ngắn.

Không dùng exact patch tiling.

Nên thực sự time-stretch hoặc extend một valid regime.

#### Missing Event

Một event/regime đáng lẽ xuất hiện nhưng bị thiếu.

Không thay bằng flat interpolation.

Nên thay bằng một **valid behavior khác** nhưng contextually wrong.

---

## 5. Difficulty Control

Mỗi anomaly generator phải có severity parameter.

Ví dụ:

```text
easy
medium
hard
```

hoặc continuous severity:

\[
s \in [0,1]
\]

Generator phải tránh hai extreme:

### Intervention quá yếu

Abnormal gần như không khác normal.

Đây là **ambiguous anomaly**, không phù hợp benchmark.

### Intervention quá mạnh

Abnormal bị tách dễ chỉ bằng local statistics.

Đây là shortcut anomaly.

Nên có cơ chế:

```text
generate
   ↓
measure intervention strength
   ↓
too weak?   -> reject
too strong? -> reject
   ↓
accept
```

---

## 6. Patch Requirements

Patching phải hỗ trợ variable-length file.

Input:

\[
X \in \mathbb{R}^{C \times T}
\]

Output:

\[
P \in \mathbb{R}^{N \times C \times W}
\]

Cần hỗ trợ:

- configurable patch size;
- configurable stride;
- overlapping patch;
- padding ở cuối nếu cần;
- lưu start index của từng patch;
- map anomaly mask từ timestep sang patch-level.

---

## 7. Masking Requirements

Masking policy cho latent prediction phải gồm ba thành phần:

\[
M =
M_{\text{random}}
\cup
M_{\text{info}}
\cup
M_{\text{block}}
\]

### Random masking

Đảm bảo coverage rộng và tránh bias.

### Information-aware masking

Dùng các statistics tính trực tiếp từ patch:

- standard deviation;
- range;
- slope;
- energy;
- derivative energy.

Không được chỉ mask patch có score cao nhất.

Nên stratify để giữ đủ:

- stable;
- transition;
- dynamic;
- extreme patches.

### Block masking

Mask các patch contiguous để buộc model dùng long-range context.

### Ablation requirement

Khi so các masking strategy:

> giữ **total mask ratio cố định**, chỉ thay composition.

---

## 8. Contrastive Views

Hai positive views phải xuất phát từ cùng một file:

\[
X \rightarrow X^{(1)}, X^{(2)}
\]

Augmentation phải bảo toàn behavioral semantics.

Có thể dùng:

- small gain perturbation;
- small offset perturbation;
- small sensor noise;
- very small temporal shift.

Không nên dùng augmentation làm thay đổi regime hoặc anomaly semantics.

---

## 9. Label và Metadata

Mỗi sample evaluation nên lưu:

```text
x
file_label
anomaly_mask
anomaly_type
anomaly_start
anomaly_end
severity
regime_sequence
regime_boundaries
seed
```

Nếu có multi-channel anomaly:

```text
affected_channels
lag / phase_shift / gain_change
```

Training normal data có thể không cần anomaly label.

---

## 10. Data Split

Ít nhất cần:

```text
train/
validation/
test/
```

### Train

Baseline đầu tiên nên ưu tiên normal-only hoặc contamination rất thấp.

### Validation

Dùng để:

- chọn threshold;
- kiểm tra representation;
- tune anomaly score fusion;
- tune masking policy.

### Test

Phải giữ anomaly generator seeds độc lập với train/validation.

Nếu có thể, giữ một số parameter ranges hoặc anomaly combinations chưa từng thấy trong train để đánh giá generalization.

---

## 11. Contamination Benchmark

Phải hỗ trợ chủ động contaminate training set:

\[
0\%, 1\%, 5\%, 10\%, 20\%
\]

Mục tiêu là đo degradation của:

- latent prediction;
- contrastive representation;
- kNN/prototype scoring;
- final anomaly detection performance.

---

## 12. Data Quality Diagnostics

Trước khi train model, phải kiểm tra ít nhất bốn nhóm.

### Normal visual check

Kiểm tra:

- regime diversity;
- transition quality;
- variable length;
- channel relation;
- natural noise.

### Anomaly visual check

Anomaly không được tạo artifact rõ ràng ngoài intended violation.

### Patch statistics

Theo dõi:

- std;
- range;
- energy;
- derivative energy;
- slope.

### Normal vs Abnormal overlap

Train một weak baseline chỉ dùng simple statistics.

Ví dụ:

```text
std
range
energy
derivative_energy
mean_abs_diff
```

Nếu weak baseline đạt gần perfect AUC thì anomaly generator có khả năng đang quá dễ.

Không yêu cầu tất cả anomaly family đều có AUC = 0.5.

Mục tiêu là tạo **difficulty spectrum**, nhưng hard anomaly phải không phụ thuộc vào obvious local shortcut.

---

## 13. Minimum Acceptance Criteria

Data pipeline được coi là đủ tốt để bắt đầu model khi:

- normal files có variable length;
- có nhiều valid regimes;
- transition normal smooth và đa dạng;
- multi-channel relation có thật;
- hard anomaly không chủ yếu là amplitude/noise shortcut;
- contextual replacement nhìn local vẫn plausible;
- wrong transition không còn plateau/step giả;
- realistic stuck vẫn có residual dynamics;
- cross-channel anomaly chỉ inject vào dynamic region;
- duration anomaly không dùng exact repetition;
- missing event được thay bằng valid but wrong behavior;
- anomaly mask chính xác;
- patch mapping chính xác;
- simple-statistics baseline không trivially solve toàn bộ benchmark;
- generator reproducible bằng seed.

---

## 14. Design Principle Cuối

Bộ dữ liệu phải buộc model học:

\[
\boxed{
\text{Normality}
=
\text{Local Pattern}
+
\text{Temporal Context}
+
\text{Cross-Channel Relation}
}
\]

thay vì:

\[
\boxed{
\text{Anomaly}
=
\text{Large Noise / Large Error / Extreme Value}
}
\]

Benchmark tốt nhất cho hướng nghiên cứu này là nơi anomaly **vẫn plausible ở local level**, nhưng vi phạm structure của normal behavior.
