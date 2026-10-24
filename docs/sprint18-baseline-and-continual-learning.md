# Sprint 18 — Baseline hiện tại và khả năng tái sử dụng cho học liên tục

Sprint 18 được kết thúc tại đây với verdict lịch sử **`UNRESOLVED`**. Tài liệu này ghi lại cách nhìn về giá trị sử dụng của kết quả hiện tại, không sửa protocol, verdict hay mở một campaign mới.

**Phương pháp hiện tại có thể tái sử dụng khi có nhãn, và hợp lý để xem là ứng viên baseline cho học liên tục. Nhưng Sprint 18 mới chứng minh khả năng phát hiện bất thường trên benchmark đã kiểm tra, chưa chứng minh khả năng học liên tục.**

Mục tiêu sử dụng được làm rõ như sau:

> Không cần giải quyết hoàn hảo mọi bất thường trong môi trường công nghiệp nhiều nhiễu. Cần học được biểu diễn hữu ích, phản ứng đủ tốt với các dạng đã biết, rồi tận dụng phản hồi có nhãn để cải thiện dần.

Với mục tiêu đó, **“không đạt recovery” không đồng nghĩa “không dùng được làm baseline”**. Gate cũ trả lời liệu có đủ bằng chứng công nhận một cấu hình phục hồi chất lượng theo protocol; nó không trực tiếp trả lời liệu biểu diễn có hữu ích cho bước học có giám sát sau này hay không.

## Khi có nhãn, có thể tận dụng lại những gì?

| Thành phần hiện tại | Cách tái sử dụng khi có nhãn |
|---|---|
| Encoder đã học từ dữ liệu không nhãn | Giữ nguyên encoder, huấn luyện một classifier nhỏ trên embedding để phân biệt khỏe/lỗi hoặc loại lỗi |
| `S_pred`, `S_pop` | Dùng làm tín hiệu đầu vào cho bộ quyết định có giám sát, hoặc hiệu chỉnh ngưỡng theo chi phí bỏ sót/báo giả |
| Checkpoint encoder | Khởi tạo để fine-tune bằng dữ liệu có nhãn, thay vì học từ đầu |
| Biểu diễn theo patch và context | Có thể hỗ trợ gán nhãn/khoanh vùng đoạn lỗi, tùy độ chi tiết của nhãn được thu thập |
| Reference bank | Có thể cập nhật bằng dữ liệu đã được xác nhận khỏe; cần chính sách kiểm soát để không đưa lỗi vào “chuẩn bình thường” |

Các cách này **khả thi về phương pháp**, nhưng hiệu quả thực tế của từng cách trên model hiện tại vẫn cần được đo.

## Lộ trình an toàn nhất

Khi bắt đầu có nhãn, ưu tiên:

**Đóng băng encoder → học một head nhỏ từ nhãn → kiểm tra trên dữ liệu thời gian sau.**

Cách này giúp trả lời câu hỏi:

> Encoder hiện tại đã giữ lại thông tin phân biệt lỗi chưa, chỉ là hai anomaly score chưa khai thác tốt — hay bản thân biểu diễn còn thiếu thông tin?

Nếu một head đơn giản đã cải thiện rõ, ta tận dụng được phần self-supervised hiện tại mà chưa phải sửa encoder. Nếu chưa đủ, mới xét fine-tune một phần hoặc toàn bộ model.

Để gọi là **học liên tục**, còn phải kiểm tra:

- Học lỗi mới có làm quên lỗi cũ không?
- Thích nghi với chương trình/chế độ vận hành mới có làm giảm chất lượng ở chế độ cũ không?
- Dữ liệu “khỏe” mới có thật sự khỏe, hay chứa lỗi chưa được phát hiện?
- Sau mỗi lần cập nhật, báo giả và bỏ sót thay đổi thế nào?

**Cập nhật reference bank liên tục không tự động giải quyết các vấn đề này; cập nhật thiếu kiểm soát còn có thể khiến model coi lỗi lặp lại là bình thường.**

## Vị trí của kết quả hiện tại

**Một baseline self-supervised đã có tín hiệu phân biệt bất thường, có các thành phần tái sử dụng được cho học có nhãn; chưa được xác nhận là baseline học liên tục hiệu quả.**

Cần phân biệt “đã biết”:

- **Các dạng lỗi đã có trong benchmark:** đã có kết quả đánh giá.
- **Các lỗi đã được con người gán nhãn và đưa vào quá trình học:** Sprint 18 chưa kiểm chứng khả năng khai thác các nhãn đó.

Yêu cầu cho bước tiếp theo nên sát mục tiêu sử dụng hơn: **chất lượng ban đầu chấp nhận được, tận dụng được ít nhãn, và cải thiện mà không quên kiến thức cũ** — thay vì yêu cầu baseline ban đầu phải giải quyết hết bài toán công nghiệp nhiều nhiễu. Đây là định hướng, chưa phải gate định lượng hoặc quyền thực thi mới.

Không sửa lại verdict lịch sử của Sprint 18. Giữ `UNRESOLVED` theo protocol cũ, rồi đánh giá riêng câu hỏi mới: **biểu diễn đã học có giúp học từ ít nhãn và cập nhật theo thời gian tốt hơn học từ đầu hay không?** Đây là phép kiểm tra trực tiếp cho giá trị tái sử dụng đang hướng tới.
