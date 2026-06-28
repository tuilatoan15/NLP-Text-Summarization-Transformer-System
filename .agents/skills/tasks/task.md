# Nhiệm vụ tích hợp 15 thuật toán & Cập nhật Bảng metrics Playground

- [x] Phân tích cấu trúc thuật toán so sánh trong frontend (`Playground.jsx`) và backend (`dashboard_service.py`).
- [x] Cập nhật model registry (`model_registry.py`) để tích hợp 9 giải thuật Hybrid (Extractive -> Abstractive).
- [x] Chỉnh sửa backend (`dashboard_service.py`) để hỗ trợ chạy so sánh và tổng hợp kết quả của nhóm Hybrid trên GPU.
- [x] Tính toán chỉ số bịa đặt `hallucination` một cách thực tế cho các mô hình sinh trong backend.
- [x] Cập nhật frontend (`Playground.jsx`) để mở rộng danh sách 15 thuật toán và hỗ trợ 3 nhóm trong bộ chọn `AlgorithmSelector`.
- [x] Định dạng lại bảng so sánh `ComparisonTable` để hiển thị đầy đủ các cột metrics nâng cao theo yêu cầu của người dùng.
- [x] Chạy kiểm thử build frontend để xác minh tính ổn định của mã nguồn.
- [x] Cập nhật báo cáo kết quả `walkthrough.md`.

# Nhiệm vụ Redesign panel Cấu hình thử nghiệm

- [x] Thiết kế component con `AlgorithmSelector.tsx` chuyên biệt bằng TypeScript:
  - [x] Thêm định nghĩa TypeScript types và Legend chỉ dẫn nhóm trực quan.
  - [x] Thêm nút chọn/hủy chọn nhanh cho TỪNG nhóm thuật toán.
  - [x] Hỗ trợ hiển thị trạng thái card chọn (nền nhạt, viền đậm, check ở góc) và chưa chọn rõ ràng.
  - [x] Thêm tooltip mô tả ngắn gọn cho từng thuật toán trong số 15 mô hình.
- [x] Refactor lại `Playground.jsx` để tích hợp giao diện mới:
  - [x] Chuyển đổi bố cục sang 2 cột cân đối: trái (~35%) chứa Upload area + panel Tóm tắt cấu hình (dạng sticky); phải (~65%) chứa Algorithm Selector.
  - [x] Thay đổi mảng thuật toán mặc định được chọn ban đầu từ 6 thuật toán sang preset 2 extractive + 1 abstractive tối ưu.
  - [x] Thêm 3 nút Preset nhanh: "Nhanh" (chỉ extractive), "Cân bằng" (extractive + vit5), "Đầy đủ" (15 thuật toán).
  - [x] Bổ sung logic ước tính thời gian chạy (Extractive: 0.02s, Hybrid: 2s, Abstractive: 2.6s) hiển thị trong panel Tóm tắt cấu hình.
  - [x] Thêm cảnh báo màu cam nếu tổng thời gian ước tính chạy > 30s.
  - [x] Thêm cảnh báo/tooltip GPU đơn nếu người dùng chọn từ 2 mô hình Transformer trở lên.
  - [x] Cập nhật nút Chạy so sánh (N) động, disable khi không có input, kèm tooltip giải thích lý do cụ thể khi bị disable.
  - [x] Kiểm tra tương thích Dark/Light mode trên toàn bộ các lớp thiết kế Tailwind.
- [x] Kiểm thử build production dự án bằng Vite thành công.
- [ ] Cập nhật báo cáo kết quả `walkthrough.md`.
