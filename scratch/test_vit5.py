import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import unicodedata
from src.model_loader import get_loaded_model
from src.preprocess import clean_generated_summary
from evaluation.output_validator import validate_output

# Load model
loaded = get_loaded_model("vit5")
model = loaded.model
tokenizer = loaded.tokenizer
device = loaded.device

text = "summarize: PHÂN HIỆU TRƯỜNG ĐẠI HỌC GTVT CỘNG HÒA XÃ HỘI CHỦNG HĨA VIỆT NAM BỘ MÔN CÔNG NGHỆ THÔNG TIN Độc lập - Tự do - Hạnh phúc BÁO CÁO TIẾN ĐỘ THỰC HIỆN ĐỒ ÁN TỐT NGHIỆP Họ tên: Nguyễn Hữu Toàn MSSV: 6351071071 Lớp: CQ. 63. CNTT Tên đề tài: Xây dựng hệ thống tóm tắt văn bản tự động sử dụng xử lý ngôn ngữ tự nhiên (NLP) và mô hình Transformer. Người hướng dẫn: Th. S Trần Phong Nhã Thời gian: Tuần từ ngày 01 tháng 04 năm 2026 đến ngày 12 tháng 04 năm 2026 Nội dung thực hiện: 1) Đề tài: Khám phá và phát triển dự án tốt nghiệp Trong tuần nghiên cứu đầu tiên, em đã tiến hành phân tích các xu hướng công nghệ nổi bật trong lĩnh vực Công nghệ Thông tin, đặc biệt là Trí tuệ Nhân tạo (AI) và Xử lý Ngôn ngữ Tự nhiên (NLP). Sau khi tham khảo nhiều nguồn tài liệu và trao đổi với giảng viên hướng dẫn, em đã lựa chọn đề tài: \"Xây dựng hệ thống tóm tắt văn bản tự động sử dụng xử lý ngôn ngữ tự nhiên (NLP) và mô hình Transformer. Lý do lựa chọn đề tài là do tính ứng dụng thực tiễn cao trong lĩnh vực giáo dục, nghiên cứu và xử lý thông tin. Đề tài cũng phù hợp với định hướng phát triển nghề nghiệp trong lĩnh vực trí tuệ nhân tạo."

encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
encoded = {k: v.to(device) for k, v in encoded.items()}

print("Generating raw...")
with torch.no_grad():
    gen_ids = model.generate(**encoded, max_new_tokens=256, num_beams=4, early_stopping=True)

decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
raw = decoded[0] if decoded else ""
normalized = unicodedata.normalize("NFC", raw)
cleaned = clean_generated_summary(normalized)

print("Raw generated:", repr(raw))
print("Normalized:", repr(normalized))
print("Cleaned:", repr(cleaned))

val = validate_output(cleaned, require_vietnamese=True)
print("Validation on cleaned:", val)
