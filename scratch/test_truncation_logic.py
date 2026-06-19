import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(r"c:\Users\ASUS\Desktop\NLP-Text-Summarization-Transformer-System")
sys.path.insert(0, str(project_root))

from backend.services.rag.summarizer import _clean_incomplete_sentence

text = '[Tóm tắt phân cấp L1 - 015_Bon_ly_do_khong_nen_bo_lo_Hoiana_Aquaman_Vietnam_2026.docx]: Bốn lý do không nên bỏ lỡ Hoiana Aquaman Vietnam 2026\n\nLần đầu đến miền Trung, Hoiana Aquaman 2026 mở nhiều cự ly, thu hút VĐV bởi mô hình nghỉ dưỡng thể thao khép kín độc đáo. Aquaman Vietnam là giải hai môn phối hợp bơi - chạy đầu tiên tại Việt Nam, thuộc hệ thống VnExpress Marathon. Sau ba mùa giải tổ chức tại Trà Cổ (Quảng Ninh), Phan Thiết (Bình Thuận) và Hồ Tràm (Bà Rịa - Vũng Tàu), giải năm nay lựa chọn quần thể Hoiana Resort & Golf (Đà Nẵng) làm điểm dừng chân tiếp theo với nhiều nâng cấp trải nghiệm cho VĐV.'

# 1. Simulate _extractive_fallback
sentences = []
for sent in re.split(r"(?<=[.!?])\s+", text):
    s = sent.strip()
    if s and len(s) > 15:
        sentences.append(s)

extracted = " ".join(sentences)
print("=== EXTRACTED ===")
print(repr(extracted))

# 2. Simulate _clean_incomplete_sentence
cleaned = _clean_incomplete_sentence(extracted)
print("\n=== CLEANED ===")
print(repr(cleaned))
