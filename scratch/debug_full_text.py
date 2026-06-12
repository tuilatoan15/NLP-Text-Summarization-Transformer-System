# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.admin_cleaner import AdministrativeDocumentCleaner, NGAY_THANG_RE, QUOC_HIEU_RE
from src.preprocess import clean_text, split_sentences
from src.utils import count_words

TEXT = (
    "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình "
    "leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập "
    "tức và mở hành lang nhân đạo cho người dân vùng chiến sự. Đại diện Mỹ phát biểu "
    "rằng Washington ủng hộ giải pháp hai nhà nước nhưng nhấn mạnh quyền tự vệ hợp "
    "pháp. Nga và Trung Quốc phản đối dự thảo nghị quyết, cho rằng văn kiện còn "
    "thiếu cân bằng. Cuộc khủng hoảng nhân đạo ngày càng nghiêm trọng khi hàng nghìn "
    "thường dân phải di tản. Các tổ chức phi chính phủ kêu gọi cộng đồng quốc tế "
    "hành động khẩn cấp để bảo vệ dân thường."
)

def main():
    print("words:", count_words(TEXT))
    print("NGAY_THANG matches:", NGAY_THANG_RE.findall(TEXT))
    admin = AdministrativeDocumentCleaner()
    print("is_admin:", admin.is_admin_document(TEXT))
    ac = admin.clean(TEXT)
    print("after admin clean words:", count_words(ac), "repr:", repr(ac[:100]))
    fc = clean_text(TEXT, aggressive=True)
    print("after full clean words:", count_words(fc))
    print("repr:", repr(fc[:150]))

if __name__ == "__main__":
    main()
