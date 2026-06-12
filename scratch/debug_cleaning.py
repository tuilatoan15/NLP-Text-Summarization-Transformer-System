# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.admin_cleaner import AdministrativeDocumentCleaner
from src.preprocess import clean_text, split_sentences
from src.utils import count_words

TEXT = (
    "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình "
    "leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập "
    "tức và mở hành lang nhân đạo cho người dân vùng chiến sự."
)

def main():
    print("TEXT chars:", len(TEXT), "words:", count_words(TEXT))
    admin = AdministrativeDocumentCleaner()
    print("is_admin_document:", admin.is_admin_document(TEXT))
    admin_cleaned = admin.clean(TEXT)
    print("after admin.clean words:", count_words(admin_cleaned))
    print("admin_cleaned repr:", repr(admin_cleaned[:200]))
    full = clean_text(TEXT, aggressive=True)
    print("after clean_text aggressive words:", count_words(full))
    print("full repr:", repr(full[:200]))
    print("sentences:", split_sentences(full))

if __name__ == "__main__":
    main()
