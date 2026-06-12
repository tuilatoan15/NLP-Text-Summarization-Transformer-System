# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEXT = (
    "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình "
    "leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập "
    "tức và mở hành lang nhân đạo cho người dân vùng chiến sự."
)

from summarizers.extractive.extractive_summarizer import EXTRACTIVE_RUNNERS
from summarizers.abstractive.abstractive_summarizer import abstractive_summarize_key
from summarizers.length_manager import SummaryLengthManager
from src.preprocess import clean_text, split_sentences
from src.utils import count_words

print("TEXT words:", count_words(TEXT))
print("cleaned words:", count_words(clean_text(TEXT, aggressive=True)))
print("analysis:", SummaryLengthManager.analyze_input(TEXT))
