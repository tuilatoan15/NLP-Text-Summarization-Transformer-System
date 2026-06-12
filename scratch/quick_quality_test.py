# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from summarizers.extractive.extractive_summarizer import EXTRACTIVE_RUNNERS
from summarizers.abstractive.abstractive_summarizer import abstractive_summarize_key
from summarizers.length_manager import SummaryLengthManager
from src.preprocess import clean_text, clean_generated_summary, split_sentences
from src.utils import count_words
from evaluation.output_validator import validate_output, detect_poor_training_output

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
    cleaned = clean_text(TEXT, aggressive=True)
    print("Cleaned words:", count_words(cleaned), "sentences:", len(split_sentences(cleaned)))
    analysis = SummaryLengthManager.analyze_input(TEXT)
    sc = SummaryLengthManager.get_extractive_sentences("auto", analysis)
    print("Sentence count setting:", sc)

    for algo in ["textrank", "lexrank", "lsa"]:
        d = EXTRACTIVE_RUNNERS[algo](TEXT, sentence_count=sc)
        print(f"\n=== {algo.upper()} ===")
        print("Summary:", d["summary"])
        for item in d.get("selected_sentences", []):
            print(f"  [{item['sentence_index']}] score={item['sentence_score']:.3f} | {item['sentence'][:80]}...")

    min_tok, max_tok = SummaryLengthManager.get_abstractive_limits("", "auto", analysis)
    for key in ["vit5", "mt5", "bartpho"]:
        raw_path = f"before_{key}"
        s = abstractive_summarize_key(key, TEXT, max_output_length=max_tok, min_output_length=min_tok)
        cleaned_s = clean_generated_summary(s)
        v = validate_output(s, require_vietnamese=key in {"vit5", "mt5"})
        print(f"\n=== {key.upper()} ===")
        print("Raw:", repr(s[:300]))
        print("After clean_generated_summary:", repr(cleaned_s[:300]))
        print("corrupted:", v["is_corrupted"], "warning:", v.get("quality_warning"))
        print("training:", detect_poor_training_output(s))

if __name__ == "__main__":
    main()
