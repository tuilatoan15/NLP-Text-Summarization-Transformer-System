import sys
import io
import torch
from pathlib import Path

# Thêm project root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from summarizers.abstractive.abstractive_summarizer import AbstractiveSummarizer
from pipeline.hybrid_summarizer import HybridSummarizer

# Fix windows console output encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sample = '''
Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình
leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập
tức và mở hành lang nhân đạo cho người dân vùng chiến sự. Đại diện Mỹ phát biểu
rằng Washington ủng hộ giải pháp hai nhà nước nhưng nhấn mạnh quyền tự vệ hợp
pháp. Nga và Trung Quốc phản đối dự thảo nghị quyết, cho rằng văn kiện còn
thiếu cân bằng. Cuộc khủng hoảng nhân đạo ngày càng nghiêm trọng khi hàng nghìn
thường dân phải di tản. Các tổ chức phi chính phủ kêu gọi cộng đồng quốc tế
hành động khẩn cấp để bảo vệ dân thường.
'''

def test_abstractive():
    print("="*60)
    print("TESTING CORE ABSTRACTIVE SUMMARIZERS")
    print("="*60)
    for model_key in ["vit5", "mt5", "bartpho"]:
        try:
            print(f"\n[*] Summarizing with {model_key.upper()}...")
            summarizer = AbstractiveSummarizer(model_name=model_key)
            summary = summarizer.summarize(sample)
            print(f"[{model_key.upper()} Output]:")
            print(repr(summary))
        except Exception as e:
            print(f"[-] Model {model_key} failed: {e}")

def test_hybrid():
    print("\n" + "="*60)
    print("TESTING HYBRID SUMMARIZER (EXTRACTIVE + ABSTRACTIVE)")
    print("="*60)
    for model_key in ["vit5", "bartpho"]:
        try:
            print(f"\n[*] Hybrid summarizing with {model_key.upper()}...")
            hybrid = HybridSummarizer(abstractive_model_key=model_key)
            summary = hybrid.summarize(sample)
            print(f"[HYBRID {model_key.upper()} Output]:")
            print(repr(summary))
        except Exception as e:
            print(f"[-] Hybrid {model_key} failed: {e}")

if __name__ == "__main__":
    test_abstractive()
    test_hybrid()
