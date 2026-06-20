"""
tests/test_trained_models.py — Integration tests to verify the local trained models can be successfully loaded and connected.
"""
import os
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from summarizers.abstractive.abstractive_summarizer import get_summarizer
from src.utils import clear_gpu_cache

SAMPLE_TEXT = """
Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình
leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập
tức và mở hành lang nhân đạo cho người dân vùng chiến sự. Đại diện Mỹ phát biểu
rằng Washington ủng hộ giải pháp hai nhà nước nhưng nhấn mạnh quyền tự vệ hợp
pháp. Nga và Trung Quốc phản đối dự thảo nghị quyết, cho rằng văn kiện còn
thiếu cân bằng. Cuộc khủng hoảng nhân đạo ngày càng nghiêm trọng khi hàng nghìn
thường dân phải di tản. Các tổ chức phi chính phủ kêu gọi cộng đồng quốc tế
hành động khẩn cấp để bảo vệ dân thường.
"""

class TestTrainedModels(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        
    def test_local_model_paths_exist(self):
        """Verify that the model target directories exist and contain model files."""
        model_names = ["vit5-finetuned", "mt5-finetuned", "bartpho-finetuned"]
        for name in model_names:
            path = self.models_dir / name
            self.assertTrue(path.exists(), f"Directory {path} does not exist!")
            self.assertTrue((path / "model.safetensors").exists(), f"Model file model.safetensors not found in {path}")
            self.assertTrue((path / "config.json").exists(), f"Config file config.json not found in {path}")

    def test_vit5_local_summarization(self):
        """Test loading and inference of local ViT5 finetuned model."""
        print("\n=== Testing local ViT5 fine-tuned model ===")
        try:
            summarizer = get_summarizer("vit5")
            summarizer.load()
            self.assertTrue(summarizer.is_loaded(), "ViT5 model was not loaded successfully.")
            
            # Run inference
            summary = summarizer.summarize(SAMPLE_TEXT, max_output_length=80, min_output_length=20)
            print(f"[+] ViT5 summary: {summary}")
            self.assertIsInstance(summary, str)
            self.assertTrue(len(summary.strip()) > 0, "ViT5 returned an empty summary.")
        finally:
            clear_gpu_cache()

    def test_mt5_local_summarization(self):
        """Test loading and inference of local mT5 finetuned model."""
        print("\n=== Testing local mT5 fine-tuned model ===")
        try:
            summarizer = get_summarizer("mt5")
            summarizer.load()
            self.assertTrue(summarizer.is_loaded(), "mT5 model was not loaded successfully.")
            
            # Run inference
            summary = summarizer.summarize(SAMPLE_TEXT, max_output_length=80, min_output_length=10)
            print(f"[+] mT5 summary: {summary}")
            self.assertIsInstance(summary, str)
            self.assertTrue(len(summary.strip()) > 0, "mT5 returned an empty summary.")
        finally:
            clear_gpu_cache()

    def test_bartpho_local_summarization(self):
        """Test loading and inference of local BARTPho finetuned model."""
        print("\n=== Testing local BARTPho fine-tuned model ===")
        try:
            summarizer = get_summarizer("bartpho")
            summarizer.load()
            self.assertTrue(summarizer.is_loaded(), "BARTPho model was not loaded successfully.")
            
            # Run inference
            summary = summarizer.summarize(SAMPLE_TEXT, max_output_length=80, min_output_length=20)
            print(f"[+] BARTPho summary: {summary}")
            self.assertIsInstance(summary, str)
            self.assertTrue(len(summary.strip()) > 0, "BARTPho returned an empty summary.")
        finally:
            clear_gpu_cache()

if __name__ == "__main__":
    unittest.main()
