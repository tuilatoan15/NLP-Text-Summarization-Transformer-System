"""
abstractive.py — Tóm tắt diễn giải (Abstractive Summarization) dùng ViT5.

Model VietAI/vit5-base là mô hình Seq2Seq dựa trên T5 được pre-train trên
tiếng Việt, phù hợp cho bài toán tóm tắt và dịch thuật tiếng Việt.

Hỗ trợ 2 chế độ:
  1. Inference: Load model từ local (./models/) hoặc Hugging Face Hub
  2. Training: Có thể fine-tune lại với dữ liệu mới (xem train/train_vit5.py)
"""

import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

from src.utils import logger, truncate_text


# ==============================================================================
# CẤU HÌNH
# ==============================================================================

# Model mặc định: ViT5 base của VietAI
DEFAULT_MODEL_NAME = "VietAI/vit5-base"

# Thư mục lưu model đã fine-tune
LOCAL_MODEL_DIR = Path("./models/vit5-finetuned")

# Cấu hình sinh văn bản
DEFAULT_MAX_INPUT_TOKENS = 512      # Giới hạn token đầu vào
DEFAULT_MAX_OUTPUT_LENGTH = 150     # Độ dài tối đa bản tóm tắt
DEFAULT_MIN_OUTPUT_LENGTH = 30      # Độ dài tối thiểu bản tóm tắt
DEFAULT_NUM_BEAMS = 4               # Beam search (tăng chất lượng, giảm tốc độ)
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3    # Tránh lặp n-gram trong output


# ==============================================================================
# CLASS ABSTRACTIVE SUMMARIZER
# ==============================================================================

class AbstractiveSummarizer:
    """
    Wrapper cho mô hình ViT5 dùng để tóm tắt diễn giải tiếng Việt.

    Tự động phát hiện và dùng GPU nếu có, fallback về CPU nếu không.
    Ưu tiên load model đã fine-tune từ local, nếu không có thì dùng Hugging Face.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        local_model_dir: Optional[str] = None,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH,
        min_output_length: int = DEFAULT_MIN_OUTPUT_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    ):
        self.model_name = model_name
        self.local_model_dir = Path(local_model_dir) if local_model_dir else LOCAL_MODEL_DIR
        self.max_input_tokens = max_input_tokens
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size

        # Phát hiện thiết bị (GPU ưu tiên, fallback CPU)
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "CUDA (GPU)" if self.device == 0 else "CPU"
        logger.info(f"Thiết bị suy luận: {device_name}")

        # Chưa load model (lazy loading để tiết kiệm RAM)
        self.tokenizer = None
        self.model = None
        self._pipeline = None

    def _resolve_model_path(self) -> str:
        """
        Quyết định dùng model local hay Hugging Face Hub.
        Ưu tiên: local fine-tuned > Hugging Face Hub
        """
        if self.local_model_dir.exists() and any(self.local_model_dir.iterdir()):
            logger.info(f"Dùng model local: {self.local_model_dir}")
            return str(self.local_model_dir)
        else:
            logger.info(f"Không tìm thấy model local, dùng Hugging Face: {self.model_name}")
            return self.model_name

    def load(self) -> None:
        """
        Load tokenizer và model vào bộ nhớ.
        Gọi hàm này một lần trước khi sử dụng.
        """
        if self._pipeline is not None:
            logger.info("Model đã được load, bỏ qua.")
            return

        model_path = self._resolve_model_path()
        logger.info(f"Đang load tokenizer từ: {model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Đang load model từ: {model_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            # Tạo pipeline cho việc sinh văn bản
            self._pipeline = pipeline(
                task="summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                framework="pt",
            )
            logger.info("✅ Load model thành công!")

        except Exception as e:
            logger.error(f"❌ Lỗi load model: {e}")
            raise RuntimeError(f"Không thể load model ViT5: {e}") from e

    def summarize(
        self,
        text: str,
        max_output_length: Optional[int] = None,
        min_output_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> str:
        """
        Sinh bản tóm tắt diễn giải từ văn bản đầu vào.

        Tự động truncate input nếu vượt quá giới hạn token.

        Args:
            text: Văn bản tiếng Việt cần tóm tắt
            max_output_length: Độ dài tối đa output (ghi đè cấu hình mặc định)
            min_output_length: Độ dài tối thiểu output
            num_beams: Số beam trong beam search

        Returns:
            Bản tóm tắt diễn giải
        """
        if not text or not text.strip():
            logger.warning("Văn bản đầu vào rỗng.")
            return ""

        # Lazy load model nếu chưa có
        if self._pipeline is None:
            self.load()

        # Truncate input nếu quá dài (tính theo từ, ước lượng token)
        # Với ViT5, 1 từ tiếng Việt ≈ 1.5 - 2 token => dùng max_words = token_limit * 0.6
        max_words = int(self.max_input_tokens * 0.6)
        text = truncate_text(text, max_words=max_words)

        # Sử dụng cấu hình override hoặc cấu hình mặc định
        _max_len = max_output_length or self.max_output_length
        _min_len = min_output_length or self.min_output_length
        _beams = num_beams or self.num_beams

        try:
            results = self._pipeline(
                text,
                max_length=_max_len,
                min_length=_min_len,
                num_beams=_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                early_stopping=True,
                do_sample=False,    # Deterministic output
            )

            summary = results[0]["summary_text"].strip()
            logger.info(f"Abstractive OK: {len(summary.split())} từ.")
            return summary

        except Exception as e:
            logger.error(f"Lỗi sinh tóm tắt: {e}")
            return ""

    def is_loaded(self) -> bool:
        """Kiểm tra model đã được load chưa."""
        return self._pipeline is not None


# ==============================================================================
# SINGLETON INSTANCE (Dùng chung trong API để tránh load model nhiều lần)
# ==============================================================================

_global_summarizer: Optional[AbstractiveSummarizer] = None


def get_summarizer(
    model_name: str = DEFAULT_MODEL_NAME,
    local_model_dir: Optional[str] = None,
) -> AbstractiveSummarizer:
    """
    Lấy instance AbstractiveSummarizer toàn cục (singleton pattern).
    Đảm bảo model chỉ được load một lần duy nhất trong quá trình chạy.
    """
    global _global_summarizer

    if _global_summarizer is None:
        logger.info("Khởi tạo AbstractiveSummarizer lần đầu...")
        _global_summarizer = AbstractiveSummarizer(
            model_name=model_name,
            local_model_dir=local_model_dir,
        )
        _global_summarizer.load()

    return _global_summarizer


def abstractive_summarize(
    text: str,
    max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH,
    min_output_length: int = DEFAULT_MIN_OUTPUT_LENGTH,
    num_beams: int = DEFAULT_NUM_BEAMS,
    local_model_dir: Optional[str] = None,
) -> str:
    """
    Hàm tiện ích: tóm tắt diễn giải, dùng singleton summarizer.

    Dùng khi muốn gọi trực tiếp mà không cần khởi tạo class.
    """
    summarizer = get_summarizer(local_model_dir=local_model_dir)
    return summarizer.summarize(
        text,
        max_output_length=max_output_length,
        min_output_length=min_output_length,
        num_beams=num_beams,
    )


# ==============================================================================
# CHẠY THỬ TRỰC TIẾP
# ==============================================================================

if __name__ == "__main__":
    sample = """
    Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình
    leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập
    tức và mở hành lang nhân đạo cho người dân vùng chiến sự. Đại diện Mỹ phát biểu
    rằng Washington ủng hộ giải pháp hai nhà nước nhưng nhấn mạnh quyền tự vệ hợp
    pháp. Nga và Trung Quốc phản đối dự thảo nghị quyết, cho rằng văn kiện còn
    thiếu cân bằng. Cuộc khủng hoảng nhân đạo ngày càng nghiêm trọng khi hàng nghìn
    thường dân phải di tản. Các tổ chức phi chính phủ kêu gọi cộng đồng quốc tế
    hành động khẩn cấp để bảo vệ dân thường.
    """

    print("=== TÓM TẮT DIỄN GIẢI (ViT5) ===")
    # Sử dụng hàm tiện ích (sẽ load model từ HF nếu chưa có local)
    summary = abstractive_summarize(sample, max_output_length=80, num_beams=2)
    print(summary)
