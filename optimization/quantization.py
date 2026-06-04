"""
quantization.py — Script tối ưu hóa mô hình bằng kỹ thuật Lượng tử hóa động (Dynamic Quantization INT8)
kết hợp định dạng ONNX Runtime. Giúp giảm dung lượng mô hình xuống 4 lần và tăng tốc độ chạy trên CPU.
"""
from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any

# Thêm root path của project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    import torch
    import onnx
    import onnxruntime
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    logger.warning("⚠️  Cần cài đặt: torch, onnx, onnxruntime để chạy lượng tử hóa.")


class ModelQuantizer:
    """
    Trình lượng tử hóa mô hình Transformer sang dạng ONNX INT8.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        from src import config
        self.output_dir = output_dir or (config.MODEL_DIR / "quantized_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def quantize_huggingface_model(
        self,
        model_name_or_path: str = "VietAI/vit5-base",
        output_filename: str = "vit5_int8.onnx",
        model_key: str | None = None
    ) -> Path:
        """
        Xuất mô hình Hugging Face (hoặc mô hình đã fine-tune nội bộ) sang ONNX và thực hiện Lượng tử hóa động INT8.
        """
        if model_key is None:
            path_lower = model_name_or_path.lower()
            if "vit5" in path_lower:
                model_key = "vit5"
            elif "bartpho" in path_lower:
                model_key = "bartpho"
            elif "mt5" in path_lower:
                model_key = "mt5"
            else:
                model_key = "vit5"

        logger.info(f"🔄 Đang tải mô hình gốc từ path/hub: {model_name_or_path} (model_key={model_key}) ...")
        t_start = time.perf_counter()

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        # 1. Tải tokenizer và model
        # T5 tiếng Việt cần use_fast=False cho vit5/bartpho, use_fast=True cho mt5
        use_fast = model_key == "mt5"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)
        
        # Kiểm tra PEFT/LoRA cục bộ
        is_peft = False
        adapter_config_file = Path(model_name_or_path) / "adapter_config.json"
        if adapter_config_file.exists():
            is_peft = True
            
        if is_peft:
            from peft import PeftModel
            import json
            logger.info("PEFT adapter found. Loading base model...")
            with open(adapter_config_file, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path")
            if not base_model_name:
                # Fallback hubs
                if model_key == "vit5":
                    base_model_name = "VietAI/vit5-base"
                elif model_key == "bartpho":
                    base_model_name = "vinai/bartpho-syllable"
                else:
                    base_model_name = "google/mt5-small"
            
            logger.info(f"Loading base model: {base_model_name}")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
            model = model.merge_and_unload()
            logger.info("PEFT adapter merged successfully for quantization.")
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            
        model.eval()

        # 2. Định nghĩa các đường dẫn file đầu ra
        onnx_model_path = self.output_dir / "temp_model_fp32.onnx"
        quantized_model_path = self.output_dir / output_filename

        logger.info("⚡ Bước 1: Xuất mô hình PyTorch sang định dạng ONNX FP32 ...")
        
        # Tạo dữ liệu giả lập (dummy inputs) cho quá trình tracing đồ thị
        dummy_text = "Hội nghị G7 diễn ra tại Hiroshima Nhật Bản."
        inputs = tokenizer(dummy_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Tạo dummy decoder input ids (đối với kiến trúc Seq2Seq)
        decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])

        # Tắt use_cache để tránh lỗi JIT tracer với EncoderDecoderCache của HuggingFace mới
        model.config.use_cache = False

        # Xuất file ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                (input_ids, attention_mask, decoder_input_ids),
                str(onnx_model_path),
                input_names=["input_ids", "attention_mask", "decoder_input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "decoder_input_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=14,
                do_constant_folding=True
            )

        logger.info(f" ONNX FP32 model exported successfully: {onnx_model_path}")
        size_fp32 = os.path.getsize(onnx_model_path) / (1024 * 1024)
        logger.info(f" Kích thước file gốc FP32: {size_fp32:.2f} MB")

        # 3. Thực hiện Dynamic Quantization sang INT8
        logger.info("⚡ Bước 2: Thực hiện Lượng tử hóa động sang dạng INT8 ...")
        
        quantize_dynamic(
            model_input=str(onnx_model_path),
            model_output=str(quantized_model_path),
            weight_type=QuantType.QInt8
        )

        # Xóa file FP32 tạm để tiết kiệm không gian đĩa
        if onnx_model_path.exists():
            os.remove(onnx_model_path)

        size_int8 = os.path.getsize(quantized_model_path) / (1024 * 1024)
        elapsed = time.perf_counter() - t_start
        
        logger.info("=" * 60)
        logger.info("🎉 HỒ SƠ LƯỢNG TỬ HÓA THÀNH CÔNG:")
        logger.info(f"  - Đường dẫn file lưu trữ: {quantized_model_path}")
        logger.info(f"  - Kích thước FP32: {size_fp32:.2f} MB")
        logger.info(f"  - Kích thước INT8: {size_int8:.2f} MB")
        logger.info(f"  - Tỷ lệ giảm kích thước: {100.0 * (1.0 - size_int8 / size_fp32):.1f}%")
        logger.info(f"  - Thời gian xử lý: {elapsed:.2f}s")
        logger.info("=" * 60)

        return quantized_model_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ONNX Dynamic Quantization for Summarization Models.")
    parser.add_argument("--model_path", default="VietAI/vit5-base", help="Hub model id or local directory.")
    parser.add_argument("--output", default="vit5_base_int8.onnx", help="Output ONNX filename.")
    parser.add_argument("--model_key", default=None, choices=["vit5", "mt5", "bartpho"], help="Explicit model architecture key.")
    args = parser.parse_args()

    quantizer = ModelQuantizer()
    try:
        quantizer.quantize_huggingface_model(
            model_name_or_path=args.model_path,
            output_filename=args.output,
            model_key=args.model_key
        )
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện chạy script lượng tử hóa: {e}", exc_info=True)
