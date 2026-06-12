#!/usr/bin/env python3
"""Comprehensive summarization quality diagnostic — 12-phase investigation harness."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.preprocess import clean_text, split_sentences, tokenize_words
from src.utils import count_words
from summarizers.extractive.extractive_summarizer import EXTRACTIVE_RUNNERS
from summarizers.abstractive.length_control import words_to_max_new_tokens

# ── 10 standard test texts ────────────────────────────────────────────────
TEST_TEXTS: dict[str, str] = {
    "short_story": (
        "Ngày xưa, có một cậu bé tên Minh sống ở làng quê nghèo. "
        "Mỗi sáng cậu dậy sớm giúp mẹ nấu cơm rồi chạy bộ đến trường. "
        "Giáo viên khen cậu chăm chỉ và luôn giúp đỡ bạn bè. "
        "Một hôm, trường tổ chức cuộc thi vẽ tranh về quê hương. "
        "Minh vẽ cánh đồng lúa vàng óng dưới ánh hoàng hôn. "
        "Bức tranh đoạt giải nhất và được treo ở phòng truyền thống. "
        "Cậu bé mỉm cười, biết rằng nỗ lực nhỏ bé cũng có thể tạo nên điều lớn lao."
    ),
    "news_article": (
        "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình "
        "leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập "
        "tức và mở hành lang nhân đạo cho người dân vùng chiến sự. Đại diện Mỹ phát biểu "
        "rằng Washington ủng hộ giải pháp hai nhà nước nhưng nhấn mạnh quyền tự vệ hợp "
        "pháp. Nga và Trung Quốc phản đối dự thảo nghị quyết, cho rằng văn kiện còn "
        "thiếu cân bằng. Cuộc khủng hoảng nhân đạo ngày càng nghiêm trọng khi hàng nghìn "
        "thường dân phải di tản. Các tổ chức phi chính phủ kêu gọi cộng đồng quốc tế "
        "hành động khẩn cấp để bảo vệ dân thường."
    ),
    "technical_doc": (
        "Hệ thống tóm tắt văn bản sử dụng kiến trúc Transformer với cơ chế attention. "
        "Mô hình encoder-decoder nhận đầu vào là chuỗi token và sinh ra chuỗi tóm tắt. "
        "Quá trình huấn luyện sử dụng hàm mất mát cross-entropy trên tập VietNews. "
        "Độ dài đầu vào tối đa là 512 token, đầu ra tối đa 128 token. "
        "Beam search với 4 beam được dùng khi suy luận. "
        "Các metric đánh giá gồm ROUGE, BERTScore và semantic similarity. "
        "Hệ thống hỗ trợ cả phương pháp trích xuất và sinh tóm tắt."
    ),
    "report": (
        "Báo cáo quý III cho thấy doanh thu tăng 12% so với cùng kỳ năm trước. "
        "Mảng công nghệ đóng góp 45% tổng doanh thu, tăng 8 điểm phần trăm. "
        "Chi phí vận hành giảm 3% nhờ tự động hóa quy trình. "
        "Số lượng khách hàng mới tăng 20%, chủ yếu từ kênh trực tuyến. "
        "Ban lãnh đạo dự báo quý IV sẽ duy trì đà tăng trưởng 10-15%. "
        "Rủi ro chính bao gồm biến động tỷ giá và thiếu hụt nhân lực kỹ thuật. "
        "Kế hoạch đầu tư R&D được duyệt thêm 5 tỷ đồng cho năm sau."
    ),
    "bullet_list": (
        "Kế hoạch triển khai gồm các bước sau:\n"
        "- Khảo sát hiện trạng hệ thống\n"
        "- Thiết kế kiến trúc mới\n"
        "- Phát triển module xử lý ngôn ngữ\n"
        "- Kiểm thử tích hợp\n"
        "- Triển khai thí điểm tại 3 đơn vị\n"
        "Thời gian dự kiến hoàn thành trong 6 tháng."
    ),
    "questions": (
        "Tại sao mô hình tóm tắt lại sinh ra văn bản lặp? "
        "Có phải do tham số repetition_penalty quá thấp không? "
        "Làm thế nào để cải thiện chất lượng cho văn bản dài? "
        "Tokenizer có ảnh hưởng đến chất lượng đầu ra không? "
        "Chúng ta nên dùng beam search hay sampling?"
    ),
    "admin_doc": (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "Độc lập - Tự do - Hạnh phúc\n"
        "Số: 123/QĐ-UBND\n"
        "QUYẾT ĐỊNH\n"
        "Về việc phê duyệt kế hoạch đào tạo nhân lực công nghệ thông tin\n"
        "Ủy ban nhân dân tỉnh căn cứ Luật Tổ chức chính quyền địa phương;\n"
        "Căn cứ nhu cầu phát triển nguồn nhân lực CNTT của tỉnh;\n"
        "Quyết định: Điều 1. Phê duyệt kế hoạch đào tạo 500 kỹ sư CNTT trong 3 năm.\n"
        "Điều 2. Giao Sở Thông tin và Truyền thông chủ trì thực hiện.\n"
        "Nơi nhận: - Như Điều 2; - Lưu VT."
    ),
    "long_article": (
        " ".join(
            [
                f"Đoạn {i + 1}: Ngành công nghệ thông tin Việt Nam tiếp tục phát triển mạnh "
                f"với nhiều startup gọi vốn thành công và xuất khẩu phần mềm tăng trưởng hai chữ số. "
                f"Chính phủ ban hành chính sách hỗ trợ chuyển đổi số cho doanh nghiệp vừa và nhỏ."
                for i in range(25)
            ]
        )
    ),
    "mixed_numbers": (
        "Trong quý I/2026, GDP tăng 6.5%, lạm phát 3.2% và tỷ giá USD/VND ở mức 25.400. "
        "Xuất khẩu đạt 95 tỷ USD, nhập khẩu 88 tỷ USD. "
        "Ngành du lịch đón 8,5 triệu lượt khách quốc tế. "
        "Đầu tư FDI đạt 5,2 tỷ USD, tăng 15% so với năm trước."
    ),
    "paragraph_heavy": (
        "Đoạn một mô tả bối cảnh nghiên cứu về xử lý ngôn ngữ tự nhiên tiếng Việt.\n\n"
        "Đoạn hai trình bày phương pháp sử dụng mô hình Transformer fine-tuned trên VietNews.\n\n"
        "Đoạn ba phân tích kết quả thực nghiệm với các metric ROUGE và BERTScore.\n\n"
        "Đoạn bốn đề xuất hướng cải thiện pipeline tiền xử lý và chunking cho văn bản dài."
    ),
}


def pipeline_stats(text: str) -> dict:
    raw_words = count_words(text)
    raw_sents = len(split_sentences(text))
    cleaned = clean_text(text, aggressive=True)
    cleaned_words = count_words(cleaned)
    cleaned_sents = split_sentences(cleaned)
    return {
        "raw_words": raw_words,
        "raw_sentences": raw_sents,
        "cleaned_words": cleaned_words,
        "cleaned_sentences": len(cleaned_sents),
        "word_loss_pct": round(100 * (1 - cleaned_words / max(1, raw_words)), 1),
        "sentence_loss_pct": round(100 * (1 - len(cleaned_sents) / max(1, raw_sents)), 1),
    }


def extractive_score_table(text: str, algo: str, sentence_count: int) -> list[dict]:
    runner = EXTRACTIVE_RUNNERS[algo]
    details = runner(text, sentence_count=sentence_count)
    selected_idxs = {item["sentence_index"] for item in details.get("selected_sentences", [])}
    rows = []
    for i, sent in enumerate(details.get("source_sentences", [])):
        score_item = next(
            (s for s in details.get("selected_sentences", []) if s["sentence_index"] == i),
            None,
        )
        rows.append(
            {
                "index": i,
                "sentence": sent[:120],
                "score": score_item["sentence_score"] if score_item else 0.0,
                "selected": i in selected_idxs,
            }
        )
    return rows


def tokenizer_diagnostics() -> dict:
    from ai_models.model_registry import ABSTRACTIVE_ALGORITHMS, resolve_model_path
    from src.model_loader import get_loaded_model

    sample = TEST_TEXTS["news_article"]
    results = {}
    for key in ("vit5", "mt5", "bartpho"):
        loaded = get_loaded_model(key)
        algo = ABSTRACTIVE_ALGORITHMS[key]
        prefixed = f"summarize: {sample}" if key in {"vit5", "mt5"} else sample
        enc = loaded.tokenizer(prefixed, return_tensors="pt", truncation=False)
        enc_trunc = loaded.tokenizer(
            prefixed, return_tensors="pt", truncation=True, max_length=config.MAX_INPUT_TOKENS
        )
        ids = enc["input_ids"][0].tolist()
        decoded_roundtrip = loaded.tokenizer.decode(ids, skip_special_tokens=True)
        results[key] = {
            "model_path": resolve_model_path(algo),
            "hub_name": algo.model_name,
            "tokenizer_class": type(loaded.tokenizer).__name__,
            "vocab_size": len(loaded.tokenizer),
            "model_embed_size": int(loaded.model.get_input_embeddings().weight.shape[0]),
            "full_token_count": len(ids),
            "truncated_token_count": len(enc_trunc["input_ids"][0]),
            "truncated": len(ids) > config.MAX_INPUT_TOKENS,
            "roundtrip_has_diacritics": any(c in decoded_roundtrip for c in "àáảãạăắằẳẵặâấầẩẫậ"),
            "roundtrip_preview": decoded_roundtrip[:200],
        }
    return results


def generation_config_dump() -> dict:
    return {
        "global": {
            "MAX_INPUT_TOKENS": config.MAX_INPUT_TOKENS,
            "MAX_OUTPUT_LENGTH": config.MAX_OUTPUT_LENGTH,
            "MIN_OUTPUT_LENGTH": config.MIN_OUTPUT_LENGTH,
            "NUM_BEAMS": config.NUM_BEAMS,
            "NO_REPEAT_NGRAM_SIZE": config.NO_REPEAT_NGRAM_SIZE,
            "ABSTRACTIVE_MAX_CHUNKS": config.ABSTRACTIVE_MAX_CHUNKS,
        },
        "per_model": config.GENERATION_CONFIGS,
    }


def run_model_outputs() -> dict:
    from summarizers.abstractive.abstractive_summarizer import abstractive_summarize_key
    from summarizers.length_manager import SummaryLengthManager

    outputs = {}
    text = TEST_TEXTS["news_article"]
    analysis = SummaryLengthManager.analyze_input(text)
    sentence_count = SummaryLengthManager.get_extractive_sentences("auto", analysis)
    min_tok, max_tok = SummaryLengthManager.get_abstractive_limits("", "auto", analysis)

    for algo in ("textrank", "lexrank", "lsa"):
        details = EXTRACTIVE_RUNNERS[algo](text, sentence_count=sentence_count)
        outputs[algo] = {
            "summary": details["summary"],
            "word_count": count_words(details["summary"]),
            "selected": len(details.get("selected_sentences", [])),
            "source_sentences": len(details.get("source_sentences", [])),
        }

    for key in ("vit5", "mt5", "bartpho"):
        t0 = time.perf_counter()
        summary = abstractive_summarize_key(
            key, text, max_output_length=max_tok, min_output_length=min_tok
        )
        elapsed = time.perf_counter() - t0
        from evaluation.output_validator import validate_output, detect_poor_training_output

        val = validate_output(summary, require_vietnamese=key in {"vit5", "mt5"})
        train_q = detect_poor_training_output(summary)
        outputs[key] = {
            "summary": summary,
            "word_count": count_words(summary),
            "elapsed_s": round(elapsed, 2),
            "validation": val,
            "training_quality": train_q,
        }
    return outputs


def main() -> None:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phases": {},
    }

    # Phase 1: Pipeline map
    report["phases"]["1_pipeline"] = {
        "steps": [
            {"step": "Document", "files": ["loaders/file_parser.py", "loaders/pdf_loader.py"]},
            {"step": "Cleaning", "files": ["preprocess/preprocessor.py:clean_text", "preprocess/admin_cleaner.py"]},
            {"step": "Sentence Splitting", "files": ["preprocess/preprocessor.py:split_sentences"]},
            {"step": "Chunking", "files": ["summarizers/abstractive/abstractive_summarizer.py:_chunk_text", "pipeline/hybrid_summarizer.py:SemanticChunker"]},
            {"step": "Summarization", "files": ["summarizers/extractive/extractive_summarizer.py", "summarizers/abstractive/abstractive_summarizer.py"]},
            {"step": "Evaluation", "files": ["evaluation/metrics.py", "backend/services/dashboard_service.py"]},
        ]
    }

    # Phase 2: Input data verification
    report["phases"]["2_input_verification"] = {
        name: pipeline_stats(text) for name, text in TEST_TEXTS.items()
    }

    # Phase 3: Sentence splitting
    report["phases"]["3_sentence_splitting"] = {}
    for name in ("questions", "bullet_list", "paragraph_heavy"):
        raw = TEST_TEXTS[name]
        report["phases"]["3_sentence_splitting"][name] = {
            "before": len(re.split(r"[.!?\n]+", raw)),
            "after_underthesea": len(split_sentences(raw)),
            "sentences": split_sentences(raw),
        }

    # Phase 4: Extractive diagnostics
    text = TEST_TEXTS["news_article"]
    from summarizers.length_manager import SummaryLengthManager
    sc = SummaryLengthManager.get_extractive_sentences("auto", SummaryLengthManager.analyze_input(text))
    report["phases"]["4_extractive"] = {
        algo: {
            "sentence_count_setting": sc,
            "score_table": extractive_score_table(text, algo, sc),
            "summary": EXTRACTIVE_RUNNERS[algo](text, sentence_count=sc)["summary"],
        }
        for algo in ("textrank", "lexrank", "lsa")
    }

    # Phase 5-7: Tokenizer + model loading
    print("Loading models for tokenizer diagnostics...")
    report["phases"]["5_tokenizer"] = tokenizer_diagnostics()
    report["phases"]["6_model_loading"] = report["phases"]["5_tokenizer"]

    # Phase 8: Generation config
    report["phases"]["8_generation_config"] = generation_config_dump()

    # Phase 9: Chunking
    from summarizers.abstractive.abstractive_summarizer import _chunk_text
    long_text = TEST_TEXTS["long_article"]
    max_words = max(180, int(config.MAX_INPUT_TOKENS * 0.55))
    chunks = _chunk_text(long_text, max_words)
    report["phases"]["9_chunking"] = {
        "max_words_per_chunk": max_words,
        "source_words": count_words(long_text),
        "chunk_count": len(chunks),
        "max_chunks_limit": config.ABSTRACTIVE_MAX_CHUNKS,
        "chunks_truncated": len(chunks) >= config.ABSTRACTIVE_MAX_CHUNKS,
        "chunk_word_counts": [count_words(c) for c in chunks],
        "mid_sentence_splits": sum(
            1 for c in chunks if c and not c.rstrip().endswith((".", "!", "?"))
        ),
    }

    # Phase 10: Model outputs
    print("Running all model outputs (may take several minutes)...")
    report["phases"]["10_model_outputs"] = run_model_outputs()

    out_path = ROOT / "storage" / "results" / "diagnostic_report_before.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Diagnostic report saved to {out_path}")


if __name__ == "__main__":
    import re
    main()
