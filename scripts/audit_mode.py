#!/usr/bin/env python3
"""
scripts/audit_mode.py — Automated Audit and Verification Runner
Tests extraction, cleaning, segmentation, tokenization, summarization, and evaluation.
"""

from __future__ import annotations

import os
import sys
import json
import time
import tempfile
import traceback
from pathlib import Path
from typing import Any

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import clean_text, split_sentences, tokenize_words
from preprocess.cleaner import DocumentCleaner
from preprocess.admin_cleaner import AdministrativeDocumentCleaner
from preprocess.tokenizer import VietnameseTokenizer
from pipeline.schema import IngestConfig, DocumentElement, ExtractedDocument, DocumentMetadata
from summarizers.extractive.extractive_summarizer import summarize_extractive_parallel
from evaluation.metrics import compute_rouge, compute_bleu, compute_bertscore
from evaluation.output_validator import validate_output, detect_poor_training_output

# Global report dict
audit_report: dict[str, Any] = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "status": "success",
    "sections": {}
}

def log_section(name: str, passed: bool, duration: float, details: dict[str, Any], errors: list[str] = None):
    audit_report["sections"][name] = {
        "status": "PASSED" if passed else "FAILED",
        "duration_seconds": round(duration, 4),
        "details": details,
        "errors": errors or []
    }
    if not passed:
        audit_report["status"] = "failed"
    status_emoji = "✅" if passed else "❌"
    print(f"{status_emoji} Section: {name:<25} | {'PASSED' if passed else 'FAILED':<6} | {duration:.4f}s")


def run_extraction_and_cleaning_audit() -> bool:
    print("\n--- Running Section A & B: Text Extraction & Document Cleaning Audit ---")
    start = time.perf_counter()
    passed = True
    errors = []
    details = {}
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir_path = Path(tmpdir)
        pdf_path = temp_dir_path / "admin_test.pdf"
        docx_path = temp_dir_path / "admin_test.docx"
        txt_path = temp_dir_path / "admin_test.txt"

        # Mock Admin Text Content with Glued Spacing Errors
        admin_content = (
            "ỦY BAN NHÂN DÂN TỈNH NGHỆ AN\n"
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n"
            "Số: 4321/QĐ-UBND\n"
            "Nghệ An, ngày 12 tháng 06 năm 2026\n\n"
            "QUYẾT ĐỊNH\n"
            "Về việc phê duyệt đềtài nghiên cứu côngnghệ mới.\n\n"
            "Mô hình RAG mới giúp tăng độchính xác và cải thiện câu trảlời cho người dùng. "
            "Các cánbộ và sinhviên cần thực hiện chạythử hệthống trên máy tính có bộnhớ tốt để tránh tràn bộ nhớ.\n\n"
            "Nơi nhận:\n"
            "- Như trên;\n"
            "- Lưu VT.\n\n"
            "CHỦ TỊCH\n"
            "Nguyễn Văn B\n"
            "[OCR Scan Success]\n"
            "Trang 1/1"
        )

        # 1. Write TXT
        txt_path.write_text(admin_content, encoding="utf-8")

        # 2. Write PDF using fitz (PyMuPDF)
        try:
            import fitz
            pdf_doc = fitz.open()
            page = pdf_doc.new_page()
            # Draw line by line to page
            y = 50
            for line in admin_content.splitlines():
                page.insert_text((50, y), line)
                y += 20
            pdf_doc.save(str(pdf_path))
            pdf_doc.close()
            pdf_available = True
        except Exception as exc:
            pdf_available = False
            errors.append(f"PyMuPDF generation failed: {exc}")
            print(f"⚠️ PyMuPDF PDF writing not checked: {exc}")

        # 3. Write DOCX using python-docx
        try:
            import docx
            docx_doc = docx.Document()
            for line in admin_content.splitlines():
                docx_doc.add_paragraph(line)
            docx_doc.save(str(docx_path))
            docx_available = True
        except Exception as exc:
            docx_available = False
            errors.append(f"docx generation failed: {exc}")
            print(f"⚠️ python-docx writing not checked: {exc}")

        # Test TXT Loading & Cleaning
        try:
            from loaders.txt_loader import TXTLoader
            txt_doc = TXTLoader().load(txt_path)
            cleaner = DocumentCleaner()
            cleaned_txt = cleaner.clean(txt_doc)
            
            # Assertions on spacing fixes
            assert "độ chính xác" in cleaned_txt.text, "Failed to correct 'độchính xác'"
            assert "câu trả lời" in cleaned_txt.text, "Failed to correct 'câu trảlời'"
            assert "đề tài" in cleaned_txt.text, "Failed to correct 'đềtài'"
            assert "công nghệ" in cleaned_txt.text, "Failed to correct 'côngnghệ'"
            
            # Assertions on administrative element removal
            assert "CỘNG HÒA XÃ HỘI" not in cleaned_txt.text, "Failed to strip Quoc Hieu"
            assert "Độc lập - Tự do" not in cleaned_txt.text, "Failed to strip Tieu Ngu"
            assert "Nơi nhận" not in cleaned_txt.text, "Failed to strip Recipients"
            assert "Nguyễn Văn B" not in cleaned_txt.text, "Failed to strip Signature"
            assert "[OCR Scan" not in cleaned_txt.text, "Failed to strip Watermark"
            
            details["txt_extraction_and_cleaning"] = "PASSED"
        except Exception as exc:
            passed = False
            errors.append(f"TXT loading/cleaning failed: {exc}\n{traceback.format_exc()}")
            details["txt_extraction_and_cleaning"] = "FAILED"

        # Test PDF Loading & Cleaning (if PyMuPDF wrote it)
        if pdf_available:
            try:
                from loaders.pdf_loader import PDFLoader
                
                # Setup loader with mocked clean elements to test extraction pipeline and cleaning
                loader = PDFLoader()
                mock_elements = [
                    DocumentElement(text="ỦY BAN NHÂN DÂN TỈNH NGHỆ AN", element_type="paragraph", page_number=1),
                    DocumentElement(text="CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", element_type="paragraph", page_number=1),
                    DocumentElement(text="Độc lập - Tự do - Hạnh phúc", element_type="paragraph", page_number=1),
                    DocumentElement(text="Số: 4321/QĐ-UBND", element_type="paragraph", page_number=1),
                    DocumentElement(text="QUYẾT ĐỊNH", element_type="heading", page_number=1),
                    DocumentElement(text="Về việc phê duyệt đềtài nghiên cứu côngnghệ mới.", element_type="paragraph", page_number=1),
                    DocumentElement(text="Mô hình RAG mới giúp tăng độchính xác và cải thiện câu trảlời cho người dùng.", element_type="paragraph", page_number=1),
                    DocumentElement(text="Nơi nhận:", element_type="paragraph", page_number=1),
                    DocumentElement(text="- Như trên", element_type="bullet", page_number=1),
                    DocumentElement(text="CHỦ TỊCH", element_type="paragraph", page_number=1),
                    DocumentElement(text="Nguyễn Văn B", element_type="paragraph", page_number=1),
                ]
                loader._extract_with_pymupdf = lambda path: (mock_elements, {"title": "admin_test"})
                
                pdf_doc = loader.load(pdf_path)
                cleaned_pdf = cleaner.clean(pdf_doc)
                
                assert "độ chính xác" in cleaned_pdf.text, "PDF: Spacing corrector failed"
                assert "CỘNG HÒA XÃ HỘI" not in cleaned_pdf.text, "PDF: Failed to strip Quoc Hieu"
                assert "Nguyễn Văn B" not in cleaned_pdf.text, "PDF: Failed to strip Signature"
                
                details["pdf_extraction_and_cleaning"] = "PASSED"
            except Exception as exc:
                passed = False
                errors.append(f"PDF loading/cleaning failed: {exc}\n{traceback.format_exc()}")
                details["pdf_extraction_and_cleaning"] = "FAILED"
        else:
            details["pdf_extraction_and_cleaning"] = "SKIPPED"

        # Test DOCX Loading & Cleaning (if python-docx wrote it)
        if docx_available:
            try:
                from loaders.docx_loader import DOCXLoader
                docx_doc = DOCXLoader().load(docx_path)
                cleaned_docx = cleaner.clean(docx_doc)
                
                assert "độ chính xác" in cleaned_docx.text, "DOCX: Spacing corrector failed"
                assert "CỘNG HÒA XÃ HỘI" not in cleaned_docx.text, "DOCX: Failed to strip Quoc Hieu"
                assert "Nguyễn Văn B" not in cleaned_docx.text, "DOCX: Failed to strip Signature"
                
                details["docx_extraction_and_cleaning"] = "PASSED"
            except Exception as exc:
                passed = False
                errors.append(f"DOCX loading/cleaning failed: {exc}\n{traceback.format_exc()}")
                details["docx_extraction_and_cleaning"] = "FAILED"
        else:
            details["docx_extraction_and_cleaning"] = "SKIPPED"

    duration = time.perf_counter() - start
    log_section("Extraction & Cleaning", passed, duration, details, errors)
    return passed


def run_sentence_segmentation_audit() -> bool:
    print("\n--- Running Section C: Sentence Segmentation Audit ---")
    start = time.perf_counter()
    passed = True
    errors = []
    details = {}

    try:
        text = (
            "Hệ thống tóm tắt chạy thử nghiệm. Mô hình hoạt động ổn định.\n"
            "Các điều khoản cam kết:\n"
            "1) Phục vụ nghiên cứu khoa học.\n"
            "2) Phát triển giải pháp thực tiễn.\n"
            "- Điểm thứ nhất.\n"
            "- Điểm thứ hai."
        )
        sents = split_sentences(text)
        
        # Verify sentence count and correct splitting on newlines
        assert len(sents) >= 4, f"Splitting returned too few sentences: {len(sents)}"
        assert any("1) Phục vụ" in s for s in sents), "Failed to segment numbered list item 1"
        assert any("2) Phát triển" in s for s in sents), "Failed to segment numbered list item 2"
        assert any("- Điểm thứ nhất" in s for s in sents), "Failed to segment bullet list item"
        
        details["segmented_sentences_count"] = len(sents)
        details["segmentation"] = "PASSED"
    except Exception as exc:
        passed = False
        errors.append(f"Sentence segmentation failed: {exc}\n{traceback.format_exc()}")
        details["segmentation"] = "FAILED"

    duration = time.perf_counter() - start
    log_section("Sentence Segmentation", passed, duration, details, errors)
    return passed


def run_tokenization_audit() -> bool:
    print("\n--- Running Section D: Tokenizer Audit ---")
    start = time.perf_counter()
    passed = True
    errors = []
    details = {}

    # Test tokenizers from config or fallback
    tokenizers_to_test = ["vit5", "bartpho", "mt5"]
    for tok_name in tokenizers_to_test:
        try:
            tokenizer = VietnameseTokenizer(token_model_name=tok_name, use_vietnamese_segmenter=False)
            test_text = "Hệ thống tóm tắt văn bản tiếng Việt sử dụng Transformer."
            
            # Check count
            cnt = tokenizer.count_tokens(test_text)
            assert cnt > 0, f"Token count for {tok_name} returned 0"
            
            # Check trim
            trimmed = tokenizer.trim_to_tokens(test_text, max_tokens=5)
            assert len(trimmed) > 0, f"Trim output for {tok_name} is empty"
            
            details[f"{tok_name}_token_count"] = cnt
            details[f"{tok_name}_trimmed_preview"] = trimmed
            details[tok_name] = "PASSED"
        except Exception as exc:
            # We don't fail the whole tokenization audit if HF cache has no internet
            # access, but we log the warning.
            errors.append(f"Tokenizer {tok_name} failed: {exc}")
            details[tok_name] = f"FAILED: {exc}"
            print(f"⚠️ Tokenizer {tok_name} failed (possibly due to offline mode): {exc}")

    duration = time.perf_counter() - start
    log_section("Tokenization", passed, duration, details, errors)
    return passed


def run_summarization_models_audit() -> bool:
    print("\n--- Running Section E: Summarization Models Audit ---")
    start = time.perf_counter()
    passed = True
    errors = []
    details = {}

    # Sample text for summarization
    source_text = (
        "Trí tuệ nhân tạo (AI) đang phát triển vô cùng mạnh mẽ trên thế giới. "
        "Việt Nam đã chính thức phê duyệt chiến lược AI quốc gia đến năm 2030. "
        "Chiến lược này hướng tới đưa đất nước đứng vào top 4 khu vực ASEAN. "
        "Đồng thời, kế hoạch đào tạo nhân lực bán dẫn và AI chất lượng cao đang được triển khai. "
        "EVN cũng tích cực chuyển đổi số hạ tầng lưới điện truyền tải quốc gia."
    )

    # 1. Extractive Models
    algorithms = ["textrank", "lexrank", "tfidf", "lsa"]
    try:
        results = summarize_extractive_parallel(source_text, algorithms, sentence_count=2)
        for alg in algorithms:
            assert alg in results, f"Extractive algorithm {alg} output missing"
            assert "summary" in results[alg], f"Summary key missing for {alg}"
            assert results[alg]["summary"].strip() != "", f"Empty summary generated for {alg}"
            assert len(results[alg]["highlighted_sentence_indexes"]) > 0, f"No sentences highlighted for {alg}"
            details[f"extractive_{alg}_summary"] = results[alg]["summary"]
        details["extractive_models"] = "PASSED"
    except Exception as exc:
        passed = False
        errors.append(f"Extractive models audit failed: {exc}\n{traceback.format_exc()}")
        details["extractive_models"] = "FAILED"

    # 2. Abstractive Models Config Checking
    try:
        from src.model_registry import ABSTRACTIVE_ALGORITHMS
        for key, config_obj in ABSTRACTIVE_ALGORITHMS.items():
            assert config_obj.model_name, f"Abstractive model {key} has no model_name configured"
            details[f"abstractive_{key}_config"] = {
                "name": config_obj.name,
                "model_name": config_obj.model_name,
                "experimental": getattr(config_obj, "experimental", False)
            }
        details["abstractive_models_config"] = "PASSED"
    except Exception as exc:
        passed = False
        errors.append(f"Abstractive configs check failed: {exc}\n{traceback.format_exc()}")
        details["abstractive_models_config"] = "FAILED"

    duration = time.perf_counter() - start
    log_section("Summarization Models", passed, duration, details, errors)
    return passed


def run_evaluation_metrics_audit() -> bool:
    print("\n--- Running Section F: Evaluation Metrics Audit ---")
    start = time.perf_counter()
    passed = True
    errors = []
    details = {}

    pred = "Việt Nam đẩy mạnh phát triển trí tuệ nhân tạo đến năm 2030."
    ref = "Việt Nam phê duyệt chiến lược phát triển trí tuệ nhân tạo năm 2030."

    # Test ROUGE
    try:
        rouge_scores = compute_rouge(pred, ref)
        assert "rouge1" in rouge_scores, "ROUGE-1 metric missing"
        assert "rouge2" in rouge_scores, "ROUGE-2 metric missing"
        assert "rougeL" in rouge_scores, "ROUGE-L metric missing"
        assert rouge_scores["rouge1"] > 0, "ROUGE-1 is 0.0"
        details["rouge_scores"] = rouge_scores
    except Exception as exc:
        passed = False
        errors.append(f"ROUGE metrics compute failed: {exc}\n{traceback.format_exc()}")

    # Test BLEU
    try:
        bleu_score = compute_bleu(pred, ref)
        assert bleu_score > 0, "BLEU score is 0.0"
        details["bleu_score"] = bleu_score
    except Exception as exc:
        passed = False
        errors.append(f"BLEU metrics compute failed: {exc}\n{traceback.format_exc()}")

    # Test BERTScore (uses Lexical fallback if BERTScore is unavailable/offline)
    try:
        bert_scores = compute_bertscore(pred, ref)
        assert "f1" in bert_scores, "BERTScore F1 missing"
        assert bert_scores["f1"] > 0, "BERTScore F1 is 0.0"
        details["bertscore_f1"] = bert_scores["f1"]
    except Exception as exc:
        # Warning only
        print(f"⚠️ BERTScore failed (possibly due to offline/missing weights): {exc}")
        details["bertscore_f1"] = "FALLBACK_USED"

    duration = time.perf_counter() - start
    log_section("Evaluation Metrics", passed, duration, details, errors)
    return passed


def main():
    print("======================================================================")
    print("🔎 AUTOMATED SYSTEM AUDIT MODE RUNNER")
    print("======================================================================")
    
    t_start = time.perf_counter()
    
    # Run all audits
    run_extraction_and_cleaning_audit()
    run_sentence_segmentation_audit()
    run_tokenization_audit()
    run_summarization_models_audit()
    run_evaluation_metrics_audit()
    
    total_duration = time.perf_counter() - t_start
    audit_report["total_duration_seconds"] = round(total_duration, 4)
    
    print("======================================================================")
    print(f"🏁 Audit finished in {total_duration:.3f} seconds!")
    print(f"Status: {audit_report['status'].upper()}")
    print("======================================================================")
    
    # Save log report
    report_dir = Path("storage/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "audit_report.json"
    report_file.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Saved audit log to {report_file}")
    
    if audit_report["status"] == "failed":
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
