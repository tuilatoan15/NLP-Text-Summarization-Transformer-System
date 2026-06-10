"""Benchmark script for Vietnamese administrative document cleaning pipeline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.admin_cleaner import AdministrativeDocumentCleaner
from evaluation.metrics import compute_rouge, compute_bertscore
from src.dashboard import summarize_all
from src.utils import count_words

# Define sample Vietnamese administrative documents (dirty version, clean version, and gold reference summary)
SAMPLE_DOCUMENTS = [
    {
        "id": "QD-1234",
        "title": "Quyết định phê duyệt dự án xây dựng cầu vượt sông Lam",
        "dirty": (
            "ỦY BAN NHÂN DÂN               CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            " TỈNH NGHỆ AN                     Độc lập - Tự do - Hạnh phúc\n"
            "Số: 1234/QĐ-UBND                  Nghệ An, ngày 10 tháng 06 năm 2026\n\n"
            "                              QUYẾT ĐỊNH\n"
            "               Về việc phê duyệt dự án xây dựng cầu vượt sông Lam\n\n"
            "Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015;\n"
            "Căn cứ Luật Đầu tư công ngày 13 tháng 6 năm 2019;\n"
            "Xét đề nghị của Sở Giao thông vận tải tỉnh Nghệ An tại Tờ trình số 567/TTr-SGTVT ngày 25 tháng 5 năm 2026.\n\n"
            "                              QUYẾT ĐỊNH:\n\n"
            "Điều 1. Phê duyệt dự án đầu tư xây dựng cầu vượt sông Lam kết nối hai bờ huyện Nam Đàn và huyện Hưng Nguyên. "
            "Tổng mức đầu tư của dự án là 500 tỷ đồng, sử dụng nguồn vốn ngân sách trung ương và địa phương. "
            "Mục tiêu của dự án là hoàn thiện hệ thống hạ tầng giao thông kết nối liên vùng, hỗ trợ phát triển kinh tế xã hội và rút ngắn thời gian di chuyển của người dân.\n\n"
            "Điều 2. Thời gian thực hiện dự án từ năm 2026 đến năm 2029. Chủ đầu tư dự án là Sở Giao thông vận tải tỉnh Nghệ An.\n\n"
            "Điều 3. Chánh Văn phòng UBND tỉnh, Giám đốc các Sở: Giao thông vận tải, Tài chính, Kế hoạch và Đầu tư và các cơ quan liên quan chịu trách nhiệm thi hành Quyết định này.\n\n"
            "Nơi nhận:                               CHỦ TỊCH\n"
            "- Như Điều 3;\n"
            "- Bộ Giao thông vận tải (để báo cáo);\n"
            "- Văn phòng Chính phủ;\n"
            "- Lưu: VT, TH.\n"
            "                                      Nguyễn Văn A\n\n"
            "--------------------------------------------------\n"
            "Hotline: 0238.1234567 - Email: ubnd@nghean.gov.vn\n"
            "[OCR Watermark - Page 1 of 1]\n"
        ),
        "clean_gold": (
            "QUYẾT ĐỊNH\n"
            "Về việc phê duyệt dự án xây dựng cầu vượt sông Lam\n\n"
            "Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015;\n"
            "Căn cứ Luật Đầu tư công ngày 13 tháng 6 năm 2019;\n"
            "Xét đề nghị của Sở Giao thông vận tải tỉnh Nghệ An tại Tờ trình số 567/TTr-SGTVT ngày 25 tháng 5 năm 2026.\n\n"
            "QUYẾT ĐỊNH:\n\n"
            "Điều 1. Phê duyệt dự án đầu tư xây dựng cầu vượt sông Lam kết nối hai bờ huyện Nam Đàn và huyện Hưng Nguyên. "
            "Tổng mức đầu tư của dự án là 500 tỷ đồng, sử dụng nguồn vốn ngân sách trung ương và địa phương. "
            "Mục tiêu của dự án là hoàn thiện hệ thống hạ tầng giao thông kết nối liên vùng, hỗ trợ phát triển kinh tế xã hội và rút ngắn thời gian di chuyển của người dân.\n\n"
            "Điều 2. Thời gian thực hiện dự án từ năm 2026 đến năm 2029. Chủ đầu tư dự án là Sở Giao thông vận tải tỉnh Nghệ An.\n\n"
            "Điều 3. Chánh Văn phòng UBND tỉnh, Giám đốc các Sở: Giao thông vận tải, Tài chính, Kế hoạch và Đầu tư và các cơ quan liên quan chịu trách nhiệm thi hành Quyết định này."
        ),
        "reference": (
            "Ủy ban nhân dân tỉnh Nghệ An phê duyệt dự án đầu tư xây dựng cầu vượt sông Lam nối huyện Nam Đàn và huyện Hưng Nguyên. "
            "Dự án có tổng mức đầu tư 500 tỷ đồng bằng ngân sách trung ương và địa phương, do Sở Giao thông vận tải làm chủ đầu tư, triển khai từ 2026 đến 2029 nhằm hoàn thiện giao thông liên vùng."
        )
    },
    {
        "id": "NQ-5678",
        "title": "Nghị quyết về phát triển kinh tế số tỉnh Quảng Ninh",
        "dirty": (
            "HỘI ĐỒNG NHÂN DÂN               CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            " TỈNH QUẢNG NINH                    Độc lập - Tự do - Hạnh phúc\n"
            "Số: 5678/NQ-HĐND                  Quảng Ninh, ngày 12 tháng 05 năm 2026\n\n"
            "                              NGHỊ QUYẾT\n"
            "               Về chương trình phát triển kinh tế số tỉnh Quảng Ninh đến năm 2030\n\n"
            "Hội đồng nhân dân tỉnh Quảng Ninh khóa XIV, kỳ họp thứ 12 thông qua:\n\n"
            "Quyết nghị:\n\n"
            "Mục 1. Mục tiêu tổng quát là thúc đẩy mạnh mẽ chuyển đổi số toàn diện, trọng tâm là phát triển kinh tế số chiếm 30% GRDP vào năm 2030. "
            "Tập trung đầu tư hạ tầng mạng 5G và băng rộng cáp quang phủ sóng toàn bộ vùng sâu, vùng xa, biên giới và hải đảo. "
            "Phát triển 5.000 doanh nghiệp công nghệ số trên địa bàn tỉnh Quảng Ninh.\n\n"
            "Mục 2. Nhiệm vụ và giải pháp bao gồm hỗ trợ đào tạo nhân lực công nghệ thông tin cho doanh nghiệp vừa và nhỏ, "
            "số hóa thủ tục hành chính công đạt mức độ 4 ở tất cả các sở ngành và địa phương cấp huyện.\n\n"
            "Mục 3. Ủy ban nhân dân tỉnh Quảng Ninh chịu trách nhiệm tổ chức triển khai thực hiện Nghị quyết này.\n\n"
            "Nơi nhận:                               CHỦ TỊCH HĐND\n"
            "- Thường trực HĐND tỉnh;\n"
            "- UBND tỉnh;\n"
            "- Các sở, ban, ngành;\n"
            "- Lưu: VT, VP.\n"
            "                                      Trần Văn B\n\n"
            "--------------------------------------------------\n"
            "Trang 1/1\n"
            "Printed by HĐND Quảng Ninh - [OCR Scan Success]\n"
        ),
        "clean_gold": (
            "NGHỊ QUYẾT\n"
            "Về chương trình phát triển kinh tế số tỉnh Quảng Ninh đến năm 2030\n\n"
            "Hội đồng nhân dân tỉnh Quảng Ninh khóa XIV, kỳ họp thứ 12 thông qua:\n\n"
            "Quyết nghị:\n\n"
            "Mục 1. Mục tiêu tổng quát là thúc đẩy mạnh mẽ chuyển đổi số toàn diện, trọng tâm là phát triển kinh tế số chiếm 30% GRDP vào năm 2030. "
            "Tập trung đầu tư hạ tầng mạng 5G và băng rộng cáp quang phủ sóng toàn bộ vùng sâu, vùng xa, biên giới và hải đảo. "
            "Phát triển 5.000 doanh nghiệp công nghệ số trên địa bàn tỉnh Quảng Ninh.\n\n"
            "Mục 2. Nhiệm vụ và giải pháp bao gồm hỗ trợ đào tạo nhân lực công nghệ thông tin cho doanh nghiệp vừa và nhỏ, "
            "số hóa thủ tục hành chính công đạt mức độ 4 ở tất cả các sở ngành và địa phương cấp huyện.\n\n"
            "Mục 3. Ủy ban nhân dân tỉnh Quảng Ninh chịu trách nhiệm tổ chức triển khai thực hiện Nghị quyết này."
        ),
        "reference": (
            "Nghị quyết của Hội đồng nhân dân tỉnh Quảng Ninh quyết định chương trình phát triển kinh tế số đến năm 2030 với mục tiêu kinh tế số chiếm 30% GRDP. "
            "Các giải pháp chính bao gồm phủ sóng mạng 5G, phát triển 5.000 doanh nghiệp số, số hóa hoàn toàn dịch vụ công cấp tỉnh và huyện."
        )
    },
    {
        "id": "BC-02",
        "title": "Báo cáo tiến độ thực hiện đồ án tốt nghiệp - Nguyễn Hữu Toàn",
        "dirty": (
            "PHÂN HIỆU TRƯỜNG ĐẠI HỌC GTVT CỘNG HÒA XÃ HỘI CHỦNGHĨA VIỆT NAM\n"
            "BỘMÔN CÔNG NGHỆTHÔNG TIN Độc lập - Tựdo - Hạnh phúc\n"
            "BÁO CÁO TIẾN ĐỘTHỰC HIỆN ĐỒÁN TỐT NGHIỆP\n"
            "Họtên: Nguyễn Hữu Toàn\n"
            "MSSV: 6351071071\n"
            "Lớp: CQ.63.CNTT\n"
            "Tên đềtài: Xây dựng hệthống tóm tắt văn bản tựđộng sửdụng xửlý ngôn ngữtự\n"
            "nhiên (NLP) và mô hình Transformer.\n"
            "Người hướng dẫn: Th. S Trần Phong Nhã\n"
            "Nội dung thực hiện: 1) Xửlý và nạp tài liệu tiếng Việt, đặc biệt là file PDF dung lượng lớn, bằng cách làm\n"
            "sạch văn bản, chia nhỏnội dung đểhệthống chạy ổn định trên máy cấu hình yếu.\n"
            "2) Xây dựng hệthống hỏi đáp tài liệu bằng mô hình Hybrid RAG, kết hợp tìm kiếm\n"
            "ngữnghĩa, tìm kiếm từkhóa và reranking đểlấy đúng đoạn văn bản liên quan trước\n"
            "khi sinh câu trảlời.\n"
            "3) Nghiên cứu module tóm tắt văn bản như tóm tắt phân cấp, tóm tắt lựa chọn và\n"
            "pipeline tóm tắt lai, kết hợp extractive và abstractive đểtạo bản tóm tắt ngắn gọn, tự\n"
            "nhiên hơn.\n"
            "4) Chuẩn bịdữliệu và công cụphục vụhuấn luyện mô hình, gồm kiểm tra tokenizer, danh sách từvựng, script train và notebook chạy thửtrên Colab.\n"
            "Kết quảđạt được: 1) Hệthống đã xửlý được tài liệu tiếng Việt dài, giảm lỗi tràn bộnhớvà hạn chếmất\n"
            "ngữcảnh khi chia nhỏvăn bản.\n"
            "2) Chức năng chatbot hỏi đáp theo tài liệu đã hoạt động, có thểtruy xuất nội dung liên\n"
            "quan và trảlời dựa trên tài liệu được nạp.\n"
            "3) Mô hình RAG đã được tích hợp theo hướng Hybrid RAG giúp tăng độchính xác\n"
            "khi tìm kiếm, hỏi đáp và tóm tắt tài liệu.\n"
            "TP. HồChí Minh, ngày... tháng... năm 20...\n"
            "XÁC NHẬN CỦA CÁN BỘHƯỚNG DẪN SINH VIÊN\n"
            "(kí, ghi rõ họtên) (kí, ghi rõ họtên)\n"
        ),
        "clean_gold": (
            "Nội dung thực hiện:\n"
            "1) Xử lý và nạp tài liệu tiếng Việt, đặc biệt là file PDF dung lượng lớn, bằng cách làm sạch văn bản, chia nhỏ nội dung để hệ thống chạy ổn định trên máy cấu hình yếu.\n"
            "2) Xây dựng hệ thống hỏi đáp tài liệu bằng mô hình Hybrid RAG, kết hợp tìm kiếm ngữ nghĩa, tìm kiếm từ khóa và reranking để lấy đúng đoạn văn bản liên quan trước khi sinh câu trả lời.\n"
            "3) Nghiên cứu module tóm tắt văn bản như tóm tắt phân cấp, tóm tắt lựa chọn và pipeline tóm tắt lai, kết hợp extractive và abstractive để tạo bản tóm tắt ngắn gọn, tự nhiên hơn.\n"
            "4) Chuẩn bị dữ liệu và công cụ phục vụ huấn luyện mô hình, gồm kiểm tra tokenizer, danh sách từ vựng, script train và notebook chạy thử trên Colab.\n\n"
            "Kết quả đạt được:\n"
            "1) Hệ thống đã xử lý được tài liệu tiếng Việt dài, giảm lỗi tràn bộ nhớ và hạn chế mất ngữ cảnh khi chia nhỏ văn bản.\n"
            "2) Chức năng chatbot hỏi đáp theo tài liệu đã hoạt động, có thể truy xuất nội dung liên quan và trả lời dựa trên tài liệu được nạp.\n"
            "3) Mô hình RAG đã được tích hợp theo hướng Hybrid RAG giúp tăng độ chính xác khi tìm kiếm, hỏi đáp và tóm tắt tài liệu."
        ),
        "reference": (
            "Báo cáo tiến độ của Nguyễn Hữu Toàn về đồ án xây dựng hệ thống tóm tắt văn bản tự động bằng NLP và Transformer. "
            "Các nội dung chính đã thực hiện bao gồm làm sạch văn bản tiếng Việt từ PDF, xây dựng chatbot Hybrid RAG hỏi đáp tài liệu, nghiên cứu module tóm tắt lai và chuẩn bị script huấn luyện mô hình."
        )
    }
]


def run_benchmark():
    print("=" * 70)
    print("  BAT DAU CHAY BENCHMARK ADMINISTRATIVE DOCUMENT CLEANING  ")
    print("=" * 70)
    print()

    cleaner = AdministrativeDocumentCleaner(clean_enabled=True)
    results = []

    # Select model for summarization
    model_name = "vit5" # ViT5 is excellent for Vietnamese

    for doc in SAMPLE_DOCUMENTS:
        doc_id = doc["id"]
        title = doc["title"]
        dirty_text = doc["dirty"]
        gold_ref = doc["reference"]

        print(f"[*] Processing document {doc_id}: {title}...")
        
        # 1. Clean the text using the cleaner
        t_clean_start = time.perf_counter()
        cleaned_text = cleaner.clean(dirty_text)
        t_clean_dur = time.perf_counter() - t_clean_start

        # 2. Words counts
        words_dirty = count_words(dirty_text)
        words_cleaned = count_words(cleaned_text)
        noise_reduction_ratio = (1 - (words_cleaned / words_dirty)) * 100

        print(f"   - Từ trước: {words_dirty} | Từ sau: {words_cleaned} | Giảm nhiễu: {noise_reduction_ratio:.2f}%")

        # 3. Generate summaries
        # Without cleaning (summarize dirty)
        print("   - Sinh tóm tắt cho văn bản CHƯA làm sạch...")
        sum_dirty_payload = summarize_all(dirty_text, reference=gold_ref, algorithms=[model_name], use_length_ratio=False)
        summary_dirty = next((r["summary"] for r in sum_dirty_payload["results"] if r["key"] == model_name), "")

        # With cleaning (summarize clean)
        print("   - Sinh tóm tắt cho văn bản ĐÃ làm sạch...")
        sum_cleaned_payload = summarize_all(cleaned_text, reference=gold_ref, algorithms=[model_name], use_length_ratio=False)
        summary_cleaned = next((r["summary"] for r in sum_cleaned_payload["results"] if r["key"] == model_name), "")

        # 4. Evaluate summaries against gold reference
        rouge_dirty = compute_rouge(summary_dirty, gold_ref)
        bert_dirty = compute_bertscore(summary_dirty, gold_ref)

        rouge_cleaned = compute_rouge(summary_cleaned, gold_ref)
        bert_cleaned = compute_bertscore(summary_cleaned, gold_ref)

        doc_result = {
            "document_id": doc_id,
            "title": title,
            "words_before": words_dirty,
            "words_after": words_cleaned,
            "noise_reduction_pct": round(noise_reduction_ratio, 2),
            "cleaning_time_s": round(t_clean_dur, 6),
            "without_cleaning": {
                "summary": summary_dirty,
                "rouge1": rouge_dirty["rouge1"],
                "rouge2": rouge_dirty["rouge2"],
                "rougeL": rouge_dirty["rougeL"],
                "bertscore_f1": bert_dirty["f1"]
            },
            "with_cleaning": {
                "summary": summary_cleaned,
                "rouge1": rouge_cleaned["rouge1"],
                "rouge2": rouge_cleaned["rouge2"],
                "rougeL": rouge_cleaned["rougeL"],
                "bertscore_f1": bert_cleaned["f1"]
            }
        }
        results.append(doc_result)
        print(f"   [OK] ROUGE-L: {rouge_dirty['rougeL']:.4f} -> {rouge_cleaned['rougeL']:.4f} | BERTScore: {bert_dirty['f1']:.4f} -> {bert_cleaned['f1']:.4f}")
        print("-" * 50)

    # Compute averages
    avg_reduction = sum(r["noise_reduction_pct"] for r in results) / len(results)
    avg_rl_before = sum(r["without_cleaning"]["rougeL"] for r in results) / len(results)
    avg_rl_after = sum(r["with_cleaning"]["rougeL"] for r in results) / len(results)
    avg_bs_before = sum(r["without_cleaning"]["bertscore_f1"] for r in results) / len(results)
    avg_bs_after = sum(r["with_cleaning"]["bertscore_f1"] for r in results) / len(results)

    summary_report = {
        "results": results,
        "averages": {
            "noise_reduction_pct": round(avg_reduction, 2),
            "rougeL_before": round(avg_rl_before, 4),
            "rougeL_after": round(avg_rl_after, 4),
            "bertscore_before": round(avg_bs_before, 4),
            "bertscore_after": round(avg_bs_after, 4)
        }
    }

    # Save to JSON
    output_dir = Path("storage/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "admin_cleaning_benchmark.json"
    output_file.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Print Table Report
    print("\n" + "=" * 76)
    print("  KẾT QUẢ BENCHMARK LÀM SẠCH VĂN BẢN HÀNH CHÍNH VIỆT NAM  ")
    print("=" * 76)
    header = f"{'Mã tài liệu':<12} {'Từ Trước':<10} {'Từ Sau':<10} {'Giảm Nhiễu':<12} {'ROUGE-L (B/A)':<15} {'BERTScore (B/A)'}"
    print(header)
    print("-" * 76)
    for r in results:
        b_rl = r["without_cleaning"]["rougeL"]
        a_rl = r["with_cleaning"]["rougeL"]
        b_bs = r["without_cleaning"]["bertscore_f1"]
        a_bs = r["with_cleaning"]["bertscore_f1"]
        print(
            f"{r['document_id']:<12} {r['words_before']:<10} {r['words_after']:<10} {r['noise_reduction_pct']:.2f}%     "
            f"{b_rl:.3f} -> {a_rl:.3f}   {b_bs:.3f} -> {a_bs:.3f}"
        )
    print("-" * 76)
    print(
        f"{'TRUNG BÌNH':<12} {'-':<10} {'-':<10} {avg_reduction:.2f}%     "
        f"{avg_rl_before:.3f} -> {avg_rl_after:.3f}   {avg_bs_before:.3f} -> {avg_bs_after:.3f}"
    )
    print("=" * 76)
    print(f"\n[OK] File báo cáo chi tiết được lưu tại: {output_file.resolve()}\n")


if __name__ == "__main__":
    run_benchmark()
