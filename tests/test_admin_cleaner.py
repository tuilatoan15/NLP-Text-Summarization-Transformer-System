"""Unit tests for AdministrativeDocumentCleaner."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess.admin_cleaner import AdministrativeDocumentCleaner
from pipeline.schema import DocumentElement


class TestAdministrativeDocumentCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = AdministrativeDocumentCleaner(clean_enabled=True)

    def test_is_admin_document_detection(self):
        # Admin document containing Quoc Hieu and Tieu Ngu
        admin_text = (
            "ỦY BAN NHÂN DÂN TỈNH NGHỆ AN\n"
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n"
            "Số: 1234/QĐ-UBND\n"
            "QUYẾT ĐỊNH Về việc phê duyệt dự án..."
        )
        self.assertTrue(self.cleaner.is_admin_document(admin_text))

        # Non-admin text (regular news article)
        news_text = (
            "Hôm nay, thời tiết tại Hà Nội rất đẹp. Người dân đổ xô ra đường vui chơi. "
            "Các tuyến phố trung tâm đông đúc xe cộ đi lại. Dự báo ngày mai thời tiết tiếp tục có nắng ấm."
        )
        self.assertFalse(self.cleaner.is_admin_document(news_text))

    def test_clean_quoc_hieu_tieu_ngu(self):
        dirty = (
            "ỦY BAN NHÂN DÂN TỈNH NGHỆ AN\n"
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n"
            "Số: 1234/QĐ-UBND\n"
            "Nghệ An, ngày 10 tháng 06 năm 2026\n\n"
            "QUYẾT ĐỊNH\n"
            "Phê duyệt dự án xây dựng..."
        )
        cleaned = self.cleaner.clean(dirty)
        self.assertNotIn("CỘNG HÒA XÃ HỘI", cleaned)
        self.assertNotIn("Độc lập - Tự do", cleaned)
        self.assertNotIn("Số: 1234/QĐ-UBND", cleaned)
        self.assertNotIn("Nghệ An, ngày 10", cleaned)
        self.assertIn("QUYẾT ĐỊNH", cleaned)

    def test_clean_noi_nhan_signature(self):
        dirty = (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n\n"
            "Nội dung chính quyết định phê duyệt dự án.\n\n"
            "Nơi nhận:\n"
            "- Như trên;\n"
            "- Bộ Giao thông;\n"
            "- Lưu VT.\n"
            "CHỦ TỊCH\n"
            "Nguyễn Văn A\n"
            "Hotline: 0238.1234567\n"
            "Trang 1/1\n"
            "[OCR Scan Success]"
        )
        cleaned = self.cleaner.clean(dirty)
        self.assertIn("Nội dung chính quyết định phê duyệt dự án.", cleaned)
        self.assertNotIn("Nơi nhận", cleaned)
        self.assertNotIn("- Như trên", cleaned)
        self.assertNotIn("CHỦ TỊCH", cleaned)
        self.assertNotIn("Nguyễn Văn A", cleaned)
        self.assertNotIn("Hotline", cleaned)
        self.assertNotIn("[OCR Scan Success]", cleaned)

    def test_clean_elements_method(self):
        elements = [
            DocumentElement(text="ỦY BAN NHÂN DÂN TỈNH NGHỆ AN", element_type="paragraph"),
            DocumentElement(text="CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", element_type="paragraph"),
            DocumentElement(text="Độc lập - Tự do - Hạnh phúc", element_type="paragraph"),
            DocumentElement(text="Nội dung chính.", element_type="paragraph"),
            DocumentElement(text="Nơi nhận:", element_type="paragraph"),
            DocumentElement(text="- Như trên", element_type="bullet"),
            DocumentElement(text="CHỦ TỊCH", element_type="paragraph"),
            DocumentElement(text="Nguyễn Văn A", element_type="paragraph"),
        ]
        full_text = "\n".join(e.text for e in elements)
        cleaned_elements = self.cleaner.clean_elements(elements, full_text)
        
        # Verify the only remaining element is the main content
        self.assertEqual(len(cleaned_elements), 1)
        self.assertEqual(cleaned_elements[0].text, "Nội dung chính.")

    def test_vietnamese_ocr_spacing_and_academic_report_cleaning(self):
        dirty = (
            "PHÂN HIỆU TRƯỜNG ĐẠI HỌC GTVT CỘNG HÒA XÃ HỘI CHỦNGHĨA VIỆT NAM\n"
            "BỘMÔN CÔNG NGHỆTHÔNG TIN Độc lập - Tựdo - Hạnh phúc\n"
            "BÁO CÁO TIẾN ĐỘTHỰC HIỆN ĐỒÁN TỐT NGHIỆP\n"
            "Họtên: Nguyễn Hữu Toàn\n"
            "MSSV: 6351071071\n"
            "Lớp: CQ.63.CNTT\n"
            "Tên đềtài: Xây dựng hệthống tóm tắt văn bản tựđộng sửdụng xửlý ngôn ngữtự\n"
            "nhiên (NLP) và mô hình Transformer.\n"
            "Người hướng dẫn: Th. S Trần Phong Nhã\n\n"
            "Nội dung thực hiện:\n"
            "1) Xử lý và nạp tài liệu tiếng Việt dài.\n"
            "2) Xây dựng hệ thống hỏi đáp bằng RAG.\n\n"
            "TP. Hồ Chí Minh, ngày... tháng... năm 20...\n"
            "XÁC NHẬN CỦA CÁN BỘHƯỚNG DẪN SINH VIÊN\n"
            "(kí, ghi rõ họtên) (kí, ghi rõ họtên)"
        )
        
        # Clean text
        cleaned = self.cleaner.clean(dirty)
        
        # Verify that OCR spacing corrector ran and administrative block was stripped
        self.assertNotIn("PHÂN HIỆU TRƯỜNG ĐẠI HỌC GTVT", cleaned)
        self.assertNotIn("CỘNG HÒA XÃ HỘI", cleaned)
        self.assertNotIn("Độc lập - Tự do", cleaned)
        self.assertNotIn("Họtên:", cleaned)
        self.assertNotIn("MSSV:", cleaned)
        self.assertNotIn("Lớp:", cleaned)
        self.assertNotIn("Người hướng dẫn:", cleaned)
        self.assertNotIn("XÁC NHẬN CỦA", cleaned)
        self.assertNotIn("(kí, ghi rõ họtên)", cleaned)
        
        # Verify main content is kept
        self.assertIn("Nội dung thực hiện", cleaned)
        self.assertIn("1) Xử lý và nạp tài liệu tiếng Việt dài.", cleaned)

    def test_vietnamese_text_normalizer_spacing_fixes(self):
        # Admin text with glued words to ensure it triggers is_admin_document detection
        dirty = (
            "ỦY BAN NHÂN DÂN TỈNH NGHỆ AN\n"
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n\n"
            "Mô hình RAG giúp tăng độchính xác và đưa ra câu trảlời nhanh chóng. "
            "Chúng tôi thực hiện nghiêncứu và pháttriển dựán tốtnghiệp về côngnghệ nghệthông tin."
        )
        cleaned = self.cleaner.clean(dirty)
        self.assertIn("độ chính xác", cleaned)
        self.assertIn("câu trả lời", cleaned)
        self.assertIn("nghiên cứu", cleaned)
        self.assertIn("phát triển", cleaned)
        self.assertIn("dự án", cleaned)
        self.assertIn("tốt nghiệp", cleaned)
        self.assertIn("công nghệ", cleaned)
        self.assertIn("nghệ thông", cleaned)


if __name__ == "__main__":
    unittest.main()
