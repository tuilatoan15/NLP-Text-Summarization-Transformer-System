"""Vietnamese administrative document cleaning pipeline."""

from __future__ import annotations

import re
from typing import Any
from utils.logger import logger

# Regex patterns for administrative document detection and cleaning
QUOC_HIEU_RE = re.compile(
    r"CỘNG\s*HÒA\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM", re.IGNORECASE
)
TIEU_NGU_RE = re.compile(
    r"Độc\s*lập\s*[-–—]\s*Tự\s*do\s*[-–—]\s*Hạnh\s*phúc", re.IGNORECASE
)
SO_HIEU_RE = re.compile(
    r"\bSố\s*:\s*\d+(?:/[A-Z0-9-]+)*\b", re.IGNORECASE
)
NGAY_THANG_RE = re.compile(
    r"\bngày\s*[\.\s]*\s*(?:tháng\s*[\.\s]*\s*)?(?:năm\s*[\.\s]*\s*\d*)?\b", re.IGNORECASE
)
CO_QUAN_KEYWORDS = [
    "ỦY BAN NHÂN DÂN", "BỘ", "SỞ", "TỔNG CỤC", "CỤC", "TÒA ÁN NHÂN DÂN",
    "VIỆN KIỂM SÁT", "ĐẢNG CỘNG SẢN", "BAN CHẤP HÀNH", "HỘI ĐỒNG NHÂN DÂN",
    "ỦY BAN MẶT TRẬN", "CÔNG TY", "PHÂN HIỆU", "BỘ MÔN", "TRƯỜNG"
]
CONTACT_KEYWORDS = [
    r"\bĐiện\s+thoại\s*:", r"\bTel\s*:", r"\bFax\s*:", r"\bEmail\s*:", r"\bWebsite\s*:", r"\bHotline\s*:"
]
CONTACT_RE = re.compile("|".join(CONTACT_KEYWORDS), re.IGNORECASE)

PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:trang\s*)?\d+(?:\s*/\s*\d+)?\s*$", re.IGNORECASE
)
PAGE_FOOTER_RE = re.compile(
    r"^\s*Quyết\s+định\s+số\s*:\s*\d+.*$|^\s*Thông\s+tư\s+số\s*:\s*\d+.*$|^\s*Nghị\s+quyết\s+số\s*:\s*\d+.*$",
    re.IGNORECASE
)

MUC_LUC_HEADER_RE = re.compile(r"^\s*MỤC\s+LỤC\s*$", re.IGNORECASE)
MUC_LUC_LINE_RE = re.compile(r"\.{4,}\s*\d+$")

NOI_NHAN_START_RE = re.compile(r"^\s*Nơi\s+nhận\s*:\s*$", re.IGNORECASE)
BULLET_LINE_RE = re.compile(r"^\s*[-+*•●▪▫‣\d+./)]\s*")

SIGNATURE_TITLE_KEYWORDS = [
    "CHỦ TỊCH", "PHÓ CHỦ TỊCH", "BỘ TRƯỞNG", "KT. BỘ TRƯỞNG", "THỨ TRƯỞNG",
    "GIÁM ĐỐC", "PHÓ GIÁM ĐỐC", "TỔNG GIÁM ĐỐC", "ĐÃ KÝ", "KÝ BỞI", "NGÀY KÝ",
    "QUYẾT NGHỊ", "TM. HỘI ĐỒNG", "TM. ỦY BAN", "XÁC NHẬN CỦA", "SINH VIÊN"
]
SIGNATURE_TITLE_RE = re.compile(
    r"^\s*(?:" + "|".join(SIGNATURE_TITLE_KEYWORDS) + r")(?:\s*.*)?$", re.IGNORECASE
)
VI_NAME_RE = re.compile(
    r"^[A-ZÀ-ỸĐ][a-zà-ỹđ]+(?:\s+[A-ZÀ-ỸĐ][a-zà-ỹđ]+){1,4}$"
)

WATERMARK_RE = re.compile(
    r"Văn\s+bản\s+gốc\s+tại|Quét\s+bởi|Printed\s+by|\[OCR.*?\]|Trang\s+\d+\s+bản\s+gốc",
    re.IGNORECASE
)
APPENDIX_RE = re.compile(
    r"^\s*(?:PHỤ\s+LỤC(?:\s+[A-Z0-9]+)?|Phụ\s+lục\s+kèm\s+theo)\s*$", re.IGNORECASE
)

ACADEMIC_METADATA_RE = re.compile(
    r"^\s*(?:Họ\s*tên|MSSV|Lớp|Người\s*hướng\s*dẫn|Tên\s*đề\s*tài|Đề\s*tài)\s*:", re.IGNORECASE
)
SIGNATURE_TEMPLATE_RE = re.compile(
    r"kí\s*,\s*ghi\s*rõ\s*họ\s*tên|\(\s*kí\b", re.IGNORECASE
)


class AdministrativeDocumentCleaner:
    """Detects and cleans administrative metadata/chrome from Vietnamese documents."""

    def __init__(self, clean_enabled: bool = True) -> None:
        self.clean_enabled = clean_enabled

    def is_admin_document(self, text: str) -> bool:
        """Heuristic to detect if a document has Vietnamese administrative or academic report formatting."""
        if not text:
            return False
        
        matches = 0
        if QUOC_HIEU_RE.search(text):
            matches += 1
        if TIEU_NGU_RE.search(text):
            matches += 1
        if SO_HIEU_RE.search(text):
            matches += 1
        if NGAY_THANG_RE.search(text):
            matches += 1
            
        academic_keywords = ["BÁO CÁO", "ĐỒ ÁN", "TỐT NGHIỆP", "MSSV", "LỚP", "HƯỚNG DẪN", "HỌ TÊN", "ĐỀ TÀI"]
        for akw in academic_keywords:
            if re.search(r"\b" + re.escape(akw) + r"\b", text, re.IGNORECASE) or akw in text.upper():
                matches += 1
                break
        
        # Check if type keywords exist
        for kw in ["QUYẾT ĐỊNH", "THÔNG TƯ", "NGHỊ QUYẾT", "CÔNG VĂN", "TỜ TRÌNH", "BÁO CÁO", "ĐỒ ÁN"]:
            if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE) or kw in text.upper():
                matches += 1
                break
                
        return matches >= 2

    def clean(self, text: str) -> str:
        """Runs the document cleaning pipeline on raw text."""
        if not self.clean_enabled or not text:
            return text
            
        if not self.is_admin_document(text):
            return text

        lines = text.splitlines()
        cleaned_lines = []
        in_recipient_block = False
        in_toc_block = False
        
        total_lines = len(lines)
        
        # Iterate and clean line-by-line
        idx = 0
        while idx < total_lines:
            line = lines[idx]
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                cleaned_lines.append("")
                idx += 1
                continue
                
            # 1. OCR watermarks and junk
            if WATERMARK_RE.search(stripped_line):
                idx += 1
                continue
                
            # 2. Page Numbers and Page Footers
            if PAGE_NUMBER_RE.match(stripped_line) or PAGE_FOOTER_RE.match(stripped_line):
                idx += 1
                continue

            # 3. Contact Info
            if CONTACT_RE.search(stripped_line):
                idx += 1
                continue

            # 4. Table of Contents (Mục lục)
            if MUC_LUC_HEADER_RE.match(stripped_line):
                in_toc_block = True
                idx += 1
                continue
            if in_toc_block:
                # If we hit dots leading to page numbers or empty text, skip
                if MUC_LUC_LINE_RE.search(stripped_line):
                    idx += 1
                    continue
                # If a normal long line starts, we exit TOC block
                if len(stripped_line.split()) > 10 and not MUC_LUC_LINE_RE.search(stripped_line):
                    in_toc_block = False

            # 5. Header (Quốc hiệu & Tiêu ngữ, Số hiệu, Cơ quan ban hành, Ngày tháng, Academic metadata)
            # Typically these are in the top 25 lines of the document
            if idx < 25:
                if QUOC_HIEU_RE.search(stripped_line) or TIEU_NGU_RE.search(stripped_line):
                    idx += 1
                    continue
                if SO_HIEU_RE.search(stripped_line) or NGAY_THANG_RE.search(stripped_line):
                    idx += 1
                    continue
                if ACADEMIC_METADATA_RE.search(stripped_line):
                    idx += 1
                    continue
                # Cơ quan ban hành (e.g. BỘ TÀI CHÍNH, ỦY BAN NHÂN DÂN TỈNH X)
                is_co_quan = False
                for keyword in CO_QUAN_KEYWORDS:
                    if stripped_line.upper().startswith(keyword):
                        is_co_quan = True
                        break
                if is_co_quan and len(stripped_line.split()) < 15:
                    idx += 1
                    continue

            # 6. Recipients block ("Nơi nhận:") at the end of document
            # Usually starts with "Nơi nhận:"
            if NOI_NHAN_START_RE.match(stripped_line):
                in_recipient_block = True
                idx += 1
                continue
                
            if in_recipient_block:
                # Inside recipients block, skip list items
                if BULLET_LINE_RE.match(stripped_line) or len(stripped_line.split()) < 5:
                    idx += 1
                    continue
                else:
                    in_recipient_block = False

            # 7. Signature blocks / Sign-off names at the bottom (last 30% of lines)
            if idx > int(total_lines * 0.65):
                # Signature template (kí, ghi rõ họtên...)
                if SIGNATURE_TEMPLATE_RE.search(stripped_line):
                    idx += 1
                    continue
                # If the line matches signature title (CHỦ TỊCH, KT. BỘ TRƯỜNG...)
                if SIGNATURE_TITLE_RE.match(stripped_line):
                    idx += 1
                    # Skip subsequent personal names following the signature title immediately
                    while idx < total_lines:
                        next_line = lines[idx].strip()
                        if not next_line:
                            idx += 1
                            continue
                        if VI_NAME_RE.match(next_line) and len(next_line.split()) <= 4:
                            idx += 1
                            break
                        break
                    continue
                    
                # Direct check for name if it's standalone in a line at the very end
                if idx > int(total_lines * 0.8) and VI_NAME_RE.match(stripped_line) and len(stripped_line.split()) <= 4:
                    idx += 1
                    continue

            # 8. Independent Appendix Headers
            if APPENDIX_RE.match(stripped_line):
                idx += 1
                continue

            cleaned_lines.append(line)
            idx += 1

        # Join lines and merge excessive newlines
        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        return cleaned_text.strip()

    def clean_elements(self, elements: list[Any], full_text: str) -> list[Any]:
        """Cleans administrative metadata from a list of DocumentElements."""
        if not self.clean_enabled or not elements:
            return elements
            
        if not self.is_admin_document(full_text):
            return elements

        cleaned: list[Any] = []
        in_recipient_block = False
        in_toc_block = False
        total_elements = len(elements)
        
        idx = 0
        while idx < total_elements:
            element = elements[idx]
            text = element.text.strip()
            
            if not text:
                idx += 1
                continue
                
            # 1. OCR watermarks and junk
            if WATERMARK_RE.search(text):
                idx += 1
                continue
                
            # 2. Page Numbers and Page Footers
            if PAGE_NUMBER_RE.match(text) or PAGE_FOOTER_RE.match(text):
                idx += 1
                continue

            # 3. Contact Info
            if CONTACT_RE.search(text):
                idx += 1
                continue

            # 4. Table of Contents (Mục lục)
            if MUC_LUC_HEADER_RE.match(text):
                in_toc_block = True
                idx += 1
                continue
            if in_toc_block:
                if MUC_LUC_LINE_RE.search(text):
                    idx += 1
                    continue
                if len(text.split()) > 10 and not MUC_LUC_LINE_RE.search(text):
                    in_toc_block = False

            # 5. Header (Quốc hiệu & Tiêu ngữ, Số hiệu, Cơ quan ban hành, Ngày tháng, Academic metadata)
            # Typically in the first 18 elements
            if idx < 18:
                if QUOC_HIEU_RE.search(text) or TIEU_NGU_RE.search(text):
                    idx += 1
                    continue
                if SO_HIEU_RE.search(text) or NGAY_THANG_RE.search(text):
                    idx += 1
                    continue
                if ACADEMIC_METADATA_RE.search(text):
                    idx += 1
                    continue
                is_co_quan = False
                for keyword in CO_QUAN_KEYWORDS:
                    if text.upper().startswith(keyword):
                        is_co_quan = True
                        break
                if is_co_quan and len(text.split()) < 15:
                    idx += 1
                    continue

            # 6. Recipients block ("Nơi nhận:") at the end of document
            if NOI_NHAN_START_RE.match(text):
                in_recipient_block = True
                idx += 1
                continue
                
            if in_recipient_block:
                if BULLET_LINE_RE.match(text) or len(text.split()) < 5:
                    idx += 1
                    continue
                else:
                    in_recipient_block = False

            # 7. Signature blocks / Sign-off names at the bottom (last 35% of elements)
            if idx > int(total_elements * 0.65):
                # Signature templates
                if SIGNATURE_TEMPLATE_RE.search(text):
                    idx += 1
                    continue
                if SIGNATURE_TITLE_RE.match(text):
                    idx += 1
                    while idx < total_elements:
                        next_el = elements[idx]
                        next_text = next_el.text.strip()
                        if not next_text:
                            idx += 1
                            continue
                        if VI_NAME_RE.match(next_text) and len(next_text.split()) <= 4:
                            idx += 1
                            break
                        break
                    continue
                if idx > int(total_elements * 0.8) and VI_NAME_RE.match(text) and len(text.split()) <= 4:
                    idx += 1
                    continue

            # 8. Appendix Headers
            if APPENDIX_RE.match(text):
                idx += 1
                continue

            cleaned.append(element)
            idx += 1
            
        return cleaned
