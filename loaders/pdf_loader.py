"""Production-oriented PDF loader with PyMuPDF, pdfplumber, unstructured, and OCR fallback."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from loaders.ocr_loader import OCRLoader
from pipeline.schema import DocumentElement, DocumentMetadata, ExtractionConfig, ExtractedDocument
from preprocess.cleaner import normalize_unicode
from utils.logger import logger
from utils.metrics import extraction_quality_score


BULLET_RE = re.compile(r"^\s*(?:[-*+•●▪▫‣]|\(?\d{1,3}[.)]|[a-zA-Z][.)])\s+")


class PDFLoader:
    """Extract structured text from PDFs while preserving page and section context."""

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or ExtractionConfig()

    def load(self, path: str | Path) -> ExtractedDocument:
        file_path = Path(path)
        warnings: list[str] = []
        engines_tried: list[str] = []

        elements, metadata_extra = self._extract_with_pymupdf(file_path)
        engines_tried.append("pymupdf")
        text = self._join_elements(elements)
        quality = extraction_quality_score(text)
        scanned = self._looks_scanned(elements, quality)

        if (not text.strip() or float(quality["score"]) < 0.30) and "pdfplumber" in self.config.pdf_fallback_engines:
            fallback_elements = self._extract_with_pdfplumber(file_path)
            engines_tried.append("pdfplumber")
            fallback_text = self._join_elements(fallback_elements)
            fallback_quality = extraction_quality_score(fallback_text)
            if float(fallback_quality["score"]) > float(quality["score"]):
                elements = fallback_elements
                text = fallback_text
                quality = fallback_quality
                warnings.append("Used pdfplumber fallback because PyMuPDF extraction quality was low.")

        if (not text.strip() or float(quality["score"]) < 0.25) and "unstructured" in self.config.pdf_fallback_engines:
            fallback_elements = self._extract_with_unstructured(file_path)
            engines_tried.append("unstructured")
            fallback_text = self._join_elements(fallback_elements)
            fallback_quality = extraction_quality_score(fallback_text)
            if float(fallback_quality["score"]) > float(quality["score"]):
                elements = fallback_elements
                text = fallback_text
                quality = fallback_quality
                warnings.append("Used unstructured fallback because parser extraction quality was low.")

        scanned = scanned or self._looks_scanned(elements, quality)
        if self.config.enable_ocr and scanned:
            ocr_elements = OCRLoader(self.config.ocr_languages).ocr_pdf_pages(file_path)
            engines_tried.append("ocr")
            ocr_text = self._join_elements(ocr_elements)
            ocr_quality = extraction_quality_score(ocr_text)
            if float(ocr_quality["score"]) > float(quality["score"]):
                elements = ocr_elements
                text = ocr_text
                quality = ocr_quality
                warnings.append("Used OCR fallback because the PDF appears scanned or image-only.")
            elif not ocr_elements:
                warnings.append("PDF appears scanned, but no OCR provider returned usable text.")

        metadata = self._metadata(file_path, metadata_extra)
        metadata.extraction_engine = "+".join(engines_tried)
        metadata.is_scanned = scanned
        metadata.quality_score = float(quality["score"])
        metadata.language = str(quality["language"])
        structure = self._build_structure(elements)
        return ExtractedDocument(
            document_id=self._document_id(file_path, text),
            metadata=metadata,
            text=text,
            elements=elements,
            structure=structure,
            warnings=warnings,
        )

    def _extract_with_pymupdf(self, path: Path) -> tuple[list[DocumentElement], dict[str, Any]]:
        try:
            import fitz
        except Exception as exc:
            logger.warning("PyMuPDF unavailable: %s", exc)
            return [], {}

        elements: list[DocumentElement] = []
        metadata: dict[str, Any] = {}
        document = fitz.open(str(path))
        try:
            metadata = dict(document.metadata or {})
            max_pages = self.config.max_pages or document.page_count
            for page_index, page in enumerate(document, start=1):
                if page_index > max_pages:
                    break
                elements.extend(self._extract_pymupdf_page(page, page_index))
        except Exception as exc:
            logger.warning("PyMuPDF extraction failed for %s: %s", path, exc)
        finally:
            document.close()
        return elements, metadata

    def _extract_pymupdf_page(self, page: Any, page_number: int) -> list[DocumentElement]:
        page_elements: list[DocumentElement] = []
        try:
            data = page.get_text("dict", sort=True)
        except Exception:
            text = page.get_text("text", sort=True)
            return [
                DocumentElement(text=text.strip(), element_type="paragraph", page_number=page_number)
            ] if text.strip() else []

        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines: list[str] = []
            max_size = 0.0
            bold_votes = 0
            span_count = 0
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not line_text:
                    continue
                lines.append(line_text)
                for span in line.get("spans", []):
                    span_count += 1
                    max_size = max(max_size, float(span.get("size") or 0))
                    font = str(span.get("font") or "").lower()
                    if "bold" in font or "black" in font:
                        bold_votes += 1
            text = normalize_unicode("\n".join(lines).strip())
            if not text:
                continue
            element_type = self._classify_block(text, max_size=max_size, bold=bold_votes > max(0, span_count // 2))
            page_elements.append(
                DocumentElement(
                    text=text,
                    element_type=element_type,
                    page_number=page_number,
                    bbox=tuple(block.get("bbox", ())) if block.get("bbox") else None,
                    metadata={"font_size": max_size},
                )
            )
        return page_elements

    def _extract_with_pdfplumber(self, path: Path) -> list[DocumentElement]:
        try:
            import pdfplumber
        except Exception as exc:
            logger.debug("pdfplumber unavailable: %s", exc)
            return []

        elements: list[DocumentElement] = []
        try:
            with pdfplumber.open(str(path)) as pdf:
                max_pages = self.config.max_pages or len(pdf.pages)
                for page_number, page in enumerate(pdf.pages[:max_pages], start=1):
                    text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
                    for paragraph in re.split(r"\n{2,}", text):
                        paragraph = paragraph.strip()
                        if paragraph:
                            elements.append(
                                DocumentElement(
                                    text=normalize_unicode(paragraph),
                                    element_type=self._classify_block(paragraph),
                                    page_number=page_number,
                                )
                            )
                    if self.config.preserve_tables:
                        for table in page.extract_tables() or []:
                            table_text = self._format_table(table)
                            if table_text:
                                elements.append(
                                    DocumentElement(
                                        text=table_text,
                                        element_type="table",
                                        page_number=page_number,
                                        metadata={"engine": "pdfplumber"},
                                    )
                                )
        except Exception as exc:
            logger.warning("pdfplumber extraction failed for %s: %s", path, exc)
        return elements

    def _extract_with_unstructured(self, path: Path) -> list[DocumentElement]:
        try:
            from unstructured.partition.pdf import partition_pdf
        except Exception as exc:
            logger.debug("unstructured unavailable: %s", exc)
            return []

        elements: list[DocumentElement] = []
        try:
            partitioned = partition_pdf(filename=str(path), infer_table_structure=self.config.preserve_tables)
            for item in partitioned:
                text = normalize_unicode(str(item).strip())
                if not text:
                    continue
                category = getattr(item, "category", "") or item.__class__.__name__
                page_number = None
                try:
                    page_number = int(item.metadata.page_number) if item.metadata.page_number else None
                except Exception:
                    page_number = None
                elements.append(
                    DocumentElement(
                        text=text,
                        element_type=self._map_unstructured_category(category, text),
                        page_number=page_number,
                        metadata={"category": category, "engine": "unstructured"},
                    )
                )
        except Exception as exc:
            logger.warning("unstructured extraction failed for %s: %s", path, exc)
        return elements

    def _metadata(self, path: Path, raw: dict[str, Any]) -> DocumentMetadata:
        pages = None
        try:
            import fitz

            with fitz.open(str(path)) as document:
                pages = document.page_count
        except Exception:
            pass
        return DocumentMetadata(
            source_path=str(path),
            source_type="pdf",
            title=(raw.get("title") or path.stem) if raw else path.stem,
            author=raw.get("author") if raw else None,
            pages=pages,
            created_at=raw.get("creationDate") if raw else None,
            modified_at=raw.get("modDate") if raw else None,
            producer=raw.get("producer") if raw else None,
            extra={k: v for k, v in raw.items() if v},
        )

    def _looks_scanned(self, elements: list[DocumentElement], quality: dict[str, Any]) -> bool:
        pages = {element.page_number for element in elements if element.page_number}
        page_count = max(1, len(pages))
        text = self._join_elements(elements)
        words_per_page = len(text.split()) / page_count
        chars_per_page = len(text) / page_count
        return (
            chars_per_page < self.config.scanned_text_min_chars_per_page
            or words_per_page < self.config.scanned_text_min_words_per_page
            or float(quality["score"]) < 0.18
        )

    @staticmethod
    def _classify_block(text: str, max_size: float = 0.0, bold: bool = False) -> str:
        stripped = text.strip()
        words = stripped.split()
        if BULLET_RE.match(stripped):
            return "bullet"
        if "|" in stripped and stripped.count("|") >= 2:
            return "table"
        if len(words) <= 14 and not re.search(r"[.!?;:]$", stripped):
            if bold or max_size >= 13 or stripped.isupper() or re.match(r"^\d+(\.\d+)*\s+", stripped):
                return "heading"
        return "paragraph"

    @staticmethod
    def _map_unstructured_category(category: str, text: str) -> str:
        lower = category.lower()
        if "title" in lower or "heading" in lower:
            return "heading"
        if "list" in lower:
            return "bullet"
        if "table" in lower:
            return "table"
        if "caption" in lower:
            return "caption"
        return PDFLoader._classify_block(text)

    @staticmethod
    def _format_table(table: list[list[Any]]) -> str:
        rows: list[str] = []
        for row in table:
            cells = [normalize_unicode(str(cell or "").strip()) for cell in row]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    @staticmethod
    def _join_elements(elements: list[DocumentElement]) -> str:
        return "\n\n".join(element.text.strip() for element in elements if element.text.strip())

    @staticmethod
    def _build_structure(elements: list[DocumentElement]) -> dict[str, Any]:
        sections: list[dict[str, Any]] = []
        current_path: list[str] = []
        for idx, element in enumerate(elements):
            if element.element_type == "heading":
                level = 1
                match = re.match(r"^(\d+(?:\.\d+)*)", element.text.strip())
                if match:
                    level = match.group(1).count(".") + 1
                current_path = current_path[: max(0, level - 1)] + [element.text.strip()]
                element.section_path = current_path.copy()
                element.level = level
                sections.append(
                    {
                        "title": element.text.strip(),
                        "level": level,
                        "page": element.page_number,
                        "element_index": idx,
                        "path": current_path.copy(),
                    }
                )
            else:
                element.section_path = current_path.copy()
        return {"sections": sections, "page_count": len({e.page_number for e in elements if e.page_number})}

    @staticmethod
    def _document_id(path: Path, text: str) -> str:
        digest = hashlib.sha1()
        digest.update(str(path.resolve()).encode("utf-8", errors="ignore"))
        digest.update(text[:8192].encode("utf-8", errors="ignore"))
        return digest.hexdigest()
