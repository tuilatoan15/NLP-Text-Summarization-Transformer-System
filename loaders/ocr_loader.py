"""OCR fallback for scanned PDFs and image-only pages."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from pipeline.schema import DocumentElement
from utils.logger import logger


class OCRLoader:
    """OCR helper with pytesseract first and EasyOCR fallback."""

    def __init__(
        self,
        languages: Iterable[str] = ("vie", "eng"),
        dpi: int = 220,
        prefer_easyocr: bool = False,
    ) -> None:
        self.languages = tuple(languages)
        self.dpi = dpi
        self.prefer_easyocr = prefer_easyocr

    def ocr_pdf_pages(self, path: str | Path, page_numbers: Iterable[int] | None = None) -> list[DocumentElement]:
        try:
            import fitz
        except Exception as exc:
            logger.warning("OCR skipped because PyMuPDF is unavailable: %s", exc)
            return []

        elements: list[DocumentElement] = []
        document = fitz.open(str(path))
        try:
            selected = set(page_numbers or range(1, document.page_count + 1))
            for page_index, page in enumerate(document, start=1):
                if page_index not in selected:
                    continue
                pix = page.get_pixmap(dpi=self.dpi, alpha=False)
                image_bytes = pix.tobytes("png")
                text = self.ocr_image_bytes(image_bytes)
                if text.strip():
                    elements.append(
                        DocumentElement(
                            text=text.strip(),
                            element_type="paragraph",
                            page_number=page_index,
                            metadata={"engine": "ocr", "dpi": self.dpi},
                        )
                    )
        finally:
            document.close()
        return elements

    def ocr_image_bytes(self, image_bytes: bytes) -> str:
        if not self.prefer_easyocr:
            text = self._ocr_with_tesseract(image_bytes)
            if text.strip():
                return text
        return self._ocr_with_easyocr(image_bytes)

    def _ocr_with_tesseract(self, image_bytes: bytes) -> str:
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(io.BytesIO(image_bytes))
            language = "+".join(self.languages)
            return pytesseract.image_to_string(image, lang=language)
        except Exception as exc:
            logger.debug("pytesseract OCR unavailable or failed: %s", exc)
            return ""

    def _ocr_with_easyocr(self, image_bytes: bytes) -> str:
        try:
            import numpy as np
            from PIL import Image

            reader = _get_easyocr_reader(self._easyocr_languages())
            image = np.array(Image.open(io.BytesIO(image_bytes)))
            lines = reader.readtext(image, detail=0, paragraph=True)
            return "\n".join(str(line) for line in lines if str(line).strip())
        except Exception as exc:
            logger.debug("easyocr OCR unavailable or failed: %s", exc)
            return ""

    def _easyocr_languages(self) -> tuple[str, ...]:
        mapping = {"vie": "vi", "eng": "en", "vi": "vi", "en": "en"}
        return tuple(dict.fromkeys(mapping.get(lang, lang) for lang in self.languages))


@lru_cache(maxsize=4)
def _get_easyocr_reader(languages: tuple[str, ...]):
    import easyocr

    return easyocr.Reader(list(languages), gpu=False)
