"""Robust plain-text loader with encoding detection."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pipeline.schema import DocumentElement, DocumentMetadata, ExtractedDocument
from preprocess.cleaner import normalize_unicode
from utils.metrics import extraction_quality_score


class TXTLoader:
    """Load TXT files using BOM, chardet, and deterministic fallbacks."""

    def load(self, path: str | Path) -> ExtractedDocument:
        file_path = Path(path)
        text, encoding = self._read_text(file_path)
        text = normalize_unicode(text)
        metadata = DocumentMetadata(
            source_path=str(file_path),
            source_type="txt",
            title=file_path.stem,
            pages=1,
            extraction_engine=f"txt:{encoding}",
        )
        quality = extraction_quality_score(text)
        metadata.quality_score = float(quality["score"])
        metadata.language = str(quality["language"])
        elements = self._build_elements(text)
        structure = self._build_structure(elements)
        return ExtractedDocument(
            document_id=self._document_id(file_path, text),
            metadata=metadata,
            text=text,
            elements=elements,
            structure={**structure, "encoding": encoding},
        )

    @staticmethod
    def _build_elements(text: str) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            element_type = TXTLoader._classify_line(stripped)
            elements.append(DocumentElement(text=stripped, element_type=element_type, page_number=1))
        return elements

    @staticmethod
    def _classify_line(text: str) -> str:
        if re.match(r"^\s*(?:[-*+•●▪▫‣]|\d{1,3}[.)])\s+", text):
            return "bullet"
        words = text.split()
        if len(words) <= 12 and not re.search(r"[.!?;:]$", text):
            if text.isupper() or re.match(r"^\d+(?:\.\d+)*\s+\w+", text):
                return "heading"
        return "paragraph"

    @staticmethod
    def _build_structure(elements: list[DocumentElement]) -> dict:
        sections: list[dict] = []
        current_path: list[str] = []
        for idx, element in enumerate(elements):
            if element.element_type == "heading":
                level = 1
                match = re.match(r"^(\d+(?:\.\d+)*)", element.text)
                if match:
                    level = match.group(1).count(".") + 1
                current_path = current_path[: max(0, level - 1)] + [element.text]
                element.section_path = current_path.copy()
                element.level = level
                sections.append(
                    {
                        "title": element.text,
                        "level": level,
                        "page": element.page_number,
                        "element_index": idx,
                        "path": current_path.copy(),
                    }
                )
            else:
                element.section_path = current_path.copy()
        return {"sections": sections}

    @staticmethod
    def _read_text(path: Path) -> tuple[str, str]:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "cp1258", "utf-16", "latin-1"):
            try:
                return raw.decode(encoding), encoding
            except UnicodeDecodeError:
                continue

        try:
            import chardet

            detection = chardet.detect(raw)
            encoding = detection.get("encoding") or "utf-8"
            return raw.decode(encoding, errors="replace"), encoding
        except Exception:
            return raw.decode("utf-8", errors="replace"), "utf-8-replace"

    @staticmethod
    def _document_id(path: Path, text: str) -> str:
        digest = hashlib.sha1()
        digest.update(str(path.resolve()).encode("utf-8", errors="ignore"))
        digest.update(text[:4096].encode("utf-8", errors="ignore"))
        return digest.hexdigest()
