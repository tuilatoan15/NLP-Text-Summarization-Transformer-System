"""Extract text from uploaded TXT, PDF, and DOCX files."""

from __future__ import annotations

from pathlib import Path

import fitz
from docx import Document

from src.preprocess import clean_text


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_text_from_file(path: str | Path) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        text = _read_txt(file_path)
    elif suffix == ".pdf":
        text = _read_pdf(file_path)
    elif suffix == ".docx":
        text = _read_docx(file_path)
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {supported}")

    return clean_extracted_text(text)


def clean_extracted_text(text: str) -> str:
    return clean_text(text or "", aggressive=True)


def _read_txt(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    document = fitz.open(str(path))
    pages = []
    try:
        for page in document:
            text = page.get_text("text", sort=True)
            if text.strip():
                pages.append(text)
    finally:
        document.close()
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    document = Document(str(path))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n\n".join(paragraphs)
