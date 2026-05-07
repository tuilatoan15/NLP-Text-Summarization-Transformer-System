"""
file_parser.py - Extract text from uploaded TXT, PDF and DOCX files.
"""

from pathlib import Path
import re
import unicodedata

import fitz
from docx import Document


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
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"([a-zà-ỹ])([A-ZÀ-Ỹ])", r"\1 \2", text)
    text = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"([a-zA-ZÀ-ỹ])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-ZÀ-ỹ])", r"\1 \2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig", errors="replace")


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
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
