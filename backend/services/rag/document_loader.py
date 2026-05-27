from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
from docx import Document


class DocumentLoader:
    supported_extensions = {".pdf", ".docx", ".txt", ".md", ".markdown"}

    def load(self, path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower()
        if suffix not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {suffix}")
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix == ".docx":
            return self._load_docx(path)
        return self._load_text(path)

    def _load_text(self, path: Path) -> dict[str, Any]:
        for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        return {"text": text, "pages": [{"page": 1, "text": text}]}

    def _load_docx(self, path: Path) -> dict[str, Any]:
        document = Document(str(path))
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)
        return {"text": text, "pages": [{"page": 1, "text": text}]}

    def _load_pdf(self, path: Path) -> dict[str, Any]:
        doc = fitz.open(str(path))
        pages: list[dict[str, Any]] = []
        try:
            for idx, page in enumerate(doc):
                text = page.get_text("text", sort=True).strip()
                if text:
                    pages.append({"page": idx + 1, "text": text})
        finally:
            doc.close()
        all_text = "\n\n".join(p["text"] for p in pages)
        return {"text": all_text, "pages": pages}

