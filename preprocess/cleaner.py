"""High-precision document cleaning for Vietnamese summarization."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass

from pipeline.schema import CleaningConfig, DocumentElement, ExtractedDocument
from utils.logger import logger
from utils.metrics import extraction_quality_score


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
PAGE_NUMBER_RE = re.compile(r"^\s*(?:trang\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?\s*$", re.IGNORECASE)
BROKEN_HYPHEN_RE = re.compile(r"([A-Za-zÀ-ỹĐđ])-\n([A-Za-zÀ-ỹĐđ])")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%)])")
SPACE_AFTER_OPEN_RE = re.compile(r"([({])\s+")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
NOISY_CHAR_RE = re.compile(r"[^\w\sÀ-ỹĐđ.,;:!?()/%+\-–—|•●▪▫‣\"']", re.UNICODE)
HEADING_TRAILING_PUNCT_RE = re.compile(r"[.!?;:]$")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


@dataclass(slots=True)
class CleanedDocument:
    text: str
    elements: list[DocumentElement]
    quality: dict
    warnings: list[str]


class DocumentCleaner:
    """Clean extracted text without destroying factual details or Vietnamese accents."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()

    def clean(self, document: ExtractedDocument) -> CleanedDocument:
        elements = deepcopy(document.elements)
        warnings = list(document.warnings)

        if self.config.remove_headers_footers:
            elements, removed = self._remove_repeated_headers_footers(elements)
            if removed:
                warnings.append(f"Removed {removed} repeated header/footer elements.")

        cleaned_elements: list[DocumentElement] = []
        seen_element_keys: set[str] = set()
        for element in elements:
            text = self.clean_element_text(element.text, element.element_type)
            if not text:
                continue
            if self.config.remove_page_numbers and PAGE_NUMBER_RE.fullmatch(text):
                continue
            if self.config.semantic_filtering and self._is_low_value_noise(text, element.element_type):
                continue
            key = self._dedupe_key(text)
            if key in seen_element_keys and len(text.split()) < 80:
                continue
            seen_element_keys.add(key)
            element.text = text
            cleaned_elements.append(element)

        if self.config.reconstruct_paragraphs:
            cleaned_elements = self._reconstruct_paragraphs(cleaned_elements)

        clean_text = self._join_elements(cleaned_elements)
        quality = extraction_quality_score(clean_text)
        if float(quality["score"]) < self.config.min_quality_score:
            warnings.append(
                f"Low extraction quality score: {quality['score']}. Consider OCR or another extraction engine."
            )
        logger.info(
            "Cleaned document %s: %s words, quality=%s",
            document.document_id[:10],
            quality.get("word_count", 0),
            quality.get("score", 0.0),
        )
        return CleanedDocument(text=clean_text, elements=cleaned_elements, quality=quality, warnings=warnings)

    def clean_element_text(self, text: str, element_type: str = "paragraph") -> str:
        if not text:
            return ""
        text = normalize_unicode(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = BROKEN_HYPHEN_RE.sub(r"\1\2", text)
        text = CONTROL_CHARS_RE.sub(" ", text)
        text = ZERO_WIDTH_RE.sub("", text)
        text = self._normalize_punctuation(text)
        if element_type != "table":
            text = NOISY_CHAR_RE.sub(" ", text)
        text = self._normalize_whitespace(text, preserve_newlines=element_type == "table")
        return text.strip()

    def _remove_repeated_headers_footers(self, elements: list[DocumentElement]) -> tuple[list[DocumentElement], int]:
        by_text_pages: dict[str, set[int]] = defaultdict(set)
        for element in elements:
            if not element.page_number:
                continue
            text = self.clean_element_text(element.text, element.element_type)
            if 0 < len(text) <= 120:
                by_text_pages[self._dedupe_key(text)].add(element.page_number)

        repeated = {
            key
            for key, pages in by_text_pages.items()
            if len(pages) >= self.config.min_repeated_header_pages and not self._looks_like_heading_key(key)
        }
        kept: list[DocumentElement] = []
        removed = 0
        for element in elements:
            key = self._dedupe_key(self.clean_element_text(element.text, element.element_type))
            if key in repeated:
                removed += 1
                continue
            kept.append(element)
        return kept, removed

    def _reconstruct_paragraphs(self, elements: list[DocumentElement]) -> list[DocumentElement]:
        reconstructed: list[DocumentElement] = []
        buffer: DocumentElement | None = None

        for element in elements:
            if element.element_type in {"heading", "bullet", "table", "caption"}:
                if buffer is not None:
                    reconstructed.append(buffer)
                    buffer = None
                reconstructed.append(element)
                continue

            if buffer is None:
                buffer = element
                continue

            if self._should_merge(buffer.text, element.text, buffer.page_number, element.page_number):
                buffer.text = f"{buffer.text} {element.text}".strip()
                if element.page_number:
                    buffer.metadata["page_end"] = element.page_number
            else:
                reconstructed.append(buffer)
                buffer = element

        if buffer is not None:
            reconstructed.append(buffer)
        return reconstructed

    @staticmethod
    def _should_merge(left: str, right: str, left_page: int | None, right_page: int | None) -> bool:
        if not left or not right:
            return False
        if left_page and right_page and abs(right_page - left_page) > 1:
            return False
        if len(left.split()) < 5:
            return False
        if HEADING_TRAILING_PUNCT_RE.search(left):
            return False
        if re.match(r"^[A-ZÀ-ỸĐ0-9][^.!?]{0,80}$", right) and len(right.split()) <= 10:
            return False
        return True

    @staticmethod
    def _normalize_punctuation(text: str) -> str:
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
            "\u2026": "...",
            "«": '"',
            "»": '"',
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        text = re.sub(r"([!?]){2,}", r"\1", text)
        text = re.sub(r"([,;:]){2,}", r"\1", text)
        text = re.sub(r"\.{3,}", "...", text)
        text = re.sub(r"(?<!\d)\.(?!\d)(?=[A-Za-zÀ-ỹĐđ])", ". ", text)
        text = re.sub(r"(\d)\s*([,.])\s*(\d)", r"\1\2\3", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str, preserve_newlines: bool = False) -> str:
        if preserve_newlines:
            lines = [MULTISPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
            return "\n".join(line for line in lines if line)
        text = text.replace("\n", " ")
        text = MULTISPACE_RE.sub(" ", text)
        text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        text = SPACE_AFTER_OPEN_RE.sub(r"\1", text)
        return text.strip()

    @staticmethod
    def _is_low_value_noise(text: str, element_type: str) -> bool:
        if element_type in {"heading", "table"}:
            return False
        stripped = text.strip()
        if len(stripped) <= 2:
            return True
        if PAGE_NUMBER_RE.fullmatch(stripped):
            return True
        if re.fullmatch(r"[-–—•\s]+", stripped):
            return True
        if len(stripped.split()) <= 2 and not re.search(r"\d", stripped):
            return True
        return False

    @staticmethod
    def _dedupe_key(text: str) -> str:
        return re.sub(r"\s+", " ", normalize_unicode(text).lower()).strip()

    @staticmethod
    def _looks_like_heading_key(key: str) -> bool:
        return bool(re.match(r"^\d+(?:\.\d+)*\s+\w+", key)) or (len(key.split()) <= 12 and key.isupper())

    @staticmethod
    def _join_elements(elements: list[DocumentElement]) -> str:
        parts: list[str] = []
        for element in elements:
            text = element.text.strip()
            if not text:
                continue
            if element.element_type == "heading":
                parts.append(text)
            elif element.element_type == "bullet":
                if re.match(r"^\s*(?:[-*+•●▪▫‣]|\d+[.)])", text):
                    parts.append(text)
                else:
                    parts.append(f"- {text}")
            else:
                parts.append(text)

        # Remove repeated consecutive paragraphs after all normalization.
        deduped: list[str] = []
        counts = Counter()
        for part in parts:
            key = DocumentCleaner._dedupe_key(part)
            counts[key] += 1
            if deduped and DocumentCleaner._dedupe_key(deduped[-1]) == key:
                continue
            deduped.append(part)
        return "\n\n".join(deduped).strip()
