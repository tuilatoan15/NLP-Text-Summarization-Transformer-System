"""Vietnamese text cleaning and dataset preprocessing utilities."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from typing import Iterable

from bs4 import BeautifulSoup

from src.utils import logger


VI_LETTER_RE = r"A-Za-zÀ-ỹĐđ"
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
HTML_ENTITY_RE = re.compile(r"&[a-zA-Z0-9#]+;")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
TOKEN_RE = re.compile(rf"[{VI_LETTER_RE}0-9]+", re.UNICODE)
NOISE_RUN_RE = re.compile(r"([^\w\sÀ-ỹĐđ.,!?;:'\"()/%+-]){2,}", re.UNICODE)


VN_STOPWORDS = {
    "và", "của", "các", "những", "một", "trong", "cho", "với", "được", "đã",
    "là", "có", "không", "này", "đó", "từ", "khi", "về", "theo", "sau",
    "trước", "tại", "để", "nhiều", "người", "năm", "ngày", "ra", "vào",
}


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def remove_html_tags(text: str) -> str:
    if not text:
        return ""
    if "<" not in text and not HTML_ENTITY_RE.search(text):
        return html.unescape(text)
    try:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ")
    except Exception:
        return re.sub(r"<[^>]+>", " ", html.unescape(text))


def normalize_punctuation(text: str) -> str:
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2026": "...",
        "«": '"', "»": '"',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"([!?]){2,}", r"\1", text)
    text = re.sub(r"([,;:]){2,}", r"\1", text)
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"\s*([,;:!?])\s*", r"\1 ", text)
    text = re.sub(r"\s+\.", ".", text)
    return text


def remove_noise_characters(text: str) -> str:
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = NOISE_RUN_RE.sub(" ", text)
    text = re.sub(r"[^\w\sÀ-ỹĐđ.,!?;:'\"()/%+\-]", " ", text, flags=re.UNICODE)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", text)
    text = re.sub(rf"([{VI_LETTER_RE}])(\d)", r"\1 \2", text)
    text = re.sub(rf"(\d)([{VI_LETTER_RE}])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def clean_text(text: str, aggressive: bool = False) -> str:
    if not text:
        return ""
    text = remove_html_tags(text)
    text = normalize_unicode(text)
    if aggressive:
        text = URL_RE.sub(" ", text)
        text = EMAIL_RE.sub(" ", text)
    text = normalize_punctuation(text)
    text = remove_noise_characters(text)
    text = normalize_whitespace(text)
    return text


def split_sentences(text: str, use_underthesea: bool = True) -> list[str]:
    text = clean_text(text) if text else ""
    if not text:
        return []

    if use_underthesea:
        try:
            from underthesea import sent_tokenize

            sentences = [s.strip() for s in sent_tokenize(text) if s and s.strip()]
            if sentences:
                return sentences
        except Exception as exc:
            logger.debug("underthesea sentence split fallback: %s", exc)

    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def tokenize_words(text: str, remove_stopwords: bool = False) -> list[str]:
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(normalize_unicode(text))]
    if remove_stopwords:
        tokens = [token for token in tokens if token not in VN_STOPWORDS and len(token) > 1]
    return tokens


def text_fingerprint(text: str) -> str:
    normalized = " ".join(tokenize_words(clean_text(text), remove_stopwords=False))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def is_probably_bad_generation(text: str) -> bool:
    if not text or len(text.strip()) < 5:
        return True
    cleaned = clean_text(text)
    tokens = tokenize_words(cleaned)
    if len(tokens) < 3:
        return True

    bracket_noise = len(re.findall(r"[\[\]{}<>|\\]", text))
    if bracket_noise > max(2, len(text) * 0.03):
        return True

    single_char_ratio = sum(1 for token in tokens if len(token) == 1) / max(1, len(tokens))
    if single_char_ratio > 0.35:
        return True

    most_common_ratio = max(tokens.count(token) for token in set(tokens)) / max(1, len(tokens))
    if len(tokens) >= 8 and most_common_ratio > 0.35:
        return True

    weird_ratio = len(re.findall(r"[^0-9A-Za-zÀ-ỹĐđ\s.,!?;:'\"()/%+\-]", text)) / max(1, len(text))
    return weird_ratio > 0.08


def clean_generated_summary(text: str) -> str:
    """
    Clean Transformer-generated text, removing SentencePiece / ViT5 artifacts.

    Handles:
    • SentencePiece ▁ (U+2581) word-boundary markers left by some tokenizers
    • <extra_id_N> sentinel tokens from T5-family models
    • Special tokens: <pad>, <unk>, <s>, </s>, <eos>
    • HTML entities that survived tokenisation
    • Repeated word sequences (word loops — sign of bad generation)
    • Trailing/leading punctuation and extra whitespace
    • Vietnamese unicode normalization and repeated phrase collapse
    """
    if not text:
        return ""

    # 1. Normalize unicode (NFC)
    import unicodedata
    text = unicodedata.normalize("NFC", text)

    # 2. Remove SentencePiece U+2581 boundary marker (▁) and common tokenizer leaks
    text = text.replace("\u2581", " ")
    text = text.replace("▁", " ")
    text = text.replace("_òng", "nòng")  # Common ViT5/mT5 tokenization error for 'nòng'
    text = text.replace("đnag", "đang")  # Typo correction

    # 3. Remove T5 sentinel tokens
    text = re.sub(r"<extra_id_\d+>", " ", text)

    # 4. Remove common special tokens that leaked through
    text = re.sub(
        r"<\s*(pad|unk|s|eos|sep|cls|mask)\s*>",
        " ", text, flags=re.IGNORECASE,
    )
    # Also handle bare token names that sometimes appear without angle brackets
    text = re.sub(r"\b(unk|pad)\b", " ", text, flags=re.IGNORECASE)

    # Clean weird characters
    text = re.sub(r"[▁█▓▒░]", " ", text)

    # 5. Remove malformed UTF fragments (e.g. invalid combinations or stray plus symbols joining parts)
    text = re.sub(r"([a-zA-ZÀ-ỹđĐ])\+([a-zA-ZÀ-ỹđĐ])", r"\1 \2", text)

    # 6. Collapse excessive repeated n-grams / phrase loops (word-loop artifact) using targeted patterns
    # Apply standard patterns first
    patterns = [
        r"(\b\w+\b)(\s+\1){2,}", # Repeated 1-gram (3 or more times)
        r"(.{2,20}?)(\1){3,}",    # Repeated phrase loops
    ]
    for pattern in patterns:
        text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE | re.UNICODE)

    for _ in range(3):  # Multiple passes to catch nested repetition loops
        # 1-word repeat: "học học học"
        text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE | re.UNICODE)
        # 2-word repeat: "học sinh học sinh"
        text = re.sub(r"\b(\w+\s+\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE | re.UNICODE)
        # 3-word repeat: "cách mạng công cách mạng công"
        text = re.sub(r"\b(\w+\s+\w+\s+\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE | re.UNICODE)
        # 4-word repeat: "cuộc cách mạng công cuộc cách mạng công"
        text = re.sub(r"\b(\w+\s+\w+\s+\w+\s+\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE | re.UNICODE)

    # 7. Standard Vietnamese text cleanup
    text = clean_text(text, aggressive=True)

    # 8. Additional malformed punctuation cleanup (e.g. " , ", " . . ", " ., ", " ,. ")
    text = re.sub(r"\s*,\s*,", ", ", text)
    text = re.sub(r"\s*\.\s*\.", ". ", text)
    text = re.sub(r"\s*,\s*\.", ". ", text)
    text = re.sub(r"\s*\.\s*,", ". ", text)
    text = re.sub(r"\s*([.,;:!?])\s*", r"\1 ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text) # Remove spaces before punctuation

    # Double check for start of summary starting with commas or dots
    text = text.lstrip(".,;:!? ")

    # 9. Telex and Delimiter cleanup for ViT5 model output
    text = post_clean_vit5_telex(text)

    return normalize_whitespace(text)


def post_clean_vit5_telex(text: str) -> str:
    """Aggressively removes telex typing leaks and delimiter artifacts left by ViT5."""
    if not text:
        return ""
    # 1. Remove custom delimiters
    text = re.sub(r"[\*\+\_\[\]\<\>\(\)\{\}\\\/\|]", " ", text)
    
    # 2. Clean telex leaks at end of words (like hãngj, kiệnj)
    pattern_telex_leak = r"\b([a-zA-ZÀ-ỹđĐ]*[áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ]+[a-zA-ZÀ-ỹđĐ]*)([jZzWwsrxf]+)\b"
    text = re.sub(pattern_telex_leak, r"\1", text)
    
    # 3. Strict uppercase tonal vowel set to strip corrupted trailing/middle capitals
    uppercase_tonal_vowels = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲÝỶỸỸ"
    pattern_uppercase_tonal_leak = rf"\b([a-zA-ZÀ-ỹđĐ]+?)([{uppercase_tonal_vowels}]+)[a-zA-ZÀ-ỹđĐ]*\b"
    text = re.sub(pattern_uppercase_tonal_leak, r"\1", text)
    
    # 4. Correct specific spelling patterns
    specific_fixes = {
        r"\bmổi\b": "mỗi",
        r"\bmổij\b": "mỗi",
        r"\bkiệnj\b": "kiện",
        r"\bnhứt\b": "nhất",
        r"\bhảng\b": "hãng",
        r"\bmăt\b": "mắt",
        r"\bnhât\b": "nhất",
        r"\bmât\b": "mất",
    }
    for pat, rep in specific_fixes.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        
    # 5. Remove single character telex leftovers
    text = re.sub(r"\b[jZzWwsrxf]\b", " ", text)
    
    return text


def clean_dataset_record(
    article: str,
    summary: str,
    min_article_words: int = 30,
    min_summary_words: int = 3,
) -> dict | None:
    if not article or not summary:
        return None

    # 1. Normalize Unicode NFC
    import unicodedata
    art_norm = unicodedata.normalize("NFC", article)
    sum_norm = unicodedata.normalize("NFC", summary)

    # 2. Duplicated space and malformed character removal
    art_norm = re.sub(r"\s+", " ", art_norm).strip()
    sum_norm = re.sub(r"\s+", " ", sum_norm).strip()
    
    # Remove weird OCR character clusters and noise
    noise_chars_pattern = r"[^\w\s.,!?;:'\"()/%+\-–—•@*]"
    art_norm = re.sub(noise_chars_pattern, "", art_norm)
    sum_norm = re.sub(noise_chars_pattern, "", sum_norm)

    # 3. Remove common headers/signatures (e.g. "VnExpress - ", "Theo...", "Ảnh:", "[A-Z]+ (VNA) -")
    art_norm = re.sub(r"^(vne|vnexpress|vietnamplus|dantri|tuoitre|vnanet)\s*[-–—:]\s*", "", art_norm, flags=re.IGNORECASE)
    art_norm = re.sub(r"\b(theo\s+[\w\s.,]+(báo|tin|đài|tổng hợp|vne|vnexpress|vnanet))\b.*$", "", art_norm, flags=re.IGNORECASE)
    art_norm = re.sub(r"\(ảnh\s*:\s*[\w\s.,-]+\)", "", art_norm, flags=re.IGNORECASE)

    # 4. Clean text natively
    article_clean = clean_text(art_norm, aggressive=True)
    summary_clean = clean_text(sum_norm, aggressive=True)

    # 5. Length Rule Enforcement (input >= 100 chars, summary >= 20 chars)
    if len(article_clean) < 100:
        return None
    if len(summary_clean) < 20:
        return None

    # Additional word count thresholds
    if len(tokenize_words(article_clean)) < min_article_words:
        return None
    if len(tokenize_words(summary_clean)) < min_summary_words:
        return None
    if is_probably_bad_generation(summary_clean):
        return None

    return {
        "article": article_clean,
        "title": summary_clean,
        "fingerprint": text_fingerprint(article_clean),
    }


def deduplicate_records(records: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for record in records:
        fp = record.get("fingerprint") or text_fingerprint(record.get("article", ""))
        if fp in seen:
            continue
        seen.add(fp)
        unique.append(record)
    return unique


def preprocess(text: str, aggressive: bool = False) -> dict:
    cleaned = clean_text(text, aggressive=aggressive)
    sentences = split_sentences(cleaned)
    issues: list[str] = []
    if is_probably_bad_generation(cleaned) and cleaned:
        issues.append("noisy_or_too_short")
    logger.info("Preprocess complete: %s words, %s sentences", len(cleaned.split()), len(sentences))
    return {
        "cleaned": cleaned,
        "sentences": sentences,
        "issues": issues,
        "garbled": "noisy_or_too_short" in issues,
    }


def detect_garbled_text(text: str, single_letter_threshold: float = 0.15) -> bool:
    tokens = tokenize_words(text)
    if not tokens:
        return False
    ratio = sum(1 for token in tokens if len(token) == 1) / len(tokens)
    return ratio >= single_letter_threshold


def fix_spaced_letters(text: str) -> str:
    return normalize_whitespace(text)
