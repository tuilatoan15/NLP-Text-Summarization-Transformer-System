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


def fix_decimal_spacing(text: str) -> str:
    text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)
    text = re.sub(r"(\d)\s*,\s*(\d)", r"\1,\2", text)
    text = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", text)
    return text


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
    text = re.sub(r"(?<!\d)\.(?!\d)(?=[A-Za-zÀ-ỹĐđ])", ". ", text)
    text = re.sub(r"\s*([,;:!?])\s*", r"\1 ", text)
    text = re.sub(r"\s+\.", ".", text)
    return fix_decimal_spacing(text)


def remove_noise_characters(text: str) -> str:
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = NOISE_RUN_RE.sub(" ", text)
    text = re.sub(r"[^\w\sÀ-ỹĐđ.,!?;:'\"()/%+\-]", " ", text, flags=re.UNICODE)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(rf"([{VI_LETTER_RE}])(\d)", r"\1 \2", text)
    text = re.sub(rf"(\d)([{VI_LETTER_RE}])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def strip_editorial_chrome(text: str) -> str:
    if not text:
        return ""

    lines: list[str] = []
    seen_headlines: set[str] = set()

    for raw_line in re.split(r"[\r\n]+", text):
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", line):
            continue
        if re.search(r"\bảnh\s*:", line, flags=re.IGNORECASE):
            continue
        if re.search(
            r"^[^.!?]{0,100},\s*tốt nghiệp ngành[^.!?]{0,120}\.\s*ảnh\s*:",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(r"^theo\s+(vne|vnexpress)", line, flags=re.IGNORECASE):
            continue

        headline_key = re.sub(r"\s+", " ", line.lower())[:140]
        if len(line) < 140 and headline_key in seen_headlines:
            continue
        if len(line) < 140:
            seen_headlines.add(headline_key)

        lines.append(line)

    return "\n".join(lines)


def is_editorial_noise_sentence(sentence: str) -> bool:
    s = (sentence or "").strip()
    if not s:
        return True
    if re.fullmatch(r"\d{1,3}", s):
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", s):
        return True
    if re.search(r"\bảnh\s*:", s, flags=re.IGNORECASE):
        return True
    if re.search(r"nhân vật cung c[ấa]p", s, flags=re.IGNORECASE) and len(s) < 220:
        return True
    if s.count("Ảnh:") + s.count("ảnh:") >= 2:
        return True
    return False


def dedupe_similar_sentences(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for sentence in sentences:
        key = re.sub(r"\s+", " ", sentence.lower().strip())[:160]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(sentence)
    return unique


def clean_text(text: str, aggressive: bool = False) -> str:
    if not text:
        return ""
    text = remove_html_tags(text)
    text = normalize_unicode(text)
    if aggressive:
        text = URL_RE.sub(" ", text)
        text = EMAIL_RE.sub(" ", text)
        text = strip_editorial_chrome(text)
    text = normalize_punctuation(text)
    text = remove_noise_characters(text)
    text = normalize_whitespace(text)
    text = fix_decimal_spacing(text)
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
    if not text:
        return ""

    import unicodedata
    text = unicodedata.normalize("NFC", text)

    text = text.replace("\u2581", " ")
    text = text.replace("▁", " ")
    text = text.replace("_òng", "nòng")
    text = text.replace("đnag", "đang")

    text = re.sub(r"<extra_id_\d+>", " ", text)

    text = re.sub(
        r"<\s*(pad|unk|s|eos|sep|cls|mask)\s*>",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(unk|pad)\b", " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[▁█▓▒░]", " ", text)

    text = re.sub(r"([a-zA-ZÀ-ỹđĐ])\+([a-zA-ZÀ-ỹđĐ])", r"\1 \2", text)

    patterns = [
        r"(\b\w+\b)(\s+\1){2,}",
        r"(.{2,20}?)(\1){3,}",
    ]
    for pattern in patterns:
        text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE | re.UNICODE)

    for _ in range(3):
        text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE | re.UNICODE)
        text = re.sub(r"\b(\w+\s+\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE | re.UNICODE)
        text = re.sub(r"\b(\w+\s+\w+\s+\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE | re.UNICODE)
        text = re.sub(r"\b(\w+\s+\w+\s+\w+\s+\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE | re.UNICODE)

    text = clean_text(text, aggressive=True)

    text = re.sub(r"\s*,\s*,", ", ", text)
    text = re.sub(r"\s*\.\s*\.", ". ", text)
    text = re.sub(r"\s*,\s*\.", ". ", text)
    text = re.sub(r"\s*\.\s*,", ". ", text)
    text = re.sub(r"\s*([.,;:!?])\s*", r"\1 ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    text = text.lstrip(".,;:!? ")

    text = post_clean_vit5_telex(text)

    text = normalize_whitespace(text)
    return fix_decimal_spacing(text)


def post_clean_vit5_telex(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\*\+\_\[\]\<\>\(\)\{\}\\\/\|]", " ", text)
    
    pattern_telex_leak = r"\b([a-zA-ZÀ-ỹđĐ]*[áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ]+[a-zA-ZÀ-ỹđĐ]*)([jZzWwsrxf]+)\b"
    text = re.sub(pattern_telex_leak, r"\1", text)
    
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

    import unicodedata
    art_norm = unicodedata.normalize("NFC", article)
    sum_norm = unicodedata.normalize("NFC", summary)

    art_norm = re.sub(r"\s+", " ", art_norm).strip()
    sum_norm = re.sub(r"\s+", " ", sum_norm).strip()
    
    noise_chars_pattern = r"[^\w\s.,!?;:'\"()/%+\-–—•@*]"
    art_norm = re.sub(noise_chars_pattern, "", art_norm)
    sum_norm = re.sub(noise_chars_pattern, "", sum_norm)

    art_norm = re.sub(r"^(vne|vnexpress|vietnamplus|dantri|tuoitre|vnanet)\s*[-–—:]\s*", "", art_norm, flags=re.IGNORECASE)
    art_norm = re.sub(r"\b(theo\s+[\w\s.,]+(báo|tin|đài|tổng hợp|vne|vnexpress|vnanet))\b.*$", "", art_norm, flags=re.IGNORECASE)
    art_norm = re.sub(r"\(ảnh\s*:\s*[\w\s.,-]+\)", "", art_norm, flags=re.IGNORECASE)

    article_clean = clean_text(art_norm, aggressive=True)
    summary_clean = clean_text(sum_norm, aggressive=True)

    if len(article_clean) < 100:
        return None
    if len(summary_clean) < 20:
        return None

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
