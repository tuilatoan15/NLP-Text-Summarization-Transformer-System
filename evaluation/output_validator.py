"""Output validator to detect corrupted Vietnamese summaries and multilingual garbage."""

from __future__ import annotations

import re
import unicodedata

from src.utils import logger


def _is_vietnamese_latin_letter(ch: str) -> bool:
    if not ch.isalpha():
        return False
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def foreign_script_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    foreign = sum(1 for ch in letters if not _is_vietnamese_latin_letter(ch))
    return foreign / len(letters)


def vietnamese_letter_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    vi_letters = len(re.findall(r"[À-ỹđĐ]", text))
    return vi_letters / len(letters)


def is_garbled_abstractive(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return True

    from src.preprocess import detect_garbled_text

    text_nfc = unicodedata.normalize("NFC", text)
    if is_multilingual_garbage(text_nfc, require_vietnamese=False):
        return True
    if detect_garbled_text(text_nfc, single_letter_threshold=0.10):
        return True

    words = text_nfc.split()
    if len(words) < 6:
        return False

    isolated_letters = sum(1 for w in words if re.fullmatch(r"[A-Za-z]", w))
    if isolated_letters >= 3:
        return True

    short_tokens = sum(1 for w in words if len(w) <= 2)
    if short_tokens / len(words) > 0.28:
        return True

    if re.search(r"\b(WỴ|nhóm!|sinh nhóm|Viet Q\.)\b", text_nfc, flags=re.IGNORECASE):
        return True

    weird_chars = len(re.findall(r"[ỖỸẸÈỴ]", text_nfc))
    if weird_chars >= 2:
        return True

    broken_ascii = sum(
        1 for w in words if re.search(r"[a-z]", w) and re.search(r"[A-Z]", w[1:])
    )
    if broken_ascii >= 4:
        return True

    return False


def is_multilingual_garbage(text: str, *, require_vietnamese: bool = False) -> bool:
    if not text or len(text.strip()) < 3:
        return True

    text_nfc = unicodedata.normalize("NFC", text)
    foreign_ratio = foreign_script_ratio(text_nfc)
    # Allow a reasonable ratio of foreign script characters (up to 15%) for mixed/scientific text
    if foreign_ratio >= 0.15:
        return True

    if require_vietnamese:
        vi_ratio = vietnamese_letter_ratio(text_nfc)
        letters = [ch for ch in text_nfc if ch.isalpha()]
        # Softened thresholds to prevent flagging short or low-accent valid texts
        if len(letters) >= 15 and vi_ratio < 0.03:
            return True
        if len(letters) >= 8 and vi_ratio == 0.0 and len(text_nfc.split()) >= 4:
            return True

    return False


def validate_output(text: str, *, require_vietnamese: bool = False) -> dict:
    if not text:
        return {"is_corrupted": True, "quality_warning": "Empty output summary."}

    text_nfc = unicodedata.normalize("NFC", text)

    if is_multilingual_garbage(text_nfc, require_vietnamese=require_vietnamese):
        ratio = foreign_script_ratio(text_nfc)
        return {
            "is_corrupted": True,
            "quality_warning": (
                f"Multilingual garbage detected (foreign script ratio={ratio:.2f})."
            ),
        }

    if is_garbled_abstractive(text_nfc):
        return {
            "is_corrupted": True,
            "quality_warning": "Garbled or corrupted abstractive generation detected.",
        }

    artifacts = ["▁", "<extra_id_", "</s>", "<pad>", "<unk>"]
    for art in artifacts:
        if art in text_nfc:
            return {
                "is_corrupted": True,
                "quality_warning": f"Detected SentencePiece artifact or token: {art}",
            }

    special_chars = re.findall(r"[+#_~*%^&|<>{}\[\]\\]", text_nfc)
    if len(special_chars) > 3:
        return {
            "is_corrupted": True,
            "quality_warning": f"Excessive random symbol count: {len(special_chars)}",
        }

    if re.search(r"[a-zA-ZÀ-ỹđĐ]\+[a-zA-ZÀ-ỹđĐ]", text_nfc):
        return {
            "is_corrupted": True,
            "quality_warning": "Detected malformed character joining via symbol (+)",
        }

    uppercase_tonal = re.findall(r"[ẰẰẮẮẰẰẤẤẦẦẬẬẼẼỮỮỰỰẴẴ]", text_nfc)
    if len(uppercase_tonal) >= 2:
        return {
            "is_corrupted": True,
            "quality_warning": "Detected character corruption or extreme tonal sequences.",
        }

    words = text_nfc.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if words[i].lower() == words[i + 1].lower() == words[i + 2].lower():
                return {
                    "is_corrupted": True,
                    "quality_warning": f"Detected repeated word loop (1-gram): '{words[i]}'",
                }

    if len(words) >= 6:
        for i in range(len(words) - 5):
            g2_1 = words[i : i + 2]
            g2_2 = words[i + 2 : i + 4]
            g2_3 = words[i + 4 : i + 6]
            if g2_1 == g2_2 == g2_3:
                return {
                    "is_corrupted": True,
                    "quality_warning": f"Detected repeated 2-gram loop: '{' '.join(g2_1)}'",
                }

    if len(words) >= 9:
        for i in range(len(words) - 8):
            g3_1 = words[i : i + 3]
            g3_2 = words[i + 3 : i + 6]
            g3_3 = words[i + 6 : i + 9]
            if g3_1 == g3_2 == g3_3:
                return {
                    "is_corrupted": True,
                    "quality_warning": f"Detected repeated 3-gram loop: '{' '.join(g3_1)}'",
                }

    vi_chars = len(re.findall(r"[À-ỹđĐ]", text_nfc))
    total_chars = len(re.findall(r"[a-zA-Z]", text_nfc))
    if total_chars > 20 and vi_chars == 0 and require_vietnamese:
        return {
            "is_corrupted": True,
            "quality_warning": "No Vietnamese characters in long alphanumeric output.",
        }

    return {"is_corrupted": False, "quality_warning": None}


def log_validation_failure(model_key: str, warning: str | None) -> None:
    if warning:
        logger.warning("[%s] Output validation failed: %s", model_key, warning)
