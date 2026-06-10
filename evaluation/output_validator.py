"""Output validator to detect corrupted Vietnamese summaries and multilingual garbage."""

from __future__ import annotations

import re
import unicodedata

from src.utils import logger

# Unicode script ranges for injection detection (non-Latin, non-Vietnamese)
_FOREIGN_BLOCK_PATTERNS = re.compile(
    r"["
    r"\u1100-\u11FF"  # Hangul Jamo (Korean)
    r"\uAC00-\uD7AF"  # Hangul Syllables (Korean)
    r"\u0B80-\u0BFF"  # Tamil
    r"\u0600-\u06FF"  # Arabic
    r"\u0900-\u097F"  # Devanagari (Hindi)
    r"\u4E00-\u9FFF"  # CJK Unified Ideographs (Chinese/Japanese)
    r"\u3040-\u30FF"  # Hiragana/Katakana (Japanese)
    r"\u0400-\u04FF"  # Cyrillic (Russian)
    r"\u0370-\u03FF"  # Greek
    r"\u0E00-\u0E7F"  # Thai
    r"]"
)


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
    if detect_garbled_text(text_nfc, single_letter_threshold=0.30):
        return True

    words = text_nfc.split()
    if len(words) < 6:
        return False

    isolated_letters = sum(1 for w in words if re.fullmatch(r"[A-Za-z]", w))
    if isolated_letters >= 3:
        return True

    short_tokens = sum(1 for w in words if len(w) <= 2)
    if short_tokens / len(words) > 0.60:
        return True

    if re.search(r"\b(WỴ|nhóm!|sinh nhóm|Viet Q\.)\b", text_nfc, flags=re.IGNORECASE):
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
        if len(letters) >= 20 and vi_ratio < 0.01:
            return True
        if len(letters) >= 35 and vi_ratio == 0.0:
            return True

    return False


def detect_poor_training_output(text: str) -> dict:
    """Detect signs that the model was poorly trained for Vietnamese.

    Returns a dict with:
      - ``is_poor_training`` (bool): True if the output shows clear signs of
        a poorly fine-tuned or undertrained model.
      - ``reason`` (str | None): Human-readable explanation of the failure.

    Catches the mT5 failure pattern: isolated Korean/Tamil/Arabic/CJK tokens
    injected into an otherwise-Vietnamese text stream, which fall *below* the
    15% foreign-ratio threshold but are still clearly wrong.
    """
    if not text or len(text.strip()) < 6:
        return {"is_poor_training": False, "reason": None}

    text_nfc = unicodedata.normalize("NFC", text)

    # 1. Find any occurrence of a non-Latin/non-Vietnamese Unicode block character
    foreign_tokens = _FOREIGN_BLOCK_PATTERNS.findall(text_nfc)
    if foreign_tokens:
        scripts_found: set[str] = set()
        for ch in foreign_tokens:
            try:
                name = unicodedata.name(ch, "")
                if "HANGUL" in name or "KOREAN" in name:
                    scripts_found.add("Korean (Hangul)")
                elif "TAMIL" in name:
                    scripts_found.add("Tamil")
                elif "ARABIC" in name:
                    scripts_found.add("Arabic")
                elif "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
                    scripts_found.add("Chinese/Japanese")
                elif "DEVANAGARI" in name:
                    scripts_found.add("Hindi (Devanagari)")
                elif "CYRILLIC" in name:
                    scripts_found.add("Cyrillic")
                elif "GREEK" in name:
                    scripts_found.add("Greek")
                elif "THAI" in name:
                    scripts_found.add("Thai")
                else:
                    scripts_found.add("Unknown foreign script")
            except Exception:
                scripts_found.add("Unknown foreign script")
        scripts_str = ", ".join(sorted(scripts_found))
        return {
            "is_poor_training": True,
            "reason": (
                f"Model output contains foreign-script tokens ({scripts_str}) mixed into Vietnamese text. "
                f"This is a strong sign of poor or incomplete fine-tuning for Vietnamese NLP."
            ),
        }

    # 2. Check for tokenizer-artifact patterns: sequences of vowel-less clusters
    # e.g. "tuha", "lytte", "vulnerக"
    words = text_nfc.split()
    # Flag words with no vowels that are entirely lowercase Latin (tokeniser debris)
    vowel_pat = re.compile(r"[aeiouàáâãèéêìíòóôõùúýăđêôơưàáảạăắặầấảẽẹèéêệếềẻếỉỊịíọốồổốộờởợởớổờớộờổỏõòóôõùúụủưứựừữửụứừữử]", re.IGNORECASE)
    debris_words = [w for w in words if re.fullmatch(r"[a-z]{3,}", w) and not vowel_pat.search(w)]
    if len(debris_words) >= 2:
        return {
            "is_poor_training": True,
            "reason": (
                f"Model output contains {len(debris_words)} vowel-less ASCII token(s) "
                f"({', '.join(debris_words[:3])}...) — likely SentencePiece tokenizer debris "
                f"from an improperly aligned vocabulary during fine-tuning."
            ),
        }

    return {"is_poor_training": False, "reason": None}


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
