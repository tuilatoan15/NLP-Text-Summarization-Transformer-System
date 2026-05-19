"""Output validator to detect corrupted Vietnamese summaries and multilingual garbage."""

import re
import unicodedata
from src.utils import logger

def validate_output(text: str) -> dict:
    """
    Validate quality of generated summary.
    Returns dict: {"is_corrupted": bool, "quality_warning": str | None}
    """
    if not text:
        return {"is_corrupted": True, "quality_warning": "Empty output summary."}

    text_nfc = unicodedata.normalize("NFC", text)
    
    # 1. Check for leftover model tokens or SentencePiece artifacts
    artifacts = ["▁", "<extra_id_", "</s>", "<pad>", "<unk>"]
    for art in artifacts:
        if art in text_nfc:
            return {
                "is_corrupted": True,
                "quality_warning": f"Detected SentencePiece artifact or token: {art}"
            }

    # 2. Check for excessive symbols (e.g., +++, ..., ???, !!!, random punctuation clusters)
    special_chars = re.findall(r"[+#_~*%^&|<>{}\[\]\\]", text_nfc)
    if len(special_chars) > 3:
        return {
            "is_corrupted": True,
            "quality_warning": f"Excessive random symbol count: {len(special_chars)}"
        }

    # 3. Check for malformed Vietnamese space or unicode errors (e.g., "nắng nóng+iết kiệm", "Ẽ Ữ Ặ")
    # Patterns of invalid combinations like letters directly attached to + or multiple consecutive capital tone vowels
    if re.search(r"[a-zA-ZÀ-ỹđĐ]\+[a-zA-ZÀ-ỹđĐ]", text_nfc):
        return {
            "is_corrupted": True,
            "quality_warning": "Detected malformed character joining via symbol (+)"
        }
        
    # Check for character map corruption like too many consecutive uppercase tone-vowels (e.g. "Ẽ Ữ Ặ")
    uppercase_tonal = re.findall(r"[ẰẰẮẮẰẰẤẤẦẦẬẬẼẼỮỮỰỰẴẴ]", text_nfc)
    if len(uppercase_tonal) >= 2:
        return {
            "is_corrupted": True,
            "quality_warning": "Detected character corruption or extreme tonal sequences."
        }

    # 4. Check for repetition loops
    # 4.1. Repeated 1-gram (words)
    words = text_nfc.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if words[i].lower() == words[i+1].lower() == words[i+2].lower():
                return {
                    "is_corrupted": True,
                    "quality_warning": f"Detected repeated word loop (1-gram): '{words[i]}'"
                }
                
    # 4.2. Repeated phrase loops (2-gram, 3-gram, 4-gram)
    # Check 2-gram loop
    if len(words) >= 6:
        for i in range(len(words) - 5):
            g2_1 = words[i:i+2]
            g2_2 = words[i+2:i+4]
            g2_3 = words[i+4:i+6]
            if g2_1 == g2_2 == g2_3:
                return {
                    "is_corrupted": True,
                    "quality_warning": f"Detected repeated 2-gram loop: '{' '.join(g2_1)}'"
                }
                
    # Check 3-gram loop
    if len(words) >= 9:
        for i in range(len(words) - 8):
            g3_1 = words[i:i+3]
            g3_2 = words[i+3:i+6]
            g3_3 = words[i+6:i+9]
            if g3_1 == g3_2 == g3_3:
                return {
                    "is_corrupted": True,
                    "quality_warning": f"Detected repeated 3-gram loop: '{' '.join(g3_1)}'"
                }

    # 5. Check for multilingual / junk ratio (low Vietnamese word ratio in generated text)
    # If the text has letters but none are common Vietnamese syllables/words
    vi_chars = len(re.findall(r"[À-ỹđĐ]", text_nfc))
    total_chars = len(re.findall(r"[a-zA-Z]", text_nfc))
    if total_chars > 20 and vi_chars == 0:
        return {
            "is_corrupted": True,
            "quality_warning": "Multilingual garbage detected (no Vietnamese characters present in alphanumeric text)."
        }

    return {"is_corrupted": False, "quality_warning": None}
