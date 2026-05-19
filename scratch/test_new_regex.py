import re

def test_new_regex():
    text = "hãngj kiệnj mổij bảnỲ tácÈ thời gianỖ gp-4 oỂ tính năngỪngÃ hình ảnhŨ"
    
    # 1. Clean telex leaks at end of words (like hãngj, kiệnj)
    pattern_telex_leak = r"\b([a-zA-ZÀ-ỹđĐ]*[áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ]+[a-zA-ZÀ-ỹđĐ]*)([jZzWwsrxf]+)\b"
    
    # 2. Strict uppercase tonal vowel set (excluding lowercase đ)
    uppercase_tonal_vowels = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲÝỶỸỸ"
    
    # Match any word containing these uppercase tonal vowels in the middle or end, and strip from that point onwards
    pattern_uppercase_tonal_leak = rf"\b([a-zA-ZÀ-ỹđĐ]+?)([{uppercase_tonal_vowels}]+)[a-zA-ZÀ-ỹđĐ]*\b"
    
    cleaned = re.sub(pattern_telex_leak, r"\1", text)
    cleaned = re.sub(pattern_uppercase_tonal_leak, r"\1", cleaned)
    
    # Specific spelling adjustments
    specific_fixes = {
        r"\bmổi\b": "mỗi",
        r"\bmổij\b": "mỗi",
        r"\bkiệnj\b": "kiện",
        r"\bnhứt\b": "nhất",
        r"\bhảng\b": "hãng",
    }
    for pat, rep in specific_fixes.items():
        cleaned = re.sub(pat, rep, cleaned, flags=re.IGNORECASE)
    
    # Remove standalone junk characters
    cleaned = re.sub(r"\b[jZzWwsrxf]\b", " ", cleaned)
    
    print("Original:", text)
    print("Cleaned :", re.sub(r"\s+", " ", cleaned).strip())

if __name__ == "__main__":
    test_new_regex()
