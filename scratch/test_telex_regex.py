import re
import unicodedata

def post_clean_vit5_telex(text: str) -> str:
    if not text:
        return ""
    
    # Print original character codes for debugging
    print("Original text codes:")
    for word in text.split():
        if "hãng" in word or "kiện" in word or "mỗi" in word or "giản" in word or "tác" in word or "bản" in word or "gian" in word:
            print(f"  {word}: {[hex(ord(c)) for c in word]}")

    # 1. Remove custom delimiters
    text = re.sub(r"[\*\+\_\[\]\<\>\(\)\{\}\\\/\|]", " ", text)
    
    # 2. Remove single character telex leftovers
    text = re.sub(r"\b[jZzWwsrxf]\b", " ", text)
    
    # 3. Strip telex markers at the end of word syllables (e.g., nhứtZ -> nhứt, kiệnj -> kiện)
    pattern_telex_leak = r"\b([a-zA-ZÀ-ỹđĐ]*[áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ]+)([jZzWwsrxf]+)\b"
    text = re.sub(pattern_telex_leak, r"\1", text)
    
    # 4. Correct specific spelling patterns
    specific_fixes = {
        r"\bmăt\b": "mắt",
        r"\bnhât\b": "nhất",
        r"\bmât\b": "mất",
        r"\bhảng\b": "hãng",
        r"\bmổij\b": "mỗi",
        r"\bkiệnj\b": "kiện",
        r"\bnhứt\b": "nhất",
        r"\bhảng\b": "hãng",
        r"\bnhứt\b": "nhất",
    }
    for pat, rep in specific_fixes.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        
    # 5. Remove any word composed solely of corrupted capital tonal marks or containing them in weird ways
    text = re.sub(r"\b[ằắằấầậẽữựẵẰẮẰẤẦẬẼỮỰẴỠẪẲỠẪẲ]\b", " ", text)
    text = re.sub(r"\b\w*[ằắằấầậẽữựẵẰẮẰẤẦẬẼỮỰẴỠẪẲỠẪẲ]+\w*\b", " ", text)
    
    return text

text = "hãngj kiệnj mổij bảnỲ tácÈ thời gianỖ gp-4 oỂ tính năngỪngÃ hình ảnhŨ"
print("Before:", text)
print("After :", post_clean_vit5_telex(text))
