"""
preprocess.py — Tiền xử lý văn bản tiếng Việt.

Bao gồm:
  - Làm sạch HTML, ký tự đặc biệt, khoảng trắng thừa
  - Chuẩn hóa Unicode về dạng NFC (quan trọng cho tiếng Việt)
  - Tách câu tiếng Việt dùng underthesea
"""

import re
import unicodedata
from bs4 import BeautifulSoup
from typing import Optional

from src.utils import logger


# ==============================================================================
# LÀM SẠCH VĂN BẢN
# ==============================================================================

def remove_html_tags(text: str) -> str:
    """
    Loại bỏ toàn bộ HTML tags khỏi văn bản.
    Sử dụng BeautifulSoup để đảm bảo xử lý đúng mọi dạng HTML phức tạp.
    """
    try:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ")
    except Exception:
        # Fallback: dùng regex nếu lxml không xử lý được
        return re.sub(r"<[^>]+>", " ", text)


def normalize_unicode(text: str) -> str:
    """
    Chuẩn hóa Unicode về dạng NFC.
    Cực kỳ quan trọng với tiếng Việt vì dấu thanh và dấu phụ có thể bị mã hóa
    theo nhiều cách khác nhau (NFD vs NFC) dẫn đến lỗi tách từ/câu.
    """
    return unicodedata.normalize("NFC", text)


def normalize_punctuation(text: str) -> str:
    """
    Chuẩn hóa dấu câu và các ký tự punctuation phổ biến.
    - Thay các dấu ngoặc kép/fancy quotes thành ASCII
    - Thay dash/dashes thành '-' chuẩn
    - Chuẩn hóa ellipsis
    - Giảm các dấu câu lặp
    """
    if not text:
        return text

    repl = {
        '“': '"', '”': '"', '«': '"', '»': '"', '‘': "'", '’': "'",
        '\u2018': "'", '\u2019': "'", '\u2013': '-', '\u2014': '-', '–': '-', '—': '-',
        '\u2026': '...', '…': '...'
    }
    for k, v in repl.items():
        text = text.replace(k, v)

    # Collapse repeated punctuation like '!!!' -> '!'
    text = re.sub(r'([!?.,;:]){2,}', r'\1', text)

    # Ensure single space after punctuation when appropriate
    text = re.sub(r"\s*([,;:.!?])\s*", r"\1 ", text)
    # Remove spaces before punctuation
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)

    # Normalize multiple dots to ellipsis representation '...'
    text = re.sub(r"\.{2,}", '...', text)

    return text.strip()


def _normalize_fullwidth(text: str) -> str:
    """Chuyển fullwidth characters về ASCII tương đương (nếu có)."""
    try:
        return unicodedata.normalize('NFC', text)
    except Exception:
        return text


def detect_invalid_input(text: str, min_words: int = 5) -> bool:
    """Phát hiện input không hợp lệ (quá ngắn hoặc chứa quá nhiều ký tự lạ)."""
    if not text or not text.strip():
        return True
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(tokens) < min_words:
        return True
    # Heuristic: nếu văn bản chứa quá ít ký tự chữ so với tổng -> nghi ngờ
    letters = re.findall(r"[A-Za-zÀ-ỹ]", text)
    if len(text) > 0 and (len(letters) / (len(text) + 1)) < 0.15:
        return True
    return False


def normalize_whitespace(text: str) -> str:
    """
    Chuẩn hóa khoảng trắng:
      - Thay thế nhiều khoảng trắng/tab/newline liên tiếp bằng một khoảng trắng
      - Loại bỏ khoảng trắng đầu/cuối dòng
    """
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"([a-zà-ỹ])([A-ZÀ-Ỹ])", r"\1 \2", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    """
    Loại bỏ các ký tự đặc biệt không cần thiết.
    
    Args:
        text: Văn bản cần xử lý
        keep_punctuation: Giữ lại dấu câu thông thường (.,!?;:) hay không
    
    Returns:
        Văn bản đã lọc
    """
    if keep_punctuation:
        # Chỉ loại bỏ ký tự không phải chữ cái, số, khoảng trắng, dấu câu cơ bản
        text = re.sub(r"[^\w\s.,!?;:\"'\-–—()[\]{}%@#/\\]", " ", text)
    else:
        text = re.sub(r"[^\w\s]", " ", text)
    return text


def remove_pdf_artifacts(text: str) -> str:
    """Loại bỏ nhiễu thường gặp khi trích xuất PDF nhưng vẫn giữ dấu tiếng Việt."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"([A-Za-zÀ-ỹ])-\s+([A-Za-zÀ-ỹ])", r"\1\2", text)
    text = re.sub(r"([A-Za-zÀ-ỹ])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-zÀ-ỹ])", r"\1 \2", text)
    return text


def remove_urls_from_text(text: str) -> str:
    """Loại bỏ URLs khỏi văn bản."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_emails(text: str) -> str:
    """Loại bỏ địa chỉ email khỏi văn bản."""
    return re.sub(r"\S+@\S+\.\S+", "", text)


def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Pipeline làm sạch văn bản đầy đủ.

    Args:
        text: Văn bản thô đầu vào (có thể chứa HTML)
        aggressive: Nếu True, loại bỏ thêm URL và email

    Returns:
        Văn bản đã được làm sạch và chuẩn hóa
    """
    if not text:
        return ""

    # 1. Loại bỏ HTML
    text = remove_html_tags(text)

    # 2. Chuẩn hóa Unicode NFC (quan trọng với tiếng Việt)
    text = normalize_unicode(text)
    text = _normalize_fullwidth(text)
    text = normalize_punctuation(text)
    text = remove_pdf_artifacts(text)

    # 3. Loại bỏ URL và email nếu cần
    if aggressive:
        text = remove_urls_from_text(text)
        text = remove_emails(text)

    # 4. Loại bỏ ký tự đặc biệt (giữ dấu câu)
    text = remove_special_chars(text, keep_punctuation=True)

    # 5. Chuẩn hóa khoảng trắng
    text = normalize_whitespace(text)

    return text


# ==============================================================================
# TÁCH CÂU TIẾNG VIỆT
# ==============================================================================

def split_sentences(text: str, use_underthesea: bool = True) -> list[str]:
    """
    Tách văn bản thành danh sách câu.

    Ưu tiên dùng underthesea (hỗ trợ tốt tiếng Việt).
    Nếu underthesea không available thì fallback dùng regex.

    Args:
        text: Văn bản đã được làm sạch
        use_underthesea: Dùng underthesea.sent_tokenize hay không

    Returns:
        Danh sách các câu
    """
    if not text:
        return []

    if use_underthesea:
        try:
            from underthesea import sent_tokenize as vi_sent_tokenize
            sentences = vi_sent_tokenize(text)
            logger.debug(f"Tách câu bằng underthesea: {len(sentences)} câu.")
            return [s.strip() for s in sentences if s.strip()]
        except ImportError:
            logger.warning("underthesea không khả dụng, dùng regex fallback.")
        except Exception as e:
            logger.warning(f"underthesea tách câu gặp lỗi: {e}. Dùng regex fallback.")

    # Fallback: tách bằng regex theo dấu câu kết thúc
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    logger.debug(f"Tách câu bằng regex: {len(sentences)} câu.")
    return sentences


# ==============================================================================
# PIPELINE TIỀN XỬ LÝ ĐẦY ĐỦ
# ==============================================================================

def preprocess(text: str, aggressive: bool = False) -> dict:
    """
    Pipeline tiền xử lý đầy đủ: làm sạch + tách câu.

    Args:
        text: Văn bản thô
        aggressive: Xóa URL và email

    Returns:
        Dict chứa:
          - 'cleaned': văn bản đã làm sạch
          - 'sentences': danh sách câu đã tách
    """
    cleaned = clean_text(text, aggressive=aggressive)

    # Phát hiện và sửa các văn bản bị tách ký tự bất thường
    garbled_detected = detect_garbled_text(cleaned)
    issues = []
    if garbled_detected:
        issues.append("garbled_spacing")
        fixed = fix_spaced_letters(cleaned)
        if fixed != cleaned:
            issues.append("fixed_spaced_letters")
            cleaned = fixed

    invalid = detect_invalid_input(cleaned)
    if invalid:
        issues.append("invalid_input")

    sentences = split_sentences(cleaned)

    logger.info(f"Tiền xử lý xong: {len(cleaned.split())} từ, {len(sentences)} câu. issues={issues}")

    return {
        "cleaned": cleaned,
        "sentences": sentences,
        "issues": issues,
        "garbled": garbled_detected,
    }


def fix_spaced_letters(text: str) -> str:
    """
    Ghép lại các ký tự/chuỗi bị tách bởi khoảng trắng bất thường.
    Đặc biệt xử lý các văn bản tiếng Việt bị lỗi tách chữ từ PDF.
    
    Ví dụ: "t ương t ác" -> "tương tác", "ch ức n ăng" -> "chức năng"
    """
    if not text or " " not in text:
        return text

    # Nếu văn bản không có dấu hiệu bị garbled thì không xử lý tránh làm hỏng văn bản chuẩn
    if not detect_garbled_text(text):
        return text

    # Bước 1: Ghép các từ bị tách bởi 1 khoảng trắng (nếu word_segmentation hỗ trợ)
    # Nhưng trước hết dùng heuristic đơn giản: 
    # Nếu một chuỗi các token ngắn đứng cạnh nhau, khả năng cao là chúng bị tách.
    tokens = text.split()
    if not tokens:
        return text

    new_tokens = []
    buffer = []
    
    def flush_buffer():
        if not buffer:
            return
        combined = "".join(buffer)
        # Thử dùng underthesea để tách từ lại cho chuẩn nếu là tiếng Việt
        try:
            from underthesea import word_tokenize
            # Chỉ dùng word_tokenize nếu chuỗi ghép lại đủ dài
            if len(combined) > 3:
                tokenized = word_tokenize(combined)
                new_tokens.extend(tokenized)
            else:
                new_tokens.append(combined)
        except Exception:
            new_tokens.append(combined)
        buffer.clear()

    for t in tokens:
        # Heuristic: Token có độ dài <= 3 hoặc chứa các ký tự đơn lẻ (n, t, h, ...)
        # Hoặc token chỉ gồm 1 nguyên âm có dấu (ú, ồ, ...)
        is_fragment = len(t) <= 3 or re.match(r"^[a-zA-ZÀ-ỹ]$|^(ch|nh|th|ph|gi|tr|qu|kh|ng|gh)$", t, re.I)
        
        if is_fragment:
            buffer.append(t)
            # Nếu buffer quá lớn (ví cả đoạn bị tách), ta vẫn để trong buffer
        else:
            # Gặp một từ "bình thường" (dài > 3), flush buffer cũ
            if buffer:
                flush_buffer()
            new_tokens.append(t)
            
    flush_buffer()
    
    result = " ".join(new_tokens)
    
    # Bước 2: Xử lý các trường hợp đặc biệt như "Xửl ý" -> "Xử lý"
    # Thêm dấu cách vào giữa chữ thường và chữ hoa nếu dính nhau (đã có trong clean_text nhưng làm lại cho chắc)
    result = re.sub(r"([a-zà-ỹ])([A-ZÀ-Ỹ])", r"\1 \2", result)
    
    # Bước 3: Chuẩn hóa lại khoảng trắng
    result = re.sub(r" {2,}", " ", result).strip()
    
    return result


def detect_garbled_text(text: str, single_letter_threshold: float = 0.15) -> bool:
    """
    Phát hiện văn bản 'garbled' (nhiễu do tách ký tự) bằng heuristics:
      - Tỉ lệ token có độ dài 1 lớn hơn ngưỡng
      - Hoặc trung bình độ dài từ nhỏ hơn một ngưỡng (nhiều token ngắn)
    """
    if not text or not text.strip():
        return False

    tokens = [t for t in re.findall(r"\S+", text)]
    if not tokens:
        return False

    single_letter = sum(1 for t in tokens if len(t) == 1 and re.match(r"^[^\W\d_]+$", t, flags=re.UNICODE))
    ratio = single_letter / len(tokens)

    avg_len = sum(len(t) for t in tokens) / len(tokens)

    # Nếu có tỉ lệ chữ đơn ký tự cao hoặc trung bình độ dài thấp => nghi ngờ
    if ratio >= single_letter_threshold or avg_len < 3.0:
        return True
    return False


# ==============================================================================
# CHẠY THỬ TRỰC TIẾP
# ==============================================================================

if __name__ == "__main__":
    sample = """
    <p>Chào mừng bạn đến với <b>hệ thống tóm tắt văn bản</b> tiếng Việt!</p>
    Đây là một   đoạn văn   có nhiều   khoảng trắng thừa.
    Hôm nay, Thủ tướng Nguyễn Xuân Phúc đã có buổi làm việc với lãnh đạo tỉnh Nghệ An.
    Ông nhấn mạnh tầm quan trọng của phát triển kinh tế bền vững. Cuộc họp diễn ra tốt đẹp!
    """
    result = preprocess(sample, aggressive=True)
    print("--- Văn bản đã làm sạch ---")
    print(result["cleaned"])
    print("\n--- Danh sách câu ---")
    for i, s in enumerate(result["sentences"], 1):
        print(f"  {i}. {s}")
