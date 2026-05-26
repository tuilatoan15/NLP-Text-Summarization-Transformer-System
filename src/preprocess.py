from typing import List
import re

# Vietnamese and common stopwords
_STOPWORDS = {
    "là", "và", "của", "trong", "có", "được", "với", "các", "cho", "này",
    "đã", "một", "những", "không", "về", "từ", "khi", "cũng", "như", "để",
    "theo", "bởi", "vì", "rằng", "thì", "mà", "hay", "hoặc", "nhưng", "nếu",
    "đó", "đây", "hơn", "hết", "vào", "ra", "lên", "xuống", "trên", "dưới",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "of", "to", "from", "that", "this", "it", "he", "she", "they", "we",
}

# Patterns that indicate editorial / boilerplate noise
_NOISE_PATTERNS = [
    re.compile(r"^\s*(ảnh|hình|nguồn|photo|image|caption)\s*:", re.IGNORECASE),
    re.compile(r"^\s*\(.*\)\s*$"),          # sentence is purely a parenthetical
    re.compile(r"^\s*[-–—•]\s*$"),          # lone bullet / dash
    re.compile(r"^\d+\s*$"),               # lone digit(s)
]


def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Clean Vietnamese text.
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    if aggressive:
        # remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        # collapse repeated punctuation
        text = re.sub(r"([!?.]){2,}", r"\1", text)
        # strip leading/trailing quotes
        text = text.strip("\"'""''")

    return text.strip()


def tokenize_words(text: str, remove_stopwords: bool = False) -> List[str]:
    """
    Basic word tokenizer.
    """
    if not text:
        return []

    tokens = text.lower().split()
    # strip punctuation from each token
    tokens = [re.sub(r"[^\w]", "", t) for t in tokens]
    tokens = [t for t in tokens if t]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    return tokens


def tokenize_sentences(text: str) -> List[str]:
    """
    Vietnamese sentence tokenizer (alias kept for backward-compat).
    """
    return split_sentences(text)


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling Vietnamese and English punctuation.
    """
    if not text:
        return []

    # split on sentence-ending punctuation followed by whitespace or end-of-string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return [s.strip() for s in sentences if s.strip()]


def is_editorial_noise_sentence(sentence: str) -> bool:
    """
    Return True if the sentence looks like editorial boilerplate / noise
    that should be excluded from summarization.
    """
    if not sentence or len(sentence.split()) < 3:
        return True
    for pattern in _NOISE_PATTERNS:
        if pattern.search(sentence):
            return True
    return False


def dedupe_similar_sentences(sentences: List[str], threshold: float = 0.85) -> List[str]:
    """
    Remove near-duplicate sentences using character-level Jaccard similarity.
    Preserves order of first occurrence.
    """
    if not sentences:
        return []

    def _ngrams(text: str, n: int = 3):
        text = text.lower()
        return set(text[i:i + n] for i in range(len(text) - n + 1))

    unique: List[str] = []
    seen_ngrams: List[set] = []

    for sent in sentences:
        ngrams = _ngrams(sent)
        duplicate = False
        for prev_ngrams in seen_ngrams:
            union = prev_ngrams | ngrams
            if not union:
                continue
            jaccard = len(prev_ngrams & ngrams) / len(union)
            if jaccard >= threshold:
                duplicate = True
                break
        if not duplicate:
            unique.append(sent)
            seen_ngrams.append(ngrams)

    return unique


def fix_decimal_spacing(text: str) -> str:
    """
    Fix spacing around decimal numbers that may have been disrupted
    (e.g. "3 . 14" -> "3.14", "100 , 000" -> "100,000").
    Also trims extra spaces before sentence-ending punctuation.
    """
    if not text:
        return ""

    # fix "3 . 14" -> "3.14"
    text = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text)
    # fix "100 , 000" -> "100,000"
    text = re.sub(r'(\d)\s+,\s+(\d)', r'\1,\2', text)
    # remove space before terminal punctuation
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)

    return text.strip()


def clean_generated_summary(text: str) -> str:
    """
    Post-process raw model output: remove prompt leakage, fix spacing,
    strip repeated sequences, and normalise whitespace.
    """
    if not text:
        return ""

    # Remove common seq2seq prompt prefixes that leak into output
    for prefix in ("summarize:", "tóm tắt:", "summary:", "abstract:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].lstrip()

    # Remove leading/trailing special tokens or angle-bracket tokens
    text = re.sub(r"<[^>]+>", " ", text)

    # Collapse repeated punctuation (e.g. "..." -> ".")
    text = re.sub(r"([.!?]){3,}", r"\1", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix spacing around punctuation
    text = fix_decimal_spacing(text)

    # Remove degenerate repeated n-gram patterns (simple heuristic)
    # e.g. "abc abc abc abc" → "abc"
    words = text.split()
    if len(words) > 6:
        deduped: List[str] = []
        window = 4
        for i, w in enumerate(words):
            recent = words[max(0, i - window):i]
            if recent.count(w) >= 3:
                continue
            deduped.append(w)
        text = " ".join(deduped)

    return text.strip()


def is_probably_bad_generation(text: str) -> bool:
    """
    Return True when the generated text is likely garbage / degenerate output.
    Heuristics:
    - Too short (< 5 words)
    - Extremely high ratio of non-ASCII / special characters
    - Repeating the same word/token more than 5 times consecutively
    """
    if not text:
        return True

    words = text.split()
    if len(words) < 5:
        return True

    # Check for runaway repetition: same token ≥ 5 times in a row
    for i in range(len(words) - 4):
        if len(set(words[i:i + 5])) == 1:
            return True

    # Check non-ASCII character ratio (threshold: > 80% for latin-script models)
    non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
    if len(text) > 0 and (non_alpha / len(text)) > 0.6:
        return True

    return False


def detect_garbled_text(text: str, single_letter_threshold: float = 0.10) -> bool:
    """
    Detect if the generated text is garbled/degenerate.
    Heuristics:
    - High ratio of single-letter words (indicative of spelling out or broken tokenization)
    - High ratio of non-alphanumeric symbols
    """
    if not text:
        return True
    
    words = text.split()
    if not words:
        return True
        
    single_letters = sum(1 for w in words if len(w) == 1 and w.isalpha())
    if single_letters / len(words) > single_letter_threshold:
        return True
        
    non_word_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) > 0 and (non_word_chars / len(text)) > 0.4:
        return True
        
    return False