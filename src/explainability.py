"""
explainability.py - Evidence highlights for extractive summaries.
"""

from collections import Counter
import math
import re

from src.preprocess import split_sentences


STOPWORDS = {
    "và", "là", "của", "có", "cho", "các", "một", "những", "được", "trong",
    "với", "khi", "từ", "đã", "này", "đó", "về", "theo", "tại", "ra",
}


def build_extractive_explanations(source_text: str, summary: str) -> dict:
    source_sentences = split_sentences(source_text)
    summary_sentences = split_sentences(summary)
    ranked = _rank_sentences(source_sentences)

    highlights = []
    for summary_sentence in summary_sentences:
        index, source_sentence, similarity = _best_match(summary_sentence, source_sentences)
        rank = ranked.get(index, len(source_sentences))
        keywords = _top_keywords(source_sentence, limit=5)
        highlights.append({
            "sentence": source_sentence,
            "summary_sentence": summary_sentence,
            "source_index": index,
            "importance_rank": rank,
            "similarity": round(similarity, 4),
            "keywords": keywords,
            "reason": _explain_reason(rank, similarity, keywords),
        })

    return {
        "sentences": source_sentences,
        "highlighted_sentence_indexes": [h["source_index"] for h in highlights],
        "highlights": highlights,
    }


def _rank_sentences(sentences: list[str]) -> dict[int, int]:
    if not sentences:
        return {}

    doc_freq = Counter()
    tokenized = []
    for sentence in sentences:
        tokens = set(_tokens(sentence))
        tokenized.append(tokens)
        doc_freq.update(tokens)

    total = len(sentences)
    scored = []
    for index, tokens in enumerate(tokenized):
        score = 0.0
        for token in tokens:
            score += math.log((total + 1) / (doc_freq[token] + 1)) + 1
        score += min(len(tokens), 30) / 30
        scored.append((score, index))

    scored.sort(reverse=True)
    return {index: rank + 1 for rank, (_, index) in enumerate(scored)}


def _best_match(sentence: str, candidates: list[str]) -> tuple[int, str, float]:
    if not candidates:
        return -1, "", 0.0
    sentence_tokens = set(_tokens(sentence))
    best = (0, candidates[0], 0.0)
    for index, candidate in enumerate(candidates):
        candidate_tokens = set(_tokens(candidate))
        similarity = _jaccard(sentence_tokens, candidate_tokens)
        if similarity > best[2]:
            best = (index, candidate, similarity)
    return best


def _tokens(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


def _top_keywords(sentence: str, limit: int) -> list[str]:
    counts = Counter(_tokens(sentence))
    return [word for word, _ in counts.most_common(limit)]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _explain_reason(rank: int, similarity: float, keywords: list[str]) -> str:
    keyword_text = ", ".join(keywords) if keywords else "các từ khóa chính"
    if rank <= 3:
        return (
            f"Câu này nằm trong nhóm câu có điểm quan trọng cao nhất và chứa "
            f"các từ khóa nổi bật: {keyword_text}."
        )
    return (
        f"Câu này khớp trực tiếp với câu trong bản tóm tắt "
        f"(độ tương đồng {similarity:.2f}) và chứa: {keyword_text}."
    )
