"""Compute real statistics from VietNews dataset records."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from statistics import mean, median, pstdev
from typing import Any, Callable

import numpy as np

from backend.services.dataset_analysis.cleaner import (
    VN_STOPWORDS,
    clean_article,
    clean_summary,
    split_sentences,
    text_fingerprint,
    tokenize_words,
)
from backend.services.dataset_analysis.loader import LoadedDataset
from backend.services.dataset_analysis.tokenizer_stats import (
    count_subword_tokens,
    count_word_tokens,
    token_stats_for_texts,
)
from evaluation.metrics import compute_rouge_batch
from src import config

# Max scatter points for JSON/charts (full stats still computed on all records)
SCATTER_VIZ_MAX = 5000
CORR_VIZ_MAX = 10000
ROUGE_FULL_MAX = 5000
NGRAM_FULL_MAX = 150000
STATS_BATCH_SIZE = 5000
TOKEN_SUBWORD_SAMPLE = 10000


def _numeric_stats(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        return {}
    vals = [float(v) for v in values]
    vals_sorted = sorted(vals)
    n = len(vals)

    def _pct(p: float) -> float:
        return round(vals_sorted[int(p * (n - 1))], 4) if n > 1 else round(vals_sorted[0], 4)

    return {
        "count": n,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "mean": round(mean(vals), 4),
        "median": round(median(vals), 4),
        "std": round(pstdev(vals), 4) if n > 1 else 0.0,
        "p25": _pct(0.25),
        "p75": _pct(0.75),
        "p95": _pct(0.95),
        "p99": _pct(0.99),
    }


def _histogram(values: list[float], bins: int = 30) -> dict[str, list]:
    if not values:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bins": [round(float(x), 2) for x in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
    }


def _lead_baseline_summary(article: str, target_words: int, *, lead_sentences: int | None = None) -> str:
    sentences = split_sentences(article, use_underthesea=False)
    if lead_sentences is not None:
        return " ".join(sentences[:lead_sentences])
    words: list[str] = []
    for sent in sentences:
        words.extend(tokenize_words(sent))
        if len(words) >= target_words:
            break
    return " ".join(words[: max(target_words, 1)])


def _extractive_coverage(article: str, summary: str) -> float:
    art = set(tokenize_words(article))
    summ = tokenize_words(summary)
    if not summ:
        return 0.0
    overlap = sum(1 for w in summ if w in art)
    return overlap / len(summ)


def _extractive_density(article: str, summary: str) -> float:
    art_tokens = tokenize_words(article)
    summ_tokens = tokenize_words(summary)
    if not art_tokens or not summ_tokens:
        return 0.0
    art_set = set(art_tokens)
    overlap = sum(1 for w in summ_tokens if w in art_set)
    return overlap / len(art_tokens)


def _novel_ngram_pct(article: str, summary: str, n: int = 1) -> float:
    art_tokens = tokenize_words(article)
    summ_tokens = tokenize_words(summary)
    if not summ_tokens:
        return 0.0
    if n == 1:
        art_set = set(art_tokens)
        novel = sum(1 for w in summ_tokens if w not in art_set)
        return novel / len(summ_tokens)
    art_ngrams = {
        tuple(art_tokens[i : i + n]) for i in range(max(0, len(art_tokens) - n + 1))
    }
    summ_ngrams = [tuple(summ_tokens[i : i + n]) for i in range(max(0, len(summ_tokens) - n + 1))]
    if not summ_ngrams:
        return 0.0
    novel = sum(1 for ng in summ_ngrams if ng not in art_ngrams)
    return novel / len(summ_ngrams)


def _zipf_data(counter: Counter, top_n: int = 50) -> list[dict[str, Any]]:
    ranked = counter.most_common()
    return [
        {"rank": i + 1, "word": word, "frequency": freq, "log_rank": round(math.log10(i + 1), 4)}
        for i, (word, freq) in enumerate(ranked[:top_n])
    ]


def _vocab_growth(tokens_stream: list[str], checkpoints: int = 30) -> list[dict[str, int]]:
    if not tokens_stream:
        return []
    step = max(1, len(tokens_stream) // checkpoints)
    seen: set[str] = set()
    curve: list[dict[str, int]] = []
    for i, tok in enumerate(tokens_stream, start=1):
        seen.add(tok)
        if i % step == 0 or i == len(tokens_stream):
            curve.append({"tokens_seen": i, "unique_vocab": len(seen)})
    return curve


def _correlation_matrix(features: dict[str, list[float]]) -> dict[str, Any]:
    keys = list(features.keys())
    if not keys or not features[keys[0]]:
        return {"labels": keys, "matrix": []}
    matrix: list[list[float]] = []
    for a in keys:
        row: list[float] = []
        va = np.array(features[a], dtype=float)
        for b in keys:
            vb = np.array(features[b], dtype=float)
            if len(va) < 2 or len(vb) < 2:
                row.append(0.0)
            else:
                corr = float(np.corrcoef(va, vb)[0, 1])
                row.append(round(corr if not math.isnan(corr) else 0.0, 4))
        matrix.append(row)
    return {"labels": keys, "matrix": matrix}


def _count_paragraphs(text: str) -> int:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return max(1, len(parts)) if text.strip() else 0


def _resolve_rouge_sample(n: int, requested: int | None) -> int:
    if requested is not None and requested > 0:
        return min(requested, n)
    if n <= ROUGE_FULL_MAX:
        return n
    return ROUGE_FULL_MAX


def compute_training_statistics() -> dict[str, Any]:
    """Training hyperparameters from config + train scripts."""
    train_script = config.PROJECT_ROOT / "train" / "train_vit5.py"
    script_defaults: dict[str, Any] = {}
    if train_script.exists():
        text = train_script.read_text(encoding="utf-8", errors="ignore")
        for key, pattern in (
            ("max_samples", r"max_samples.*?default=(\d+)"),
            ("epochs", r"epochs.*?default=(\d+)"),
            ("batch_size", r"batch_size.*?default=(\d+)"),
            ("learning_rate", r"lr.*?default=([\d.e-]+)"),
        ):
            m = re.search(pattern, text)
            if m:
                script_defaults[key] = m.group(1)

    return {
        "dataset_name": config.DATASET_NAME,
        "max_train_samples": config.MAX_TRAIN_SAMPLES,
        "validation_ratio": config.VALIDATION_RATIO,
        "train_batch_size": config.TRAIN_BATCH_SIZE,
        "eval_batch_size": config.EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": config.LEARNING_RATE,
        "num_epochs": config.NUM_EPOCHS,
        "weight_decay": config.WEIGHT_DECAY,
        "warmup_steps": config.WARMUP_STEPS,
        "max_input_tokens": config.MAX_INPUT_TOKENS,
        "max_target_tokens": config.MAX_TARGET_TOKENS,
        "default_model": config.DEFAULT_MODEL_NAME,
        "use_fp16": config.USE_FP16,
        "optimizer": "AdamW",
        "scheduler": "linear_warmup",
        "script_defaults": script_defaults,
        "source": "src.config + train/train_vit5.py",
    }


def compute_all_statistics(
    loaded: LoadedDataset,
    *,
    rouge_sample_size: int | None = None,
    ngram_sample_size: int | None = None,
    seed: int = 42,
    progress_cb: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    records = loaded.records
    n = len(records)
    is_full = loaded.limit_per_split is None

    if progress_cb:
        progress_cb("statistics:prepare", 0.05)

    art_sentences: list[int] = []
    sum_sentences: list[int] = []
    art_paragraphs: list[int] = []
    title_lens: list[int] = []
    art_words: list[int] = []
    sum_words: list[int] = []
    art_chars: list[int] = []
    sum_chars: list[int] = []
    art_tokens: list[int] = []
    sum_tokens: list[int] = []
    fingerprints: list[str] = []
    titles: list[str] = []
    category_counter: Counter = Counter()

    word_counter: Counter = Counter()
    bigram_counter: Counter = Counter()
    trigram_counter: Counter = Counter()
    stopword_hits: Counter = Counter()
    all_tokens: list[str] = []

    ngram_n = n if (is_full and n <= NGRAM_FULL_MAX) else min(n, ngram_sample_size or 5000)
    ngram_index_set = (
        set(rng.sample(range(n), ngram_n)) if ngram_n < n else set(range(n))
    )

    for batch_start in range(0, n, STATS_BATCH_SIZE):
        batch = records[batch_start : batch_start + STATS_BATCH_SIZE]
        for local_i, r in enumerate(batch):
            global_i = batch_start + local_i
            a = clean_article(r.article)
            s = clean_summary(r.abstract)
            titles.append(r.title)
            if r.category:
                category_counter[r.category] += 1

            art_sentences.append(len(split_sentences(a, use_underthesea=False)))
            sum_sentences.append(len(split_sentences(s, use_underthesea=False)))
            art_paragraphs.append(_count_paragraphs(a))
            title_lens.append(len(tokenize_words(r.title)))
            aw = len(tokenize_words(a))
            sw = len(tokenize_words(s))
            art_words.append(aw)
            sum_words.append(sw)
            art_chars.append(len(a))
            sum_chars.append(len(s))
            art_tokens.append(count_word_tokens(a))
            sum_tokens.append(count_word_tokens(s))
            fingerprints.append(text_fingerprint(a))

            if global_i in ngram_index_set:
                tokens = tokenize_words(a, remove_stopwords=False)
                all_tokens.extend(tokens)
                word_counter.update(tokens)
                for t in tokens:
                    if t in VN_STOPWORDS:
                        stopword_hits[t] += 1
                for i in range(len(tokens) - 1):
                    bigram_counter[(tokens[i], tokens[i + 1])] += 1
                for i in range(len(tokens) - 2):
                    trigram_counter[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1

        if progress_cb and n:
            progress_cb("statistics:metrics", 0.05 + 0.35 * min(1.0, (batch_start + len(batch)) / n))

    compression = [
        (sw / aw if aw > 0 else 0.0) for aw, sw in zip(art_words, sum_words)
    ]

    vocab_size = len(word_counter)
    rare_threshold = 2
    rare_words = sum(1 for _, c in word_counter.items() if c <= rare_threshold)

    if progress_cb:
        progress_cb("statistics:quality", 0.4)

    fp_counter = Counter(fingerprints)
    duplicates = sum(c - 1 for c in fp_counter.values() if c > 1)
    empty_articles = sum(1 for w in art_words if w == 0)
    empty_summaries = sum(1 for w in sum_words if w == 0)
    missing_title = sum(1 for t in titles if not t.strip())
    very_short_art = sum(1 for w in art_words if w < 30)
    very_long_art = sum(1 for w in art_words if w > 1500)
    very_short_sum = sum(1 for w in sum_words if w < 5)
    very_long_sum = sum(1 for w in sum_words if w > 200)

    art_mean, art_std = (mean(art_words), pstdev(art_words)) if art_words else (0, 0)
    outliers = sum(1 for w in art_words if art_std and abs(w - art_mean) > 3 * art_std)

    if progress_cb:
        progress_cb("statistics:rouge", 0.55)

    def _article_at(i: int) -> str:
        return clean_article(records[i].article)

    def _summary_at(i: int) -> str:
        return clean_summary(records[i].abstract)

    rouge_sample_n = _resolve_rouge_sample(n, rouge_sample_size)
    rouge_indices = rng.sample(range(n), rouge_sample_n) if n > rouge_sample_n else list(range(n))

    rouge_baselines: dict[str, Any] = {"sample_size": rouge_sample_n, "full_dataset": rouge_sample_n == n}
    baseline_specs = (
        ("lead_words_proportional", lambda i: _lead_baseline_summary(_article_at(i), sum_words[i] or 20)),
        ("lead_1", lambda i: _lead_baseline_summary(_article_at(i), sum_words[i] or 20, lead_sentences=1)),
        ("lead_3", lambda i: _lead_baseline_summary(_article_at(i), sum_words[i] or 20, lead_sentences=3)),
    )
    refs = [_summary_at(i) for i in rouge_indices]
    for name, fn in baseline_specs:
        preds = [fn(i) for i in rouge_indices]
        scores = compute_rouge_batch(preds, refs)
        rouge_baselines[name] = scores

    coverages = [_extractive_coverage(_article_at(i), _summary_at(i)) for i in rouge_indices]
    densities = [_extractive_density(_article_at(i), _summary_at(i)) for i in rouge_indices]
    novel_unigram = [_novel_ngram_pct(_article_at(i), _summary_at(i), 1) for i in rouge_indices]
    novel_bigram = [_novel_ngram_pct(_article_at(i), _summary_at(i), 2) for i in rouge_indices]

    extractive_metrics = {
        "sample_size": rouge_sample_n,
        "avg_coverage": round(mean(coverages), 4) if coverages else 0,
        "avg_density": round(mean(densities), 4) if densities else 0,
        "avg_novel_unigram_pct": round(mean(novel_unigram), 4) if novel_unigram else 0,
        "avg_novel_bigram_pct": round(mean(novel_bigram), 4) if novel_bigram else 0,
    }

    scatter_n = min(SCATTER_VIZ_MAX, n)
    scatter_idx = rng.sample(range(n), scatter_n) if n > scatter_n else list(range(n))
    scatter_points = [
        {
            "article_words": art_words[i],
            "summary_words": sum_words[i],
            "compression_ratio": round(compression[i], 4),
        }
        for i in scatter_idx
    ]

    if scatter_points:
        x = np.array([p["article_words"] for p in scatter_points], dtype=float)
        y = np.array([p["compression_ratio"] for p in scatter_points], dtype=float)
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            regression = {
                "slope": round(float(slope), 6),
                "intercept": round(float(intercept), 6),
                "equation": f"y = {slope:.6f}x + {intercept:.6f}",
                "scatter_sample_size": scatter_n,
            }
        else:
            regression = {"scatter_sample_size": scatter_n}
    else:
        regression = {}

    corr_n = min(CORR_VIZ_MAX, n)
    corr_features = {
        "article_words": [float(art_words[i]) for i in range(corr_n)],
        "summary_words": [float(sum_words[i]) for i in range(corr_n)],
        "article_sentences": [float(art_sentences[i]) for i in range(corr_n)],
        "compression_ratio": [float(compression[i]) for i in range(corr_n)],
    }

    if progress_cb:
        progress_cb("statistics:tokens", 0.75)

    token_text_sample = min(n, TOKEN_SUBWORD_SAMPLE) if n > TOKEN_SUBWORD_SAMPLE else n
    if token_text_sample < n:
        token_idx = rng.sample(range(n), token_text_sample)
    else:
        token_idx = list(range(n))
    art_texts = [clean_article(records[i].article) for i in token_idx]
    sum_texts = [clean_summary(records[i].abstract) for i in token_idx]

    token_statistics = {
        "articles": token_stats_for_texts(art_texts, sample_size=0),
        "summaries": token_stats_for_texts(sum_texts, sample_size=0),
        "subword_sample_size": token_text_sample,
        "subword_full_dataset": token_text_sample == n,
    }

    overview = {
        "dataset_name": loaded.dataset_name,
        "total_documents": n,
        "total_summaries": n,
        "splits": loaded.splits,
        "split_raw_counts": getattr(loaded, "split_raw_counts", loaded.splits),
        "total_raw_samples": loaded.total_raw_samples,
        "limit_per_split": loaded.limit_per_split,
        "full_dataset": is_full,
        "source": loaded.source,
        "columns": loaded.columns,
        "total_sentences_articles": sum(art_sentences),
        "total_sentences_summaries": sum(sum_sentences),
        "total_words_articles": sum(art_words),
        "total_words_summaries": sum(sum_words),
        "vocab_size": vocab_size,
        "unique_words": vocab_size,
        "avg_article_words": round(mean(art_words), 2) if art_words else 0,
        "avg_summary_words": round(mean(sum_words), 2) if sum_words else 0,
        "avg_article_sentences": round(mean(art_sentences), 2) if art_sentences else 0,
        "avg_summary_sentences": round(mean(sum_sentences), 2) if sum_sentences else 0,
        "avg_article_paragraphs": round(mean(art_paragraphs), 2) if art_paragraphs else 0,
        "avg_title_words": round(mean(title_lens), 2) if title_lens else 0,
        "avg_compression_ratio": round(mean(compression), 4) if compression else 0,
        "avg_reduction_pct": round(100 * (1 - mean(compression)), 2) if compression else 0,
    }

    document_stats = {
        "sentences": _numeric_stats(art_sentences),
        "words": _numeric_stats(art_words),
        "chars": _numeric_stats(art_chars),
        "tokens": _numeric_stats(art_tokens),
        "paragraphs": _numeric_stats(art_paragraphs),
    }
    summary_stats = {
        "sentences": _numeric_stats(sum_sentences),
        "words": _numeric_stats(sum_words),
        "chars": _numeric_stats(sum_chars),
        "tokens": _numeric_stats(sum_tokens),
        "compression": _numeric_stats(compression),
        "title_words": _numeric_stats(title_lens),
    }

    vocabulary = {
        "unique_words": vocab_size,
        "rare_words_count": rare_words,
        "rare_threshold": rare_threshold,
        "stopword_total_hits": sum(stopword_hits.values()),
        "top_stopwords": [{"word": w, "count": c} for w, c in stopword_hits.most_common(30)],
        "top_100_words": [{"word": w, "count": c} for w, c in word_counter.most_common(100)],
        "top_30_bigrams": [
            {"ngram": " ".join(ng), "count": c} for ng, c in bigram_counter.most_common(30)
        ],
        "top_30_trigrams": [
            {"ngram": " ".join(ng), "count": c} for ng, c in trigram_counter.most_common(30)
        ],
        "zipf": _zipf_data(word_counter, top_n=50),
        "vocab_growth": _vocab_growth(all_tokens),
        "ngram_sample_size": ngram_n,
        "ngram_full_dataset": ngram_n == n,
    }

    quality = {
        "duplicates": duplicates,
        "empty_articles": empty_articles,
        "empty_summaries": empty_summaries,
        "missing_titles": missing_title,
        "very_short_articles": very_short_art,
        "very_long_articles": very_long_art,
        "very_short_summaries": very_short_sum,
        "very_long_summaries": very_long_sum,
        "outliers_3sigma": outliers,
        "valid_pairs": n,
    }

    length_distribution = {
        "article_words": _histogram(art_words),
        "summary_words": _histogram(sum_words),
        "article_sentences": _histogram(art_sentences),
        "compression_ratio": _histogram(compression),
    }

    compression_statistics = {
        "overall": _numeric_stats(compression),
        "scatter_sample": scatter_points,
        "regression": regression,
        "scatter_sample_size": scatter_n,
    }

    correlation = _correlation_matrix(corr_features)
    correlation["sample_size"] = corr_n

    word_frequency = {
        "top_200": [{"word": w, "count": c} for w, c in word_counter.most_common(200)],
        "total_tokens": len(all_tokens),
    }

    category_stats = {
        "available": bool(category_counter),
        "categories": [
            {"name": cat, "count": cnt}
            for cat, cnt in category_counter.most_common(50)
        ],
    }

    training_statistics = compute_training_statistics()

    if progress_cb:
        progress_cb("statistics:done", 1.0)

    return {
        "overview": overview,
        "document_stats": document_stats,
        "summary_stats": summary_stats,
        "vocabulary": vocabulary,
        "quality": quality,
        "length_distribution": length_distribution,
        "token_statistics": token_statistics,
        "compression_statistics": compression_statistics,
        "correlation": correlation,
        "word_frequency": word_frequency,
        "category_stats": category_stats,
        "rouge_baseline": rouge_baselines,
        "extractive_metrics": extractive_metrics,
        "training_statistics": training_statistics,
    }
