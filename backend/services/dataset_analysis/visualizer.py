"""Generate PNG charts for VietNews dataset analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from backend.services.dataset_analysis.exporter import CHARTS_DIR
from src.utils import logger


def _save_fig(name: str) -> str:
    path = CHARTS_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return str(path)


def _hist_from_distribution(dist: dict[str, list], title: str, xlabel: str, filename: str) -> str | None:
    bins = dist.get("bins") or []
    counts = dist.get("counts") or []
    if not bins or not counts:
        return None
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
    plt.figure(figsize=(8, 4.5))
    plt.bar(centers, counts, width=(bins[1] - bins[0]) * 0.9, color="#0ea5e9", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Số mẫu")
    plt.grid(axis="y", alpha=0.3)
    return _save_fig(filename)


def generate_all_charts(stats: dict[str, Any]) -> dict[str, str]:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    charts: dict[str, str] = {}

    overview = stats.get("overview", {})
    splits = overview.get("splits", {})
    if splits:
        plt.figure(figsize=(6, 5))
        labels = list(splits.keys())
        values = [splits[k] for k in labels]
        colors = ["#6366f1", "#10b981", "#f59e0b", "#f43f5e"][: len(labels)]
        plt.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
        plt.title("Phân chia Train / Val / Test")
        charts["split_pie"] = _save_fig("split_pie")

    length_dist = stats.get("length_distribution", {})
    for key, title, xlab, fname in (
        ("article_words", "Phân phối độ dài bài viết (từ)", "Số từ", "hist_article_words"),
        ("summary_words", "Phân phối độ dài tóm tắt (từ)", "Số từ", "hist_summary_words"),
        ("compression_ratio", "Phân phối tỷ lệ nén", "Tỷ lệ nén", "hist_compression"),
    ):
        p = _hist_from_distribution(length_dist.get(key, {}), title, xlab, fname)
        if p:
            charts[fname] = p

    vocab = stats.get("vocabulary", {})
    top_words = vocab.get("top_100_words", [])[:20]
    if top_words:
        plt.figure(figsize=(9, 5))
        words = [w["word"] for w in top_words][::-1]
        counts = [w["count"] for w in top_words][::-1]
        plt.barh(words, counts, color="#6366f1")
        plt.title("Top 20 từ phổ biến")
        plt.xlabel("Tần suất")
        charts["bar_top_words"] = _save_fig("bar_top_words")

    zipf = vocab.get("zipf", [])
    if zipf:
        plt.figure(figsize=(7, 4.5))
        ranks = [z["rank"] for z in zipf]
        freqs = [z["frequency"] for z in zipf]
        plt.loglog(ranks, freqs, "o-", color="#10b981", markersize=4)
        plt.title("Phân phối Zipf (log-log)")
        plt.xlabel("Hạng (rank)")
        plt.ylabel("Tần suất")
        plt.grid(True, which="both", alpha=0.3)
        charts["zipf_line"] = _save_fig("zipf_line")

    growth = vocab.get("vocab_growth", [])
    if growth:
        plt.figure(figsize=(7, 4.5))
        xs = [g["tokens_seen"] for g in growth]
        ys = [g["unique_vocab"] for g in growth]
        plt.plot(xs, ys, color="#0ea5e9", linewidth=2)
        plt.title("Tăng trưởng từ vựng")
        plt.xlabel("Tokens đã xem")
        plt.ylabel("Từ vựng duy nhất")
        plt.grid(alpha=0.3)
        charts["vocab_growth_line"] = _save_fig("vocab_growth_line")

    comp = stats.get("compression_statistics", {})
    scatter = comp.get("scatter_sample", [])
    if scatter:
        plt.figure(figsize=(7, 5))
        x = [p["article_words"] for p in scatter]
        y = [p["compression_ratio"] for p in scatter]
        plt.scatter(x, y, alpha=0.35, s=12, color="#6366f1")
        reg = comp.get("regression", {})
        if reg.get("slope") is not None:
            xs = np.linspace(min(x), max(x), 50)
            ys = reg["slope"] * xs + reg["intercept"]
            plt.plot(xs, ys, color="#f43f5e", linewidth=2, label=reg.get("equation", "regression"))
            plt.legend()
        plt.title("Tỷ lệ nén vs độ dài bài viết")
        plt.xlabel("Số từ bài viết")
        plt.ylabel("Tỷ lệ nén")
        plt.grid(alpha=0.3)
        charts["compression_scatter"] = _save_fig("compression_scatter")

    corr = stats.get("correlation", {})
    labels = corr.get("labels", [])
    matrix = corr.get("matrix", [])
    if labels and matrix:
        plt.figure(figsize=(6, 5))
        arr = np.array(matrix, dtype=float)
        im = plt.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("Ma trận tương quan")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        charts["correlation_heatmap"] = _save_fig("correlation_heatmap")

    doc_stats = stats.get("document_stats", {})
    sum_stats = stats.get("summary_stats", {})
    if doc_stats.get("words") and sum_stats.get("words"):
        plt.figure(figsize=(7, 4.5))
        metrics = ["min", "mean", "median", "max"]
        art_vals = [doc_stats["words"].get(m, 0) for m in metrics]
        sum_vals = [sum_stats["words"].get(m, 0) for m in metrics]
        x = np.arange(len(metrics))
        w = 0.35
        plt.bar(x - w / 2, art_vals, w, label="Bài viết", color="#0ea5e9")
        plt.bar(x + w / 2, sum_vals, w, label="Tóm tắt", color="#10b981")
        plt.xticks(x, metrics)
        plt.title("So sánh độ dài từ (min/mean/median/max)")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        charts["bar_length_compare"] = _save_fig("bar_length_compare")

    # Box plot from raw histogram centers (approximation)
    aw = length_dist.get("article_words", {})
    sw = length_dist.get("summary_words", {})
    if aw.get("bins") and sw.get("bins"):
        plt.figure(figsize=(6, 4.5))
        art_data = []
        for c, lo, hi in zip(aw["counts"], aw["bins"][:-1], aw["bins"][1:]):
            art_data.extend([((lo + hi) / 2)] * c)
        sum_data = []
        for c, lo, hi in zip(sw["counts"], sw["bins"][:-1], sw["bins"][1:]):
            sum_data.extend([((lo + hi) / 2)] * c)
        if art_data and sum_data:
            plt.boxplot([art_data, sum_data], labels=["Bài viết", "Tóm tắt"])
            plt.title("Box plot độ dài (từ)")
            plt.grid(axis="y", alpha=0.3)
            charts["box_length"] = _save_fig("box_length")

    # Word cloud via matplotlib if word list available
    wf = stats.get("word_frequency", {}).get("top_200", [])[:80]
    if wf:
        try:
            from wordcloud import WordCloud

            freq = {w["word"]: w["count"] for w in wf}
            wc = WordCloud(
                width=900,
                height=450,
                background_color="white",
                colormap="viridis",
                font_path=None,
            ).generate_from_frequencies(freq)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title("Word Cloud — tần suất từ")
            charts["wordcloud"] = _save_fig("wordcloud")
        except Exception as exc:
            logger.warning("WordCloud skipped: %s", exc)

    # N-gram bar
    bigrams = vocab.get("top_30_bigrams", [])[:15]
    if bigrams:
        plt.figure(figsize=(9, 5))
        labels_ng = [b["ngram"] for b in bigrams][::-1]
        vals = [b["count"] for b in bigrams][::-1]
        plt.barh(labels_ng, vals, color="#a855f7")
        plt.title("Top 15 bigrams")
        plt.xlabel("Tần suất")
        charts["bar_bigrams"] = _save_fig("bar_bigrams")

    quality = stats.get("quality", {})
    if quality:
        plt.figure(figsize=(7, 4.5))
        q_labels = [
            "Trùng lặp",
            "Rỗng (article)",
            "Rỗng (summary)",
            "Quá ngắn",
            "Quá dài",
            "Outliers",
        ]
        q_vals = [
            quality.get("duplicates", 0),
            quality.get("empty_articles", 0),
            quality.get("empty_summaries", 0),
            quality.get("very_short_articles", 0),
            quality.get("very_long_articles", 0),
            quality.get("outliers_3sigma", 0),
        ]
        plt.bar(q_labels, q_vals, color="#f43f5e", alpha=0.8)
        plt.title("Chỉ số chất lượng dữ liệu")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Số mẫu")
        plt.grid(axis="y", alpha=0.3)
        charts["bar_quality"] = _save_fig("bar_quality")

    logger.info("Generated %s charts in %s", len(charts), CHARTS_DIR)
    return charts
