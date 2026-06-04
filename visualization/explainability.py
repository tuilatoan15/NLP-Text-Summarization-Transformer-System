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


def build_sentence_ranking_graph(
    source_text: str,
    algorithm: str = "textrank",
    top_edges: int = 40,
) -> dict:
    from summarizers.extractive.extractive_summarizer import (
        EXTRACTIVE_RUNNERS,
        _cosine_similarity,
        _pagerank,
        _prepare_sentences,
        _sentence_matrix,
        summarize_extractive_algorithm,
    )


    key = algorithm.strip().lower()
    if key not in EXTRACTIVE_RUNNERS:
        raise KeyError(f"Unsupported explainability algorithm: {algorithm}")

    sentences = _prepare_sentences(source_text)
    details = summarize_extractive_algorithm(source_text, key, sentence_count=min(8, max(1, len(sentences))))
    selected_indexes = set(details.get("highlighted_sentence_indexes") or [])

    nodes = [
        {
            "id": f"s{idx}",
            "label": sentence[:120],
            "index": idx,
            "score": next(
                (s.get("sentence_score", 0.0) for s in details.get("selected_sentences", []) if s.get("sentence_index") == idx),
                0.0,
            ),
            "selected": idx in selected_indexes,
        }
        for idx, sentence in enumerate(sentences)
    ]

    edges: list[dict] = []
    if len(sentences) >= 2:
        matrix, _ = _sentence_matrix(sentences)
        if key == "lexrank":
            similarity = _cosine_similarity(matrix)
            import numpy as np

            threshold = float(np.mean(similarity[similarity > 0])) if np.any(similarity > 0) else 0.1
            graph = np.where(similarity >= threshold, similarity, 0.0)
            scores = _pagerank(graph)
        elif key in {"lsa", "tfidf"}:
            details_only = EXTRACTIVE_RUNNERS[key](source_text, min(8, len(sentences)))
            score_map = {s["sentence_index"]: s.get("sentence_score", 0.0) for s in details_only.get("selected_sentences", [])}
            scores = [score_map.get(i, 0.0) for i in range(len(sentences))]
            similarity = _cosine_similarity(matrix)
        else:
            similarity = _cosine_similarity(matrix)
            scores = _pagerank(similarity)

        flat = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                weight = float(similarity[i, j]) if similarity.size else 0.0
                if weight > 0.05:
                    flat.append((weight, i, j))
        flat.sort(reverse=True)
        for weight, i, j in flat[:top_edges]:
            edges.append({"source": f"s{i}", "target": f"s{j}", "weight": round(weight, 4)})

        for idx, node in enumerate(nodes):
            if isinstance(scores, list):
                node["rank_score"] = round(float(scores[idx]) if idx < len(scores) else 0.0, 4)
            else:
                import numpy as np

                node["rank_score"] = round(float(scores[idx]), 4) if idx < len(scores) else 0.0

    return {
        "algorithm": algorithm,
        "nodes": nodes,
        "edges": edges,
        "selected_sentence_indexes": list(selected_indexes),
        "summary": details.get("summary", ""),
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


def visualize_centrality_graph(
    source_text: str,
    algorithm: str = "textrank",
    top_edges: int = 25,
    output_path: str | None = None,
) -> str:
    """Generate and save a visual Sentence Centrality Graph image for thesis report."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Build the graph JSON representation first
        graph_data = build_sentence_ranking_graph(source_text, algorithm, top_edges)
        
        G = nx.Graph()
        
        # Add nodes with attributes
        node_colors = []
        node_sizes = []
        labels = {}
        
        for node in graph_data["nodes"]:
            node_id = node["id"]
            rank_score = node.get("rank_score", 0.0)
            selected = node["selected"]
            
            G.add_node(node_id, label=node["label"], rank_score=rank_score, selected=selected)
            
            # Label is just first 30 characters with node id prefix
            labels[node_id] = f"S{node['index']} ({rank_score:.2f})"
            
            # Selected nodes are highlighted
            if selected:
                node_colors.append("#ff5722")  # Deep Orange
                node_sizes.append(1000 + rank_score * 5000)
            else:
                node_colors.append("#2196f3")  # Blue
                node_sizes.append(400 + rank_score * 3000)
                
        # Add edges
        for edge in graph_data["edges"]:
            G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
            
        plt.figure(figsize=(10, 8), dpi=150)
        
        # Compute layout
        pos = nx.spring_layout(G, k=0.45, seed=42)
        
        # Draw edges with opacity based on similarity weight
        edge_list = list(G.edges(data=True))
        if edge_list:
            edge_weights = [e[2]["weight"] for e in edge_list]
            max_weight = max(edge_weights) if edge_weights else 1.0
            for u, v, d in edge_list:
                w = d["weight"]
                # Alpha between 0.1 and 0.8
                alpha = 0.1 + 0.7 * (w / max_weight)
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], 
                    width=1.0 + w * 3.0, 
                    alpha=alpha, 
                    edge_color="#9e9e9e"
                )
                
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors, 
            node_size=node_sizes, 
            edgecolors="#333333", 
            linewidths=1.0
        )
        
        # Draw labels with clear font
        nx.draw_networkx_labels(
            G, pos, labels, 
            font_size=8, 
            font_weight="bold", 
            font_color="#ffffff",
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
        )
        
        plt.title(
            f"Sentence Centrality Relation Graph ({algorithm.upper()})\nSelected summary sentences highlighted in Orange", 
            fontsize=12, fontweight="bold", pad=15
        )
        plt.axis("off")
        plt.tight_layout()
        
        if output_path is None:
            from src import config
            output_path = str(config.RESULTS_DIR / f"{algorithm}_sentence_graph.png")
            
        plt.savefig(output_path, format="png", bbox_inches="tight")
        plt.close()
        
        from src.utils import logger
        logger.info(f"Successfully generated and saved centrality graph image: {output_path}")
        return output_path
        
    except Exception as exc:
        from src.utils import logger
        logger.error(f"Failed to generate centrality graph visualization: {exc}", exc_info=True)
        return ""

