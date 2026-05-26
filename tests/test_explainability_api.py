from __future__ import annotations

from pathlib import Path

from src.document_intelligence import DocumentIntelligenceService
from src.explainability import build_sentence_ranking_graph


def test_explain_extractive_graph(tmp_path: Path) -> None:
    text = (
        "Nhu cầu điện tăng cao trong mùa nắng nóng. "
        "Thủy điện miền Bắc vận hành thận trọng. "
        "Cần theo dõi phụ tải giờ cao điểm."
    )
    doc = tmp_path / "sample.txt"
    doc.write_text(text, encoding="utf-8")
    service = DocumentIntelligenceService(tmp_path / "store")
    payload = service.ingest_file(doc, include_embeddings=True, embedding_model="hash")
    explain = service.explain_extractive(payload["document_id"], "textrank")
    assert explain["ranking_graph"]["nodes"]
    assert explain["ranking_graph"]["summary"]


def test_build_sentence_ranking_graph_tfidf() -> None:
    text = "Alpha beta gamma. Beta gamma delta. Epsilon zeta eta."
    graph = build_sentence_ranking_graph(text, algorithm="tfidf")
    assert graph["algorithm"] == "tfidf"
    assert len(graph["nodes"]) >= 2
