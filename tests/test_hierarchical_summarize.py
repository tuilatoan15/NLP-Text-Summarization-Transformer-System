from __future__ import annotations

from pathlib import Path

from summarizers.hierarchical import hierarchical_summarize


def test_hierarchical_extractive_map() -> None:
    chunks = [
        {"chunk_id": "c1", "section_path": ["A"], "text": "Nhu cầu điện tăng cao trong mùa nắng nóng tại miền Bắc."},
        {"chunk_id": "c2", "section_path": ["B"], "text": "Thủy điện vận hành thận trọng do mực nước hồ chứa thấp."},
    ]
    source = " ".join(c["text"] for c in chunks)
    result = hierarchical_summarize(source, chunks, use_extractive_map=True)
    assert result["global_summary"]
    assert len(result["chunk_summaries"]) == 2
    assert result["strategy"] == "map-reduce"
