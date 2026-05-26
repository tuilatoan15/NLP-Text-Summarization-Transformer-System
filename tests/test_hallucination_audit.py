from __future__ import annotations

from evaluation.hallucination import audit_summary


def test_audit_summary_semantic_fields() -> None:
    source = "Thủy điện miền Bắc vận hành thận trọng do mực nước thấp."
    summary = "Thủy điện miền Bắc cần vận hành thận trọng."
    result = audit_summary(summary, source, mode="fast")
    assert "semantic_coverage" in result
    assert "sentence_audits" in result
    assert result["hallucination_risk"] in {"low", "medium", "high"}
