from __future__ import annotations

from pathlib import Path

from src.document_intelligence import DocumentIntelligenceService


def test_document_intelligence_ingest_search_assets(tmp_path: Path) -> None:
    document_path = tmp_path / "evn.txt"
    document_path.write_text(
        "TỔNG QUAN\n\n"
        "Nhu cầu tiêu thụ điện trong mùa nắng nóng tiếp tục tăng cao tại nhiều địa phương.\n\n"
        "Các nhà máy thủy điện miền Bắc phải vận hành thận trọng do mực nước hồ chứa chưa phục hồi.\n\n"
        "KHUYẾN NGHỊ\n\n"
        "- Theo dõi phụ tải giờ cao điểm\n"
        "- Duy trì nguồn điện dự phòng",
        encoding="utf-8",
    )
    service = DocumentIntelligenceService(tmp_path / "store")

    payload = service.ingest_file(document_path, include_embeddings=True, embedding_model="hash")
    search = service.semantic_search(payload["document_id"], "thủy điện miền Bắc", top_k=2)
    citations = service.ground_summary("Thủy điện miền Bắc cần vận hành thận trọng.", payload["chunks"])

    assert payload["document_id"]
    assert payload["analysis_assets"]["overview"]["document_overview"]
    assert payload["visualization"]["chunk_graph"]["nodes"]
    assert search["results"]
    assert citations[0]["evidence"]
