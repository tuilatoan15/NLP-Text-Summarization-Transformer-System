"""Small Vietnamese semantic chunking demo without loading embedding models."""

from __future__ import annotations

import json

from pipeline.schema import ChunkingConfig, DocumentElement
from preprocess.chunker import SemanticChunker


def main() -> None:
    elements = [
        DocumentElement("TỔNG QUAN NGÀNH ĐIỆN", "heading", page_number=1, level=1),
        DocumentElement(
            "Nhu cầu tiêu thụ điện trong mùa nắng nóng tăng mạnh tại nhiều địa phương. "
            "Các khu vực đô thị ghi nhận phụ tải cao vào giờ cao điểm.",
            "paragraph",
            page_number=1,
            section_path=["TỔNG QUAN NGÀNH ĐIỆN"],
        ),
        DocumentElement(
            "Các nhà máy thủy điện miền Bắc được yêu cầu vận hành thận trọng do mực nước hồ chứa chưa phục hồi hoàn toàn. "
            "Điều này làm giảm dư địa huy động nguồn thủy điện trong một số thời điểm.",
            "paragraph",
            page_number=1,
            section_path=["TỔNG QUAN NGÀNH ĐIỆN"],
        ),
        DocumentElement("KHUYẾN NGHỊ VẬN HÀNH", "heading", page_number=2, level=1),
        DocumentElement(
            "- Theo dõi sát mực nước hồ chứa\n- Ưu tiên tiết kiệm điện giờ cao điểm\n- Bổ sung nguồn dự phòng khi phụ tải tăng",
            "bullet",
            page_number=2,
            section_path=["KHUYẾN NGHỊ VẬN HÀNH"],
        ),
    ]
    chunker = SemanticChunker(
        ChunkingConfig(
            target_tokens=55,
            min_tokens=20,
            max_tokens=80,
            overlap_tokens=12,
            semantic_model_name=None,
        )
    )
    chunks = chunker.chunk("demo-vietnamese-power", elements)
    print(json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
