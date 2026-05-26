from __future__ import annotations

from pathlib import Path

from pipeline.ingest_pipeline import IngestPipeline
from pipeline.schema import (
    ChunkingConfig,
    CleaningConfig,
    DocumentElement,
    DocumentMetadata,
    EmbeddingConfig,
    ExtractedDocument,
    IngestConfig,
)
from preprocess.chunker import SemanticChunker
from preprocess.cleaner import DocumentCleaner


def test_cleaner_removes_repeated_headers_and_page_numbers() -> None:
    document = ExtractedDocument(
        document_id="doc-1",
        metadata=DocumentMetadata(source_path="x.pdf", source_type="pdf", pages=2),
        text="",
        elements=[
            DocumentElement("Báo cáo vận hành điện", "paragraph", page_number=1),
            DocumentElement("1", "footer", page_number=1),
            DocumentElement("Nhu cầu tiêu thụ điện tăng cao trong mùa nắng nóng.", "paragraph", page_number=1),
            DocumentElement("Báo cáo vận hành điện", "paragraph", page_number=2),
            DocumentElement("2", "footer", page_number=2),
            DocumentElement("Nhà máy thủy điện vận hành thận trọng do mực nước hồ thấp.", "paragraph", page_number=2),
        ],
    )

    cleaned = DocumentCleaner(CleaningConfig(min_repeated_header_pages=2)).clean(document)

    assert "Báo cáo vận hành điện" not in cleaned.text
    assert "\n\n1\n\n" not in cleaned.text
    assert "Nhu cầu tiêu thụ điện tăng cao" in cleaned.text
    assert "mực nước hồ thấp" in cleaned.text


def test_chunker_respects_sentence_boundaries_and_context() -> None:
    elements = [
        DocumentElement("TỔNG QUAN", "heading", page_number=1, level=1, section_path=["TỔNG QUAN"]),
        DocumentElement(
            "Nhu cầu tiêu thụ điện tăng cao trong mùa nắng nóng. "
            "Các địa phương phải theo dõi phụ tải giờ cao điểm. "
            "Nhà máy thủy điện vận hành thận trọng do mực nước hồ chứa thấp.",
            "paragraph",
            page_number=1,
            section_path=["TỔNG QUAN"],
        ),
    ]
    chunker = SemanticChunker(
        ChunkingConfig(target_tokens=20, min_tokens=8, max_tokens=34, overlap_tokens=6)
    )

    chunks = chunker.chunk("doc-2", elements)

    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.text[-1] in ".!?" or "TỔNG QUAN" in chunk.text for chunk in chunks)
    assert chunks[0].section_path == ["TỔNG QUAN"]


def test_txt_ingest_pipeline_without_embeddings(tmp_path: Path) -> None:
    path = tmp_path / "dien_luc.txt"
    path.write_text(
        "TỔNG QUAN\n\n"
        "Tập đoàn Điện lực Việt Nam cho biết nhu cầu tiêu thụ điện trong mùa nắng nóng tiếp tục tăng cao.\n\n"
        "Các nhà máy thủy điện ở miền Bắc được yêu cầu vận hành thận trọng do mực nước hồ chứa chưa phục hồi.",
        encoding="utf-8",
    )
    config = IngestConfig(
        chunking=ChunkingConfig(target_tokens=60, min_tokens=10, max_tokens=90, overlap_tokens=10),
        embedding=EmbeddingConfig(model_name="hash", show_progress=False),
        enable_embeddings=False,
    )

    result = IngestPipeline(config).ingest(path)
    payload = result.to_dict()

    assert payload["document_id"]
    assert payload["metadata"]["source_type"] == "txt"
    assert "nhu cầu tiêu thụ điện" in payload["clean_text"]
    assert payload["chunks"]
    assert payload["embeddings"] is None
    assert payload["structure"]["chunking"]["strategy"]
