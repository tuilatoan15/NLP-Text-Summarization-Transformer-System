#!/usr/bin/env python3
"""Regenerate docs/flowcharts/00-main-sequence-diagram.drawio with proper UML activations."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from uml_sequence_builder import Message, Participant, SequenceDiagram  # noqa: E402

OUT = ROOT / "docs" / "flowcharts" / "00-main-sequence-diagram.drawio"

SUMMARIZE = SequenceDiagram(
    title="Luồng chính: Tóm tắt văn bản — Playground so sánh thuật toán",
    participants=[
        Participant("user", "Người dùng", 60, "actor"),
        Participant("fe", "Giao diện Playground", 340, "box"),
        Participant("api", "Máy chủ", 620, "box"),
        Participant("svc", "Dịch vụ tóm tắt", 900, "box"),
        Participant("model", "Mô hình AI", 1180, "box"),
        Participant("store", "Kho lưu trữ", 1460, "box"),
    ],
    messages=[
        Message("user", "fe", "Nhập văn bản, chọn thuật toán, nhấn Chạy", "sync"),
        Message("fe", "fe", "Giao diện chuẩn bị yêu cầu so sánh", "self"),
        Message("fe", "api", "Gửi yêu cầu so sánh lên máy chủ", "sync"),
        Message("api", "api", "Máy chủ kiểm tra văn bản và thiết lập SSE", "self"),
        Message("api", "svc", "Máy chủ chuyển yêu cầu sang dịch vụ", "sync"),
        Message("api", "fe", "SSE: thông báo bắt đầu (start)", "async"),
        Message("svc", "api", "Dịch vụ trả kết quả từng thuật toán", "return"),
        Message("api", "fe", "SSE: cập nhật tiến trình (running/done)", "async"),
        Message("svc", "model", "Dịch vụ gọi mô hình AI tóm tắt", "sync"),
        Message("model", "svc", "Mô hình trả kết quả và chỉ số", "return"),
        Message("svc", "store", "Dịch vụ lưu kết quả vào kho lưu trữ", "sync"),
        Message("api", "fe", "SSE: thông báo hoàn tất (finished)", "async"),
        Message("fe", "fe", "Giao diện cập nhật bảng so sánh", "self"),
    ],
    y_start=120,
    y_gap=38,
    page_w=1680,
    extra_frames=[
        {
            "label": "vòng lặp [mỗi thuật toán]",
            "x": 40,
            "y": 380,
            "w": 1460,
            "h": 172,
        }
    ],
)

RAG = SequenceDiagram(
    title="Luồng chính: RAG Chat — hỏi đáp tài liệu với phản hồi liên tục",
    participants=[
        Participant("user", "Người dùng", 60, "actor"),
        Participant("fe", "Giao diện Chat", 320, "box"),
        Participant("api", "Máy chủ", 580, "box"),
        Participant("svc", "Dịch vụ RAG Chat", 840, "box"),
        Participant("repo", "Kho hội thoại", 1100, "box"),
        Participant("vec", "Kho vector", 1360, "box"),
        Participant("ret", "Bộ truy xuất lai", 1620, "box"),
        Participant("gen", "Bộ sinh có căn cứ", 1880, "box"),
        Participant("llm", "Mô hình LLM", 2140, "box"),
    ],
    messages=[
        Message("user", "fe", "Người dùng gửi câu hỏi", "sync"),
        Message("fe", "fe", "Giao diện hiển thị tin nhắn và chỗ trả lời tạm", "self"),
        Message("fe", "fe", "Giao diện chuẩn bị yêu cầu hỏi đáp", "self"),
        Message("fe", "api", "Gửi yêu cầu hỏi đáp lên máy chủ", "sync"),
        Message("api", "svc", "Máy chủ chuyển yêu cầu sang dịch vụ RAG", "sync"),
        Message("svc", "svc", "Dịch vụ chuẩn bị phiên hỏi đáp", "self"),
        Message("svc", "repo", "Dịch vụ tạo hoặc mở hội thoại, lưu câu hỏi", "sync"),
        Message("repo", "svc", "Kho hội thoại trả lịch sử tin nhắn", "return"),
        Message("svc", "svc", "Dịch vụ phân loại ý định câu hỏi", "self"),
        Message("svc", "svc", "Dịch vụ chuyển câu hỏi thành vector", "self"),
        Message("svc", "vec", "Dịch vụ truy vấn kho vector", "sync"),
        Message("vec", "svc", "Kho vector trả danh sách đoạn văn ứng viên", "return"),
        Message("svc", "ret", "Dịch vụ xếp hạng và chọn đoạn phù hợp", "sync"),
        Message("ret", "svc", "Bộ truy xuất trả các đoạn được chọn", "return"),
        Message("svc", "svc", "Dịch vụ nén và tổ chức ngữ cảnh", "self"),
        Message("svc", "gen", "Dịch vụ soạn lời nhắc từ câu hỏi", "sync"),
        Message("api", "fe", "SSE: sự kiện tiến trình (stage)", "async"),
        Message("svc", "gen", "Dịch vụ yêu cầu sinh câu trả lời có căn cứ", "sync"),
        Message("gen", "llm", "Bộ sinh gọi LLM tạo từng phần trả lời", "sync"),
        Message("api", "fe", "SSE: truyền từng phần trả lời (token)", "async"),
        Message("fe", "fe", "Giao diện cập nhật tiến trình xử lý", "self"),
        Message("svc", "svc", "Dịch vụ đánh giá độ tin cậy câu trả lời", "self"),
        Message("svc", "repo", "Dịch vụ lưu câu trả lời và trích dẫn", "sync"),
        Message("api", "fe", "SSE: thông báo hoàn tất (done)", "async"),
        Message("fe", "fe", "Giao diện hiển thị câu trả lời và trích dẫn", "self"),
    ],
    y_start=120,
    y_gap=32,
    page_w=2340,
)


def _extract_diagram_xml(full: str) -> str:
    start = full.index("<diagram")
    end = full.rindex("</diagram>") + len("</diagram>")
    return full[start:end]


def validate(path: Path) -> bool:
    try:
        tree = ET.parse(path)
        return len(tree.getroot().findall("diagram")) == 2
    except ET.ParseError:
        return False


def main() -> None:
    d1 = _extract_diagram_xml(SUMMARIZE.build_drawio("diag-main-summarize-seq", "A. Tóm tắt"))
    d2 = _extract_diagram_xml(RAG.build_drawio("diag-main-rag-seq", "B. RAG Chat"))
    content = (
        '<mxfile host="app.diagrams.net" modified="2026-07-04T04:00:00.000Z" '
        'agent="UML-Seq-Builder" version="22.1.0" type="device">\n'
        f"  {d1}\n\n  {d2}\n\n</mxfile>\n"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content, encoding="utf-8")
    print(f"Wrote {OUT} ({'valid' if validate(OUT) else 'INVALID'})")
    print(f"  Tab A steps: {len(SUMMARIZE.messages)}")
    print(f"  Tab B steps: {len(RAG.messages)}")


if __name__ == "__main__":
    main()
