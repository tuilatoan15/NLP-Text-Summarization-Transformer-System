#!/usr/bin/env python3
"""Generate docs/flowcharts/00-full-project-sequence.drawio — master UML sequence."""

from __future__ import annotations

import html
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from uml_sequence_builder import (  # noqa: E402
    Message,
    Participant,
    SequenceDiagram,
)

OUT_DRAWIO = ROOT / "docs" / "flowcharts" / "00-full-project-sequence.drawio"
OUT_SVG = ROOT / "docs" / "flowcharts" / "00-full-project-sequence.svg"

TITLE = "Sơ đồ tuần tự tổng thể — Hệ thống NLP Tóm tắt văn bản"

PARTICIPANTS = [
    Participant("user", "Người dùng", 60, "actor"),
    Participant("fe", "Giao diện Web", 320, "box"),
    Participant("api", "Máy chủ API", 580, "box"),
    Participant("sum", "Dịch vụ tóm tắt", 840, "box"),
    Participant("rag", "Dịch vụ RAG", 1100, "box"),
    Participant("ai", "Mô hình AI", 1360, "box"),
    Participant("store", "Kho lưu trữ", 1620, "box"),
]

# Compact 18 steps — short Vietnamese labels (unchanged count)
MESSAGES = [
    Message("api", "ai", "Khởi động máy chủ, nạp mô hình AI", "sync"),
    Message("user", "fe", "Nhập văn bản trên Playground", "sync"),
    Message("fe", "api", "Gửi yêu cầu so sánh tóm tắt", "sync"),
    Message("api", "sum", "Dịch vụ tóm tắt AI, lưu kho", "sync"),
    Message("api", "fe", "SSE: kết quả so sánh về giao diện", "async"),
    Message("user", "fe", "Tải tài liệu lên trang Chat", "sync"),
    Message("api", "rag", "Nhúng vector AI và lưu kho", "sync"),
    Message("user", "fe", "Đặt câu hỏi về tài liệu", "sync"),
    Message("fe", "api", "Gửi câu hỏi hỏi đáp RAG", "sync"),
    Message("api", "rag", "Truy xuất ngữ cảnh, sinh trả lời", "sync"),
    Message("api", "fe", "SSE: luồng token về giao diện", "async"),
    Message("user", "fe", "Nạp tài liệu Document Intelligence", "sync"),
    Message("api", "sum", "Tóm tắt phân cấp AI, lưu kho", "sync"),
    Message("api", "store", "Tìm kiếm ngữ nghĩa trên tài liệu", "sync"),
    Message("user", "fe", "Mở trang phân tích bộ dữ liệu", "sync"),
    Message("api", "fe", "Trả báo cáo thống kê VietNews", "return"),
    Message("user", "fe", "Mở Dashboard và Benchmark", "sync"),
    Message("api", "fe", "Trả dữ liệu tổng quan hệ thống", "return"),
]

PAGE_W = 2100
Y_GAP = 30


def build_svg(diag: SequenceDiagram) -> str:
    x_map = diag._x()
    ys = diag._assign_y()
    act_periods = diag._compute_activations(ys)
    w, h = PAGE_W, max(ys) + 110 if ys else 740

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" style="background:#fff">',
        "<style>text{fill:#000;font-family:Arial,sans-serif;font-size:11px}"
        ".title{font-size:15px;font-weight:bold}</style>",
        f'<text x="{w/2}" y="32" text-anchor="middle" class="title">{html.escape(TITLE)}</text>',
    ]

    box_w = 130
    header_y = 60
    for p in diag.participants:
        x = p.x
        if p.kind == "actor":
            lines.append(f'<circle cx="{x}" cy="{header_y + 15}" r="14" fill="#fff" stroke="#000"/>')
            lines.append(f'<line x1="{x}" y1="{header_y + 29}" x2="{x}" y2="{header_y + 48}" stroke="#000"/>')
            lines.append(f'<line x1="{x-12}" y1="{header_y + 36}" x2="{x+12}" y2="{header_y + 36}" stroke="#000"/>')
            lines.append(f'<text x="{x}" y="{header_y + 62}" text-anchor="middle">{html.escape(p.label)}</text>')
        else:
            lines.append(
                f'<rect x="{x - box_w//2}" y="{header_y}" width="{box_w}" height="40" '
                f'fill="#fff" stroke="#000"/>'
            )
            lines.append(
                f'<text x="{x}" y="{header_y + 24}" text-anchor="middle" font-weight="bold">'
                f"{html.escape(p.label)}</text>"
            )

    bottom_y = max(ys) + 50
    for p in diag.participants:
        x = p.x
        lines.append(
            f'<line x1="{x}" y1="120" x2="{x}" y2="{bottom_y}" stroke="#000" stroke-dasharray="4,4"/>'
        )

    for pid, periods in act_periods.items():
        x = x_map[pid]
        for period in periods:
            ax = x - 5 + period.depth * 3
            hh = max(period.y1 - period.y0, 10)
            lines.append(
                f'<rect x="{ax}" y="{period.y0}" width="10" height="{hh}" '
                f'fill="#fff" stroke="#000" opacity="0.6"/>'
            )

    for i, (msg, y) in enumerate(zip(MESSAGES, ys)):
        num = i + 1
        text = f"{num}. {msg.label}"
        sx, dx = x_map[msg.src], x_map[msg.dst]
        dashed = msg.kind in ("async", "return")
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        marker = "url(#open)" if dashed else "url(#closed)"

        if msg.kind == "self":
            lines.append(
                f'<path d="M {sx+5} {y} L {sx+50} {y} L {sx+50} {y+18} L {sx+5} {y+18}" '
                f'fill="none" stroke="#000"{dash} marker-end="{marker}"/>'
            )
            lx, ly = sx + 58, y + 10
        else:
            mid_y = y + 14 if abs(sx - dx) > 400 and msg.kind == "sync" else y
            if mid_y != y:
                lines.append(
                    f'<path d="M {sx} {y} L {sx} {mid_y} L {dx} {mid_y} L {dx} {y}" '
                    f'fill="none" stroke="#000"{dash} marker-end="{marker}"/>'
                )
            else:
                lines.append(
                    f'<line x1="{sx}" y1="{y}" x2="{dx}" y2="{y}" stroke="#000"{dash} '
                    f'marker-end="{marker}"/>'
                )
            lx = min(sx, dx) + abs(dx - sx) * 0.35
            ly = y - 6

        lines.append(f'<text x="{lx}" y="{ly}">{html.escape(text)}</text>')

    lines.extend(
        [
            "<defs>",
            '<marker id="closed" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">',
            '<path d="M0,0 L8,3 L0,6 Z" fill="#000"/>',
            "</marker>",
            '<marker id="open" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">',
            '<path d="M0,0 L8,3 L0,6" fill="none" stroke="#000"/>',
            "</marker>",
            "</defs>",
            "</svg>",
        ]
    )
    return "\n".join(lines)


def validate_drawio(path: Path) -> bool:
    try:
        ET.parse(path)
        text = path.read_text(encoding="utf-8")
        return text.strip().startswith("<mxfile") and "</mxfile>" in text
    except ET.ParseError:
        return False


def main() -> None:
    diag = SequenceDiagram(
        title=TITLE,
        participants=PARTICIPANTS,
        messages=MESSAGES,
        y_start=120,
        y_gap=Y_GAP,
        page_w=PAGE_W,
    )
    drawio = diag.build_drawio("diag-full-project-seq", "Tổng thể")
    OUT_DRAWIO.parent.mkdir(parents=True, exist_ok=True)
    OUT_DRAWIO.write_text(drawio, encoding="utf-8")
    OUT_SVG.write_text(build_svg(diag), encoding="utf-8")

    ok = validate_drawio(OUT_DRAWIO)
    print(f"drawio: {OUT_DRAWIO} ({'valid' if ok else 'INVALID'})")
    print(f"svg:    {OUT_SVG}")
    print(f"steps:  {len(MESSAGES)}, participants: {len(PARTICIPANTS)}")


if __name__ == "__main__":
    main()
