#!/usr/bin/env python3
"""Update draw.io flowcharts: Vietnamese labels, step numbers, B&W styling."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

FLOWCHARTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "flowcharts"

# Participant / label translations for 00-main-sequence-diagram
PARTICIPANT_VI = {
    "User": "Người dùng",
    "Playground.jsx": "Giao diện Playground",
    "FastAPI&#xa;api/main.py": "API FastAPI&#xa;(main.py)",
    "dashboard_service&#xa;stream_compare / summarize_all": "Dịch vụ dashboard&#xa;stream_compare / summarize_all",
    "Models&#xa;TextRank / ViT5 / mT5 / BARTPho": "Mô hình&#xa;TextRank / ViT5 / mT5 / BARTPho",
    "Storage&#xa;storage/results": "Lưu trữ&#xa;storage/results",
    "Celery&#xa;(async_mode)": "Celery&#xa;(chế độ bất đồng bộ)",
    "Chat.tsx&#xa;ragApi.ts": "Giao diện Chat&#xa;(ragApi.ts)",
    "FastAPI&#xa;document_chat.py": "API FastAPI&#xa;(document_chat.py)",
    "RAGChatService&#xa;service.py": "Dịch vụ RAG Chat&#xa;(service.py)",
    "RAGRepository&#xa;rag_chat.db": "Kho hội thoại&#xa;(rag_chat.db)",
    "VectorStore&#xa;Chroma": "Vector store&#xa;(Chroma)",
    "HybridRetriever&#xa;BM25+RRF+Rerank": "Bộ truy xuất lai&#xa;BM25 + RRF + Rerank",
    "GroundedGenerator&#xa;generator.py": "Bộ sinh có căn cứ&#xa;(generator.py)",
    "LLM / Model&#xa;Gemini·Ollama·ViT5": "Mô hình LLM&#xa;Gemini / Ollama / ViT5",
}

# Message text patches for sequence diagram (key fragments → Vietnamese)
MSG_VI_PATCHES = [
    (r"^2\. handleRun\(\) → streamCompareSummaries\(\)$",
     "2. handleRun() → gọi streamCompareSummaries()"),
    (r"^POST /summarize \(SummarizeRequest\)$", "15. POST /summarize (SummarizeRequest)"),
    (r"^resolve_algorithm\(model_name\) → \[model_key, textrank\]$",
     "16. resolve_algorithm(model_name) → [model_key, textrank]"),
    (r"^\[async\] summarize_task\.delay\(\) → queued \+ job_id$",
     "17. [bất đồng bộ] summarize_task.delay() → hàng đợi + job_id"),
    (r"^\[sync\] _compare_or_400_async → run_in_executor$",
     "18. [đồng bộ] _compare_or_400_async → run_in_executor"),
    (r"^summarize_all\(\) → _run_all_parallel\(\)$",
     "19. summarize_all() → _run_all_parallel()"),
    (r"^_legacy_response\(\) → SummarizeResponse JSON$",
     "20. _legacy_response() → JSON SummarizeResponse"),
    (r"^generator\.stream_answer\(general_chat=true\)$",
     "10. generator.stream_answer(chế độ chat chung)"),
    (r"^stream tokens$", "11. Luồng token"),
    (r"^2\. setMessages\(user \+ assistant placeholder\)$",
     "2. setMessages(tin nhắn người dùng + placeholder trợ lý)"),
    (r"^3\. streamRagChat\(RAGChatRequest\)$",
     "3. streamRagChat(RAGChatRequest)"),
]

# Start/end label translations for flowcharts 01-22
START_END_VI = {
    "Client: POST /summarize": "1. Client: POST /summarize",
    "HTTP 200 response": "Phản hồi HTTP 200",
    "Client: POST /summarize/compare/stream": "1. Client: POST /summarize/compare/stream",
    "Kết thúc SSE stream": "Kết thúc luồng SSE",
    "Client: POST /summarize/compare": "1. Client: POST /summarize/compare",
    "Trả compare payload hoặc job status": "Trả payload so sánh hoặc trạng thái job",
    "FastAPI lifespan startup": "1. Khởi động lifespan FastAPI",
    "Server lắng nghe / tắt gracefully": "Máy chủ lắng nghe / tắt an toàn",
}

BW_VERTEX = "fillColor=#ffffff;strokeColor=#000000;fontColor=#000000"
BW_TEXT = "strokeColor=none;fillColor=none;fontColor=#000000"
BW_EDGE = "strokeColor=#000000;fontColor=#000000"
BW_ACTIVATION = "fillColor=#ffffff;strokeColor=#000000;opacity=40"
BW_LIFELINE = "endArrow=none;dashed=1;html=1;strokeColor=#000000"


def strip_step_prefix(text: str) -> str:
    return re.sub(r"^\d+\.\s*", "", text.strip())


def apply_bw_style(style: str, cell_type: str) -> str:
    """Normalize style string to B&W."""
    if not style:
        return style

    # Remove colored properties
    style = re.sub(r"fillColor=#[0-9a-fA-F]{3,8};?", "", style)
    style = re.sub(r"strokeColor=#[0-9a-fA-F]{3,8};?", "", style)
    style = re.sub(r"fontColor=#[0-9a-fA-F]{3,8};?", "", style)

    if cell_type == "text":
        base = BW_TEXT
    elif cell_type == "edge":
        base = BW_EDGE
    elif cell_type == "lifeline":
        base = BW_LIFELINE
    elif cell_type == "activation":
        base = BW_ACTIVATION
    else:
        base = BW_VERTEX

    # Preserve non-color style props
    preserved = []
    for part in style.split(";"):
        part = part.strip()
        if not part:
            continue
        key = part.split("=")[0] if "=" in part else part
        if key in ("fillColor", "strokeColor", "fontColor"):
            continue
        preserved.append(part)

    merged = base
    if preserved:
        merged = base + ";" + ";".join(preserved)
    return merged


def get_y(cell: ET.Element) -> float:
    geom = cell.find("mxGeometry")
    if geom is None:
        return 0.0
    y = geom.get("y")
    if y is not None:
        return float(y)
    # edge message y from sourcePoint
    sp = geom.find("mxPoint[@as='sourcePoint']")
    if sp is not None and sp.get("y"):
        return float(sp.get("y"))
    return 0.0


def classify_cell(cell: ET.Element) -> str:
    style = cell.get("style", "")
    if cell.get("edge") == "1":
        if "endArrow=none" in style and "dashed=1" in style:
            return "lifeline"
        return "edge"
    if "strokeColor=none" in style and "fillColor=none" in style and cell.get("value"):
        return "text"
    if cell.get("value") == "" and "opacity" in style:
        return "activation"
    return "vertex"


def is_step_vertex(cell: ET.Element) -> bool:
    if cell.get("edge") == "1":
        return False
    style = cell.get("style", "")
    val = cell.get("value") or ""
    if not val.strip():
        return False
    if "strokeColor=none" in style and "fillColor=none" in style:
        return False
    if "shape=umlFrame" in style or "shape=note" in style:
        return False
    if "shape=umlActor" in style:
        return False
    return True


def is_sequence_message(cell: ET.Element) -> bool:
    if cell.get("edge") != "1":
        return False
    style = cell.get("style", "")
    val = cell.get("value") or ""
    if not val.strip():
        return False
    if "endArrow=none" in style and "dashed=1" in style:
        return False
    return "endArrow" in style


def number_flowchart_steps(root: ET.Element) -> None:
    steps = [c for c in root.findall("mxCell") if is_step_vertex(c)]
    steps.sort(key=get_y)
    for i, cell in enumerate(steps, start=1):
        raw = strip_step_prefix(cell.get("value", ""))
        cell.set("value", f"{i}. {raw}")


def number_sequence_messages(root: ET.Element) -> None:
    msgs = [c for c in root.findall("mxCell") if is_sequence_message(c)]
    msgs.sort(key=get_y)
    for i, cell in enumerate(msgs, start=1):
        raw = strip_step_prefix(cell.get("value", ""))
        # Apply message patches
        for pattern, repl in MSG_VI_PATCHES:
            if re.match(pattern, raw) or raw == strip_step_prefix(repl):
                raw = strip_step_prefix(repl)
                break
        cell.set("value", f"{i}. {raw}")


def translate_participants(root: ET.Element) -> None:
    for cell in root.findall("mxCell"):
        val = cell.get("value")
        if val in PARTICIPANT_VI:
            cell.set("value", PARTICIPANT_VI[val])


def translate_frames_and_notes(root: ET.Element) -> None:
    replacements = {
        "loop [mỗi thuật toán: extractive song song → abstractive tuần tự]":
            "vòng lặp [mỗi thuật toán: extractive song song → abstractive tuần tự]",
        "alt POST /summarize (client trực tiếp — không qua Playground)":
            "nhánh thay thế POST /summarize (client trực tiếp — không qua Playground)",
        "alt [async_mode = true]": "nhánh thay thế [async_mode = true]",
        "alt [intent]": "nhánh thay thế [ý định truy vấn]",
        "opt [GENERAL — không retrieval]": "tùy chọn [CHAT CHUNG — không truy xuất]",
        "else [DOCUMENT_QA / SUMMARIZE — retrieval pipeline]":
            "ngược lại [HỎI ĐÁP TÀI LIỆU / TÓM TẮT — pipeline truy xuất]",
        "opt [POST /rag/chat sync — service.chat()]":
            "tùy chọn [POST /rag/chat đồng bộ — service.chat()]",
    }
    for cell in root.findall("mxCell"):
        val = cell.get("value")
        if not val:
            continue
        for old, new in replacements.items():
            if old in val:
                cell.set("value", val.replace(old, new))
                break


def apply_bw_to_root(root: ET.Element) -> None:
    for cell in root.findall("mxCell"):
        style = cell.get("style", "")
        if not style:
            continue
        ctype = classify_cell(cell)
        cell.set("style", apply_bw_style(style, ctype))


def process_flowchart_file(path: Path) -> None:
    tree = ET.parse(path)
    root_el = tree.getroot()
    is_sequence = path.name == "00-main-sequence-diagram.drawio"

    for diagram in root_el.findall("diagram"):
        model = diagram.find("mxGraphModel/root")
        if model is None:
            continue
        if is_sequence:
            translate_participants(model)
            translate_frames_and_notes(model)
            number_sequence_messages(model)
        else:
            number_flowchart_steps(model)
        apply_bw_to_root(model)

    tree.write(path, encoding="unicode", xml_declaration=False)
    # Restore mxfile format (ET may not preserve exact formatting)
    content = path.read_text(encoding="utf-8")
    if not content.startswith("<mxfile"):
        content = content
    path.write_text(content, encoding="utf-8")


def validate_mxfile(path: Path) -> bool:
    try:
        ET.parse(path)
        text = path.read_text(encoding="utf-8")
        return text.strip().startswith("<mxfile") and "</mxfile>" in text
    except ET.ParseError:
        return False


def main() -> None:
    files = sorted(FLOWCHARTS_DIR.glob("*.drawio"))
    updated = 0
    for f in files:
        process_flowchart_file(f)
        if validate_mxfile(f):
            updated += 1
            print(f"OK: {f.name}")
        else:
            print(f"INVALID: {f.name}")

    print(f"\nTotal updated: {updated}/{len(files)}")


if __name__ == "__main__":
    main()
