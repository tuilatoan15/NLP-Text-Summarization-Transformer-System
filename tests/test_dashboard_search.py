"""Tests for GET /search summarize history over full source text."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from backend.services.system_service import search_dashboard


@pytest.fixture()
def temp_results_dir():
    temp_dir = tempfile.mkdtemp()
    results_dir = Path(temp_dir) / "results"
    results_dir.mkdir(parents=True)
    yield results_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def _write_compare_record(results_dir: Path, result_id: str, payload: dict) -> None:
    path = results_dir / f"{result_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_search_finds_keyword_in_full_source_text(temp_results_dir):
    result_id = "a" * 32
    long_tail = "từ khóa đặc biệt chỉ có ở cuối văn bản nguồn"
    payload = {
        "type": "compare",
        "result_id": result_id,
        "text_preview": "Đoạn đầu văn bản không chứa từ khóa.",
        "full_text": f"Đoạn đầu văn bản không chứa từ khóa. {'x' * 600} {long_tail}",
        "reference_text": "Tham chiếu ngắn.",
        "results": [
            {"key": "textrank", "algorithm": "TextRank", "summary": "Tóm tắt mẫu."},
        ],
        "best_model": {"algorithm": "TextRank", "key": "textrank"},
        "ranking": [{"algorithm": "TextRank", "key": "textrank"}],
    }
    _write_compare_record(temp_results_dir, result_id, payload)

    with mock.patch("backend.services.analytics_service.RESULT_DIR", temp_results_dir):
        data = search_dashboard("từ khóa đặc biệt", limit=10)

    hits = [r for r in data["results"] if r.get("type") == "summarize_history"]
    assert len(hits) == 1
    assert hits[0]["result_id"] == result_id
    assert hits[0]["match_field"] == "source"
    assert hits[0]["link"] == f"/summarize?result={result_id}"
    assert "từ khóa đặc biệt" in hits[0]["subtitle"].lower()


def test_search_finds_algorithm_summary(temp_results_dir):
    result_id = "b" * 32
    payload = {
        "type": "compare",
        "result_id": result_id,
        "text_preview": "Nguồn không liên quan.",
        "full_text": "Nguồn không liên quan.",
        "reference_text": "",
        "results": [
            {
                "key": "vit5",
                "algorithm": "ViT5",
                "summary": "Mexico vượt qua vòng bảng nhờ sai lầm thủ môn.",
            },
        ],
        "best_model": {"algorithm": "ViT5", "key": "vit5"},
        "ranking": [{"algorithm": "ViT5", "key": "vit5"}],
    }
    _write_compare_record(temp_results_dir, result_id, payload)

    with mock.patch("backend.services.analytics_service.RESULT_DIR", temp_results_dir):
        data = search_dashboard("Mexico vượt", limit=10)

    hits = [r for r in data["results"] if r.get("type") == "summarize_history"]
    assert len(hits) == 1
    assert hits[0]["match_field"] == "summary"
    assert hits[0]["link"] == f"/summarize?result={result_id}"


def test_search_preserves_other_result_types(temp_results_dir):
    with mock.patch("backend.services.analytics_service.RESULT_DIR", temp_results_dir):
        data = search_dashboard("textrank", limit=10)

    types = {r["type"] for r in data["results"]}
    assert "algorithm" in types
