"""
perf.py — Tiện ích đo latency và dynamic top-k cho pipeline RAG.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageTimer:
    """Ghi nhận thời gian từng giai đoạn pipeline."""

    stages: dict[str, float] = field(default_factory=dict)
    _marks: dict[str, float] = field(default_factory=dict)

    def start(self, name: str) -> None:
        self._marks[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        start = self._marks.pop(name, None)
        if start is None:
            return 0.0
        elapsed = time.perf_counter() - start
        self.stages[name] = self.stages.get(name, 0.0) + elapsed
        return elapsed

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + seconds

    def total(self) -> float:
        return sum(self.stages.values())

    def report(self) -> dict[str, Any]:
        total = self.total() or 1e-9
        breakdown = {
            name: {
                "seconds": round(sec, 4),
                "percent": round(100.0 * sec / total, 1),
            }
            for name, sec in sorted(self.stages.items(), key=lambda x: -x[1])
        }
        return {
            "total_seconds": round(total, 4),
            "stages": breakdown,
        }


def compute_dynamic_top_k(
    *,
    base_top_k: int,
    query: str,
    document_ids: list[str] | None,
    intent: str,
    min_k: int = 3,
    max_k: int = 12,
) -> int:
    """
    Điều chỉnh top_k theo độ dài câu hỏi, số tài liệu và intent.
    Câu hỏi dài / đa tài liệu → tăng k; câu ngắn → giảm k để nhanh hơn.
    """
    k = base_top_k
    word_count = len(query.split())
    if word_count <= 6:
        k -= 1
    elif word_count >= 20:
        k += 1
    if document_ids:
        if len(document_ids) >= 3:
            k += 1
        elif len(document_ids) == 1:
            k = max(min_k, k - 1)
    if intent == "SUMMARIZE":
        k += 1
    return max(min_k, min(max_k, k))
