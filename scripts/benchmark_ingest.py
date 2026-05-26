"""CLI wrapper for ingest and embedding benchmark.

Run:
    python -m scripts.benchmark_ingest data/report.pdf --query "rủi ro vận hành thủy điện"
"""

from __future__ import annotations

from embeddings.benchmark import main


if __name__ == "__main__":
    main()
