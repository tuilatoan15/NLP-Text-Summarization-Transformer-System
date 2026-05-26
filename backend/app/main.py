"""Production entrypoint: `python -m backend.app.main` (wraps legacy `api.main`)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import app, lifespan  # noqa: E402
from src import config  # noqa: E402
from src.utils import logger  # noqa: E402

__all__ = ["app", "lifespan"]


def run() -> None:
    import uvicorn

    logger.info("Starting backend via backend.app.main on %s:%s", config.API_HOST, config.API_PORT)
    uvicorn.run(
        "backend.app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    run()
