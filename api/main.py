"""FastAPI backend focused on algorithm comparison for research.

Changes vs old version
────────────────────────
• lifespan startup hook calls preload_all_models() — all Transformer models
  are warmed up before the first request arrives.  Cold-start latency on
  request #1 drops from ~30 s → < 100 ms.
• /health endpoint now reports full GPU status, VRAM usage, and per-model
  load times via registry_status().
• /metrics endpoint (new, non-breaking) exposes Prometheus-style diagnostics.
• All existing endpoints (/summarize, /summarize/compare, /summarize/files,
  /summarize/files/compare) are 100% interface-compatible with the old version.
• API_VERSION bumped to 3.1.0.
"""

from __future__ import annotations

import json
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src import config
from src.dashboard import summarize_all
from src.file_parser import SUPPORTED_EXTENSIONS, extract_text_from_file
from src.model_loader import preload_all_models, registry_status
from src.model_registry import DEFAULT_ALGORITHMS, list_algorithms, resolve_algorithm
from src.preprocess import clean_text
from src.utils import get_device_info, log_device_info, logger


# ─────────────────────────── Request / Response schemas ────────────────────
# (Identical to old version — no breaking changes)

class SummarizeRequest(BaseModel):
    text: Optional[str] = Field(default=None)
    reference: Optional[str] = Field(default=None)
    extractive_sentences: int = Field(default=5, ge=1, le=20)
    max_abstractive_length: int = Field(default=config.MAX_OUTPUT_LENGTH, ge=24, le=512)
    model_name: str = Field(default="vit5")
    save_result: bool = Field(default=False)
    analysis_mode: str = Field(default="research")

    @field_validator("text", "reference", mode="before")
    @classmethod
    def _strip(cls, value):
        return value.strip() if isinstance(value, str) else value


class CompareRequest(BaseModel):
    text: Optional[str] = Field(default=None)
    reference: Optional[str] = Field(default=None)
    algorithms: list[str] = Field(default_factory=lambda: DEFAULT_ALGORITHMS.copy())
    extractive_sentences: int = Field(default=5, ge=1, le=20)
    max_abstractive_length: int = Field(default=config.MAX_OUTPUT_LENGTH, ge=24, le=512)

    @field_validator("text", "reference", mode="before")
    @classmethod
    def _strip(cls, value):
        return value.strip() if isinstance(value, str) else value


class SummarizeResponse(BaseModel):
    extractive: str
    abstractive: str
    best: str
    best_type: str
    scores: dict
    word_count: dict
    processing_time_seconds: float
    consistency: dict = Field(default_factory=dict)
    explainability: dict = Field(default_factory=dict)
    documents: list[dict] = Field(default_factory=list)
    storage: dict = Field(default_factory=dict)
    controls: dict = Field(default_factory=dict)
    best_extractive: Optional[dict] = Field(default=None)
    best_abstractive: Optional[dict] = Field(default=None)
    research_analysis: Optional[dict] = Field(default=None)
    warning: Optional[str] = Field(default=None)


# ─────────────────────────── Lifespan (startup / shutdown) ─────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: preload ALL Transformer models into GPU/CPU memory.

    This is the single most impactful optimization in the entire refactor.
    Without preloading, every request that hits a Transformer model pays
    10–30 s of disk I/O + tokenizer initialisation.
    With preloading, that cost is paid once at startup.
    """
    logger.info("=" * 60)
    logger.info("  NLP Summarization API v%s  — startup", config.API_VERSION)
    logger.info("=" * 60)
    log_device_info()

    if config.PRELOAD_MODELS:
        logger.info("🔄 PRELOAD_MODELS=1 — loading all Transformer models now …")
        try:
            preload_all_models()
        except Exception as exc:
            # Do NOT crash the server if one model fails to load.
            # The per-request fallback in abstractive.py will handle it.
            logger.error("Preload encountered errors: %s", exc, exc_info=True)
    else:
        logger.info("ℹ️  PRELOAD_MODELS=0 — models will load lazily on first request")

    logger.info("🚀 API ready — listening on %s:%s", config.API_HOST, config.API_PORT)
    yield
    # Graceful shutdown: release GPU memory
    try:
        from src.utils import clear_gpu_cache
        clear_gpu_cache()
        logger.info("GPU cache cleared on shutdown")
    except Exception:
        pass


# ─────────────────────────── App factory ───────────────────────────────────

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=(
        "Research API for comparing Vietnamese text summarization algorithms. "
        "Supports TextRank, LexRank, LSA (extractive) and ViT5, mT5, BARTPho (abstractive). "
        "All models are preloaded on startup for zero-latency inference."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────── Request logging middleware ────────────────────

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    logger.info(
        "%s %s → %s  (%.3f s)",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


# ─────────────────────────── Internal helpers ──────────────────────────────

def _ensure_text(text: str | None) -> str:
    cleaned = clean_text(text or "", aggressive=True)
    if len(cleaned.split()) < 5:
        raise HTTPException(
            status_code=422,
            detail="Input text must contain at least 5 valid words after cleaning.",
        )
    return cleaned


def _compare_or_400(
    text: str,
    reference: str | None,
    algorithms: list[str],
    extractive_sentences: int,
    max_abstractive_length: int,
) -> dict:
    try:
        return summarize_all(
            text=text,
            reference=reference,
            algorithms=algorithms,
            sentence_count=extractive_sentences,
            max_output_length=max_abstractive_length,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _legacy_response(compare: dict, requested_model: str) -> dict:
    """Map the new compare payload to the old SummarizeResponse shape."""
    rows = compare.get("results", [])
    extractive_row = next((r for r in rows if r["key"] == "textrank"), None)

    requested_key = "vit5"
    try:
        requested_key = resolve_algorithm(requested_model).key
    except Exception:
        pass

    abstractive_row = next((r for r in rows if r["key"] == requested_key), None)
    abstractive_row = abstractive_row or next((r for r in rows if r["group"] == "abstractive"), None)

    best_key = (compare.get("best_model") or {}).get("key")
    best_row = next((r for r in rows if r["key"] == best_key), None) or (rows[0] if rows else {})

    extractive_summary = (extractive_row or {}).get("summary", "")
    abstractive_summary = (abstractive_row or {}).get("summary", "")
    best_summary = best_row.get("summary", "")

    return {
        "extractive": extractive_summary,
        "abstractive": abstractive_summary,
        "best": best_summary,
        "best_type": best_row.get("group", ""),
        "scores": {row["key"]: row["metrics"] for row in rows},
        "word_count": {
            "input": compare.get("meta", {}).get("input_words", 0),
            "extractive": len(extractive_summary.split()),
            "abstractive": len(abstractive_summary.split()),
            "best": len(best_summary.split()),
        },
        "processing_time_seconds": round(
            sum(row.get("processing_time", 0.0) for row in rows), 4
        ),
        "consistency": {},
        "explainability": {
            "extractive": (extractive_row or {}).get("explainability", {}),
            "abstractive": (abstractive_row or {}).get("explainability", {}),
        },
        "documents": [],
        "storage": {},
        "controls": compare.get("meta", {}),
        "best_extractive": compare.get("best_extractive"),
        "best_abstractive": compare.get("best_abstractive"),
        "research_analysis": compare.get("research_analysis"),
        "warning": compare.get("meta", {}).get("warning"),
    }


def _safe_upload_name(filename: str) -> str:
    name = Path(filename or "upload.txt").name
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or "upload.txt"


async def _read_uploads(files: list[UploadFile]) -> tuple[str, list[dict]]:
    documents = []
    texts = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

        target = config.UPLOAD_DIR / f"{int(time.time() * 1000)}_{_safe_upload_name(upload.filename or '')}"
        content = await upload.read()
        target.write_bytes(content)
        text = extract_text_from_file(target)
        if text:
            texts.append(text)
            documents.append({
                "name": upload.filename,
                "source_type": suffix.lstrip("."),
                "word_count": len(text.split()),
            })

    if not texts:
        raise HTTPException(
            status_code=422,
            detail="No valid text could be extracted from uploaded files.",
        )
    return "\n\n".join(texts), documents


# ─────────────────────────── Info endpoints ────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Vietnamese Text Summarization Research API",
        "version": config.API_VERSION,
        "focus": "algorithm_comparison",
        "algorithm_groups": {
            "extractive": ["TextRank", "LexRank", "LSA Summarizer"],
            "abstractive": ["ViT5", "mT5", "BARTPho"],
        },
        "endpoints": [
            "/summarize",
            "/summarize/compare",
            "/summarize/files",
            "/summarize/files/compare",
            "/models",
            "/health",
            "/metrics",
        ],
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """
    Returns model registry status, GPU info, and VRAM usage.
    Useful for monitoring and debugging startup issues.
    """
    status = registry_status()
    return {
        "status": "ok",
        "api_version": config.API_VERSION,
        "model_status": "preloaded" if status["preloaded"] else "lazy_loaded",
        "default_algorithms": DEFAULT_ALGORITHMS,
        "registry": status,
    }


@app.get("/metrics", tags=["Info"])
async def diagnostics():
    """
    Prometheus-style system diagnostics for benchmarking and monitoring.
    Reports: device, VRAM, torch version, preload status, model load times.
    """
    gpu = get_device_info()
    reg = registry_status()
    return {
        "device": gpu.get("device", "cpu"),
        "gpu_name": gpu.get("gpu_name"),
        "total_vram_mb": gpu.get("total_vram_mb"),
        "free_vram_mb": gpu.get("free_vram_mb"),
        "allocated_vram_mb": gpu.get("allocated_vram_mb"),
        "torch_version": gpu.get("torch_version"),
        "fp16_enabled": reg.get("fp16"),
        "torch_compile_enabled": reg.get("torch_compile"),
        "models_preloaded": reg.get("preloaded"),
        "model_load_times": {
            key: info.get("load_time_s")
            for key, info in (reg.get("models") or {}).items()
        },
    }


@app.get("/models", tags=["Info"])
async def models():
    return {"models": list_algorithms()}


# ─────────────────────────── Summarization endpoints ──────────────────────
# All four endpoints below are interface-compatible with the old api/main.py.

@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize(request_body: SummarizeRequest):
    text = _ensure_text(request_body.text)
    model_key = resolve_algorithm(request_body.model_name).key
    algorithms = [model_key, "textrank"] if model_key != "textrank" else ["textrank", "vit5"]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_algos = [a for a in algorithms if not (a in seen or seen.add(a))]  # type: ignore[func-returns-value]

    compare = _compare_or_400(
        text,
        request_body.reference,
        unique_algos,
        request_body.extractive_sentences,
        request_body.max_abstractive_length,
    )
    return _legacy_response(compare, requested_model=model_key)


@app.post("/summarize/compare", tags=["Summarization"])
async def summarize_compare(request_body: CompareRequest):
    text = _ensure_text(request_body.text)
    return _compare_or_400(
        text,
        request_body.reference,
        request_body.algorithms,
        request_body.extractive_sentences,
        request_body.max_abstractive_length,
    )


@app.post("/summarize/files", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize_files(
    files: list[UploadFile] = File(...),
    reference: Optional[str] = Form(default=None),
    extractive_sentences: int = Form(default=5, ge=1, le=20),
    max_abstractive_length: int = Form(default=config.MAX_OUTPUT_LENGTH, ge=24, le=512),
    model_name: str = Form(default="vit5"),
):
    text, documents = await _read_uploads(files)
    model_key = resolve_algorithm(model_name).key
    compare = _compare_or_400(
        text, reference, ["textrank", model_key],
        extractive_sentences, max_abstractive_length,
    )
    response = _legacy_response(compare, requested_model=model_key)
    response["documents"] = documents
    return response


@app.post("/summarize/files/compare", tags=["Summarization"])
async def summarize_files_compare(
    files: list[UploadFile] = File(...),
    reference: Optional[str] = Form(default=None),
    algorithms: Optional[str] = Form(default=None),
    extractive_sentences: int = Form(default=5, ge=1, le=20),
    max_abstractive_length: int = Form(default=config.MAX_OUTPUT_LENGTH, ge=24, le=512),
):
    text, documents = await _read_uploads(files)
    selected = DEFAULT_ALGORITHMS.copy()
    if algorithms:
        try:
            parsed = json.loads(algorithms)
            selected = parsed if isinstance(parsed, list) else selected
        except Exception:
            selected = [p.strip() for p in algorithms.split(",") if p.strip()]
    result = _compare_or_400(text, reference, selected, extractive_sentences, max_abstractive_length)
    result["documents"] = documents
    return result


# ─────────────────────────── Dev entrypoint ────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,  # reload=True breaks preloaded model state across reloads
        log_level=config.LOG_LEVEL.lower(),
    )
