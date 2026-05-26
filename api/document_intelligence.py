"""FastAPI router for AI Document Intelligence workflows."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src import config
from backend.services.document_service import DocumentService
from src.document_intelligence import DEFAULT_ANALYSIS_ALGORITHMS
from src.file_parser import SUPPORTED_EXTENSIONS
from src.utils import logger

service = DocumentService()


router = APIRouter(prefix="/documents", tags=["Document Intelligence"])


class DocumentCompareRequest(BaseModel):
    reference: Optional[str] = None
    algorithms: list[str] = Field(default_factory=lambda: DEFAULT_ANALYSIS_ALGORITHMS.copy())
    target_length_ratio: int = Field(default=30, ge=10, le=100)
    extractive_sentences: int = Field(default=5, ge=1, le=20)
    max_abstractive_length: int = Field(default=180, ge=24, le=512)


class HierarchicalRequest(BaseModel):
    model_key: str = Field(default="vit5")
    use_extractive_map: bool = Field(default=False)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


@router.get("")
async def list_documents(limit: int = Query(default=30, ge=1, le=100)):
    return {"items": await service.list_documents_async(limit=limit)}


@router.get("/embedding-models")
async def list_embedding_models():
    from embeddings.embedder import EmbeddingModelRegistry

    return {"models": EmbeddingModelRegistry.list_models()}


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    include_embeddings: bool = Form(default=True),
    embedding_model: Optional[str] = Form(default=None),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

    target = config.UPLOAD_DIR / f"{int(time.time() * 1000)}_{_safe_upload_name(file.filename or 'upload')}"
    content = await file.read()
    target.write_bytes(content)
    try:
        payload = await service.ingest_file_async(target, include_embeddings=include_embeddings, embedding_model=embedding_model)
        return payload
    except Exception as exc:
        logger.exception("Document ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{document_id}")
async def get_document(document_id: str):
    try:
        return await service.get_document_async(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{document_id}/search")
async def semantic_search(document_id: str, request: SearchRequest):
    try:
        return service.semantic_search(document_id, request.query, request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{document_id}/compare")
async def compare_summaries(document_id: str, request: DocumentCompareRequest):
    try:
        return await service.compare_summaries_async(
            document_id,
            reference=request.reference,
            algorithms=request.algorithms,
            target_length_ratio=request.target_length_ratio,
            extractive_sentences=request.extractive_sentences,
            max_abstractive_length=request.max_abstractive_length,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document comparison failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{document_id}/explainability")
async def document_explainability(document_id: str, algorithm: str = Query(default="textrank")):
    try:
        return service.explain_extractive(document_id, algorithm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{document_id}/summarize/hierarchical")
async def hierarchical_summary(document_id: str, request: HierarchicalRequest):
    try:
        return await _to_thread(
            service.hierarchical_summarize,
            document_id,
            model_key=request.model_key,
            use_extractive_map=request.use_extractive_map,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{document_id}/podcast/tts")
async def export_podcast_tts(document_id: str):
    from backend.services.tts_service import TTSService
    from backend.db.repository import DocumentRepository

    try:
        payload = await service.get_document_async(document_id)
        script = (payload.get("analysis_assets") or {}).get("podcast") or {}
        tts = TTSService().export_podcast(document_id, script)
        repo = DocumentRepository()
        if repo.enabled and tts.get("audio_uri"):
            await repo.save_podcast_script(document_id, {**script, **tts}, tts.get("audio_uri"))
        return tts
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{document_id}/assets")
async def document_assets(document_id: str):
    try:
        payload = await service.get_document_async(document_id)
        return payload.get("analysis_assets") or service.generate_assets(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{document_id}/visualization")
async def document_visualization(document_id: str):
    try:
        payload = await service.get_document_async(document_id)
        return payload.get("visualization") or service.build_visualization(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.websocket("/{document_id}/stream")
async def document_stream(websocket: WebSocket, document_id: str):
    await websocket.accept()
    try:
        payload = await service.get_document_async(document_id)
        await websocket.send_json({"event": "loaded", "document_id": document_id, "metadata": payload.get("metadata", {})})
        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            if message.get("type") == "search":
                result = service.semantic_search(document_id, str(message.get("query", "")), int(message.get("top_k", 5)))
                await websocket.send_json({"event": "search_result", "data": result})
            elif message.get("type") == "assets":
                await websocket.send_json({"event": "assets", "data": payload.get("analysis_assets", {})})
            else:
                await websocket.send_json({"event": "error", "detail": "Unsupported stream message type"})
    except FileNotFoundError:
        await websocket.send_json({"event": "error", "detail": "Document not found"})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.exception("Document websocket failed")
        try:
            await websocket.send_json({"event": "error", "detail": str(exc)})
        except Exception:
            pass


@router.get("/{document_id}/report/html")
async def export_report_html(document_id: str):
    from fastapi.responses import HTMLResponse
    from backend.services.report_generator import ReportGenerator

    try:
        payload = await service.get_document_async(document_id)
        compare_data = None
        try:
            compare_data = await service.compare_summaries_async(document_id)
        except Exception:
            pass
        
        html_content = ReportGenerator().generate_html(payload, compare_data)
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{document_id}/report/markdown")
async def export_report_markdown(document_id: str):
    from fastapi.responses import PlainTextResponse
    from backend.services.report_generator import ReportGenerator

    try:
        payload = await service.get_document_async(document_id)
        compare_data = None
        try:
            compare_data = await service.compare_summaries_async(document_id)
        except Exception:
            pass
        
        md_content = ReportGenerator().generate_markdown(payload, compare_data)
        return PlainTextResponse(content=md_content, status_code=200)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _to_thread(fn, *args, **kwargs):
    import asyncio

    return await asyncio.to_thread(fn, *args, **kwargs)


def _safe_upload_name(filename: str) -> str:
    name = Path(filename or "upload.txt").name
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or "upload.txt"
