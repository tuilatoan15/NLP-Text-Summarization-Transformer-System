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
from src.document_intelligence import DEFAULT_ANALYSIS_ALGORITHMS, service
from src.file_parser import SUPPORTED_EXTENSIONS
from src.utils import logger


router = APIRouter(prefix="/documents", tags=["Document Intelligence"])


class DocumentCompareRequest(BaseModel):
    reference: Optional[str] = None
    algorithms: list[str] = Field(default_factory=lambda: DEFAULT_ANALYSIS_ALGORITHMS.copy())
    target_length_ratio: int = Field(default=30, ge=10, le=100)
    extractive_sentences: int = Field(default=5, ge=1, le=20)
    max_abstractive_length: int = Field(default=180, ge=24, le=512)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


@router.get("")
async def list_documents(limit: int = Query(default=30, ge=1, le=100)):
    return {"items": service.list_documents(limit=limit)}


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
        payload = await _to_thread(service.ingest_file, target, include_embeddings=include_embeddings, embedding_model=embedding_model)
        return payload
    except Exception as exc:
        logger.exception("Document ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{document_id}")
async def get_document(document_id: str):
    try:
        return service.get_document(document_id)
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
        return await _to_thread(
            service.compare_summaries,
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


@router.get("/{document_id}/assets")
async def document_assets(document_id: str):
    try:
        payload = service.get_document(document_id)
        return payload.get("analysis_assets") or service.generate_assets(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{document_id}/visualization")
async def document_visualization(document_id: str):
    try:
        payload = service.get_document(document_id)
        return payload.get("visualization") or service.build_visualization(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.websocket("/{document_id}/stream")
async def document_stream(websocket: WebSocket, document_id: str):
    await websocket.accept()
    try:
        payload = service.get_document(document_id)
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
        await websocket.send_json({"event": "error", "detail": str(exc)})


async def _to_thread(fn, *args, **kwargs):
    import asyncio

    return await asyncio.to_thread(fn, *args, **kwargs)


def _safe_upload_name(filename: str) -> str:
    name = Path(filename or "upload.txt").name
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or "upload.txt"
