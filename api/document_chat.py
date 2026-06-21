from __future__ import annotations

import re
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services.rag import RAGChatService
from src import config
from src.file_parser import SUPPORTED_EXTENSIONS


router = APIRouter(prefix="/rag", tags=["RAG Chat"])
service = RAGChatService()


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    conversation_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    retrieval_mode: str = Field(default="hybrid")
    use_reranking: bool = Field(default=False)
    embedding_model: str = Field(default=config.DEFAULT_EMBEDDING_MODEL)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


@router.get("/embedding-models")
async def list_embedding_models():
    return {"models": service.list_embedding_models()}


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=80),
    embedding_model: str = Form(default=config.DEFAULT_EMBEDDING_MODEL),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size")

    filename = _safe_upload_name(file.filename or "upload.txt")
    target = config.UPLOAD_DIR / f"rag_{int(time.time() * 1000)}_{filename}"
    target.write_bytes(await file.read())
    return await _to_thread(
        service.upload_document,
        path=target,
        filename=file.filename or filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )


@router.get("/documents")
async def list_documents():
    return {"items": service.list_documents()}


@router.delete("/documents/all")
async def delete_all_documents():
    try:
        service.delete_all_documents()
        return {"ok": True, "message": "Đã xóa toàn bộ tài liệu RAG"}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Không thể xóa tất cả tài liệu: {e}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    await _to_thread(service.delete_document, document_id)
    return {"ok": True}


@router.get("/conversations")
async def list_conversations(limit: int = Query(default=50, ge=1, le=200)):
    return {"items": service.list_conversations()[:limit]}


@router.get("/conversations/{conversation_id}/messages")
async def list_messages(conversation_id: str):
    return {"items": service.list_messages(conversation_id)}


class SummarizeDocsRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1)
    query: str = Field(default="Tóm tắt nội dung chính của các tài liệu")


@router.post("/chat")
async def chat(request: ChatRequest):
    return await _to_thread(service.chat, **request.model_dump())


@router.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    async def event_gen():
        async for event in service.stream_chat(**request.model_dump()):
            yield event

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/documents/summarize")
async def summarize_documents(request: SummarizeDocsRequest):
    return await _to_thread(service.summarize_documents, **request.model_dump())


@router.post("/documents/summarize/stream")
async def stream_summarize_documents(request: SummarizeDocsRequest):
    async def event_gen():
        async for event in service.stream_summarize_documents(**request.model_dump()):
            yield event

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _to_thread(fn, *args, **kwargs):
    import asyncio

    return await asyncio.to_thread(fn, *args, **kwargs)


def _safe_upload_name(filename: str) -> str:
    name = Path(filename or "upload.txt").name
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or "upload.txt"

