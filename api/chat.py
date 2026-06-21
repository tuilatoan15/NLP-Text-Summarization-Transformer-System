from __future__ import annotations

from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.rag import RAGChatService

router = APIRouter(prefix="/api/chat", tags=["Chat History"])
service = RAGChatService()


class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New chat"
    user_id: Optional[str] = None


class RenameConversationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=150)


class SaveMessageRequest(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)
    citations: Optional[list[dict[str, Any]]] = None
    confidence: Optional[float] = None
    retrieval_threshold: Optional[float] = None
    model_used: Optional[str] = None
    evaluation: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


@router.get("/conversations")
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    try:
        items = service.repository.list_conversations(limit=limit, offset=offset)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lấy danh sách cuộc trò chuyện: {e}")


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    try:
        conv = service.repository.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện")
        
        messages = service.repository.list_messages(conversation_id)
        conv["messages"] = messages
        return conv
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lấy chi tiết cuộc trò chuyện: {e}")


@router.post("/conversations")
async def create_conversation(request: CreateConversationRequest):
    try:
        conv = service.repository.create_conversation(
            title=request.title or "New chat",
            user_id=request.user_id
        )
        return conv
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể tạo cuộc trò chuyện: {e}")


@router.put("/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, request: RenameConversationRequest):
    try:
        conv = service.repository.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện")
        
        service.repository.rename_conversation(conversation_id, request.title)
        return {"ok": True, "title": request.title}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể đổi tên cuộc trò chuyện: {e}")


@router.delete("/conversations/all")
async def delete_all_conversations():
    try:
        service.repository.delete_all_conversations()
        return {"ok": True, "message": "Đã xóa toàn bộ cuộc trò chuyện"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể xóa tất cả cuộc trò chuyện: {e}")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    try:
        conv = service.repository.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện")
        
        service.repository.delete_conversation(conversation_id)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể xóa cuộc trò chuyện: {e}")


@router.post("/conversations/{conversation_id}/messages")
async def save_message(conversation_id: str, request: SaveMessageRequest):
    try:
        conv = service.repository.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện")
        
        msg_id = service.repository.append_message(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            citations=request.citations,
            confidence=request.confidence,
            retrieval_threshold=request.retrieval_threshold,
            model_used=request.model_used,
            evaluation=request.evaluation,
            metadata=request.metadata
        )
        
        # Trigger auto title generation if it's the 2nd to 4th message
        service._trigger_auto_title(conversation_id)
        
        return {"ok": True, "message_id": msg_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu tin nhắn: {e}")


@router.get("/search")
async def search_conversations(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    try:
        items = service.repository.search_conversations(query=q, limit=limit, offset=offset)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tìm kiếm cuộc trò chuyện: {e}")
