"""
api/main.py — FastAPI server cho hệ thống tóm tắt văn bản tiếng Việt.

Endpoints:
  GET  /           — Health check & thông tin hệ thống
  GET  /health     — Kiểm tra trạng thái model
  POST /summarize  — Tóm tắt văn bản (chính)

Dùng singleton pattern để model chỉ được load một lần khi server khởi động,
tránh lãng phí tài nguyên RAM/VRAM.
"""

import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Thêm project root vào Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.utils import logger
from src.crawler import crawl_articles, merge_texts
from src.preprocess import preprocess
from src.extractive import extractive_summarize
from src.abstractive import get_summarizer
from src.selector import select_best_summary


# ==============================================================================
# PYDANTIC SCHEMAS (Request / Response)
# ==============================================================================

class SummarizeRequest(BaseModel):
    """Schema cho request POST /summarize"""

    text: Optional[str] = Field(
        default=None,
        description="Văn bản tiếng Việt trực tiếp cần tóm tắt",
        example="Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp...",
    )
    urls: Optional[list[str]] = Field(
        default=None,
        description="Danh sách URL bài báo cần crawl và tóm tắt",
        example=["https://vnexpress.net/some-article"],
    )
    reference: Optional[str] = Field(
        default=None,
        description=(
            "Văn bản tham chiếu để tính ROUGE (tuỳ chọn). "
            "Nếu không cung cấp, sẽ dùng văn bản gốc làm reference."
        ),
    )
    extractive_sentences: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Số câu trong bản tóm tắt trích xuất",
    )
    max_abstractive_length: int = Field(
        default=150,
        ge=30,
        le=512,
        description="Độ dài tối đa bản tóm tắt diễn giải (số token)",
    )

    @validator("text", "reference", pre=True)
    def strip_string(cls, v):
        return v.strip() if isinstance(v, str) else v

    class Config:
        schema_extra = {
            "example": {
                "text": "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập tức và mở hành lang nhân đạo cho người dân vùng chiến sự.",
                "urls": [],
                "extractive_sentences": 3,
                "max_abstractive_length": 100,
            }
        }


class RougeScoreDetail(BaseModel):
    rouge1: float
    rouge2: float
    rougeL: float
    rougeLsum: float
    length_score: float
    combined_score: float


class SummarizeResponse(BaseModel):
    """Schema cho response POST /summarize"""
    extractive:  str = Field(description="Bản tóm tắt trích xuất (TextRank)")
    abstractive: str = Field(description="Bản tóm tắt diễn giải (ViT5)")
    best:        str = Field(description="Bản tóm tắt được chọn (tốt nhất)")
    best_type:   str = Field(description="Loại tóm tắt tốt nhất: 'extractive' hoặc 'abstractive'")
    scores: dict      = Field(description="Điểm ROUGE chi tiết cho cả hai bản tóm tắt")
    word_count: dict  = Field(description="Số từ của mỗi bản tóm tắt")
    processing_time_seconds: float = Field(description="Thời gian xử lý (giây)")


# ==============================================================================
# LIFESPAN: LOAD MODEL KHI KHỞI ĐỘNG SERVER
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
    - startup: Load model abstractive vào bộ nhớ
    - shutdown: Giải phóng tài nguyên (nếu cần)
    """
    # === STARTUP ===
    logger.info("🚀 Server khởi động. Đang load model ViT5...")
    try:
        summarizer = get_summarizer()
        app.state.summarizer = summarizer
        logger.info("✅ Model ViT5 đã sẵn sàng!")
    except Exception as e:
        logger.error(f"❌ Không thể load model: {e}")
        logger.warning("API sẽ khởi động nhưng abstractive summarization không khả dụng.")
        app.state.summarizer = None

    yield  # Server chạy ở đây

    # === SHUTDOWN ===
    logger.info("Server đang tắt. Dọn dẹp tài nguyên...")


# ==============================================================================
# KHỞI TẠO FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="🇻🇳 Vietnamese Multi-Document Summarization API",
    description=(
        "Hệ thống tóm tắt văn bản tiếng Việt đa tài liệu.\n\n"
        "Hỗ trợ:\n"
        "- **Extractive**: Trích xuất câu quan trọng (TextRank)\n"
        "- **Abstractive**: Sinh câu mới (ViT5 Transformer)\n"
        "- **ROUGE Evaluation**: Đánh giá và chọn bản tóm tắt tốt nhất"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cấu hình CORS (cho phép frontend gọi API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# MIDDLEWARE: REQUEST LOGGING
# ==============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log thông tin mỗi request vào/ra."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration:.2f}s)"
    )
    return response


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Thông tin API và danh sách endpoints."""
    return {
        "name": "Vietnamese Text Summarization API",
        "version": "1.0.0",
        "description": "Hệ thống tóm tắt văn bản tiếng Việt đa tài liệu",
        "endpoints": {
            "POST /summarize": "Tóm tắt văn bản hoặc danh sách URLs",
            "GET /health":     "Kiểm tra trạng thái server và model",
            "GET /docs":       "Tài liệu Swagger UI",
        },
        "model": "VietAI/vit5-base",
        "methods": ["TextRank (Extractive)", "ViT5 (Abstractive)"],
    }


@app.get("/health", tags=["Info"])
async def health_check(request: Request):
    """Kiểm tra trạng thái server và model."""
    summarizer = getattr(request.app.state, "summarizer", None)
    model_status = "ready" if (summarizer and summarizer.is_loaded()) else "not_loaded"

    return {
        "status": "ok",
        "model_status": model_status,
        "model_name": "VietAI/vit5-base",
    }


@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    tags=["Summarization"],
    summary="Tóm tắt văn bản tiếng Việt",
)
async def summarize(request_body: SummarizeRequest, request: Request):
    """
    **Tóm tắt văn bản tiếng Việt** từ text trực tiếp hoặc danh sách URLs.

    Quy trình xử lý:
    1. Crawl URLs (nếu có) và gộp với text trực tiếp
    2. Tiền xử lý: làm sạch HTML, chuẩn hóa Unicode
    3. Tóm tắt trích xuất (TextRank)
    4. Tóm tắt diễn giải (ViT5)
    5. Đánh giá ROUGE và chọn bản tốt nhất

    **Lưu ý**: Cần ít nhất một trong hai: `text` hoặc `urls`.
    """
    start_time = time.time()

    # --- Kiểm tra input ---
    if not request_body.text and not request_body.urls:
        raise HTTPException(
            status_code=422,
            detail="Cần cung cấp ít nhất một trong hai: 'text' hoặc 'urls'.",
        )

    # --- Bước 1: Thu thập văn bản ---
    all_texts = []

    # Crawl URLs nếu có
    if request_body.urls:
        logger.info(f"Crawling {len(request_body.urls)} URL(s)...")
        crawled_texts = crawl_articles(request_body.urls)
        all_texts.extend(crawled_texts)

    # Thêm text trực tiếp
    if request_body.text:
        all_texts.append(request_body.text)

    if not all_texts:
        raise HTTPException(
            status_code=400,
            detail="Không thu thập được nội dung từ các nguồn đã cung cấp. "
                   "Kiểm tra lại URLs hoặc nội dung text.",
        )

    # Gộp tất cả văn bản lại
    raw_text = merge_texts(all_texts)

    # --- Bước 2: Tiền xử lý ---
    logger.info("Tiền xử lý văn bản...")
    processed = preprocess(raw_text, aggressive=True)
    clean = processed["cleaned"]

    if not clean or len(clean.split()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Văn bản sau khi xử lý quá ngắn (< 10 từ). Vui lòng cung cấp nội dung dài hơn.",
        )

    # --- Bước 3: Tóm tắt trích xuất ---
    logger.info("Tóm tắt trích xuất (TextRank)...")
    extractive = extractive_summarize(
        clean,
        sentence_count=request_body.extractive_sentences,
    )

    # --- Bước 4: Tóm tắt diễn giải ---
    logger.info("Tóm tắt diễn giải (ViT5)...")
    summarizer = getattr(request.app.state, "summarizer", None)

    if summarizer and summarizer.is_loaded():
        abstractive = summarizer.summarize(
            clean,
            max_output_length=request_body.max_abstractive_length,
        )
    else:
        logger.warning("Model không khả dụng, dùng extractive làm fallback cho abstractive.")
        abstractive = extractive

    # Fallback nếu abstractive trống
    if not abstractive:
        abstractive = extractive

    # --- Bước 5: Đánh giá và chọn bản tốt nhất ---
    logger.info("Đánh giá và chọn bản tóm tắt tốt nhất...")
    reference = request_body.reference if request_body.reference else clean

    selection = select_best_summary(
        extractive_summary=extractive,
        abstractive_summary=abstractive,
        reference=reference,
    )

    # --- Kết quả ---
    processing_time = round(time.time() - start_time, 2)

    response = {
        "extractive":  extractive,
        "abstractive": abstractive,
        "best":        selection["best_summary"],
        "best_type":   selection["best_type"],
        "scores":      selection["scores"],
        "word_count": {
            "input":       len(clean.split()),
            "extractive":  len(extractive.split()),
            "abstractive": len(abstractive.split()),
            "best":        len(selection["best_summary"].split()),
        },
        "processing_time_seconds": processing_time,
    }

    logger.info(
        f"✅ Hoàn tất: input={len(clean.split())} từ, "
        f"best={selection['best_type']} ({processing_time}s)"
    )

    return response


# ==============================================================================
# CHẠY TRỰC TIẾP (DEV MODE)
# ==============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,       # Tự reload khi code thay đổi (development)
        log_level="info",
    )
