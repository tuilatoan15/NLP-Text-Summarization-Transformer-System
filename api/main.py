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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.utils import logger
from src.crawler import crawl_article, merge_texts
from src.preprocess import preprocess
from src.extractive import extractive_summarize_with_details
from src.abstractive import get_summarizer, resolve_model_name
from src.selector import select_best_summary
from src.file_parser import extract_text_from_file
from src.explainability import build_extractive_explanations
from src.fact_check import check_consistency
from src.summary_control import enforce_word_limit, resolve_summary_controls
from src.storage import persist_result, save_upload_file


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
    length_control: str = Field(
        default="auto",
        description="auto | 20_percent | 50_percent | 100_words | 200_words",
    )
    model_name: str = Field(
        default="vit5",
        description="vit5/t5 hoặc bart. Có thể truyền tên model Hugging Face bất kỳ.",
    )
    save_result: bool = Field(
        default=True,
        description="Lưu kết quả ra storage JSON và MongoDB nếu MONGO_URI được cấu hình.",
    )
    analysis_mode: str = Field(
        default="fast",
        description="fast hoặc full. Fast chỉ kiểm chứng top summary sentences để giảm thời gian.",
    )

    @validator("text", "reference", pre=True)
    def strip_string(cls, v):
        return v.strip() if isinstance(v, str) else v

    class Config:
        json_schema_extra = {
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
    consistency: dict = Field(default_factory=dict)
    explainability: dict = Field(default_factory=dict)
    documents: list[dict] = Field(default_factory=list)
    storage: dict = Field(default_factory=dict)
    controls: dict = Field(default_factory=dict)


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
# PIPELINE HELPERS
# ==============================================================================

def _summarize_clean_text(
    clean: str,
    request: Request,
    extractive_sentences: int,
    max_abstractive_length: int,
    length_control: str,
    model_name: str,
    reference: Optional[str] = None,
    analysis_mode: str = "fast",
) -> dict:
    step_start = time.time()
    controls = resolve_summary_controls(
        clean,
        length_control=length_control,
        extractive_sentences=extractive_sentences,
        max_abstractive_length=max_abstractive_length,
    )

    extractive_details = extractive_summarize_with_details(
        clean,
        sentence_count=controls["extractive_sentences"],
    )
    extractive = extractive_details["summary"]
    extractive = enforce_word_limit(extractive, controls["target_words"])

    resolved_model_name = resolve_model_name(model_name)
    summarizer = getattr(request.app.state, "summarizer", None)
    if resolved_model_name != "VietAI/vit5-base":
        try:
            summarizer = get_summarizer(model_name=resolved_model_name)
        except Exception as exc:
            logger.warning(f"Không load được model {resolved_model_name}: {exc}")
            summarizer = None

    if summarizer and summarizer.is_loaded():
        abstractive = summarizer.summarize(
            clean,
            max_output_length=controls["max_abstractive_length"],
            num_beams=2,
        )
    else:
        logger.warning("Model không khả dụng, dùng extractive làm fallback cho abstractive.")
        abstractive = extractive

    if not abstractive:
        abstractive = extractive
    abstractive = enforce_word_limit(abstractive, controls["target_words"])

    reference_text = reference if reference else clean
    selection = select_best_summary(
        extractive_summary=extractive,
        abstractive_summary=abstractive,
        reference=reference_text,
    )
    best = enforce_word_limit(selection["best_summary"], controls["target_words"])

    consistency = check_consistency(best, clean, mode=analysis_mode if analysis_mode in {"fast", "full"} else "fast")
    explainability = build_extractive_explanations(clean, extractive)
    explainability["extractive_details"] = extractive_details

    logger.info(
        "Summary pipeline done: model=%s mode=%s best=%s consistency=%s time=%.2fs",
        resolved_model_name,
        analysis_mode,
        selection["best_type"],
        consistency.get("consistency_percent"),
        time.time() - step_start,
    )

    return {
        "extractive": extractive,
        "abstractive": abstractive,
        "best": best,
        "best_type": selection["best_type"],
        "scores": selection["scores"],
        "word_count": {
            "input": len(clean.split()),
            "extractive": len(extractive.split()),
            "abstractive": len(abstractive.split()),
            "best": len(best.split()),
        },
        "consistency": consistency,
        "explainability": explainability,
        "controls": {
            **controls,
            "model_name": resolved_model_name,
            "analysis_mode": analysis_mode,
        },
    }


def _summarize_documents(
    documents: list[dict],
    request: Request,
    extractive_sentences: int,
    max_abstractive_length: int,
    length_control: str,
    model_name: str,
    reference: Optional[str] = None,
    save_result: bool = True,
    analysis_mode: str = "fast",
) -> dict:
    start_time = time.time()

    valid_documents = []
    for doc in documents:
        processed = preprocess(doc["text"], aggressive=True)
        clean = processed["cleaned"]
        if clean and len(clean.split()) >= 10:
            valid_documents.append({**doc, "clean_text": clean})

    if not valid_documents:
        raise HTTPException(
            status_code=400,
            detail="Không thu thập được nội dung hợp lệ. Văn bản cần tối thiểu 10 từ sau tiền xử lý.",
        )

    document_results = []
    for doc in valid_documents:
        result = _summarize_clean_text(
            doc["clean_text"],
            request=request,
            extractive_sentences=extractive_sentences,
            max_abstractive_length=max_abstractive_length,
            length_control=length_control,
            model_name=model_name,
            reference=reference,
            analysis_mode=analysis_mode,
        )
        document_results.append({
            "name": doc["name"],
            "source_type": doc["source_type"],
            "word_count": len(doc["clean_text"].split()),
            "summary": result["best"],
            "summary_type": result["best_type"],
            "consistency_score": result["consistency"]["consistency_score"],
            "consistency_status": result["consistency"]["status"],
            "explainability": result["explainability"],
            "consistency": result["consistency"],
            "storage_path": doc.get("storage_path"),
        })

    raw_text = merge_texts([doc["clean_text"] for doc in valid_documents])
    combined = _summarize_clean_text(
        raw_text,
        request=request,
        extractive_sentences=extractive_sentences,
        max_abstractive_length=max_abstractive_length,
        length_control=length_control,
        model_name=model_name,
        reference=reference,
        analysis_mode=analysis_mode,
    )

    response = {
        **combined,
        "documents": document_results,
        "processing_time_seconds": round(time.time() - start_time, 2),
        "storage": {},
    }

    if save_result:
        response["storage"] = persist_result(response)

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
    if not request_body.text and not request_body.urls:
        raise HTTPException(
            status_code=422,
            detail="Cần cung cấp ít nhất một trong hai: 'text' hoặc 'urls'.",
        )

    documents = []
    if request_body.urls:
        logger.info(f"Crawling {len(request_body.urls)} URL(s)...")
        for url in request_body.urls:
            text = crawl_article(url)
            if text:
                documents.append({
                    "name": url,
                    "source_type": "url",
                    "text": text,
                })

    if request_body.text:
        documents.append({
            "name": "direct_text",
            "source_type": "text",
            "text": request_body.text,
        })

    return _summarize_documents(
        documents,
        request=request,
        extractive_sentences=request_body.extractive_sentences,
        max_abstractive_length=request_body.max_abstractive_length,
        length_control=request_body.length_control,
        model_name=request_body.model_name,
        reference=request_body.reference,
        save_result=request_body.save_result,
        analysis_mode=request_body.analysis_mode,
    )


@app.post(
    "/summarize/files",
    response_model=SummarizeResponse,
    tags=["Summarization"],
    summary="Upload TXT/PDF/DOCX và tóm tắt nhiều tài liệu",
)
async def summarize_files(
    request: Request,
    files: list[UploadFile] = File(...),
    reference: Optional[str] = Form(default=None),
    extractive_sentences: int = Form(default=5, ge=1, le=20),
    max_abstractive_length: int = Form(default=150, ge=30, le=512),
    length_control: str = Form(default="auto"),
    model_name: str = Form(default="vit5"),
    analysis_mode: str = Form(default="fast"),
    save_result: bool = Form(default=True),
):
    if not files:
        raise HTTPException(status_code=422, detail="Cần upload ít nhất một file.")

    documents = []
    for upload in files:
        try:
            await upload.seek(0)
            saved_path = save_upload_file(upload.file, upload.filename)
            text = extract_text_from_file(saved_path)
            documents.append({
                "name": upload.filename,
                "source_type": saved_path.suffix.lower().lstrip("."),
                "text": text,
                "storage_path": str(saved_path),
            })
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(f"Không xử lý được file {upload.filename}: {exc}")
            raise HTTPException(
                status_code=400,
                detail=f"Không xử lý được file {upload.filename}: {exc}",
            ) from exc

    return _summarize_documents(
        documents,
        request=request,
        extractive_sentences=extractive_sentences,
        max_abstractive_length=max_abstractive_length,
        length_control=length_control,
        model_name=model_name,
        reference=reference,
        save_result=save_result,
        analysis_mode=analysis_mode,
    )


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
