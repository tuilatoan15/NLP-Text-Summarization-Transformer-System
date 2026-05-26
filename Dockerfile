# ============================================================
# Dockerfile — Vietnamese Text Summarization API
# Base: Python 3.11 slim để giảm kích thước image
# ============================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="thesis@nlp.vn"
LABEL description="Vietnamese NLP Text Summarization System"

# Tránh interactive prompts từ apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Thư mục làm việc
WORKDIR /app

# Cài các gói hệ thống cần thiết (cho pdfminer, lxml, v.v.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài Python dependencies trước (để tận dụng Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p logs storage/results storage/uploads cache/dashboard models/checkpoints data

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" || true

# Expose port API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Lệnh chạy server (production mode, 1 worker vì model dùng RAM lớn)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
