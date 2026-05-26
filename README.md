# 🇻🇳 Hệ thống Nghiên cứu Tóm tắt Văn bản Tiếng Việt Đa Phương pháp (Transformer-based & Graph-based)
### Đồ án Tốt nghiệp / Nghiên cứu Khoa học — Vietnamese NLP Text Summarization System

Hệ thống tóm tắt văn bản tiếng Việt toàn diện, tích hợp các thuật toán **Extractive** cổ điển và **Abstractive** hiện đại (Transformers) sử dụng FastAPI ở backend và React ở frontend. Hệ thống hỗ trợ xử lý tăng tốc bằng GPU, song song hóa đa luồng, đánh giá đa chiều thời gian thực và phân tích nghiên cứu trực quan.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B--cu124-red)](https://pytorch.org)

---

## 🏗️ Kiến trúc & Tính năng Nổi bật của Hệ thống

1. **6 Thuật toán Tóm tắt Tích hợp**:
   * **Extractive (Trích xuất)**: TextRank, LexRank, LSA Summarizer (chạy song song đa luồng).
   * **Abstractive (Trừu tượng)**: ViT5 (Fine-tuned), BARTPho (Syllable), mT5 (Experimental Baseline).
2. **GPU & FP16 Inference Backend**:
   * Tự động phát hiện GPU CUDA và kích hoạt Mixed Precision (`fp16`) giúp tăng tốc sinh văn bản từ **5-8 lần** (Inference time $\le 5$ giây trên GPU RTX 3050 Laptop).
   * Cơ chế **Singleton Model Registry** cho phép nạp trước (preload) các mô hình Transformers vào VRAM khi khởi động server, tránh trễ ở request đầu tiên.
   * Quản lý tài nguyên GPU an toàn qua cơ chế Semaphore (chạy Sequential đối với các mô hình Abstractive) tránh lỗi tràn bộ nhớ VRAM (CUDA Out-of-Memory).
3. **Đánh giá Công bằng (Evaluation Fairness)**:
   * Tự động tắt các độ đo trùng lặp (ROUGE, BLEU) và chuyển sang **BERTScore** + **Semantic Similarity (SBERT)** để xếp hạng khi không có Reference Summary (Tránh score inflation cho extractive).
   * Hiển thị warning thông báo trên cả API và giao diện React.
4. **Lọc lỗi giải mã ViT5 chuyên sâu (`post_clean_vit5_telex`)**:
   * Tích hợp bộ dọn dẹp lỗi telex dính trong dataset (`hãngj` -> `hãng`, `kiệnj` -> `kiện`).
   * Xóa các delimiter lỗi do logit penalty (`+`, `_`, `*`) và các ký tự hoa dính bất thường (`tính năngỪngÃ` -> `tính năng`).
   * Cơ chế tự động kích hoạt **Greedy Decoding Fallback** khi văn bản sinh không đạt tiêu chuẩn chất lượng.

---

## 📁 Cấu trúc Dự án

```
NLP-Text-Summarization-Transformer-System/
│
├── api/
│   └── main.py                ← FastAPI server & lifespan (Preload & Diagnostics)
│
├── src/
│   ├── config.py              ← Quản lý cấu hình tập trung (Env-based)
│   ├── abstractive.py         ← Trình sinh văn bản Transformers (ViT5, BARTPho, mT5)
│   ├── extractive.py          ← Trình tóm tắt trích xuất song song (ThreadPoolExecutor)
│   ├── model_loader.py        ← Singleton ModelRegistry, sửa lỗi tokenizer & vocab mismatch
│   ├── output_validator.py    ← Bộ kiểm tra lỗi lặp và ký tự rác tự động của Transformers
│   ├── preprocess.py          ← Chuẩn hóa văn bản Unicode NFC, dọn dẹp Telex ViT5 nâng cao
│   ├── evaluate.py            ← Đánh giá ROUGE, BLEU, BERTScore, Semantic Similarity
│   ├── explainability.py      ← Giải thích cơ chế lựa chọn câu của Extractive
│   ├── dashboard.py           ← Multi-algorithm orchestrator & SSE generator
│   └── utils.py               ← Logging, kiểm tra thông tin GPU & VRAM
│
├── scripts/
│   ├── train.py               ← Pipeline fine-tune ViT5 tối ưu (seq2seq, smooth loss)
│   └── crawl_vnexpress.py     ← Script crawl data VNExpress tự động
│
├── frontend/                  ← Ứng dụng React/Vite/CSS
├── models/                    ← Thư mục chứa checkpoints ViT5 đã fine-tune (cục bộ)
└── requirements.txt
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Cài đặt Môi trường (Hỗ trợ GPU CUDA)

```bash
# Clone dự án
git clone https://github.com/your-repo/NLP-Text-Summarization-Transformer-System
cd NLP-Text-Summarization-Transformer-System

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate

# Uninstall torch CPU-only (nếu có) và cài đặt phiên bản CUDA tương thích (cu124 cho CUDA 12.x)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Cài đặt các thư viện phụ thuộc khác
pip install -r requirements.txt

# Tải dữ liệu NLTK bổ trợ
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Khởi chạy Backend API Server

Backend hỗ trợ tự động tải trước các mô hình để sẵn sàng inference. Khởi chạy bằng lệnh sau:

```bash
$env:PYTHONIOENCODING="utf-8"
python -m api.main
```

* **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)
* **Diagnostics Endpoint**: [http://localhost:8000/metrics](http://localhost:8000/metrics) (Theo dõi VRAM, tốc độ load mô hình và phiên bản CUDA đang chạy)

### 3. Khởi chạy Frontend React

Mở một terminal mới:

```bash
cd frontend
npm install
npm run dev
```

* Truy cập giao diện tại: [http://localhost:5173](http://localhost:5173)

---

## 🧠 Fine-tune ViT5

Hệ thống hỗ trợ script huấn luyện tùy biến mô hình ViT5 trên GPU sử dụng Hugging Face Seq2SeqTrainer:

```bash
# Huấn luyện trên dataset VNExpress từ Hub Hugging Face
python scripts/train.py --epochs 3 --batch_size 2 --lr 2e-5

# Chỉ chạy test nhanh với 100 mẫu
python scripts/train.py --epochs 1 --batch_size 1 --max_samples 100
```

Các checkpoint sẽ được tự động lưu trong thư mục `models/vit5-finetuned/`.

---

## 📊 So sánh Metrics Đánh giá

| Độ đo | Khoảng giá trị | Mô tả |
|---|---|---|
| **ROUGE-1 / 2 / L** | 0.0 - 1.0 | Tỉ lệ trùng khớp n-gram từ đơn, từ đôi và chuỗi con chung dài nhất |
| **BLEU** | 0.0 - 1.0 | Độ chuẩn xác n-gram (thường dùng trong dịch máy và sinh văn bản) |
| **BERTScore F1** | 0.0 - 1.0 | Độ tương đồng ngữ nghĩa sâu sử dụng embeddings của PhoBERT/mBERT |
| **Semantic Sim** | 0.0 - 1.0 | Độ tương đồng vector câu sử dụng mô hình Sentence-BERT (SBERT) |

---

## 👨‍💻 Nghiên cứu & Bản quyền

Dự án này phục vụ cho mục đích nghiên cứu học thuật và thực nghiệm các kỹ thuật xử lý ngôn ngữ tự nhiên tiếng Việt nâng cao.

*© 2026 — Đồ án Nghiên cứu NLP Tiếng Việt.*
<!-- Document Intelligence upgrade summary -->

## AI Document Intelligence Upgrade

This project now includes a research-grade Document Intelligence layer for "De tai 18: Tom tat van ban tu dong bang xu ly ngon ngu tu nhien".

Core additions:

- Production ingest pipeline for PDF/DOCX/TXT with PyMuPDF, pdfplumber, unstructured, python-docx, Mammoth and OCR fallback.
- Semantic, heading-aware, token-aware chunking for long-document summarization and RAG.
- Embedding wrapper and benchmark support for BGE-M3, multilingual-E5, Jina, Vietnamese SBERT and PhoBERT SimCSE.
- FastAPI Document Intelligence router: `/documents/ingest`, `/documents/{document_id}/search`, `/documents/{document_id}/compare`, `/documents/{document_id}/assets`, `/documents/{document_id}/visualization`, and websocket streaming.
- NotebookLM-style generated assets: overview, research report, quiz, flashcards, mindmap, presentation outline, podcast script, infographic, timeline and entity graph.
- Citation grounding for every summary sentence and factual consistency checks.
- New React research workspace at `/documents` with upload, semantic search, comparison charts, chunk graph, embedding map and citation viewer.
- Docker Compose includes PostgreSQL, Redis, MinIO and ChromaDB service definitions.

Architecture details: `docs/DOCUMENT_INTELLIGENCE_ARCHITECTURE.md`.
Database schema: `docs/DATABASE_SCHEMA.sql`.
Ingest details: `docs/INGEST_PIPELINE.md`.
