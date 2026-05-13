# 🇻🇳 Hệ thống Tóm tắt Văn bản Tiếng Việt Đa Tài liệu
### Đồ án Tốt nghiệp — Xử lý Ngôn ngữ Tự nhiên (NLP)

> Hệ thống tóm tắt văn bản tiếng Việt toàn diện, kết hợp **Extractive** (TextRank, LSA, LexRank) và **Abstractive** (ViT5, T5, BART, Pegasus) với đánh giá ROUGE, BLEU, BERTScore và Semantic Similarity theo thời gian thực.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🏗️ Kiến trúc Hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                 │
│   Text trực tiếp │ URL crawling │ Upload TXT/PDF/DOCX           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING MODULE                           │
│   HTML removal → Unicode normalization → Sentence splitting     │
└────────────────────────────┬────────────────────────────────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
┌──────────────────────┐     ┌──────────────────────────────────┐
│  EXTRACTIVE ENGINE   │     │      ABSTRACTIVE ENGINE          │
│  • TextRank (sumy)   │     │  • VietAI/vit5-base (Fine-tuned) │
│  • LSA (sumy)        │     │  • t5-small                      │
│  • LexRank (sumy)    │     │  • facebook/bart-large-cnn       │
└──────────┬───────────┘     │  • google/pegasus-xsum           │
           │                 └──────────────┬───────────────────┘
           └─────────────┬──────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EVALUATION MODULE                             │
│   ROUGE-1/2/L │ BLEU │ BERTScore │ Semantic Similarity         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              POST-PROCESSING & ANALYSIS                         │
│   Consistency Checker │ Explainability │ Selector (Best)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
               ┌─────────────┴──────────────┐
               ▼                            ▼
┌──────────────────────┐     ┌──────────────────────────────────┐
│   FastAPI REST API   │     │     React Frontend (Vite)         │
│   /summarize         │     │  • So sánh bảng có progress bar  │
│   /summarize/compare │     │  • ROUGE / BERTScore charts      │
│   /summarize/files   │     │  • Sentence highlighting         │
│   /dashboard         │     │  • Dark mode, responsive         │
└──────────────────────┘     └──────────────────────────────────┘
```

---

## 📊 Kết quả Benchmark (VnExpress Dataset — 100 samples)

| Thuật toán | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Thời gian avg |
|---|---|---|---|---|---|
| **TextRank** | 0.4231 | 0.1872 | 0.3654 | 0.8102 | 0.12s |
| **LSA** | 0.3987 | 0.1654 | 0.3421 | 0.7989 | 0.14s |
| **LexRank** | 0.4089 | 0.1743 | 0.3512 | 0.8034 | 0.15s |
| **ViT5 (base)** | 0.3812 | 0.1543 | 0.3287 | 0.8321 | 4.21s |
| **ViT5 (fine-tuned)** | **0.4512** | **0.2143** | **0.3987** | **0.8567** | 4.35s |
| **T5-small** | 0.2987 | 0.1021 | 0.2654 | 0.7812 | 2.54s |
| **BART** | 0.3654 | 0.1432 | 0.3123 | 0.8154 | 5.12s |

> 📝 *Số liệu trên tập validation `thanhnew2001/vnexpress`. ViT5 fine-tuned đạt kết quả tốt nhất trên cả ROUGE và BERTScore.*

---

## 📁 Cấu trúc Dự án

```
NLP-Text-Summarization-Transformer-System/
│
├── api/
│   ├── __init__.py
│   └── main.py                ← FastAPI server (endpoints, lifespan)
│
├── src/
│   ├── config.py              ← Cấu hình tập trung (env-based)
│   ├── abstractive.py         ← ViT5/T5/BART abstractive summarizer
│   ├── extractive.py          ← TextRank / LexRank
│   ├── preprocess.py          ← Tiền xử lý tiếng Việt
│   ├── evaluate.py            ← ROUGE, BLEU, BERTScore, Semantic Sim
│   ├── selector.py            ← Chọn bản tóm tắt tốt nhất
│   ├── fact_check.py          ← Consistency checker
│   ├── explainability.py      ← Giải thích câu được chọn
│   ├── dashboard.py           ← Multi-algorithm orchestrator + SSE
│   ├── analytics.py           ← Dashboard metrics
│   ├── benchmark.py           ← Script chạy benchmark hàng loạt
│   ├── crawler.py             ← Thu thập bài báo từ URL
│   ├── file_parser.py         ← Đọc TXT/PDF/DOCX
│   ├── storage.py             ← Lưu kết quả JSON/MongoDB
│   └── utils.py               ← Logging, helpers
│
├── train/
│   ├── dataset_loader.py      ← Load VnExpress dataset
│   └── train_vit5.py          ← Fine-tune pipeline
│
├── frontend/
│   └── src/
│       ├── main.jsx           ← React app (comparison, charts, SSE)
│       └── styles.css         ← Design system CSS
│
├── tests/
│   ├── test_preprocess.py     ← Unit tests tiền xử lý
│   ├── test_evaluate.py       ← Unit tests ROUGE/BLEU/BERTScore
│   ├── test_extractive.py     ← Unit tests TextRank/LexRank
│   └── test_api.py            ← Integration tests FastAPI
│
├── models/                    ← Model đã fine-tune (sau training)
├── data/                      ← Dataset nội bộ (CSV)
├── storage/results/           ← Kết quả benchmark đã lưu
├── Dockerfile                 ← Production container
├── docker-compose.yml         ← Multi-service setup
├── pyproject.toml             ← Pytest config
└── requirements.txt
```

---

## ⚙️ Cài đặt & Chạy

### 1. Cài đặt môi trường

```bash
git clone https://github.com/your-repo/NLP-Text-Summarization-Transformer-System
cd NLP-Text-Summarization-Transformer-System

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt

# Tải NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Chạy Backend API

```bash
python -m api.main
# Hoặc:
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

| URL | Mô tả |
|---|---|
| http://localhost:8000 | Health check |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/redoc | ReDoc |

### 3. Chạy Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### 4. Chạy với Docker (Production)

```bash
# Build & start tất cả services
docker-compose up --build -d

# Xem logs
docker-compose logs -f api
```

---

## 🧠 Fine-tune ViT5

```bash
# Dùng dataset VnExpress từ Hugging Face (tự động tải)
python -m train.train_vit5 --max_samples 5000 --epochs 3 --batch_size 2

# Test nhanh với 100 samples
python -m train.train_vit5 --max_samples 100 --epochs 1

# Dùng dataset CSV nội bộ
python -m train.train_vit5 --local_data data/dataset.csv --max_samples 5000

# Chạy trên CPU (tắt GPU)
set CUDA_VISIBLE_DEVICES=-1
python -m train.train_vit5 --max_samples 2000 --batch_size 2
```

Kết quả lưu tại `models/vit5-finetuned/eval_results.json`.

---

## 🔬 Chạy Benchmark

```bash
# Đánh giá 100 samples từ tập validation VnExpress
python -m src.benchmark --samples 100 --model vit5

# So sánh tất cả models
python -m src.benchmark --samples 100 --model vit5
python -m src.benchmark --samples 100 --model bart
python -m src.benchmark --samples 100 --model t5

# Kết quả lưu tại storage/results/benchmark_*.json
```

---

## 🧪 Chạy Unit Tests

```bash
# Tất cả tests
pytest

# Tests cụ thể
pytest tests/test_evaluate.py -v
pytest tests/test_extractive.py -v
pytest tests/test_api.py -v

# Với coverage report
pip install pytest-cov
pytest --cov=src --cov=api --cov-report=html
```

---

## 📡 API Reference

### POST `/summarize`
Tóm tắt văn bản với 1 model.

```json
{
  "text": "Văn bản tiếng Việt...",
  "urls": ["https://vnexpress.net/..."],
  "model_name": "vit5",
  "extractive_sentences": 5,
  "max_abstractive_length": 150,
  "length_control": "auto"
}
```

### POST `/summarize/compare/stream`
So sánh nhiều thuật toán với SSE streaming.

```json
{
  "text": "Văn bản...",
  "algorithms": ["textrank", "lsa", "lexrank", "vit5", "bart"],
  "extractive_sentences": 5
}
```

### POST `/summarize/files/compare/stream`
Upload files và streaming compare.

```
Form fields: files (multi), algorithms (JSON array)
```

---

## 📈 Metrics Giải thích

| Metric | Phạm vi | Ý nghĩa |
|---|---|---|
| **ROUGE-1** | 0–1 | Overlap unigrams (từ đơn) |
| **ROUGE-2** | 0–1 | Overlap bigrams (cặp từ) |
| **ROUGE-L** | 0–1 | Longest Common Subsequence |
| **BLEU** | 0–1 | Precision n-gram (dịch máy) |
| **BERTScore F1** | 0–1 | Contextual embedding similarity |
| **Semantic Sim.** | 0–1 | Cosine similarity (Sentence-BERT) |

> **BERTScore** (Zhang et al., 2020) — Metric hiện đại sử dụng BERT embeddings, tương quan tốt hơn với đánh giá con người so với ROUGE.

---

## 🔎 Tính năng Nổi bật

- ✅ **7 thuật toán** so sánh đồng thời (Extractive + Abstractive)
- ✅ **4 metrics** đánh giá: ROUGE, BLEU, BERTScore, Semantic Similarity
- ✅ **SSE Streaming** — Kết quả hiển thị realtime từng thuật toán
- ✅ **Multi-document** — Upload nhiều file TXT/PDF/DOCX cùng lúc
- ✅ **Crawling** — Tóm tắt trực tiếp từ URL bài báo
- ✅ **Consistency Checker** — Phát hiện câu không nhất quán với nguồn
- ✅ **Explainability** — Highlight câu nguồn được chọn và lý do
- ✅ **Fine-tuning** — Pipeline hoàn chỉnh để fine-tune ViT5
- ✅ **Docker** — Deploy production-ready

---

## 📚 Tài liệu Tham khảo

- [VietAI/vit5-base](https://huggingface.co/VietAI/vit5-base) — ViT5 pre-trained tiếng Việt
- [BERTScore (Zhang et al., 2020)](https://arxiv.org/abs/1904.09675)
- [TextRank (Mihalcea & Tarau, 2004)](https://aclanthology.org/W04-3252/)
- [ROUGE (Lin, 2004)](https://aclanthology.org/W04-1013/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [sumy](https://github.com/miso-belica/sumy) — Extractive Summarization
- [underthesea](https://github.com/undertheseanlp/underthesea) — Vietnamese NLP
- [FastAPI](https://fastapi.tiangolo.com)

---

## 👨‍💻 Thông tin Tác giả

Đồ án tốt nghiệp — **Hệ thống Tóm tắt Văn bản Tiếng Việt Đa Tài liệu**  
Chuyên ngành: Công nghệ Thông tin / Khoa học Máy tính  
Hướng nghiên cứu: Xử lý Ngôn ngữ Tự nhiên (NLP) — Transformer Models

---

*© 2026 — Đồ án Tốt nghiệp NLP*
