# Hệ Thống Tóm Tắt Văn Bản Tiếng Việt Đa Luồng & Hỏi Đáp Tài Liệu Đa Tầng ChatRAG

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite-61DAFB?style=flat&logo=react&logoColor=white)](https://react.dev)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-FF4B4B?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange?style=flat)](https://huggingface.co)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Compatible-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)

Dự án này là một hệ thống toàn diện giải quyết hai bài toán cốt lõi trong xử lý ngôn ngữ tự nhiên (NLP) cho tiếng Việt: **Tóm tắt văn bản đa thuật toán (Multi-Algorithm Summarization)** và **Hỏi đáp thông minh trên tài liệu dài (Advanced ChatRAG - Retrieval-Augmented Generation)**.

Hệ thống tích hợp song song các thuật toán trích xuất (Extractive) cổ điển và các mô hình sinh (Abstractive) học sâu dựa trên kiến trúc Sequence-to-Sequence (Seq2Seq) Transformer đã được tinh chỉnh (fine-tuned) trên dữ liệu tiếng Việt. Hệ thống cung cấp một **Pipeline Tóm tắt lai nhiều tầng (Hybrid Summarization)** và **Hệ thống RAG nâng cao** sử dụng phân tách ngữ nghĩa (Semantic Chunking) và chỉ mục phân cấp cây tóm tắt (RAPTOR-lite) nhằm tối ưu hóa khả năng hỏi đáp trên tài liệu dài, ngăn ngừa tràn bộ nhớ VRAM GPU.

---

## 📌 Mục Lục
- [📌 Tính Năng Cốt Lõi](#-tính-năng-cốt-lõi)
- [📊 Kết Quả Thực Nghiệm & So Sánh](#-kết-quả-thực-nghiệm--so-sánh-empirical-benchmark-results)
- [📁 Cấu Trúc Thư Mục Dự Án](#-cấu-trúc-thư-mục-dự-án-project-structure)
- [🛠️ Hướng Dẫn Cài Đặt](#%EF%B8%8F-hướng-dẫn-cài-đặt-installation-guide)
- [🔌 Tài Liệu API Tóm Tắt Nhanh](#-tài-liệu-api-tóm-tắt-nhanh)

---

## 📌 Tính Năng Cốt Lõi

1. **Tóm tắt văn bản đa thuật toán:**
   * **Extractive (Trích xuất):** TextRank, LexRank, LSA (SVD Matrix), TF-IDF.
   * **Abstractive (Mô hình sinh):** ViT5 (base), BARTPho (syllable-level), mT5 (baseline).
   * **Hybrid (Lai ghép):** Kết hợp trích xuất câu chính ở Giai đoạn 1 và mô hình sinh viết lại mượt mà ở Giai đoạn 2. Giúp giảm ~35-50% độ trễ và loại bỏ lỗi tràn bộ nhớ ngữ cảnh GPU trên văn bản dài.
2. **Hỏi đáp tài liệu thông minh (Advanced ChatRAG):**
   * **Semantic Chunking:** Phân mảnh văn bản dựa trên khoảng cách tương đồng ngữ nghĩa động thay vì phân chia số lượng ký tự cứng.
   * **Hybrid Search & RRF:** Kết hợp tìm kiếm từ khóa chính xác (BM25) và tìm kiếm ngữ nghĩa (PhoBERT-SimCSE Dense Embedding) thông qua Reciprocal Rank Fusion (RRF).
   * **Reranking:** Tái xếp hạng các phân đoạn bằng mô hình Cross-Encoder (`bge-reranker-v2-m3`).
   * **RAPTOR-lite Indexing:** Phân cụm hỗn hợp Gaussian (GMM) và tóm tắt đệ quy để xây dựng chỉ mục cây phân cấp tài liệu, hỗ trợ trả lời các câu hỏi tổng quan toàn văn bản.
   * **Trích nguồn tham chiếu:** Trích dẫn chính xác nguồn thông tin (tên file, số trang) để giảm thiểu ảo giác thông tin.

---

## 📊 Kết Quả Thực Nghiệm & So Sánh (Empirical Benchmark Results)

Dưới đây là kết quả thực nghiệm chi tiết thu được khi chạy thử nghiệm trên tập mẫu kiểm thử **5.000 tài liệu VietNews** (seed=42) trên cấu hình cục bộ (CPU Intel64 Family 6 Model 154 Stepping 3, RAM 15.63GB, GPU NVIDIA GeForce RTX 3050 Ti Laptop 4GB VRAM):

| Hạng | Model/Giải thuật | Nhóm | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | Thời gian TB | Tỷ lệ nén | Độ trung thực | Điểm tổng hợp |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | TextRank | Extractive | 41% | 21% | 28% | 72% | 0.01s | 23% | 100% | 61% |
| 2 | LexRank | Extractive | 42% | 21% | 28% | 72% | 0.01s | 22% | 100% | 61% |
| 3 | LSA Summarizer | Extractive | 46% | 20% | 29% | 71% | 0.01s | 18% | 100% | 61% |
| 4 | ViT5 (Fine-tuned) | Abstractive | 60% | 28% | 38% | 71% | 2.49s | 10% | 83% | 58% |
| 5 | TextRank ➔ ViT5 | Hybrid | 58% | 26% | 37% | 70% | 1.64s | 9% | 83% | 57% |
| 6 | LexRank ➔ ViT5 | Hybrid | 58% | 26% | 37% | 70% | 1.61s | 9% | 83% | 57% |
| 7 | LSA ➔ ViT5 | Hybrid | 56% | 23% | 35% | 69% | 1.57s | 9% | 83% | 56% |
| 8 | BARTPho (Fine-tuned) | Abstractive | 54% | 20% | 33% | 68% | 3.38s | 9% | 86% | 56% |
| 9 | TextRank ➔ mT5 | Hybrid | 55% | 22% | 33% | 69% | 1.99s | 9% | 83% | 56% |
| 10 | TextRank ➔ BARTPho | Hybrid | 53% | 19% | 33% | 68% | 1.86s | 8% | 87% | 55% |
| 11 | LexRank ➔ BARTPho | Hybrid | 53% | 19% | 33% | 68% | 1.68s | 8% | 87% | 55% |
| 12 | LexRank ➔ mT5 | Hybrid | 54% | 22% | 33% | 69% | 1.89s | 9% | 83% | 55% |
| 13 | mT5 (Baseline) | Abstractive | 55% | 21% | 33% | 69% | 2.73s | 9% | 82% | 55% |
| 14 | LSA ➔ mT5 | Hybrid | 52% | 19% | 32% | 68% | 1.70s | 8% | 84% | 55% |
| 15 | LSA ➔ BARTPho | Hybrid | 51% | 16% | 31% | 67% | 1.50s | 8% | 86% | 54% |

### Nhận Xét Kết Luận Nghiên Cứu:
1. **Chất lượng sinh vượt trội của ViT5:** Mô hình `VIT5` (Fine-tuned) vẫn duy trì điểm ROUGE-L cao nhất (**0.3802**), sinh văn bản mượt mà, trôi chảy và đạt điểm BLEU vượt trội (**0.1548**).
2. **Hiệu năng của mô hình Hybrid:** Các mô hình lai (Hybrid) giúp tối ưu hóa đáng kể thời gian suy diễn. Ví dụ, `BARTPho` khi chạy độc lập có độ trễ lên tới 5.20s nhưng khi kết hợp qua mô hình lai `LEXRANK ➔ BARTPHO` độ trễ giảm xuống còn **2.31s** (rút ngắn hơn **55% thời gian**). Đồng thời, cơ chế lọc câu chính ở Giai đoạn 1 loại bỏ triệt để nguy cơ tràn VRAM GPU.
3. **Thế mạnh của Extractive:** Các phương pháp trích xuất (`LEXRANK`, `TEXTRANK`) dẫn đầu bảng xếp hạng Composite Score (**0.6143** và **0.6139**) nhờ độ trễ cực thấp (dưới **20ms**), độ phủ ngữ nghĩa tốt và đạt độ trung thực thông tin tuyệt đối (`Faithfulness = 100%`).

---

## 📁 Cấu Trúc Thư Mục Dự Án (Project Structure)

```
NLP-Text-Summarization-Transformer-System/
├── ai_models/                      # Quản lý vòng đời nạp/offload mô hình trên GPU/CPU
├── api/                            # FastAPI Routers (Endpoints: /summarize, /chat, /research)
├── backend/                        # Cấu hình Database & Dịch vụ ChatRAG, Hybrid Retriever, Reranker
├── configs/                        # Cài đặt tham số mặc định của mô hình và phân mảnh (JSON)
├── docs/                           # Sơ đồ Draw.io XML tối giản đen trắng và báo cáo học thuật
├── evaluation/                     # Module tính toán metrics (ROUGE, BLEU, BERTScore, Hallucination)
├── frontend/                       # Client Single Page Application (React + Vite + Zustand + Tailwind)
├── loaders/                        # Trích xuất và đọc tài liệu PDF, DOCX, TXT
├── notebooks/                      # Các notebook huấn luyện Colab cho ViT5 / BARTPho / mT5
├── pipeline/                       # Pipeline tóm tắt lai ghép (hybrid_summarizer.py)
├── scripts/                        # Các script chạy huấn luyện (train.py) và benchmark tự động
└── storage/                        # Cơ sở dữ liệu SQLite cục bộ, ChromaDB index và kết quả thực nghiệm
```

---

## 🛠️ Hướng Dẫn Cài Đặt (Installation Guide)

### 1. Triển khai trên Windows (PowerShell)
1. **Yêu cầu:** Python 3.11.x - 3.12.x, Node.js LTS, Git.
2. **Tải dự án và cài đặt môi trường Python:**
   ```powershell
   git clone https://github.com/tuilatoan15/NLP-Text-Summarization-Transformer-System.git
   cd NLP-Text-Summarization-Transformer-System
   python -m venv venv
   venv\Scripts\activate
   pip install --no-cache-dir -r requirements.txt
   ```
3. **Cài đặt PyTorch tối ưu CUDA (Nếu có GPU NVIDIA):**
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
4. **Cấu hình biến môi trường:**
   Sao chép tệp mẫu `copy .env.example .env` và cấu hình các khóa cần thiết.
5. **Khởi chạy hệ thống:**
   ```powershell
   .\run_project.bat
   ```
   Tập lệnh tự động chạy song song **Backend FastAPI** (cổng `8000`) và **Frontend Vite** (cổng `5173`).

### 2. Triển khai tự động bằng Docker Compose
Khởi chạy đồng bộ ứng dụng và toàn bộ hạ tầng (PostgreSQL, Redis, Qdrant Vector DB, Celery Workers):
```bash
docker-compose up -d --build
```
* **Frontend UI:** `http://localhost:5173`
* **Swagger API Docs:** `http://localhost:8000/docs`
* **Qdrant Dashboard:** `http://localhost:6333/dashboard`

---

## 🔌 Tài Liệu API Tóm Tắt Nhanh

### 1. Tóm Tắt Văn Bản Đơn (`POST /summarize`)
* **Request:**
```json
{
  "text": "Đoạn văn bản tiếng Việt cần tóm tắt...",
  "model_name": "vit5",
  "extractive_sentences": 3,
  "max_abstractive_length": 150
}
```
* **Response:**
```json
{
  "summary": "Nội dung tóm tắt được sinh ra...",
  "processing_time": 2.45,
  "metrics": {
    "rougeL": 0.3827,
    "composite_score": 0.5800
  }
}
```

### 2. Hỏi Đáp RAG dạng Luồng (`POST /rag/chat/stream`)
* **Request:**
```json
{
  "query": "Bị cáo Nguyễn Văn A bị phạt bao nhiêu năm tù?",
  "conversation_id": "chat-session-uuid",
  "document_ids": ["doc-uuid"],
  "use_reranking": true
}
```
* **Response:** Trả về luồng văn bản thời gian thực (Server-Sent Events) từng token một và cấu trúc nguồn trích dẫn.