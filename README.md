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

Dưới đây là kết quả thực nghiệm chi tiết thu được khi chạy thử nghiệm trên tập mẫu kiểm thử **1.000 tài liệu VietNews** (seed=42) trên cấu hình cục bộ (GPU NVIDIA GeForce RTX 3050 Ti Laptop 4GB VRAM, RAM 16GB):

| Hạng | Mô hình / Giải thuật | Nhóm | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore | Sem. Sim | Latency (s) | Composite Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **TEXTRANK** | Extractive | 0.4149 | 0.2056 | 0.2804 | 0.0758 | 0.7199 | 0.8172 | **0.0100s** | **0.6141** |
| 2 | **LEXRANK** | Extractive | 0.4225 | 0.2085 | 0.2815 | 0.0777 | 0.7207 | 0.8178 | **0.0061s** | **0.6140** |
| 3 | **LSA** | Extractive | 0.4584 | 0.1994 | 0.2913 | 0.0782 | 0.7099 | 0.8026 | **0.0064s** | **0.6051** |
| 4 | **VIT5** | Abstractive | 0.5968 | 0.2822 | 0.3827 | 0.1588 | 0.7074 | 0.7473 | 2.4933s | **0.5800** |
| 5 | **TEXTRANK ➔ VIT5** | Hybrid | 0.5839 | 0.2611 | 0.3678 | 0.1431 | 0.7025 | 0.7389 | 1.6442s | **0.5734** |
| 6 | **LEXRANK ➔ VIT5** | Hybrid | 0.5811 | 0.2587 | 0.3665 | 0.1408 | 0.7014 | 0.7380 | 1.6140s | **0.5723** |
| 7 | **LSA ➔ VIT5** | Hybrid | 0.5638 | 0.2303 | 0.3456 | 0.1217 | 0.6932 | 0.7252 | 1.5663s | **0.5613** |
| 8 | **BARTPHO** | Abstractive | 0.5392 | 0.1964 | 0.3341 | 0.0808 | 0.6783 | 0.7746 | 3.3798s | **0.5582** |
| 9 | **TEXTRANK ➔ MT5** | Hybrid | 0.5464 | 0.2193 | 0.3341 | 0.1141 | 0.6894 | 0.7232 | 1.9941s | **0.5559** |
| 10 | **TEXTRANK ➔ BARTPHO**| Hybrid | 0.5302 | 0.1888 | 0.3278 | 0.0779 | 0.6771 | 0.7655 | 1.8637s | **0.5549** |
| 11 | **LEXRANK ➔ BARTPHO** | Hybrid | 0.5287 | 0.1873 | 0.3260 | 0.0771 | 0.6771 | 0.7653 | 1.6847s | **0.5540** |
| 12 | **LEXRANK ➔ MT5** | Hybrid | 0.5436 | 0.2173 | 0.3324 | 0.1118 | 0.6883 | 0.7200 | 1.8936s | **0.5538** |
| 13 | **MT5** | Abstractive | 0.5484 | 0.2104 | 0.3295 | 0.1069 | 0.6862 | 0.7136 | 2.7282s | **0.5506** |
| 14 | **LSA ➔ MT5** | Hybrid | 0.5242 | 0.1922 | 0.3177 | 0.0948 | 0.6805 | 0.7089 | 1.7027s | **0.5456** |
| 15 | **LSA ➔ BARTPHO** | Hybrid | 0.5102 | 0.1621 | 0.3074 | 0.0677 | 0.6692 | 0.7500 | 1.4959s | **0.5422** |

### Nhận Xét Kết Luận Nghiên Cứu:
1. **Chất lượng sinh vượt trội của ViT5:** Mô hình `VIT5` (Fine-tuned) đạt điểm ROUGE-L cao nhất (**0.3827**), sinh văn bản cực kỳ mượt mà và tự nhiên.
2. **Hiệu năng của mô hình Hybrid:** Các mô hình lai (Hybrid) giảm thiểu hơn **35% - 50% độ trễ (latency)** so với mô hình sinh thuần túy (ví dụ: `BARTPho` giảm từ 3.38s xuống còn **1.68s** khi kết hợp với `LexRank`). Đồng thời, cơ chế trích lọc câu chính ở Giai đoạn 1 giúp triệt tiêu hoàn toàn nguy cơ sập VRAM GPU.
3. **Thế mạnh của Extractive:** Đạt độ trung thực thông tin (`Faithfulness = 100%`) và tốc độ đáp ứng siêu nhanh (<10ms).

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