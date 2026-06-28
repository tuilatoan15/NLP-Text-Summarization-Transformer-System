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

| Hạng | Mô hình / Giải thuật | Nhóm | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore | Sem. Sim | Latency (s) | Composite Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **LEXRANK** | Extractive | 0.4258 | 0.2102 | 0.2835 | 0.0773 | 0.7210 | 0.8169 | **0.0093s** | **0.6143** |
| 2 | **TEXTRANK** | Extractive | 0.4198 | 0.2076 | 0.2820 | 0.0758 | 0.7196 | 0.8165 | **0.0198s** | **0.6139** |
| 3 | **LSA** | Extractive | 0.4604 | 0.2008 | 0.2931 | 0.0775 | 0.7101 | 0.8008 | **0.0088s** | **0.6051** |
| 4 | **VIT5** | Abstractive | 0.5953 | 0.2778 | 0.3802 | 0.1548 | 0.7067 | 0.7457 | 2.6564s | **0.5788** |
| 5 | **TEXTRANK ➔ VIT5** | Hybrid | 0.5819 | 0.2610 | 0.3676 | 0.1421 | 0.7013 | 0.7387 | 1.9019s | **0.5730** |
| 6 | **LEXRANK ➔ VIT5** | Hybrid | 0.5822 | 0.2600 | 0.3663 | 0.1412 | 0.7011 | 0.7386 | 1.8453s | **0.5726** |
| 7 | **LSA ➔ VIT5** | Hybrid | 0.5615 | 0.2296 | 0.3458 | 0.1202 | 0.6925 | 0.7248 | 1.7865s | **0.5612** |
| 8 | **BARTPHO** | Abstractive | 0.5414 | 0.1995 | 0.3356 | 0.0798 | 0.6784 | 0.7761 | 5.1996s | **0.5594** |
| 9 | **TEXTRANK ➔ MT5** | Hybrid | 0.5474 | 0.2201 | 0.3345 | 0.1117 | 0.6884 | 0.7225 | 2.6709s | **0.5557** |
| 10 | **LEXRANK ➔ MT5** | Hybrid | 0.5462 | 0.2187 | 0.3332 | 0.1107 | 0.6880 | 0.7212 | 2.7206s | **0.5549** |
| 11 | **TEXTRANK ➔ BARTPHO**| Hybrid | 0.5275 | 0.1879 | 0.3262 | 0.0749 | 0.6759 | 0.7653 | 2.4403s | **0.5537** |
| 12 | **MT5** | Abstractive | 0.5508 | 0.2158 | 0.3330 | 0.1105 | 0.6870 | 0.7174 | 3.2492s | **0.5532** |
| 13 | **LEXRANK ➔ BARTPHO** | Hybrid | 0.5269 | 0.1867 | 0.3253 | 0.0744 | 0.6758 | 0.7647 | 2.3137s | **0.5531** |
| 14 | **LSA ➔ MT5** | Hybrid | 0.5219 | 0.1924 | 0.3183 | 0.0946 | 0.6800 | 0.7059 | 3.0224s | **0.5447** |
| 15 | **LSA ➔ BARTPHO** | Hybrid | 0.5093 | 0.1671 | 0.3104 | 0.0675 | 0.6703 | 0.7518 | 2.1272s | **0.5438** |

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