# Hệ Thống Tóm Tắt Văn Bản Tiếng Việt Đa Luồng Kết Hợp Kế Thừa Và Học Sâu (Seq2Seq Transformers) & Hỏi Đáp Tài Liệu Đa Tầng ChatRAG (RAPTOR-lite)

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite-61DAFB?style=flat&logo=react&logoColor=white)](https://react.dev)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-FF4B4B?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange?style=flat)](https://huggingface.co)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Compatible-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dự án này là một hệ thống nghiên cứu khoa học và phát triển phần mềm cấp công nghiệp, giải quyết toàn diện hai bài toán cốt lõi của xử lý ngôn ngữ tự nhiên (NLP) cho tiếng Việt: **Tóm tắt văn bản đa thuật toán (Multi-Algorithm Document Summarization)** và **Hỏi đáp thông minh trên tài liệu dài (Advanced ChatRAG - Retrieval-Augmented Generation)**. 

Hệ thống tích hợp side-by-side các giải thuật trích xuất (Extractive) toán học cổ điển vững chắc và các mô hình sinh (Abstractive) học sâu dựa trên kiến trúc Sequence-to-Sequence (Seq2Seq) Transformer đã được tinh chỉnh (fine-tuned) trên các tập dữ liệu báo chí tiếng Việt quy mô lớn. Đồng thời, hệ thống phát triển một **Pipeline Tóm tắt lai nhiều tầng (Hybrid Summarization)** và **Hệ thống RAG nâng cao** tích hợp cơ chế phân tách ngữ nghĩa (Semantic Chunking) và chỉ mục phân cấp cây tóm tắt (RAPTOR-lite) để tối ưu hóa khả năng hỏi đáp trên tài liệu dài mà không bị tràn bộ nhớ VRAM hay mất mát ngữ cảnh.

---

## 📌 Mục Lục

1. [Kiến Trúc Hệ Thống (System Architecture)](#1-kien-truc-he-thong-system-architecture)
2. [Cấu Trúc Thư Mục Dự Án (Project Directory Tree)](#2-cau-truc-thu-muc-du-an-project-directory-tree)
3. [Giải Thích Thuật Toán Chi Tiết (Algorithmic Analysis)](#3-giai-thich-thuat-toan-chi-tiet-algorithmic-analysis)
4. [Hệ Thống Chỉ Số Đánh Giá & Công Thức Toán Học (Evaluation Metrics & Mathematical Formulations)](#4-he-thong-chi-so-danh-gia--cong-thuc-toan-hoc-evaluation-metrics--mathematical-formulations)
5. [Điểm Số Tổng Hợp Xếp Hạng pgBestModel (Composite Score Formulation)](#5-diem-so-tong-hop-xep-hang-pgbestmodel-composite-score-formulation)
6. [Tập Dữ Liệu & Tiền Xử Lý (Datasets & Preprocessing)](#6-tap-du-lieu--tien-xu-ly-datasets--preprocessing)
7. [Kết Quả Thực Nghiệm & So Sánh (Empirical Benchmark Results)](#7-ket-qua-thuc-nghiem--so-sanh-empirical-benchmark-results)
8. [Advanced RAG Pipeline (Hệ Thống RAG Nâng Cao)](#8-advanced-rag-pipeline-he-thong-rag-nang-cao)
9. [Phân Tích Chỉ Mục Cây Phân Cấp RAPTOR (RAPTOR Hierarchical Indexing)](#9-phan-tich-chi-muc-cay-phan-cap-raptor-raptor-hierarchical-indexing)
10. [Pipeline Tóm Tắt Lai (Hybrid Summarization Pipeline)](#10-pipeline-tom-tat-lai-hybrid-summarization-pipeline)
11. [Quản Lý Cuộc Trò Chuyện (Conversation Management)](#11-quan-ly-cuoc-tro-chuyen-conversation-management)
12. [Tài Liệu Hướng Dẫn API (API Documentation)](#12-tai-lieu-huong-dan-api-api-documentation)
13. [Hướng Dẫn Cài Đặt (Installation Guide)](#13-huong-dan-cai-dat-installation-guide)
14. [Kế Hoạch Phát Triển & Bản Đồ Đường Đi (Roadmap & Future Enhancements)](#14-ke-hoach-phat-trien--ban-do-duong-di-roadmap--future-enhancements)

---

## 🏗️ 1. Kiến Trúc Hệ Thống (System Architecture)

Hệ thống được xây dựng theo mô hình hướng dịch vụ (Service-Oriented Architecture - SOA) kết hợp cơ chế lập lịch xử lý nền bất đồng bộ (Celery + Redis) và lưu trữ vector chuyên dụng (ChromaDB/Qdrant).

### Sơ Đồ Kiến Trúc Hệ Thống Tổng Thể (ASCII Architecture Diagram)

```
+---------------------------------------------------------------------------------------------------+
|                                      USER INTERFACE (FRONTEND)                                    |
|                                                                                                   |
|             +------------------+     +------------------+     +--------------------+              |
|             |  React SPA (Vite)|<--->|  Zustand Store   |<--->|  Tailwind CSS UI   |              |
|             +------------------+     +------------------+     +--------------------+              |
+-----------------------------------------------┬────────────────────────────────-------------------+
                                                │ REST API / JSON (Port 5173 -> 8000)
                                                ▼
+---------------------------------------------------------------------------------------------------+
|                                      BACKEND ENGINE (FASTAPI)                                     |
|                                                                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  |                                      FastAPI Routers                                        |  |
|  |  +------------------+  +------------------+  +----------------------+  +-----------------+  |  |
|  |  |   /summarize     |  |    /research     |  |     /rag/chat        |  |   /documents    |  |  |
|  |  +------------------+  +------------------+  +----------------------+  +-----------------+  |  |
|  +--------------------------------------------┬────────────────────────────────----------------+  |
|                                               │
|                                               ▼
|  +---------------------------------------------------------------------------------------------+  |
|  |                                      Service Layer                                          |  |
|  |  +----------------------------+  +----------------------------+  +-----------------------+  |  |
|  |  | DocumentService            |  | AnalyticsService           |  | RAGChatService        |  |  |
|  |  +----------------------------+  +----------------------------+  +-----------------------+  |  |
|  +----------------────────────────────────────┬────────────────────────────────────────────────+  |
|                                               │
|                                               ▼
|  +---------------------------------------------------------------------------------------------+  |
|  |                                      Computation Engines                                    |  |
|  |                                                                                             |  |
|  |  [ Extractive Algorithms ]        [ Abstractive Models ]           [ Advanced RAG Engine ]  |  |
|  |    - TextRank                     - ViT5 (Fine-tuned)                - Chunking Pipeline    |  |
|  |    - LexRank                      - BARTPho (Fine-tuned)             - Embedding Service    |  |
|  |    - LSA (SVD Matrix)             - mT5 (Multilingual)               - Hybrid Retriever     |  |
|  |    - TF-IDF Ranker                - LLM API (Gemini/OpenAI)          - Cross-Reranker       |  |
|  |                                                                      - RAPTOR Indexer       |  |
|  |                                                                      - Grounded Generator   |  |
|  |                                                                                             |  |
|  |  * GPU Semaphore Lock Lock (_GPU_LOCK = Semaphore(1)) prevents simultaneous CUDA VRAM usage* |  |
|  +--------------------------------------------┬────────────────────────────────────────────────+  |
+-----------------------------------------------┼---------------------------------------------------+
                                                │
                                                ▼
+---------------------------------------------------------------------------------------------------+
|                                   DATA & STORAGE LAYER (INFRASTRUCTURE)                           |
|                                                                                                   |
|  +-----------------------+  +------------------------+  +------------------+  +----------------+  |
|  |  Relational Database  |  |    Vector Database     |  |   Message Broker |  |  File Storage  |  |
|  |  - SQLite (RAG chat)  |  |  - ChromaDB (Local)    |  |  - Redis         |  |  - MinIO       |  |
|  |  - Postgres (Eval)    |  |  - Qdrant (Docker)     |  |  - Celery Queue  |  |  - Local FS    |  |
|  +-----------------------+  +------------------------+  +------------------+  +----------------+  |
+---------------------------------------------------------------------------------------------------+
```

---

## 📁 2. Cấu Trúc Thư Mục Dự Án (Project Directory Tree)

Dưới đây là sơ đồ chi tiết cấu trúc thư mục của dự án kèm theo mô tả vai trò kỹ thuật của từng thư mục:

```
NLP-Text-Summarization-Transformer-System/
├── ai_models/                      # Quản lý và tải các mô hình deep learning
│   ├── model_loader.py             # Quản lý vòng đời nạp/cache mô hình trên GPU, tránh quá tải VRAM
│   └── model_registry.py           # Khai báo danh sách checkpoint HuggingFace và tham số suy diễn mặc định
├── api/                            # Các FastAPI Routers (Giao diện lập trình ứng dụng)
│   ├── main.py                     # Entrypoint khởi chạy server API chính, thiết lập CORS và các hooks hệ thống
│   ├── research.py                 # Endpoint phục vụ chạy thử nghiệm, đối sánh thuật toán phục vụ nghiên cứu
│   ├── chat.py                     # Quản lý các endpoint hội thoại thông thường
│   ├── document_chat.py            # API dành riêng cho RAG chatbot và truyền dữ liệu dạng luồng (Streaming API)
│   └── document_intelligence.py    # API phục vụ phân tích cấu trúc tài liệu PDF/DOCX và các tác vụ nặng
├── backend/                        # Module nghiệp vụ cốt lõi của Backend
│   ├── db/                         # Cấu hình kết nối và schema cơ sở dữ liệu (SQLAlchemy, PostgreSQL, SQLite)
│   └── services/                   # Các dịch vụ nghiệp vụ chính
│       └── rag/                    # Thành phần lõi của RAG (retriever, reranker, vector_store, raptor)
├── configs/                        # Chứa các file cấu hình định dạng JSON
│   ├── models.json                 # Cài đặt mặc định của 7 thuật toán tóm tắt (số câu, chiều dài đầu ra...)
│   └── ingest.json                 # Cài đặt tham số phân mảnh văn bản (chunk size, overlap)
├── data/                           # Lưu trữ dữ liệu thô (raw data) và tập dữ liệu phục vụ huấn luyện
├── docs/                           # Tài liệu thiết kế hệ thống, sơ đồ và báo cáo học thuật
├── embeddings/                     # Bộ nhúng tạo vector ngữ nghĩa (SentenceTransformer wrappers)
├── evaluation/                     # Module tính toán chỉ số chất lượng văn bản tóm tắt
│   ├── metrics.py                  # Mã nguồn tính toán ROUGE, BLEU, BERTScore, Faithfulness, Coverage
│   ├── readability.py              # Thống kê phân tích độ phức tạp, độ dài câu và từ của văn bản tiếng Việt
│   └── hallucination.py            # Đánh giá rủi ro bịa đặt thông tin sử dụng kỹ thuật Natural Language Inference
├── frontend/                       # Mã nguồn ứng dụng Client Web SPA (React + Vite + Zustand + Tailwind)
│   ├── src/
│   │   ├── components/             # Các thành phần giao diện dùng chung (charts, modal, file uploaders)
│   │   ├── pages/                  # Các trang Overview, Playground, Compare, Chat, Analytics, Settings
│   │   └── styles.css              # Tệp cấu hình các tokens thiết kế giao diện (CSS variables, animations)
├── loaders/                        # Module đọc và phân tích cấu trúc tài liệu định dạng (PDF, DOCX, TXT)
├── pipeline/                       # Kịch bản luồng xử lý tự động
│   └── hybrid_summarizer.py        # Triển khai giải thuật tóm tắt lai nhiều tầng (Extractive -> Abstractive)
├── scripts/                        # Các tập lệnh thực thi tác vụ nghiên cứu, phân tích dữ liệu
│   ├── run_research_benchmark.py   # Tự động hóa đánh giá benchmark trên tập VietNews
│   ├── run_evaluation.py           # Đánh giá nhanh các mô hình
│   └── train.py                    # Huấn luyện (Fine-tuning) các checkpoint Seq2Seq trên GPU bằng HuggingFace Trainer
├── storage/                        # Thư mục lưu trữ cục bộ (local storage)
│   ├── document_intelligence/      # Nơi lưu SQLite DB, file index ChromaDB và dữ liệu tài liệu dạng JSON
│   └── results/                    # Báo cáo thực nghiệm định dạng CSV, JSON và các biểu đồ trực quan hóa
└── workers/                        # Hàng đợi Celery Worker để xử lý bất đồng bộ các tác vụ nặng
```

---

## 🤖 3. Giải Thích Thuật Toán Chi Tiết (Algorithmic Analysis)

Dự án tích hợp 11 thuật toán và mô hình NLP. Dưới đây là phân tích chi tiết nguyên lý hoạt động, quy trình xử lý, ưu/nhược điểm và độ phức tạp tính toán của từng thuật toán dựa trên mã nguồn thực tế.

---

### A. Nhóm thuật toán trích xuất (Extractive Summarization)

#### 1. TextRank

##### Nguyên lý hoạt động
TextRank là thuật toán xếp hạng văn bản dựa trên lý thuyết đồ thị vô hướng, được lấy cảm hứng từ thuật toán PageRank của Google. Thay vì liên kết giữa các trang web qua liên kết ngược (backlink), TextRank xây dựng đồ thị tương đồng câu, trong đó các đỉnh ($V$) đại diện cho các câu và các cạnh ($E$) có trọng số thể hiện mức độ tương đồng lexical giữa hai câu.

##### Pipeline xử lý
1. Tách văn bản đầu vào thành danh sách câu $\{S_1, S_2, ..., S_n\}$.
2. Chuẩn hóa từng câu (loại bỏ ký tự đặc biệt, đưa về chữ thường).
3. Tính toán trọng số tương đồng $W(S_i, S_j)$ giữa mọi cặp câu dựa trên số từ trùng lặp (unigram overlap) chuẩn hóa theo độ dài của hai câu:
   $$\text{Similarity}(S_i, S_j) = \frac{| \{w \in S_i\} \cap \{w \in S_j\} |}{\log(|S_i|) + \log(|S_j|)}$$
4. Xây dựng ma trận kề đại diện cho đồ thị tương đồng.
5. Chạy thuật toán lặp PageRank để tính điểm hội tụ trọng số của từng đỉnh cho đến khi sai số nhỏ hơn ngưỡng $\epsilon$ ($10^{-4}$):
   $$PR(V_i) = (1 - d) + d \times \sum_{V_j \in \text{In}(V_i)} \frac{W(V_j, V_i)}{\sum_{V_k \in \text{Out}(V_j)} W(V_j, V_k)}$$
   (Với hệ số suy giảm $d = 0.85$).
6. Sắp xếp các câu theo điểm trọng số giảm dần và trích xuất top $k$ câu làm bản tóm tắt.

##### Ưu điểm
* Không yêu cầu huấn luyện trước (Unsupervised).
* Hoạt động cực kỳ nhanh, tiêu tốn ít bộ nhớ.
* Giữ được nguyên văn cấu trúc câu tiếng Việt chuẩn xác.

##### Nhược điểm
* Dễ gặp hiện tượng tóm tắt rời rạc, thiếu sự liên kết mạch lạc giữa các câu được trích xuất.
* Bị ảnh hưởng bởi các câu dài chứa nhiều từ phổ biến (nhiễu).

##### Độ phức tạp
* Xây dựng ma trận tương đồng: $\mathcal{O}(N^2 \times L)$ với $N$ là số lượng câu và $L$ là độ dài trung bình của câu.
* Tính PageRank: $\mathcal{O}(I \times N^2)$ với $I$ là số lượng vòng lặp hội tụ.

---

#### 2. LexRank

##### Nguyên lý hoạt động
LexRank là một thuật toán dựa trên đồ thị tương tự TextRank nhưng sử dụng độ tương đồng cosine (Cosine Similarity) trên vector tần suất từ TF-IDF để đo mức độ liên quan giữa các câu. LexRank đưa ra khái niệm tính điểm trung tâm (centrality) của câu dựa trên mạng lưới kết nối đồ thị vượt ngưỡng (thresholded similarity).

##### Pipeline xử lý
1. Phân tách văn bản thành danh sách câu và tính toán vector đặc trưng TF-IDF cho từng câu.
2. Xây dựng ma trận tương đồng bằng cách tính Cosine Similarity giữa các vector TF-IDF:
   $$\text{CosineSim}(S_i, S_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$$
3. Nhị phân hóa ma trận kề bằng cách áp dụng một ngưỡng (threshold) tương đồng $t$ (mặc định $t = 0.1$). Các cạnh có điểm số dưới $t$ sẽ bị đưa về 0.
4. Tính toán điểm trung tâm của mỗi câu dựa trên bậc của đỉnh (degree centrality):
   $$Centrality(i) = \sum_{j} A_{ij}$$
5. Áp dụng PageRank trên ma trận kề chuẩn hóa để tìm phân phối xác suất dừng ổn định của xích Markov đại diện cho đồ thị câu.
6. Trích xuất các câu có điểm số cao nhất.

##### Ưu điểm
* Loại bỏ được hiện tượng các câu trùng lặp ngẫu nhiên do chia sẻ các từ dừng (stopwords) nhờ có thành phần IDF giảm thiểu trọng số từ phổ biến.
* Khớp ngữ nghĩa tốt hơn TextRank trên văn bản dài.

##### Nhược điểm
* Bản tóm tắt vẫn mang tính chắp vá và có thể bị lặp thông tin nếu hai câu tương đồng cùng đạt điểm centrality cao mà không có cơ chế khử trùng lặp (redundancy removal).

##### Độ phức tạp
* Tính TF-IDF và Cosine: $\mathcal{O}(N^2 \times V)$ với $V$ là kích thước từ vựng của văn bản.
* PageRank: $\mathcal{O}(I \times N^2)$.

---

#### 3. Latent Semantic Analysis (LSA)

##### Nguyên lý hoạt động
LSA sử dụng giải thuật đại số tuyến tính - Phân tích suy biến ma trận (Singular Value Decomposition - SVD) để phân tích không gian ngữ nghĩa ẩn. Bằng cách ánh xạ ma trận từ-câu (Term-Document Matrix) ban đầu sang một không gian ngữ nghĩa có chiều thấp hơn, LSA nắm bắt được cấu trúc chủ đề ẩn và chọn ra các câu mang nhiều thông tin đại diện nhất cho các chủ đề đó.

##### Pipeline xử lý
1. Xây dựng ma trận từ-câu $A$ kích thước $M \times N$, trong đó $M$ là số từ vựng độc nhất, $N$ là số lượng câu. Giá trị ô $A_{i,j}$ là trọng số TF-IDF của từ $i$ trong câu $j$.
2. Thực hiện phân tích trị riêng (SingVD) ma trận $A$:
   $$A = U \Sigma V^\top$$
   Trong đó:
   * $U$ ($M \times R$) là ma trận trực giao biểu diễn mối liên hệ từ - chủ đề ẩn.
   * $\Sigma$ ($R \times R$) là ma trận đường chéo chứa các trị riêng suy biến (singular values) biểu thị mức độ quan trọng của từng chủ đề.
   * $V^\top$ ($R \times N$) biểu diễn mối liên hệ câu - chủ đề ẩn.
3. Chọn $k$ trị riêng lớn nhất tương ứng với các chủ đề ẩn chiếm ưu thế nhất của văn bản.
4. Với mỗi chủ đề ẩn (cột tương ứng của ma trận $V$), chọn ra câu có hệ số đóng góp lớn nhất (giá trị tuyệt đối lớn nhất).
5. Tổng hợp các câu được chọn từ các chủ đề ẩn khác nhau để tạo thành văn bản tóm tắt.

##### Ưu điểm
* Bắt được mối quan hệ đồng nghĩa ẩn (latent semantics) giữa các từ mà các giải thuật lexical thông thường bỏ qua.
* Đảm bảo tính đa dạng thông tin cao do mỗi câu đại diện cho một chủ đề ẩn khác nhau.

##### Nhược điểm
* Việc xác định số lượng chủ đề tối ưu $k$ là một siêu tham số nhạy cảm.
* Thuật toán nhạy cảm với cấu trúc câu quá ngắn hoặc chứa từ viết tắt.

##### Độ phức tạp
* Phân tích SVD: $\mathcal{O}(M \times N \times \min(M, N))$. Với các tài liệu văn bản thông thường, đây là mức độ phức tạp tính toán rất nhỏ.

---

#### 4. TF-IDF Sentence Ranker

##### Nguyên lý hoạt động
Đây là thuật toán baseline trích xuất đơn giản nhất dựa trên thống kê tần suất từ. Điểm số của một câu được tính bằng tổng các trọng số TF-IDF của các từ độc nhất có mặt trong câu đó, sau đó chuẩn hóa theo độ dài câu để tránh thiên vị câu quá dài.

##### Pipeline xử lý
1. Tính toán tần suất từ (Term Frequency - TF) trong từng câu và tần suất tài liệu nghịch đảo (Inverse Document Frequency - IDF) trên toàn bộ văn bản.
2. Với mỗi câu $S_i$, tính điểm số:
   $$Score(S_i) = \frac{\sum_{w \in S_i} \text{TF}(w, S_i) \times \text{IDF}(w)}{\log(|S_i|)}$$
3. Sắp xếp các câu theo điểm số giảm dần và trích xuất top $k$ câu.

##### Ưu điểm
* Tốc độ thực thi cực kỳ nhanh (thời gian tính toán thực tế dưới $10$ ms).
* Phù hợp để làm baseline đối sánh.

##### Nhược điểm
* Hoàn toàn bỏ qua cấu trúc cú pháp, ngữ nghĩa ngữ cảnh và mối quan hệ giữa các câu.
* Bản tóm tắt có xu hướng rời rạc nhất trong số các phương pháp extractive.

##### Độ phức tạp
* Khởi tạo và chấm điểm: $\mathcal{O}(N \times L)$ với $N$ là số câu, $L$ là độ dài trung bình của câu.

---

### B. Nhóm mô hình sinh học sâu (Abstractive Transformer Models)

#### 5. ViT5

##### Nguyên lý hoạt động
ViT5 là mô hình ngôn ngữ dựa trên kiến trúc Sequence-to-Sequence (Encoder-Decoder) Transformer được VietAI tiền huấn luyện (pretrained) chuyên biệt cho tiếng Việt. ViT5 kế thừa thiết kế của mô hình T5 (Text-to-Text Transfer Transformer) và được huấn luyện trên khối lượng dữ liệu tiếng Việt khổng lồ, đặc biệt tối ưu cho các tác vụ sinh văn bản (Text Generation) như dịch máy và tóm tắt văn bản.

##### Pipeline xử lý
1. Nhận chuỗi ký tự đầu vào và đưa qua bộ mã hóa Tokenizer dựa trên SentencePiece để chuyển đổi thành chuỗi token IDs.
2. Đưa qua Encoder để tạo ra các biểu diễn ngữ nghĩa dạng vector ngữ cảnh đa tầng (Contextualized Embeddings).
3. Decoder tự hồi quy (Autoregressive Decoder) sẽ sinh ra bản tóm tắt tiếng Việt từng token một. Sử dụng cơ chế tìm kiếm Beam Search để sinh chuỗi tối ưu nhất:
   $$P(Y|X) = \prod_{i=1}^{m} P(y_i | y_{<i}, X)$$
4. Tokenizer giải mã chuỗi token IDs đầu ra thành văn bản tiếng Việt hoàn chỉnh.

##### Ưu điểm
* Sinh ra văn bản tóm tắt tự nhiên, trôi chảy, diễn đạt lại ý (paraphrasing) tốt như con người viết.
* Có hiểu biết sâu sắc về ngữ pháp, cấu trúc và ngữ nghĩa tiếng Việt.

##### Nhược điểm
* Yêu cầu tài nguyên tính toán lớn (GPU VRAM).
* Gặp giới hạn về độ dài đầu vào tối đa (Max Input Tokens là $1024$). Nếu vượt quá sẽ bị cắt cụt văn bản (truncation).
* Tốc độ suy diễn chậm do tính chất tự hồi quy từng token một của Decoder.

##### Độ phức tạp
* Thời gian suy diễn: $\mathcal{O}(L_{in} \times d_{model} + L_{out} \times L_{in} \times d_{model})$ với $L_{in}$, $L_{out}$ là độ dài chuỗi đầu vào/đầu ra và $d_{model}$ là số chiều ẩn của mạng Transformer.

---

#### 6. BARTPho

##### Nguyên lý hoạt động
BARTPho là mô hình sequence-to-sequence tự hồi quy đầu tiên được huấn luyện pretrain cho tiếng Việt, dựa trên kiến trúc của mô hình BART (Bidirectional and Auto-Regressive Transformers). BARTPho được thiết kế theo mô hình Denoising Autoencoder, kết hợp cả bộ mã hóa hai chiều (Bidirectional Encoder giống BERT) và bộ giải mã tự hồi quy trái-sang-phải (Autoregressive Decoder giống GPT). BARTPho sử dụng đơn vị từ vựng là âm tiết tiếng Việt (Syllable-level), giúp nắm bắt cực kỳ chính xác cấu trúc ngôn ngữ đơn lập của tiếng Việt.

##### Pipeline xử lý
1. Tokenizer mức âm tiết chuyển đổi văn bản đầu vào thành danh sách các token IDs.
2. Encoder hai chiều xây dựng biểu diễn ngữ cảnh toàn diện cho toàn bộ văn bản đầu vào.
3. Decoder tự hồi quy sinh ra các âm tiết tiếng Việt tiếp theo dựa trên cơ chế Attention chéo (Cross-Attention) qua các lớp đại diện ngữ nghĩa của Encoder.
4. Áp dụng cấu hình suy diễn GenerationConfig (num_beams, repetition_penalty) để tối ưu hóa việc chọn từ và tránh lặp từ.
5. Giải mã chuỗi âm tiết đầu ra thành văn bản tóm tắt hoàn chỉnh.

##### Ưu điểm
* Đặc biệt xuất sắc trong việc duy trì cấu trúc ngữ pháp và trật tự từ vựng tiếng Việt mức âm tiết.
* Điểm số BERTScore và Semantic Similarity cao nhất trong các mô hình abstractive độc lập.

##### Nhược điểm
* Tốc độ tính toán chậm nhất trong số các mô hình do kiến trúc BART lớn đòi hỏi nhiều tham số tính toán hơn.
* Nhạy cảm với lỗi tràn bộ nhớ GPU (CUDA OOM) nếu không có cơ chế phân tách và giới hạn đầu vào chặt chẽ.

##### Độ phức tạp
* Tương đương kiến trúc Encoder-Decoder Transformer tiêu chuẩn: $\mathcal{O}(L_{in} \cdot d_{model} + L_{out} \cdot L_{in} \cdot d_{model})$.

---

#### 7. mT5

##### Nguyên lý hoạt động
mT5 (Multilingual T5) là phiên bản đa ngôn ngữ của mô hình T5 do Google phát triển, được huấn luyện trên tập dữ liệu mC4 bao gồm hơn 101 ngôn ngữ (trong đó có tiếng Việt). mT5 sử dụng chung một cấu trúc Text-to-Text để xử lý mọi tác vụ NLP bằng cách thêm tiền tố tác vụ (task prefix) vào đầu câu đầu vào.

##### Pipeline xử lý
1. Tiền xử lý văn bản và thêm tiền tố chỉ định tác vụ, ví dụ: `"summarize: [Văn bản tiếng Việt]"` vào đầu chuỗi đầu vào.
2. Tokenizer đa ngôn ngữ chuyển đổi chuỗi thành các token IDs.
3. Encoder đa ngôn ngữ ánh xạ chuỗi vào không gian biểu diễn ngữ nghĩa đa ngôn ngữ chung.
4. Decoder sinh ra tóm tắt tiếng Việt. Do mT5 được huấn luyện đa ngôn ngữ, decoder có xu hướng sử dụng kiến thức ngữ nghĩa chéo từ các ngôn ngữ khác để bổ trợ ngữ nghĩa khi sinh.

##### Ưu điểm
* Khả năng hiểu ngôn ngữ đa dạng, có thể tóm tắt dịch thuật chéo giữa các ngôn ngữ.
* Phù hợp để làm baseline đối sánh chuẩn quốc tế.

##### Nhược điểm
* Do từ vựng dùng chung cho 101 ngôn ngữ (Multilingual Vocabulary), mật độ token dành riêng cho tiếng Việt bị loãng, dẫn đến việc giải mã đôi khi gặp lỗi chính tả hoặc từ ngữ không thuần Việt nếu không được fine-tune sâu.
* Chi phí huấn luyện và kích thước mô hình lớn nhưng hiệu năng tiếng Việt độc lập không bằng các mô hình chuyên biệt như ViT5 hay BARTPho.

##### Độ phức tạp
* Tương đương kiến trúc T5 tiêu chuẩn.

##### Thông số huấn luyện & Tinh chỉnh mô hình (Fine-tuning & Training Specifications)

Cả ba mô hình tóm tắt sinh (Abstractive: ViT5, mT5, BARTPho) đều được huấn luyện tinh chỉnh (fine-tuned) trên môi trường Google Colab phục vụ báo cáo tốt nghiệp khoa học:
* **Phần cứng huấn luyện**: Google Colab với GPU NVIDIA T4 (16GB VRAM) và CPU Intel Xeon.
* **Thời gian huấn luyện (Training Duration)**: **Hơn 6 giờ** cho mỗi mô hình (chạy 3 epochs đầy đủ trên tập dữ liệu tinh chỉnh).
* **Tập dữ liệu sử dụng**: 30,000 mẫu bài viết tiếng Việt từ tập dữ liệu `nam194/vietnews` (tỷ lệ phân chia 90% train / 10% validation).
* **Siêu tham số (Hyperparameters)**:
  - Batch size: 2 per device (tích lũy gradient accumulation steps = 4, tương đương batch size thực tế = 8).
  - Tốc độ học (Learning Rate): $5 \times 10^{-5}$ với thuật toán tối ưu AdamW.
  - Weight Decay: $0.01$ và Warmup Steps: 100.
  - Chế độ huấn luyện: Mixed Precision (FP16) để tăng tốc và tiết kiệm VRAM.
  - Các script huấn luyện tương ứng: [train.py](file:///c:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/scripts/train.py) và các file notebook Colab: [Colab_ViT5_VietNews_30k_3Epochs.ipynb](file:///c:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/Colab_ViT5_VietNews_30k_3Epochs.ipynb), [Colab_BARTPho_VietNews_30k_3Epochs.ipynb](file:///c:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/Colab_BARTPho_VietNews_30k_3Epochs.ipynb), [Colab_mT5_VietNews_30k_3Epochs.ipynb](file:///c:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/Colab_mT5_VietNews_30k_3Epochs.ipynb).

---

### C. Nhóm thuật toán tìm kiếm và RAG (Retrieval & RAG Algorithms)

```
                            +--------------------------+
                            |     Truy vấn (Query)     |
                            +------------┬-------------+
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
      [ Sparse Keyword Search ]                    [ Dense Semantic Search ]
         - Okapi BM25 Scoring                         - PhoBERT-SimCSE / BGE-M3
         - Tokenized Overlap                          - Vector Cosine Similarity
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                            [ Reciprocal Rank Fusion ]
                               - RRF Score Formula
                               - Candidate Pre-Rerank Top 8
                                         │
                                         ▼
                            [ Cross-Encoder Reranking ]
                               - bge-reranker-v2-m3
                               - Threshold Filtering >= 0.35
                                         │
                                         ▼
                            +--------------------------+
                            |   Top 4 RAG Contexts     |
                            +--------------------------+
```

#### 8. Okapi BM25 (Sparse Keyword Search)

##### Nguyên lý hoạt động
BM25 (Best Matching 25) là thuật toán xếp hạng tìm kiếm dựa trên mô hình không gian vector lexical. Khác với TF-IDF thông thường, BM25 chuẩn hóa tần suất xuất hiện của từ khóa trong tài liệu bằng cách giới hạn mức độ bão hòa tần suất từ (term frequency saturation) và chuẩn hóa theo độ dài của tài liệu (document length normalization).

##### Pipeline xử lý
1. Nhận câu hỏi $Q$ và phân tách thành tập các từ khóa $\{q_1, q_2, ...\}$.
2. Với mỗi phân đoạn tài liệu (chunk) $D$, tính toán điểm số BM25:
   $$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \times \frac{f(q_i, D) \times (k_1 + 1)}{f(q_i, D) + k_1 \times \left(1 - b + b \times \frac{|D|}{\text{avgdl}}\right)}$$
   Trong đó:
   * $f(q_i, D)$ là tần suất của từ $q_i$ trong chunk $D$.
   * $|D|$ là độ dài chunk $D$ tính bằng số từ.
   * $\text{avgdl}$ là độ dài trung bình của tất cả các chunk trong hệ thống.
   * $k_1$ (thiết lập $1.5$) kiểm soát giới hạn bão hòa tần suất từ.
   * $b$ (thiết lập $0.75$) kiểm soát mức độ chuẩn hóa độ dài tài liệu.
3. IDF của từ $q_i$ được tính bằng:
   $$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$
   Với $N$ là tổng số chunk và $n(q_i)$ là số lượng chunk chứa từ $q_i$.
4. Sắp xếp các chunk theo điểm BM25 giảm dần.

##### Ưu điểm
* Tìm kiếm từ khóa chính xác tuyệt đối (ví dụ tên riêng, mã số luật, ngày tháng, thuật ngữ chuyên ngành).
* Thuật toán cực kỳ nhanh và hoạt động ổn định trên CPU.

##### Nhược điểm
* Hoàn toàn không hiểu ngữ nghĩa và từ đồng nghĩa (ví dụ truy vấn "xe hơi" sẽ không khớp với chunk chứa "ô tô" nếu dùng BM25 thuần túy).

##### Độ phức tạp
* Truy vấn: $\mathcal{O}(|Q| \times N_{contains})$ với $N_{contains}$ là số lượng chunk chứa các từ khóa trong truy vấn.

---

#### 9. Semantic Search (Dense Retrieval)

##### Nguyên lý hoạt động
Semantic Search biểu diễn cả truy vấn và các chunk văn bản thành các vector dense có số chiều cố định (embedding vectors) trong một không gian ngữ nghĩa chung nhờ các mô hình học sâu như PhoBERT-SimCSE hoặc BGE-M3. Mức độ liên quan ngữ nghĩa giữa truy vấn và chunk được đo bằng khoảng cách cosine giữa hai vector này.

##### Pipeline xử lý
1. Mô hình hóa câu hỏi thành vector truy vấn $\mathbf{q}$ bằng cách đưa qua SentenceTransformer.
2. Với mỗi chunk $D_i$ đã được vector hóa trước đó thành $\mathbf{d}_i$ và lưu trong Vector DB:
3. Tính điểm Cosine Similarity:
   $$\text{CosineSim}(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\| \|\mathbf{d}_i\|}$$
4. Truy xuất top $k$ chunk có điểm số cosine cao nhất.

##### Ưu điểm
* Hiểu ngữ nghĩa sâu sắc, xử lý xuất sắc các bài toán paraphrase, từ đồng nghĩa và ngữ cảnh câu hỏi.
* Ít bị ảnh hưởng bởi lỗi chính tả nhỏ của người dùng.

##### Nhược điểm
* Yêu cầu tính toán vector tốn tài nguyên (GPU suy diễn).
* Đôi khi bỏ qua các từ khóa chính xác (exact keywords) nếu các từ đó không đóng góp lớn vào ngữ nghĩa tổng thể của vector câu.

##### Độ phức tạp
* Tính vector query: $\mathcal{O}(L_{query} \times d_{model})$.
* Tìm kiếm K-Lân cận gần nhất (KNN): $\mathcal{O}(N \times d_{vector})$ với tìm kiếm tuyến tính hoặc $\mathcal{O}(\log(N) \times d_{vector})$ khi sử dụng cấu trúc chỉ mục HNSW trong Qdrant/ChromaDB.

---

#### 10. Hybrid Search (BM25 + Dense RRF)

##### Nguyên lý hoạt động
Hybrid Search kết hợp thế mạnh của cả hai phương pháp tìm kiếm: Tìm kiếm từ khóa chính xác (Okapi BM25) và Tìm kiếm ngữ nghĩa sâu (Dense Semantic Search). Để kết hợp điểm số của hai phương pháp có miền giá trị và phân phối khác nhau, hệ thống sử dụng thuật toán trộn thứ hạng **Reciprocal Rank Fusion (RRF)**.

##### Pipeline xử lý
1. Chạy song song truy vấn trên BM25 và Dense Semantic Search để thu được hai danh sách xếp hạng độc lập cho cùng tập hợp ứng viên chunk.
2. Với mỗi chunk $d$ có mặt trong một hoặc cả hai danh sách, tính điểm RRF kết hợp:
   $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + r_m(d)}$$
   Trong đó:
   * $r_m(d)$ là thứ hạng (rank, bắt đầu từ $1$) của chunk $d$ trong danh sách kết quả của phương pháp tìm kiếm $m$. Nếu chunk không có mặt trong danh sách, rank của nó được coi là vô hạn và đóng góp của nó bằng $0$.
   * $k$ là hằng số làm mịn (smoothing constant), mặc định đặt bằng $60.0$ theo chuẩn công nghiệp để tránh việc thứ hạng đầu tiên chiếm trọng số quá áp đảo.
3. Sắp xếp lại toàn bộ các chunk theo điểm số RRF giảm dần và lấy top ứng viên (RETRIEVAL_PRE_RERANK_TOP_K = 8).

##### Ưu điểm
* Đạt độ chính xác (Precision) và độ bao phủ (Recall) vượt trội so với việc chỉ dùng đơn lẻ một phương pháp.
* Cân bằng hoàn hảo giữa tính chính xác từ khóa và sự hiểu biết ngữ nghĩa.

##### Nhược điểm
* Tăng thời gian xử lý do phải chạy cả hai tiến trình tìm kiếm song song.

##### Độ phức tạp
* Bằng tổng độ phức tạp của BM25 và Dense Retrieval cộng thêm bước sắp xếp RRF: $\mathcal{O}(N_{candidates} \log(N_{candidates}))$.

---

#### 11. Cross-Encoder Reranker

##### Nguyên lý hoạt động
Các mô hình tìm kiếm thông thường (Bi-Encoders) sinh vector cho truy vấn và tài liệu độc lập với nhau để tìm kiếm nhanh. Tuy nhiên, điều này giới hạn việc tương tác thông tin chéo giữa câu hỏi và tài liệu. Cross-Encoder nhận đồng thời cả truy vấn và tài liệu đầu vào để tính toán cơ chế tự chú ý (Self-Attention) toàn diện trên mọi token của cả hai chuỗi, cho điểm số độ liên quan chính xác sâu sắc.

##### Pipeline xử lý
1. Nhận danh sách top 8 ứng viên từ bước Hybrid Search.
2. Định dạng đầu vào cho Cross-Encoder dạng cặp: `[CLS] Query [SEP] Chunk Text [SEP]`.
3. Đưa qua mô hình Transformer Reranker (`BAAI/bge-reranker-v2-m3`).
4. Lấy xác suất phân loại nhị phân đầu ra tại token `[CLS]` biểu thị mức độ liên quan thực tế giữa query và chunk.
5. Sắp xếp lại danh sách ứng viên theo điểm số rerank này.
6. Loại bỏ các chunk có điểm số dưới ngưỡng lọc an toàn (RETRIEVAL_THRESHOLD = 0.35).
7. Giữ lại tối đa top 4 chunks chất lượng nhất đưa vào prompt ngữ cảnh.

##### Ưu điểm
* Đạt độ chính xác cực kỳ cao, loại bỏ gần như hoàn toàn các kết quả trùng lặp ngữ nghĩa nông (false positives) từ bước Bi-Encoder.
* Cải thiện trực tiếp chất lượng câu trả lời RAG nhờ lọc sạch nhiễu ngữ cảnh.

##### Nhược điểm
* Chi phí tính toán cực kỳ lớn, không thể chạy trên tập dữ liệu hàng triệu chunk trực tiếp mà bắt buộc phải chạy ở bước tái xếp hạng (Rerank) sau khi đã lọc qua bước Hybrid Search.

##### Độ phức tạp
* $\mathcal{O}(K \times (L_{query} + L_{chunk})^2 \times d_{model})$ với $K$ là số lượng ứng viên đưa vào rerank ($K = 8$).

---

## 🧮 4. Hệ Thống Chỉ Số Đánh Giá & Công Thức Toán Học (Evaluation Metrics)

Hệ thống đánh giá khoa học dựa trên 9 chỉ số chất lượng văn bản. Dưới đây là định nghĩa toán học chi tiết của từng chỉ số:

### 1. ROUGE-1 (Unigram Overlap)
ROUGE-1 đo lường sự trùng lặp của các unigram (từ đơn) giữa bản tóm tắt sinh tự động (Candidate) và bản tóm tắt chuẩn (Reference).
* **Công thức toán học:**
  $$\text{ROUGE-1}_{\text{Recall}} = \frac{\sum_{S \in \{\text{Reference}\}} \sum_{\text{gram}_1 \in S} \text{Count}_{\text{match}}(\text{gram}_1)}{\sum_{S \in \{\text{Reference}\}} \sum_{\text{gram}_1 \in S} \text{Count}(\text{gram}_1)}$$
  $$\text{ROUGE-1}_{\text{Precision}} = \frac{\sum_{S \in \{\text{Candidate}\}} \sum_{\text{gram}_1 \in S} \text{Count}_{\text{match}}(\text{gram}_1)}{\sum_{S \in \{\text{Candidate}\}} \sum_{\text{gram}_1 \in S} \text{Count}(\text{gram}_1)}$$
  $$\text{ROUGE-1}_{\text{F1}} = 2 \cdot \frac{\text{ROUGE-1}_{\text{Precision}} \cdot \text{ROUGE-1}_{\text{Recall}}}{\text{ROUGE-1}_{\text{Precision}} + \text{ROUGE-1}_{\text{Recall}}}$$
* **Ý nghĩa:** Đánh giá mức độ bảo tồn từ vựng cốt lõi.
* **Miền giá trị:** $[0.0, 1.0]$. Giá trị càng gần $1.0$ thể hiện bản tóm tắt chứa đầy đủ các từ đơn quan trọng có trong bản gốc.
* **Cách diễn giải:** Điểm ROUGE-1 $> 0.55$ cho thấy sự tương thích từ vựng rất cao.

### 2. ROUGE-2 (Bigram Overlap)
ROUGE-2 đo lường sự trùng lặp của các bigram (cặp từ kề nhau) nhằm đánh giá tính mạch lạc cục bộ.
* **Công thức toán học:**
  $$\text{ROUGE-2}_{\text{Recall}} = \frac{\sum_{S \in \{\text{Reference}\}} \sum_{\text{gram}_2 \in S} \text{Count}_{\text{match}}(\text{gram}_2)}{\sum_{S \in \{\text{Reference}\}} \sum_{\text{gram}_2 \in S} \text{Count}(\text{gram}_2)}$$
* **Ý nghĩa:** Đánh giá tính trôi chảy và mức độ giữ lại cấu trúc từ ghép tiếng Việt (vốn gồm nhiều âm tiết đi liền nhau).
* **Miền giá trị:** $[0.0, 1.0]$.
* **Cách diễn giải:** ROUGE-2 thường thấp hơn ROUGE-1; điểm $> 0.30$ được coi là tốt.

### 3. ROUGE-L (Longest Common Subsequence)
ROUGE-L dựa trên chuỗi con chung dài nhất (LCS) giữa hai câu. Khác với unigrams/bigrams, LCS không yêu cầu các từ phải kề nhau mà chỉ cần xuất hiện đúng thứ tự tương đối.
* **Công thức toán học:**
  $$R_{\text{LCS}} = \frac{\text{LCS}(\text{Ref}, \text{Cand})}{m}, \quad P_{\text{LCS}} = \frac{\text{LCS}(\text{Ref}, \text{Cand})}{n}$$
  $$\text{ROUGE-L} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}$$
  (Với $m$ là độ dài Reference, $n$ là độ dài Candidate, và $\beta = \frac{P_{\text{LCS}}}{R_{\text{LCS}}}$).
* **Ý nghĩa:** Đo mức độ tương đồng cấu trúc câu tổng thể của bản tóm tắt.
* **Miền giá trị:** $[0.0, 1.0]$.
* **Cách diễn giải:** Extractive thường có ROUGE-L cao hơn Abstractive do sao chép nguyên văn cấu trúc câu.

### 4. BLEU (Bilingual Evaluation Understudy)
BLEU tính toán tỷ lệ trùng khớp n-gram ($n=1..4$) kết hợp với một hình phạt độ dài ngắn (Brevity Penalty - BP) để tránh thiên vị văn bản tóm tắt quá ngắn.
* **Công thức toán học:**
  $$\text{BLEU} = \text{BP} \cdot \exp \left( \sum_{n=1}^{N} w_n \ln p_n \right)$$
  $$\text{BP} = \begin{cases} 1 & \text{nếu } c > r \\ \exp\left(1 - \frac{r}{c}\right) & \text{nếu } c \le r \end{cases}$$
  (Với $p_n$ là điểm Precision của n-gram, $w_n = 1/N$ là trọng số, $c$ là độ dài Candidate, $r$ là độ dài Reference).
* **Ý nghĩa:** Đánh giá độ chính xác sinh chuỗi so với mẫu chuẩn.
* **Miền giá trị:** $[0.0, 1.0]$.
* **Cách diễn giải:** Điểm BLEU $> 0.35$ cho thấy chất lượng tóm tắt ở mức nghiên cứu học thuật cao.

### 5. BERTScore
BERTScore tính toán sự tương đồng ngữ nghĩa mềm giữa các token của hai văn bản bằng cách căn chỉnh các vector nhúng ngữ cảnh từ mô hình RoBERTa.
* **Công thức toán học:**
  $$\text{BERTScore}_{\text{Recall}} = \frac{1}{|y|} \sum_{y_i \in y} \max_{x_j \in x} \mathbf{E}_{y_i}^\top \mathbf{E}_{x_j}$$
  $$\text{BERTScore}_{\text{Precision}} = \frac{1}{|x|} \sum_{x_j \in x} \max_{y_i \in y} \mathbf{E}_{y_i}^\top \mathbf{E}_{x_j}$$
  $$\text{BERTScore}_{\text{F1}} = 2 \cdot \frac{\text{BERTScore}_{\text{Precision}} \cdot \text{BERTScore}_{\text{Recall}}}{\text{BERTScore}_{\text{Precision}} + \text{BERTScore}_{\text{Recall}}}$$
* **Ý nghĩa:** Khắc phục nhược điểm của ROUGE/BLEU khi chấm điểm thấp cho các câu diễn đạt đồng nghĩa (paraphrase).
* **Miền giá trị:** $[0.0, 1.0]$.
* **Cách diễn giải:** Điểm số $> 0.82$ biểu thị tính tương đồng ngữ nghĩa sâu sắc.

### 6. Semantic Similarity
Sử dụng mô hình SentenceTransformer để nhúng cả đoạn văn bản thành vector và tính toán Cosine Similarity trực tiếp.
* **Công thức toán học:**
  $$\text{Sim}_{\text{SBERT}}(\text{Cand}, \text{Ref}) = \frac{\mathbf{v}_{\text{cand}} \cdot \mathbf{v}_{\text{ref}}}{\|\mathbf{v}_{\text{cand}}\| \|\mathbf{v}_{\text{ref}}\|}$$
  Chuẩn hóa về $[0.0, 1.0]$:
  $$\text{SemanticSimilarity} = \frac{\text{Sim}_{\text{SBERT}} + 1.0}{2.0}$$
* **Ý nghĩa:** Đo mức độ trùng khớp ý tưởng vĩ mô toàn văn.
* **Miền giá trị:** $[0.0, 1.0]$.

### 7. Faithfulness (Độ trung thực sự thật)
Đo lường tính chính xác về mặt thông tin của bản tóm tắt, phát hiện ảo giác sinh chữ (hallucination) bằng cách đối chiếu từng câu của bản tóm tắt với văn bản nguồn gốc.
* **Công thức toán học:**
  $$\text{Faithfulness} = \frac{1}{|S_{\text{generated}}|} \sum_{s \in S_{\text{generated}}} \max_{d \in D_{\text{source}}} \text{CosineSimilarity}(\mathbf{e}_s, \mathbf{e}_d)$$
* **Ý nghĩa:** Đánh giá xem có câu nào trong bản tóm tắt tự sinh bịa đặt thông tin so với tài liệu gốc hay không.
* **Miền giá trị:** $[0.0, 1.0]$.
* **Cách diễn giải:** Điểm số $> 0.90$ được xem là an toàn và ít rủi ro ảo giác.

### 8. Coverage (Độ bao phủ từ khóa)
* **Công thức toán học:**
  $$\text{Coverage} = \frac{|T_{\text{generated}} \cap T_{\text{source}}|}{|T_{\text{source}}|}$$
  (Với $T$ là tập hợp các từ khóa nội dung có độ dài ký tự $> 2$ và không thuộc danh sách từ dừng).
* **Ý nghĩa:** Đo tỷ lệ từ khóa thông tin gốc được giữ lại.
* **Miền giá trị:** $[0.0, 1.0]$.

### 9. Compression Ratio Score
Đo mức độ cô đọng thông tin dựa trên độ dài so với độ dài tối ưu (CR mục tiêu = 0.25).
* **Công thức toán học:**
  $$\text{CR} = \frac{\text{Words}(\text{Generated})}{\text{Words}(\text{Source})}$$
  $$\text{Score}_{\text{compression}} = \max \left(0.0, 1.0 - \frac{|\text{CR} - 0.25|}{0.25} \right)$$
* **Ý nghĩa:** Đánh giá tính cô đọng thông tin, phạt các bản tóm tắt quá dài hoặc quá ngắn.
* **Miền giá trị:** $[0.0, 1.0]$.

---

## 🧮 5. Điểm Số Tổng Hợp Xếp Hạng pgBestModel (Composite Score)

Để đánh giá và chọn lựa mô hình tối ưu nhất một cách tự động, hệ thống sử dụng điểm số kết hợp **Composite Score** được khai báo tại [src/config.py](file:///c:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/src/config.py). Điểm số này phân phối trọng số cho cả khía cạnh trùng lặp từ vựng (lexical overlap), ngữ nghĩa mềm (token semantic), sự tương đồng ý tưởng toàn văn (sentence embedding semantic), độ phủ thông tin (coverage), độ trung thực (faithfulness), và sự trôi chảy mạch lạc (readability fluency):

$$\mathcal{S}_{\text{composite}} = 0.25 \cdot \text{ROUGE-L} + 0.25 \cdot \text{BERTScore} + 0.20 \cdot \text{SemanticSimilarity} + 0.15 \cdot \text{Faithfulness} + 0.10 \cdot \text{Coverage} + 0.05 \cdot \text{Fluency}$$

### Giải thích ý nghĩa trọng số khoa học
*   **0.25 ROUGE-L (Longest Common Subsequence)**: Đánh giá độ trùng khớp cấu trúc ngữ pháp tuần tự ở mức độ câu, đo lường khả năng giữ nguyên cấu trúc hành văn chuẩn mực.
*   **0.25 BERTScore (Token Contextual Similarity)**: Đánh giá sự tương đồng ngữ nghĩa mềm ở mức độ token ngữ cảnh bằng mô hình RoBERTa, tránh phạt oan các mô hình Abstractive khi sử dụng từ đồng nghĩa hoặc diễn đạt lại (paraphrasing).
*   **0.20 Semantic Similarity (Sentence Embeddings Cosine)**: Đo lường sự tương đồng ý tưởng vĩ mô toàn văn thông qua SBERT, đảm bảo văn bản tóm tắt giữ đúng nội dung cốt lõi của bản gốc.
*   **0.15 Faithfulness (Factual Consistency)**: Đánh giá tính chính xác sự thật của các câu tự sinh đối chiếu với tài liệu gốc nhằm hạn chế tối đa hiện tượng bịa đặt thông tin (ảo giác - hallucination).
*   **0.10 Coverage (Information Coverage)**: Đo lường tỷ lệ các thực thể, danh từ riêng và từ khóa nội dung gốc được giữ lại trong bản tóm tắt.
*   **0.05 Fluency (Language Model Fluency)**: Đánh giá độ tự nhiên, trôi chảy ngữ pháp của văn bản tóm tắt thông qua điểm số perplexity (PPL) từ mô hình GPT-2 tiếng Việt.

---

### Chứng minh toán học cho công thức Điểm số tổng hợp (Mathematical Proof & Properties)

Để đảm bảo tính chuyên nghiệp của đồ án tốt nghiệp và tính nghiêm ngặt về mặt khoa học, dưới đây là các chứng minh tính chất toán học của chỉ số $\mathcal{S}_{\text{composite}}$:

#### 1. Định nghĩa chuẩn tắc (Convex Combination & Boundedness)
Đặt $M = (M_1, M_2, M_3, M_4, M_5, M_6)$ là vector chứa 6 chỉ số đánh giá thành phần được định nghĩa trên miền $[0, 1]^6$:
$$M_1 = \text{ROUGE-L}, \quad M_2 = \text{BERTScore}, \quad M_3 = \text{SemanticSimilarity}$$
$$M_4 = \text{Faithfulness}, \quad M_5 = \text{Coverage}, \quad M_6 = \text{Fluency}$$
Đặt $W = (w_1, w_2, w_3, w_4, w_5, w_6) = (0.25, 0.25, 0.20, 0.15, 0.10, 0.05)$ là vector trọng số.
Ta có:
$$w_i \ge 0, \quad \forall i \in \{1..6\} \quad \text{và} \quad \sum_{i=1}^{6} w_i = 1.0$$
Do đó, hàm số $\mathcal{S}_{\text{composite}}(x) = \sum_{i=1}^{6} w_i M_i(x)$ là một **tổ hợp lồi (Convex Combination)** của các chỉ số thành phần.

**Hệ quả (Tính bị chặn - Boundedness):**
$$\forall x, \quad \mathcal{S}_{\text{composite}}(x) \in [0.0, 1.0]$$
*Chứng minh:*
Vì $M_i(x) \in [0, 1], \forall i \in \{1..6\}$:
$$\mathcal{S}_{\text{composite}}(x) = \sum_{i=1}^{6} w_i M_i(x) \le \sum_{i=1}^{6} w_i \cdot 1.0 = 1.0 \cdot \sum_{i=1}^{6} w_i = 1.0$$
$$\mathcal{S}_{\text{composite}}(x) = \sum_{i=1}^{6} w_i M_i(x) \ge \sum_{i=1}^{6} w_i \cdot 0.0 = 0.0$$
Điều này chứng minh điểm số Composite Score luôn chuẩn hóa trong khoảng $[0\%, 100\%]$, thích hợp hiển thị trực quan mà không bị bão hòa.

#### 2. Tính đơn điệu nghiêm ngặt và Tối ưu Pareto (Strict Monotonicity & Pareto Efficiency)
Giả sử có hai bản tóm tắt $x$ và $y$ được sinh ra:
$$\text{Nếu } M_i(x) \ge M_i(y), \forall i \in \{1..6\} \quad \text{và} \quad \exists j \text{ sao cho } M_j(x) > M_j(y)$$
Thì:
$$\mathcal{S}_{\text{composite}}(x) > \mathcal{S}_{\text{composite}}(y)$$
*Chứng minh:*
$$\mathcal{S}_{\text{composite}}(x) - \mathcal{S}_{\text{composite}}(y) = \sum_{i=1}^{6} w_i \left(M_i(x) - M_i(y)\right)$$
Vì $M_i(x) - M_i(y) \ge 0, \forall i \ne j$ và $w_i > 0$:
$$\mathcal{S}_{\text{composite}}(x) - \mathcal{S}_{\text{composite}}(y) \ge w_j \left(M_j(x) - M_j(y)\right) > 0$$
$$\Rightarrow \mathcal{S}_{\text{composite}}(x) > \mathcal{S}_{\text{composite}}(y)$$
Điều này chứng minh bất kỳ sự cải thiện nào ở một trong các chiều đánh giá mà không làm giảm các chiều khác đều làm tăng điểm tổng hợp, bảo đảm tính tối ưu Pareto (Pareto-optimal) cho bảng xếp hạng mô hình.

#### 3. Cơ chế Triệt tiêu Ảo giác (Hallucination Mitigation Barrier)
Một vấn đề nghiêm trọng của các mô hình sinh (Abstractive) là hiện tượng bịa đặt thông tin không có trong văn bản gốc nhưng câu văn vẫn rất trôi chảy và trùng lặp nhiều từ khóa (ROUGE-L và Fluency cao). Công thức Composite Score ngăn chặn điều này bằng cách gán trọng số Faithfulness ($w_4 = 0.15$).

**Định lý (Giới hạn trên cho mô hình ảo giác):**
Nếu một bản tóm tắt bị mất tính trung thực hoàn toàn ($M_4(x) \to 0$):
$$\mathcal{S}_{\text{composite}}(x) \le 1.0 - w_4 = 0.85$$
Nói cách khác, một mô hình bịa đặt thông tin sẽ bị chặn trên ở mức điểm **0.85** kể cả khi đạt điểm tuyệt đối 1.0 ở cả 5 tiêu chí còn lại. Điều này thiết lập một "rào cản an toàn" (safety barrier) bảo vệ hệ thống RAG và tóm tắt luôn ưu tiên các mô hình có độ trung thực cao.

---

### Trích dẫn Tài liệu Khoa học liên quan (Scientific References)

Ý tưởng kết hợp đa chỉ số đánh giá bằng tổ hợp lồi để tăng tính tương đồng với đánh giá của con người (human alignment) dựa trên các nghiên cứu khoa học uy tín:
1.  **G-Eval / HEval Framework (Liu et al., 2023)**: Xác nhận việc đánh giá chất lượng văn bản sinh cần phân rã thành nhiều chiều (fluency, consistency, coherence, relevance) thay vì chỉ sử dụng ROUGE.
2.  **BERTScore (Zhang et al., 2020)**: Chứng minh tính hiệu quả của việc so sánh cosine vector nhúng token context của BERT để đánh giá ngữ nghĩa mềm (paraphrasing).
3.  **Sentence-BERT (Reimers & Gurevych, 2019)**: Ứng dụng cosine similarity trên vector nhúng toàn câu để so sánh ý nghĩa vĩ mô.
4.  **FactCC / Factuality Evaluation (Kryscinski et al., 2020)**: Đề xuất kiểm tra thực tế (factual consistency) độc lập để ngăn chặn hiện tượng sinh ảo giác ở các mô hình seq2seq.

---

## 📂 6. Tập Dữ Liệu & Tiền Xử Lý (Datasets & Preprocessing)

Dự án sử dụng và tích hợp trực tiếp hai tập dữ liệu chuẩn cho tiếng Việt:

### 1. Tập dữ liệu VietNews (`nam194/vietnews`)
*   **Mô tả:** Tập dữ liệu tóm tắt văn bản tiếng Việt lớn nhất hiện nay, bao gồm các bài báo thu thập từ VnExpress, Tuổi Trẻ, Người Đưa Tin...
*   **Quy mô toàn phần:** **143,816 mẫu** (Bài viết - Tóm tắt).
*   **Thống kê phân tích thực tế** (Thống kê thực hiện trên tập mẫu validation 9,000 mẫu bằng script [scripts/dataset_stats.py](file:///c:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/scripts/dataset_stats.py)):
    *   **Tổng số lượng mẫu phân tích:** 9,000 mẫu.
    *   **Độ dài trung bình văn bản gốc:** **423.9 từ** (Min: 55 từ, Max: 1,693 từ).
    *   **Độ dài trung bình bản tóm tắt:** **32.0 từ** (Min: 10 từ, Max: 88 từ).
    *   **Tỷ lệ nén tự nhiên (Natural Compression Ratio):** **9.0%** (Văn bản tóm tắt chỉ chiếm 9% độ dài văn bản gốc).

### 2. Tập dữ liệu VnExpress (`thanhnew2001/vnexpress`)
*   Bao gồm các bài viết được thu thập và phân loại theo chủ đề đa dạng. Được sử dụng để đánh giá khả năng tổng quát hóa của mô hình và chạy tinh chỉnh (Fine-tuning) tập trung.

### Các kỹ thuật tiền xử lý đặc thù tiếng Việt
Mã nguồn tiền xử lý tại [src/preprocess.py](file:///c:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/src/preprocess.py) thực hiện:
1.  **Unicode Normalization:** Chuẩn hóa toàn bộ văn bản về dạng Unicode NFC nhằm giải quyết triệt để lỗi gõ dấu tiếng Việt (ví dụ dấu chữ "òa" có thể tạo từ hai mã ký tự khác nhau).
2.  **Word Segmentation:** Sử dụng thư viện `pyvi` để thực hiện tách từ ghép tiếng Việt (ví dụ: *"học sinh học sinh học"* thành *"học_sinh học sinh_học"*). Điều này cực kỳ quan trọng cho mô hình BM25 để phân biệt chính xác thực thể.
3.  **Noise Filtering:** Loại bỏ các cụm từ thừa của ban biên tập như *"Ảnh minh họa"*, *"TPO - ..."*, các đường link và khoảng trắng thừa.

---

## 📊 7. Kết Quả Thực Nghiệm & So Sánh (Empirical Benchmark Results)

Dưới đây là kết quả thực nghiệm chi tiết thu được khi chạy thử nghiệm trên máy chủ local có cấu hình CPU đa nhân và GPU NVIDIA GeForce RTX 3050 Ti Laptop (4GB VRAM).

### Bảng Kết Quả Đánh Giá Benchmark Toàn Diện (VietNews - 10.000 mẫu test)
*(Số liệu được kết xuất từ kết quả chạy thực nghiệm thực tế tại [storage/results/leaderboard_benchmark.csv](file:///c:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/storage/results/leaderboard_benchmark.csv))*

| Mã Mô Hình | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore | Sem. Sim. | Latency (s) | Throughput (w/s) | Faithfulness | Coverage | Composite Score | Xếp hạng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lsa_bartpho` (Hybrid) | **0.7252** | **0.3838** | 0.4260 | **0.3618** | **0.9309** | **0.9052** | 9.9131s | 29.23 | 0.9611 | 0.9147 | **0.7774** | **#1** |
| `lexrank_bartpho` (Hybrid) | 0.7175 | 0.3788 | 0.4191 | 0.3560 | 0.9251 | 0.8988 | 9.7277s | 28.55 | 0.9555 | 0.9091 | 0.7712 | #2 |
| `textrank_bartpho` (Hybrid) | 0.7108 | 0.3697 | 0.4094 | 0.3477 | 0.9201 | 0.8901 | 9.5807s | 30.45 | 0.9482 | 0.9020 | 0.7632 | #3 |
| `lsa_vit5` (Hybrid) | 0.6052 | 0.2815 | 0.3882 | 0.3305 | 0.9021 | 0.8759 | 8.5803s | 36.80 | 0.9383 | 0.8921 | 0.7476 | #4 |
| `lexrank_vit5` (Hybrid) | 0.5970 | 0.2734 | 0.3809 | 0.3240 | 0.8971 | 0.8711 | 8.4041s | 35.86 | 0.9298 | 0.8835 | 0.7409 | #5 |
| `bartpho` (Abstractive) | 0.7052 | 0.3655 | 0.4009 | 0.3404 | 0.9096 | 0.8796 | 37.8054s | 10.53 | 0.8904 | 0.8458 | 0.7393 | #6 |
| `textrank_vit5` (Hybrid) | 0.5917 | 0.2680 | 0.3753 | 0.3185 | 0.8917 | 0.8651 | 8.2800s | 38.18 | 0.9200 | 0.8739 | 0.7340 | #7 |
| `lsa` (Extractive) | 0.4705 | 0.3699 | **0.4501** | 0.3821 | 0.7500 | 0.7206 | 0.3266s | 1837.22 | **1.0000** | **0.9459** | 0.7223 | #8 |
| `lexrank` (Extractive) | 0.4510 | 0.3500 | 0.4294 | 0.3653 | 0.7302 | 0.7000 | 0.1930s | 2911.17 | **1.0000** | 0.9452 | 0.7079 | #9 |
| `vit5` (Abstractive) | 0.5876 | 0.2549 | 0.3631 | 0.3084 | 0.8801 | 0.8496 | 30.2687s | 14.83 | 0.8406 | 0.7980 | 0.7013 | #10 |
| `textrank` (Extractive) | 0.4301 | 0.3191 | 0.4103 | 0.3489 | 0.7097 | 0.6806 | **0.1313s** | **4628.79** | **1.0000** | 0.9454 | 0.6942 | #11 |
| `mt5` (Abstractive) | 0.0656 | 0.0363 | 0.0635 | 0.0577 | 0.5200 | 0.4796 | 33.1577s | 18.39 | 0.1796 | 0.1732 | 0.2702 | #12 |

### Phân tích chi tiết các mô hình nổi bật
*   **Mô hình mạnh nhất (Best Quality):** `lsa_bartpho` (Điểm Composite Score cao nhất: **0.7774**). Sự kết hợp này mang lại kết quả tóm tắt xuất sắc nhất về cả cấu trúc ngữ nghĩa ngữ cảnh tiếng Việt và sự mạch lạc của câu chữ.
*   **Mô hình nhanh nhất (Fastest):** `textrank` (Độ trễ trung bình: **0.1313 giây**, tốc độ xử lý **4628.79 từ/giây**). Phù hợp cho các hệ thống thời gian thực cần phản hồi ngay lập tức trên CPU.
*   **Mô hình cân bằng nhất (Best Trade-off):** Nhóm các mô hình lai (Hybrid) như `lsa_vit5` và `textrank_bartpho`. Chúng giúp rút ngắn thời gian suy diễn của các mô hình sinh từ ~37.8 giây xuống còn **~8 - 9 giây** (giảm gần 75% độ trễ) trong khi vẫn duy trì điểm BERTScore ngữ nghĩa cao ($>0.89$) và độ an toàn không bịa đặt (Faithfulness $>92\%$).

---

## 🔍 8. Advanced RAG Pipeline (Hệ Thống RAG Nâng Cao)

Hệ thống ChatRAG được triển khai toàn diện để xử lý hỏi đáp ngữ cảnh trên tài liệu tiếng Việt dài thông qua các bước xử lý nghiêm ngặt:

```
[ PDF/DOCX ] ──► [ PyMuPDF/python-docx ] ──► [ Unicode NFC Clean ]
                                                   │
                                                   ▼
[ Semantic Chunks ] ◄── [ Cosine Breakpoints ] ◄── [ pyvi Tokenization ]
        │
        ▼
[ PhoBERT-SimCSE 768-D ] ──► [ ChromaDB / Qdrant ]
                                   │
                                   ▼
[ Query (BM25 + Dense) ] ──► [ RRF Fusion ] ──► [ bge-reranker-v2-m3 ]
                                                       │
                                                       ▼
[ Citations Output ] ◄── [ Grounded Generation ] ◄── [ Context Top 4 ]
```

1.  **Document Ingestion & Parsing:**
    Tải lên tài liệu PDF, DOCX, TXT. Bộ parser sử dụng `PyMuPDF` (cho PDF) và `python-docx` (cho Word) để đọc văn bản gốc, thu thập thông tin về cấu trúc chương mục và số trang.
2.  **Unicode & Text Cleaning:**
    Văn bản thô được chạy bộ lọc chuẩn hóa Unicode NFC và làm sạch các nhiễu như header, footer, số trang lặp lại.
3.  **Semantic Chunking:**
    Thay vì phân chia đoạn văn bản theo độ dài ký tự cố định làm đứt gãy ý nghĩa, hệ thống triển khai cơ chế **Semantic Chunking**. Văn bản được tách thành các câu đơn độc lập, đưa qua mô hình embedding để tính toán vector. Sau đó, tính toán khoảng cách tương đồng cosine giữa các câu liền kề:
    $$\text{Dist}_i = 1.0 - \text{CosineSimilarity}(\mathbf{v}_{S_i}, \mathbf{v}_{S_{i+1}})$$
    Các điểm ranh giới chủ đề (semantic breakpoints) được phát hiện khi điểm khoảng cách vượt quá ngưỡng biến thiên động (Dynamic Threshold) dựa trên độ lệch chuẩn:
    $$\tau = \mu + 1.2 \times \sigma$$
    Từ đó, hệ thống gộp các câu có cùng chủ đề vào một chunk ngữ nghĩa thống nhất.
4.  **Dense Embedding Generation:**
    Mỗi chunk được chuyển thành vector đặc trưng 768 chiều bằng mô hình `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`.
5.  **Vector Store indexing:**
    Các vector cùng với nội dung văn bản và metadata (filename, page, chunk_index) được lưu vào cơ sở dữ liệu vector persistent (ChromaDB hoặc Qdrant).
6.  **Hybrid Retrieval & RRF:**
    Khi người dùng đặt câu hỏi, hệ thống chạy truy vấn song song:
    *   Truy vấn Keyword chính xác bằng **BM25** trên cơ sở dữ liệu SQLite cục bộ.
    *   Truy vấn ngữ nghĩa bằng **Dense Cosine similarity** trên Vector DB.
    Sau đó, hai danh sách được trộn hạng bằng công thức **Reciprocal Rank Fusion (RRF)** để lấy top 8 ứng viên.
7.  **Cross-Encoder Reranking:**
    Top 8 ứng viên cùng câu hỏi được đưa qua Cross-Encoder `BAAI/bge-reranker-v2-m3` để chấm điểm lại mức độ liên quan thực tế. Chỉ giữ lại tối đa top 4 phân đoạn có điểm số $> 0.35$.
8.  **Grounded Generation & Citation:**
    Mô hình Generator (cục bộ hoặc API) nhận prompt ngữ cảnh gồm top 4 phân đoạn đã lọc và sinh ra câu trả lời cuối cùng. Với mỗi phần thông tin đưa ra, hệ thống đối chiếu và tạo chỉ mục trích dẫn nguồn cụ thể (ví dụ: `[Tài liệu A, trang 5]`) để người dùng dễ dàng kiểm chứng thông tin sự thật.

---

## 🌲 9. Phân Tích Chỉ Mục Cây Phân Cấp RAPTOR (RAPTOR Hierarchical Indexing)

Đối với các tài liệu dài hàng chục trang (báo cáo thường niên, tài liệu luật), hệ thống RAG thông thường chỉ truy xuất được các đoạn nhỏ vụn vặt và bỏ qua bức tranh toàn cảnh (global query). Để giải quyết vấn đề này, dự án triển khai kỹ thuật **RAPTOR-lite (Recursive Abstractive Processing for Tree-Organized Retrieval)**.

### Sơ Đồ Cấu Trúc Chỉ Mục Cây RAPTOR (ASCII Tree Diagram)

```
                            [ Root Node: Toàn bộ tài liệu ]
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       [ Parent Node L2 - Cụm A ]                    [ Parent Node L2 - Cụm B ]
        (Abstractive Summary A)                       (Abstractive Summary B)
             │                                             │
      ┌──────┴──────┐                               ┌──────┴──────┐
      ▼             ▼                               ▼             ▼
 [ Node L1-1 ] [ Node L1-2 ]                   [ Node L1-3 ] [ Node L1-4 ]
 (Summary 1)   (Summary 2)                     (Summary 3)   (Summary 4)
      │             │                               │             │
  ┌───┴───┐     ┌───┴───┐                       ┌───┴───┐     ┌───┴───┐
  ▼       ▼     ▼       ▼                       ▼       ▼     ▼       ▼
[Leaf 1][Leaf 2][Leaf 3][Leaf 4]              [Leaf 5][Leaf 6][Leaf 7][Leaf 8]
(Base chunks gốc - Level 0)                   (Base chunks gốc - Level 0)
```

### Quy trình xây dựng cây RAPTOR (Tree Construction Pipeline)
1.  **Leaf Nodes (Level 0):** Văn bản tài liệu gốc được chia thành các chunk cơ bản (leaf nodes).
2.  **Soft Clustering (GMM):** Vector hóa các nút ở tầng hiện tại. Sử dụng thuật toán phân cụm hỗn hợp Gaussian (**Gaussian Mixture Model - GMM**) bằng Numpy để gom nhóm các vector tương đồng. Khác với K-Means phân cụm cứng, GMM hỗ trợ phân cụm mềm chéo (overlapping clustering). Một nút có xác suất thuộc về một cụm lớn hơn ngưỡng $\theta = 0.15$ sẽ được gán cho cụm đó, giúp bảo tồn các phân đoạn mang chủ đề giao thoa chéo.
3.  **Recursive Summarization:** Với mỗi cụm được tạo ra, gom toàn bộ văn bản của các nút con lại và đưa qua mô hình Generator để viết một bản tóm tắt ý chính của cụm đó (Abstractive Summary Node).
4.  **Parent Nodes creation:** Bản tóm tắt cụm này được lưu lại thành một nút cha (Parent Node) ở tầng tiếp theo (Level 1), tiến hành vector hóa nút cha này.
5.  **Recursion:** Tiếp tục lặp lại quá trình phân cụm và tóm tắt trên các nút tầng mới cho đến khi số lượng nút ở tầng hiện tại quá ít (nhỏ hơn 3) hoặc đạt số tầng tối đa (Max Levels = 3).
6.  **Query Traversal:** Khi người dùng hỏi một câu hỏi tổng quan (ví dụ: *"Các chủ đề chính được thảo luận trong báo cáo này là gì?"*), hệ thống sẽ tìm kiếm trên cả các nút cha (Summary Nodes) và nút lá (Leaf Nodes) để đưa ra câu trả lời mang tính bao quát và tổng hợp cao.

---

## 🔄 10. Pipeline Tóm Tắt Lai (Hybrid Summarization Pipeline)

Khi xử lý các văn bản cực dài bằng các mô hình học sâu sinh văn bản (Abstractive Transformers như ViT5, BARTPho), lập trình viên thường đối mặt với hai thách thức lớn:
1.  **CUDA Out-of-Memory (OOM):** Độ phức tạp tính toán cơ chế self-attention tăng theo hàm mũ bình phương độ dài đầu vào $\mathcal{O}(L^2)$, gây tràn bộ nhớ VRAM nhanh chóng.
2.  **Context Truncation:** Các mô hình đều có giới hạn token đầu vào tối đa (thường là $1024$ tokens). Các phần văn bản phía sau giới hạn này sẽ bị cắt bỏ hoàn toàn, gây mất mát thông tin nghiêm trọng.

Hệ thống của chúng tôi giải quyết vấn đề này bằng **Pipeline Tóm tắt lai nhiều tầng (Hybrid Summarization Pipeline)**:

```
[ Văn bản gốc cực dài ]
         │
         ▼
[ Giai đoạn 1: Extractive Filter ] (LSA / TextRank / LexRank)
   - Chọn ra N câu có trọng số ngữ nghĩa cao nhất
   - Nén 45% - 50% độ dài văn bản nguồn
         │
         ▼
[ Giai đoạn 2: Context Compression ]
   - Gom các câu được chọn theo đúng trật tự thời gian xuất hiện gốc
         │
         ▼
[ Giai đoạn 3: Abstractive Transformer ] (ViT5 / BARTPho)
   - Đưa văn bản đã nén vào Seq2Seq Model để sinh tóm tắt tự nhiên
         │
         ▼
[ Văn bản tóm tắt cuối cùng ]
```

### Giải thích tại sao Hybrid Pipeline cải thiện hiệu năng hệ thống:
*   **Giảm 75% độ trễ (Latency reduction):** Nhờ giảm kích thước đầu vào từ sớm, Decoder tự hồi quy của Transformer phải xử lý ít bước tính toán attention hơn, rút ngắn thời gian sinh từ ~37.8 giây xuống còn **~9.9 giây**.
*   **Triệt tiêu lỗi CUDA OOM:** Độ dài đầu vào được ép dưới ngưỡng an toàn (thường là $512$ tokens), đảm bảo VRAM GPU luôn hoạt động ổn định dưới mức giới hạn tối đa.
*   **Tăng tính trung thực thông tin (Faithfulness):** Các giải thuật extractive ở Giai đoạn 1 đóng vai trò như bộ lọc loại bỏ các câu nhiễu ngoài lề, thông tin quảng cáo, giúp mô hình sinh ở Bước 2 tập trung hoàn toàn vào nội dung cốt lõi và không bị phân tâm dẫn tới bịa đặt thông tin.

---

## 💬 11. Quản Lý Cuộc Trò Chuyện (Conversation Management)

Ứng dụng hỗ trợ hệ thống quản lý lịch sử hội thoại toàn diện tương tự như ChatGPT để nâng cao trải nghiệm người dùng cuối:

*   **Phân nhóm lịch sử theo thời gian (Chronological Grouping):** Lịch sử trò chuyện ở Sidebar tự động phân nhóm khoa học theo các mốc: *Hôm nay, Hôm qua, 7 ngày gần nhất, 30 ngày gần nhất, Cũ hơn*.
*   **Tự động sinh tiêu đề hội thoại (Auto Title Generation):** Khi người dùng bắt đầu cuộc hội thoại mới và cuộc trò chuyện đạt từ 2 đến 4 tin nhắn đầu tiên, hệ thống sẽ tự động kích hoạt một luồng xử lý chạy ngầm (background process). Luồng này gửi câu hỏi đầu tiên tới mô hình ngôn ngữ để tự động sinh ra một tiêu đề cô đọng từ 5 đến 10 từ tiếng Việt phù hợp ngữ cảnh, sau đó cập nhật lại cơ sở dữ liệu SQLite cục bộ mà không làm gián đoạn luồng chat. Nếu tiến trình này bị lỗi, hệ thống sẽ tự động lấy 50 ký tự đầu tiên làm tiêu đề.
*   **Tìm kiếm lịch sử trò chuyện (Full-Text Search):** Người dùng có thể tìm kiếm nhanh lại các cuộc trò chuyện cũ. Thanh tìm kiếm sẽ lọc song song theo tiêu đề cuộc trò chuyện và nội dung chi tiết của từng tin nhắn bên trong bằng truy vấn SQL tối ưu.
*   **Xóa cascade (Cascade Deletion):** Hỗ trợ xóa cuộc trò chuyện. Khi thực hiện xóa, hệ thống sẽ hiển thị modal xác nhận và tự động xóa sạch các bản ghi tin nhắn và trích dẫn liên quan trong database.

---

## 🔌 12. Tài Liệu Hướng Dẫn API (API Documentation)

FastAPI tự động cung cấp tài liệu Swagger UI đầy đủ tại `http://localhost:8000/docs`. Dưới đây là mô tả chi tiết một số endpoints cốt lõi kèm theo ví dụ JSON Payload thực tế.

### 1. Endpoint Tóm Tắt Đơn (`POST /summarize`)
*   **Chức năng:** Tóm tắt văn bản đầu vào sử dụng một mô hình được chỉ định.
*   **Request Body JSON:**
```json
{
  "text": "Trong hai ngày 18 và 19-6, khu vực Bắc Bộ và Trung Bộ tiếp tục xảy ra nắng nóng gay gắt diện rộng với nhiệt độ cao nhất phổ biến từ 37 đến 39 độ C, có nơi trên 40 độ C. Độ ẩm tương đối thấp nhất phổ biến 40-45%. Cảnh báo cấp độ rủi ro thiên tai do nắng nóng ở cấp 1.",
  "model_name": "vit5",
  "extractive_sentences": 3,
  "max_abstractive_length": 120,
  "target_length_ratio": 30,
  "use_length_ratio": false
}
```
*   **Response JSON:**
```json
{
  "summary": "Dự báo khu vực Bắc Bộ và Trung Bộ tiếp tục nắng nóng gay gắt diện rộng với nhiệt độ cao nhất từ 37 đến 39 độ C.",
  "algorithm": "vit5",
  "word_count": 27,
  "processing_time": 4.1205,
  "metrics": {
    "rouge1": 0.5876,
    "rouge2": 0.2549,
    "rougeL": 0.3631,
    "bleu": 0.3084,
    "bertscore_f1": 0.8801,
    "semantic_similarity": 0.8496,
    "faithfulness": 0.8406,
    "coverage": 0.7980,
    "compression_ratio": 0.23,
    "composite_score": 0.7013
  }
}
```

---

### 2. Endpoint So Sánh Đa Thuật Toán (`POST /summarize/compare`)
*   **Chức năng:** So sánh side-by-side kết quả tóm tắt, thời gian chạy và metrics của nhiều mô hình cùng lúc.
*   **Request Body JSON:**
```json
{
  "text": "Văn bản tiếng Việt dài cần phân tích đối sánh chi tiết giữa các mô hình...",
  "reference": "Bản tóm tắt chuẩn của con người phục vụ tính toán metrics...",
  "algorithms": ["textrank", "lsa", "vit5", "bartpho"],
  "extractive_sentences": 5,
  "max_abstractive_length": 150
}
```
*   **Response JSON:**
```json
{
  "results": [
    {
      "algorithm": "textrank",
      "summary": "Kết quả tóm tắt trích xuất bằng TextRank...",
      "processing_time": 0.1235,
      "metrics": { "rougeL": 0.4103, "composite_score": 0.6942 }
    },
    {
      "algorithm": "lsa",
      "summary": "Kết quả tóm tắt trích xuất bằng LSA...",
      "processing_time": 0.2987,
      "metrics": { "rougeL": 0.4501, "composite_score": 0.7223 }
    },
    {
      "algorithm": "vit5",
      "summary": "Kết quả tóm tắt sinh bằng ViT5...",
      "processing_time": 4.2504,
      "metrics": { "rougeL": 0.3631, "composite_score": 0.7013 }
    }
  ],
  "best_model": {
    "key": "lsa",
    "composite_score": 0.7223,
    "reason": "Mô hình lsa đạt điểm số Composite cao nhất trong các mô hình hợp lệ."
  }
}
```

---

### 3. Endpoint Chat RAG Luồng (`POST /rag/chat/stream`)
*   **Chức năng:** Truyền dữ liệu dạng luồng (streaming) câu trả lời chatbot từng token một theo chuẩn Server-Sent Events (SSE).
*   **Request Body JSON:**
```json
{
  "query": "Bị cáo Nguyễn Văn A bị phạt bao nhiêu năm tù?",
  "conversation_id": "8f2e9fea-017b-40ac-85eb-50da3e3ae068",
  "document_ids": ["d3217418-3599-42d7-aa7b-5a48cf6d82e1"],
  "top_k": 4,
  "threshold": 0.35,
  "retrieval_mode": "hybrid",
  "use_reranking": true
}
```
*   **Response (Server-Sent Events Stream):**
```
data: {"event": "token", "content": "Theo bản án hình sự", "conversation_id": "8f2e9fea-017b-40ac-85eb-50da3e3ae068"}

data: {"event": "token", "content": "Theo bản án hình sự sơ thẩm, bị cáo Nguyễn Văn A", "conversation_id": "8f2e9fea-017b-40ac-85eb-50da3e3ae068"}

...

data: {"event": "done", "response": {"conversation_id": "8f2e9fea-017b-40ac-85eb-50da3e3ae068", "answer": "Theo bản án hình sự sơ thẩm, bị cáo Nguyễn Văn A bị xử phạt 5 năm tù về tội lừa đảo chiếm đoạt tài sản.", "confidence": 0.9450, "grounded": true, "retrieved_context": [{"chunk_id": "chunk_0", "filename": "ban_an.docx", "page": 1, "text": "Xử phạt bị cáo Nguyễn Văn A 05 năm tù...", "combined_score": 88.5}]}}
```

---

## ⚙️ 13. Hướng Dẫn Cài Đặt (Installation Guide)

Dự án tương thích hoàn toàn với các hệ điều hành chính (Windows, Ubuntu/Linux) và hỗ trợ ảo hóa qua Docker.

### A. Triển khai trên môi trường Windows (PowerShell)
1.  **Cài đặt các công cụ yêu cầu:**
    *   Tải và cài đặt Python phiên bản **3.11.x** hoặc **3.12.x** (Lưu ý: Không dùng Python 3.13 do một số thư viện PyTorch/Transformers chưa tương thích hoàn toàn).
    *   Cài đặt Node.js phiên bản LTS mới nhất.
2.  **Khởi tạo dự án:**
    *   Tải mã nguồn về máy:
        ```powershell
        git clone https://github.com/tuilatoan15/NLP-Text-Summarization-Transformer-System.git
        cd NLP-Text-Summarization-Transformer-System
        ```
3.  **Thiết lập Virtual Environment & Cài đặt thư viện Python:**
    ```powershell
    python -m venv venv
    venv\Scripts\activate
    pip install --no-cache-dir -r requirements.txt
    ```
4.  **Cài đặt PyTorch tối ưu CUDA (Nếu máy có GPU NVIDIA):**
    ```powershell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```
5.  **Cấu hình Environment Variables:**
    *   Sao chép tệp cấu hình mẫu: `copy .env.example .env`
    *   Mở tệp `.env` và thiết lập các biến môi trường phù hợp (ví dụ: `PRELOAD_MODELS=0` để tải mô hình theo yêu cầu tiết kiệm RAM khởi động).
6.  **Khởi chạy hệ thống bằng tệp Batch:**
    *   Chạy tệp khởi động tự động:
        ```powershell
        .\run_project.bat
        ```
    *   Tệp này sẽ tự động kích hoạt môi trường ảo, chạy backend FastAPI trên cổng `8000`, sau đó truy cập thư mục `frontend` và khởi chạy Vite dev server trên cổng `5173`.

---

### B. Triển khai trên môi trường Linux (Ubuntu)
1.  **Cài đặt các gói phụ thuộc hệ thống:**
    ```bash
    sudo apt update && sudo apt install -y python3-pip python3-venv nodejs npm git
    ```
2.  **Khởi tạo Virtual Environment và cài đặt dependency:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install --no-cache-dir -r requirements.txt
    ```
3.  **Khởi chạy Backend Server:**
    ```bash
    python3 -m api.main
    ```
4.  **Cài đặt và chạy Frontend:**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

---

### C. Triển khai tự động bằng Docker Compose
Để chạy toàn bộ hệ thống tích hợp đầy đủ cơ sở dữ liệu PostgreSQL, cache Redis, cơ sở dữ liệu vector Qdrant và Celery Workers:
1.  Đảm bảo máy của bạn đã cài đặt Docker và Docker Compose.
2.  Chạy lệnh build và khởi động tại thư mục gốc của dự án:
    ```bash
    docker-compose up -d --build
    ```
3.  Docker sẽ tự động tải các images, thiết lập mạng nội bộ giữa các dịch vụ và ánh xạ cổng ra ngoài:
    *   FastAPI Swagger Docs: `http://localhost:8000/docs`
    *   Giao diện người dùng React: `http://localhost:5173`
    *   Qdrant Dashboard: `http://localhost:6333/dashboard`

---

## 🗺️ 14. Kế Hoạch Phát Triển & Bản Đồ Đường Đi (Roadmap & Future Enhancements)

Dựa trên các phần mã nguồn được đánh dấu `TODO` và định hướng phát triển thực tế, chúng tôi xây dựng kế hoạch nâng cấp hệ thống như sau:

*   `[ ]` **Semantic Chunking nâng cao:** Thay thế phương pháp chia văn bản theo số câu cố định bằng việc sử dụng embedding model để tính cosine similarity giữa các câu liền kề và cắt chunk tại các điểm ranh giới thay đổi chủ đề ngữ nghĩa (Semantic Breakpoints).
*   `[ ]` **Triển khai Agentic RAG:** Tích hợp cơ chế tự động lập kế hoạch tìm kiếm (Query rewriting/expansion) và tự sửa lỗi câu trả lời bằng LLM Agent (Self-RAG) khi LLM Judge chấm điểm câu trả lời có độ tin cậy thấp hoặc phát hiện rủi ro ảo giác sự thật.
*   `[ ]` **Phục hồi đồ thị liên kết thực thể (Entity-Graph RAG):** Trích xuất các thực thể chính (người, địa danh, tổ chức) trong tài liệu tiếng Việt và xây dựng đồ thị tri thức (Knowledge Graph) liên kết giữa chúng nhằm phục vụ cho các câu hỏi suy luận sâu sắc.
*   `[ ]` **Tích hợp mô hình ngôn ngữ lớn tiếng Việt cục bộ (Local LLM):** Tinh chỉnh và tối ưu hóa suy diễn các mô hình chuyên biệt tiếng Việt như `PhoGPT`, `ViGemma` hoặc `VinaLlama` chạy trực tiếp trên máy chủ cục bộ bằng thư viện vLLM để bảo mật dữ liệu tuyệt đối cho doanh nghiệp.