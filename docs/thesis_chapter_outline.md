# Dàn Ý Chi Tiết Luận Văn Tốt Nghiệp

## Tên Đề Tài (Chính Thức)

> **"Nghiên Cứu và Phát Triển Hệ Thống Tóm Tắt Văn Bản Tiếng Việt Sử Dụng Mô Hình Transformer: So Sánh 6 Thuật Toán Tóm Tắt Trích Rút và Diễn Giải kết hợp Hệ Thống Hỏi Đáp ChatRAG trên Bộ Dữ Liệu Tin Tức VietNews"**

---

## LỜI MỞ ĐẦU (1-2 trang)

---

## CHƯƠNG 1: GIỚI THIỆU (5-8 trang)

### 1.1 Bối cảnh và Động lực Nghiên cứu
- Bùng nổ thông tin trong kỷ nguyên số hóa
- Thách thức đặc thù của tiếng Việt trong NLP:
  - Ngôn ngữ đơn lập, âm tiết phân tách bằng khoảng trắng
  - Thiếu các bộ dữ liệu quy mô lớn chất lượng cao
  - Giới hạn của các mô hình đa ngôn ngữ tổng quát

### 1.2 Mục Tiêu Nghiên Cứu
- **Mục tiêu 1**: So sánh toàn diện 6 thuật toán tóm tắt văn bản tiếng Việt (3 extractive + 3 abstractive)
- **Mục tiêu 2**: Fine-tune 3 mô hình Transformer (ViT5, BARTPho, mT5) trên bộ dữ liệu VietNews
- **Mục tiêu 3**: Xây dựng hệ thống ChatRAG hỏi đáp thông minh dựa trên nội dung bài báo
- **Mục tiêu 4**: Phát triển hệ thống đánh giá tự động đa tiêu chí (ROUGE, BERTScore, Semantic Similarity)

### 1.3 Phạm Vi Nghiên Cứu
- **Dữ liệu**: Bộ dữ liệu VietNews (`nam194/vietnews`) — 143,816 bài báo tiếng Việt
- **Lĩnh vực**: Báo chí trực tuyến (Tuổi Trẻ, VnExpress, Người Đưa Tin)
- **Ngôn ngữ**: Tiếng Việt hiện đại
- **Phần cứng**: Google Colab T4 GPU (16GB VRAM)

### 1.4 Đóng Góp Khoa Học
1. Bộ thực nghiệm so sánh toàn diện đầu tiên trên 6 thuật toán tóm tắt tiếng Việt trên VietNews
2. Kết quả fine-tuning 3 mô hình Transformer trên bộ dữ liệu 143k+ mẫu
3. Kiến trúc hệ thống RAG kết hợp tóm tắt và hỏi đáp cho báo chí tiếng Việt
4. Bộ công cụ đánh giá tự động kết hợp ROUGE + BERTScore + Semantic Similarity

### 1.5 Cấu Trúc Luận Văn
*Giới thiệu tóm tắt 6 chương còn lại.*

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT (15-20 trang)

### 2.1 Tóm Tắt Văn Bản (Text Summarization)

#### 2.1.1 Định nghĩa và Phân loại
- **Tóm tắt trích xuất (Extractive)**: Chọn câu quan trọng từ văn bản gốc
- **Tóm tắt diễn giải (Abstractive)**: Sinh câu mới tổng hợp nội dung
- **Tóm tắt đơn tài liệu vs. đa tài liệu**

#### 2.1.2 Các Thuật Toán Trích Xuất Cổ Điển
- **TextRank** (Mihalcea & Tarau, 2004): Đồ thị PageRank dựa trên tương đồng câu
- **LexRank** (Erkan & Radev, 2004): Eigenvector Centrality với Cosine TF-IDF
- **TF-IDF Ranking**: Điểm câu dựa trên tần suất từ có trọng số

#### 2.1.3 Thách Thức Đặc Thù Tiếng Việt
- Phân tách từ ghép (word segmentation)
- Đồng âm, đa nghĩa từ tiếng Việt
- Thiếu dữ liệu huấn luyện chất lượng cao

### 2.2 Kiến Trúc Transformer

#### 2.2.1 Self-Attention Mechanism (Vaswani et al., 2017)
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### 2.2.2 Mô Hình T5 và Text-to-Text Framework (Raffel et al., 2020)
- Kiến trúc Encoder-Decoder
- Prefix-based task formulation: `"summarize: <input>"`

#### 2.2.3 Mô Hình BART (Lewis et al., 2020)
- Denoising Autoencoder pre-training
- Seq2Seq với bidirectional encoder

### 2.3 Các Mô Hình Tiếng Việt

#### 2.3.1 ViT5 (Tran et al., 2022)
- Mô hình T5 được pre-train trên 50GB văn bản tiếng Việt
- Vocabulary 32,000 SentencePiece tokens
- `VietAI/vit5-base`: 248M tham số

#### 2.3.2 BARTPho (Tran et al., 2021)
- BART syllable-level cho tiếng Việt của VinAI
- Tokenization ở cấp âm tiết (syllable-level)
- `vinai/bartpho-syllable`: 396M tham số

#### 2.3.3 mT5 (Xue et al., 2021)
- Multilingual T5 từ Google
- Hỗ trợ 101 ngôn ngữ
- `google/mt5-small`: 300M tham số

### 2.4 Retrieval-Augmented Generation (RAG)

#### 2.4.1 Kiến Trúc RAG (Lewis et al., 2020)
- Dense Passage Retrieval + Generative Model
- Kết hợp tri thức ngoài và khả năng sinh ngôn ngữ

#### 2.4.2 Hybrid Search
$$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Sim}_{\text{semantic}} + (1-\alpha) \cdot \text{BM25}_{\text{norm}}$$

### 2.5 Các Thước Đo Đánh Giá

#### 2.5.1 ROUGE (Lin, 2004)
- ROUGE-1: Unigram overlap (F1)
- ROUGE-2: Bigram overlap (F1)
- ROUGE-L: Longest Common Subsequence

#### 2.5.2 BERTScore (Zhang et al., 2020)
- Token-level similarity với BERT embeddings
- Xử lý đồng nghĩa tốt hơn ROUGE

#### 2.5.3 Semantic Similarity (SBERT)
- Sentence-level cosine similarity
- Đánh giá ý nghĩa tổng thể

---

## CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT (15-20 trang)

### 3.1 Kiến Trúc Hệ Thống Tổng Thể
*[Sơ đồ kiến trúc dual-path: Playground + ChatRAG]*

### 3.2 Pipeline Xử Lý Dữ Liệu

#### 3.2.1 Tiền Xử Lý Văn Bản Tiếng Việt
- Loại bỏ HTML tags, noise
- Chuẩn hóa Unicode (NFC)
- Tách câu cho tiếng Việt

#### 3.2.2 Phát Hiện và Loại Bỏ Trùng Lặp

### 3.3 Phương Pháp Fine-Tuning

#### 3.3.1 Chiến Lược Chung
- Framework: HuggingFace Transformers + Seq2SeqTrainer
- Optimizer: AdamW với weight_decay=0.01
- Scheduler: Linear warmup → decay
- Early Stopping: patience=3

#### 3.3.2 Cấu Hình ViT5
| Siêu tham số | Giá trị |
|-------------|---------|
| Batch size | 4 × gradient_acc=4 = 16 effective |
| Learning rate | 3e-5 |
| Max input tokens | 512 |
| Max target tokens | 128 |
| Epochs | 3-5 |
| Prefix | `"summarize: "` |

#### 3.3.3 Cấu Hình BARTPho
| Siêu tham số | Giá trị |
|-------------|---------|
| Batch size | 8 × gradient_acc=2 = 16 effective |
| Learning rate | 5e-5 |
| Scheduler | Cosine |
| Prefix | Không dùng (BART không cần) |

#### 3.3.4 Cấu Hình mT5
| Siêu tham số | Giá trị |
|-------------|---------|
| Tokenizer | T5Tokenizer (fast) |
| Batch size | 8 × gradient_acc=2 = 16 effective |
| Learning rate | 5e-5 |

### 3.4 Mô-Đun Đánh Giá Tự Động
*[Mô tả pipeline evaluation.metrics.py]*

### 3.5 Kiến Trúc ChatRAG
- Chunking Strategy (sentence-boundary aware)
- Embedding: PhoBERT-SimCSE hoặc BAAI/bge-m3
- Vector Store: ChromaDB / Qdrant
- Reranker: BAAI/bge-reranker-v2-m3
- Generator: Local (ViT5/BARTPho) / Cloud (Gemini/OpenAI)

---

## CHƯƠNG 4: BỘ DỮ LIỆU VÀ THỰC NGHIỆM (20-25 trang)

### 4.1 Bộ Dữ Liệu VietNews

#### 4.1.1 Nguồn Gốc và Đặc Điểm
- **Nguồn**: `nam194/vietnews` (HuggingFace Hub)
- **Xuất xứ**: Bài báo tiếng Việt thu thập từ Tuổi Trẻ, VnExpress, Người Đưa Tin
- **Cấu trúc**: `{guid, title, abstract, article}`

#### 4.1.2 Phân Tích Thống Kê (Số liệu thực tế)

| Chỉ số | Train | Validation | Test |
|--------|-------|-----------|------|
| Số mẫu | 99,134 | 22,184 | 22,498 |
| Độ dài bài viết TB | 406.5 từ | 435.7 từ | 429.5 từ |
| Độ dài abstract TB | 32.4 từ | 31.8 từ | 31.8 từ |
| Tỷ lệ nén TB | 9.31% | 8.76% | 8.92% |
| Percentile 95 (bài) | 785 từ | 804 từ | 812 từ |

#### 4.1.3 Tiền Xử Lý Dữ Liệu
- Loại bỏ mẫu rỗng
- Loại bỏ trùng lặp (deduplication)
- Giới hạn samples để thực nghiệm nhanh

### 4.2 Môi Trường Thực Nghiệm

| Thành phần | Chi tiết |
|------------|---------|
| GPU | NVIDIA T4 (16GB VRAM) — Google Colab |
| CPU | Không xác định (Colab) |
| RAM | 12GB |
| Framework | PyTorch 2.x + HuggingFace Transformers 4.38+ |
| Python | 3.10 |
| OS | Ubuntu 22.04 (Colab) |

### 4.3 Kết Quả Thực Nghiệm

#### 4.3.1 Kết Quả Extractive (200 mẫu test — ĐÃ ĐO)

| Thuật Toán | ROUGE-1 | ROUGE-2 | ROUGE-L | Thời gian |
|-----------|:-------:|:-------:|:-------:|:---------:|
| **LexRank** | **0.4404** | 0.2130 | **0.2848** | 0.00s |
| TextRank | 0.4288 | **0.2143** | 0.2800 | 0.02s |
| TF-IDF | 0.3624 | 0.1971 | 0.2481 | 0.00s |

#### 4.3.2 Kết Quả Abstractive (Cập nhật sau fine-tuning)

| Thuật Toán | ROUGE-1 | ROUGE-2 | ROUGE-L | Ghi chú |
|-----------|:-------:|:-------:|:-------:|---------|
| ViT5 | *TBD* | *TBD* | *TBD* | Fine-tuned trên VietNews |
| BARTPho | *TBD* | *TBD* | *TBD* | Fine-tuned trên VietNews |
| mT5 | *TBD* | *TBD* | *TBD* | Fine-tuned trên VietNews |

#### 4.3.3 Phân Tích Định Tính (Case Study)
*[Thêm 3-5 ví dụ cụ thể — bài báo gốc, abstract tham chuẩn, tóm tắt của từng mô hình]*

### 4.4 Phân Tích và Thảo Luận
- So sánh extractive vs. abstractive
- Ưu nhược điểm của từng phương pháp với tiếng Việt
- Ảnh hưởng của fine-tuning

---

## CHƯƠNG 5: XÂY DỰNG HỆ THỐNG DEMO (10-15 trang)

### 5.1 Kiến Trúc Phần Mềm
- Backend: FastAPI + Python 3.10
- Frontend: Vite + TypeScript + React
- Communication: REST API (JSON)

### 5.2 Giao Diện Người Dùng
#### 5.2.1 Playground — So Sánh 6 Thuật Toán
*[Screenshot giao diện + mô tả]*

#### 5.2.2 ChatRAG — Hỏi Đáp Thông Minh
*[Screenshot giao diện + mô tả]*

#### 5.2.3 Dashboard — Thống Kê Hệ Thống
*[Screenshot dashboard]*

### 5.3 API Backend
- `POST /api/summarize` — Tóm tắt với một thuật toán
- `POST /api/playground/compare` — So sánh nhiều thuật toán
- `POST /api/rag/chat` — Hỏi đáp ChatRAG
- `POST /api/documents/upload` — Upload và xử lý tài liệu

### 5.4 Đánh Giá Hiệu Năng Hệ Thống
- Thời gian phản hồi trung bình (ms)
- Throughput (requests/second)
- Sử dụng bộ nhớ GPU

---

## CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (5-8 trang)

### 6.1 Tổng Kết Kết Quả Đạt Được
1. Đã so sánh thực nghiệm 3 thuật toán extractive với kết quả ROUGE-1 đạt tới 0.44
2. Đã fine-tune 3 mô hình Transformer (ViT5, BARTPho, mT5) trên VietNews
3. Đã xây dựng hệ thống ChatRAG tích hợp hoàn chỉnh
4. Đã phát triển pipeline đánh giá tự động đa tiêu chí

### 6.2 Hạn Chế
- Giới hạn phần cứng (Colab T4, 16GB VRAM)
- Chưa đánh giá METEOR và human evaluation
- ChatRAG phụ thuộc vào chất lượng embedding

### 6.3 Hướng Phát Triển
1. Fine-tune với LoRA để giảm tài nguyên tính toán
2. Mở rộng sang bài toán tóm tắt đa tài liệu
3. Tích hợp human evaluation protocol
4. Thử nghiệm với LLM lớn hơn (ViT5-large, BARTPho-word)
5. Mở rộng ChatRAG hỗ trợ multi-turn conversation

---

## TÀI LIỆU THAM KHẢO (20-30 mục)

### Bắt buộc phải có:
1. Vaswani, A. et al. (2017). "Attention is All You Need." *NeurIPS 2017*.
2. Raffel, C. et al. (2020). "Exploring the Limits of Transfer Learning with T5." *JMLR, 21(140)*.
3. Lewis, M. et al. (2020). "BART: Denoising Seq2Seq Pre-training for NLG." *ACL 2020*.
4. Tran, D. Q. et al. (2021). "BARTPho: Pre-trained Seq2Seq Models for Vietnamese." *EMNLP 2021*.
5. Tran, L. et al. (2022). "ViT5: Pretrained Text-to-Text Transformer for Vietnamese." *NAACL 2022*.
6. Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." *ACL Workshop 2004*.
7. Zhang, T. et al. (2020). "BERTScore: Evaluating Text Generation with BERT." *ICLR 2020*.
8. Mihalcea, R. & Tarau, P. (2004). "TextRank: Bringing Order into Text." *EMNLP 2004*.
9. Erkan, G. & Radev, D. (2004). "LexRank: Graph-based Lexical Centrality for NLP." *JAIR, 22*.
10. Lewis, P. et al. (2020). "RAG for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
11. Hu, E. J. et al. (2022). "LoRA: Low-Rank Adaptation of LLMs." *ICLR 2022*.
12. Xue, L. et al. (2021). "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer." *NAACL 2021*.
13. Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT." *EMNLP 2019*.
14. Nguyen, V. H. et al. (2019). "VietNews Dataset." *GitHub/HuggingFace: nam194/vietnews*.
15. Robertson, S. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25." *Foundations and Trends in IR*.

---

## PHỤ LỤC

### Phụ Lục A: Hướng Dẫn Cài Đặt và Chạy Hệ Thống
### Phụ Lục B: Cấu Hình Siêu Tham Số Chi Tiết
### Phụ Lục C: Bảng Kết Quả Thực Nghiệm Đầy Đủ
### Phụ Lục D: Ví Dụ Đầu Vào/Đầu Ra Của Từng Mô Hình
