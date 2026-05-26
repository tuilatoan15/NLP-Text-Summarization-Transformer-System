# Methodology Template — Đề tài 18

## 1. Bài toán nghiên cứu

Tóm tắt văn bản tiếng Việt theo hai hướng:

- **Extractive**: TextRank, LexRank, LSA, TF-IDF
- **Abstractive**: ViT5, mT5, BARTPho

## 2. Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | VNExpress / tài liệu PDF-DOCX nội bộ |
| Tiền xử lý | Unicode NFC, loại noise, dedupe câu |
| Chia tập | `datasets/scripts/vn_dataset_pipeline.py` |

## 3. Pipeline hệ thống

1. Ingest có cấu trúc (PDF/DOCX/TXT, OCR fallback)
2. Semantic chunking (heading-aware, overlap, token-aware)
3. Embedding + retrieval (BGE-M3 / E5 / hash fallback)
4. So sánh thuật toán + citation grounding
5. Đánh giá ROUGE, BERTScore, semantic similarity, factual consistency

## 4. Thiết kế thí nghiệm

- Biến độc lập: thuật toán, độ dài tóm tắt, có/không reference
- Biến phụ thuộc: ROUGE-L, BERTScore F1, latency, hallucination risk
- Kiểm soát: cùng pipeline ingest, cùng tỷ lệ độ dài đầu ra

## 5. Hạn chế

- Abstractive cần GPU và checkpoint fine-tune
- ROUGE/BLEU không phản ánh đúng khi thiếu reference summary
