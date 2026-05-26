# Agentic AI Document Ingest Pipeline

Pipeline này được thiết kế cho bài toán NLP Text Summarization ưu tiên độ chính xác nội dung, factual consistency và khả năng mở rộng sang RAG summarization.

## Kiến trúc

```text
loaders/
  pdf_loader.py      PyMuPDF chính, pdfplumber/unstructured fallback, OCR nếu scanned
  docx_loader.py     python-docx chính, Mammoth fallback
  txt_loader.py      detect encoding cho TXT/MD
  ocr_loader.py      pytesseract trước, EasyOCR fallback

preprocess/
  cleaner.py         Unicode NFC, bỏ header/footer, ghép đoạn, giữ bullet/table
  tokenizer.py       underthesea/pyvi, tiktoken/transformers token counting
  chunker.py         heading + paragraph + semantic boundary + token-aware overlap

embeddings/
  embedder.py        SentenceTransformer wrapper, GPU, fp16, batch, hash fallback
  benchmark.py       benchmark extract/chunk/embed/retrieval/summarization proxy

pipeline/
  ingest_pipeline.py orchestration end-to-end
  schema.py          typed dataclasses
```

Output chuẩn:

```json
{
  "document_id": "...",
  "metadata": {},
  "clean_text": "...",
  "chunks": [],
  "embeddings": [],
  "structure": {}
}
```

## Vì Sao Chọn Thư Viện

PyMuPDF (`fitz`) là parser chính cho PDF vì nhanh, ổn định, lấy được block, page, bbox và metadata. Nhược điểm là bảng phức tạp hoặc PDF layout nhiều cột đôi khi cần fallback.

`pdfplumber` mạnh ở bảng và layout dựa trên tọa độ. Nhược điểm là chậm hơn PyMuPDF.

`unstructured` phù hợp tài liệu phức tạp vì phân loại element như title, narrative text, list item, table. Nhược điểm là dependency nặng và tốc độ thấp hơn.

`python-docx` ổn định cho DOCX, giữ paragraph, style, heading, table. Mammoth dùng fallback khi cần preserve format HTML/list tốt hơn.

`pytesseract` thường chính xác tốt cho OCR tiếng Việt nếu máy đã cài Tesseract + language pack `vie`. EasyOCR là fallback dễ dùng hơn trong môi trường Python, nhưng chậm và tốn RAM/GPU hơn.

`underthesea` được hỗ trợ cho sentence segmentation tiếng Việt, `pyvi` là fallback. Mặc định pipeline dùng regex Unicode nhanh để tránh treo import trong môi trường thiếu model; bật segmenter chuyên sâu bằng `"use_vietnamese_segmenter": true` hoặc biến môi trường `INGEST_USE_VI_SEGMENTER=1`.

`sentence-transformers` là API ổn định nhất để chạy BGE, E5, Jina, SBERT và SimCSE. Pipeline hỗ trợ fp16, batch embedding và prefix đúng cho E5.

## Embedding Recommendation

Khuyến nghị benchmark theo thứ tự:

1. `BAAI/bge-m3`: lựa chọn mặc định tốt cho multilingual retrieval, long-context và tiếng Việt.
2. `intfloat/multilingual-e5-large`: semantic quality cao; nhớ dùng prefix `query:` và `passage:`.
3. `jinaai/jina-embeddings-v3`: hiện đại, mạnh cho nhiều task; cần `trust_remote_code`.
4. `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`: baseline tiếng Việt tốt cho sentence similarity.
5. `keepitreal/vietnamese-sbert`: nhẹ hơn, phù hợp máy yếu hoặc benchmark nhanh.

Không chọn model embedding chỉ theo điểm similarity. Hãy đo thêm retrieval accuracy, chunk coherence và tác động lên summary cuối.

## Chunking Để Không Mất Context

Pipeline không split cứng theo ký tự. Nó ưu tiên:

- Heading boundary để giữ cấu trúc section.
- Paragraph boundary để giữ ý trọn vẹn.
- Sentence boundary khi paragraph quá dài.
- Overlap theo token, không overlap mù theo ký tự.
- Metadata `section_path`, `page_start`, `page_end`, `source_element_ids` để summary/RAG truy vết nguồn.

Với summarization, chunk quá nhỏ gây mất lập luận; chunk quá lớn gây loãng retrieval. Mặc định `target_tokens=420`, `max_tokens=640`, `overlap_tokens=64` là điểm cân bằng cho tài liệu tiếng Việt. PDF báo cáo dài có thể dùng `target_tokens=360`, `overlap_tokens=80`.

## Giảm Hallucination Khi Summarization

Best practices:

- Giữ `metadata`, `page_start/page_end` và `section_path` trong prompt tóm tắt.
- Tóm tắt theo chunk trước, sau đó tổng hợp hierarchical summary.
- Với abstractive summarization, yêu cầu model chỉ dùng facts từ chunk retrieved.
- Dùng retrieval top-k theo từng section thay vì top-k toàn tài liệu nếu tài liệu có nhiều chủ đề.
- Dùng extractive summary làm factual anchors trước khi abstractive rewrite.
- Log chunk ids trong output để audit factual consistency.

## Tối Ưu GPU VRAM Thấp

- Giảm `embedding.batch_size` xuống 2 hoặc 4.
- Bật `use_fp16=true`.
- Giảm `max_seq_length` còn 4096 nếu không cần long context.
- Chạy extraction/chunking CPU trước, embedding theo batch sau.
- Không benchmark nhiều model cùng lúc trên GPU nhỏ; chạy từng model và giải phóng cache.
- Nếu chỉ test pipeline, đặt `"model_name": "hash"` hoặc `--no-embeddings`.

## Cách Chạy

Ingest một tài liệu:

```bash
python -m scripts.ingest_document data/bao-cao.pdf --output storage/ingest/bao-cao.json --pretty
```

Ingest không embedding để kiểm tra extraction/chunking:

```bash
python -m scripts.ingest_document data/bao-cao.pdf --no-embeddings --pretty
```

Ví dụ PDF tiếng Việt lớn, low VRAM:

```bash
python -m scripts.example_ingest_large_pdf data/bao-cao-dien-luc.pdf --output storage/ingest/dien_luc.json
```

Demo semantic chunking:

```bash
python -m scripts.demo_semantic_chunking
```

Benchmark embedding models:

```bash
python -m scripts.benchmark_ingest data/bao-cao.pdf ^
  --query "nhu cầu tiêu thụ điện và vận hành thủy điện" ^
  --reference-summary "Nhu cầu điện tăng cao, thủy điện miền Bắc cần vận hành thận trọng do mực nước hồ chứa thấp." ^
  --output storage/results/ingest_benchmark.json
```

## Benchmark Metrics

Pipeline benchmark hiện đo:

- Extraction speed.
- Extraction quality score.
- Chunk count, average chunk tokens.
- Chunk coherence bằng adjacent embedding similarity.
- Embedding speed và chunks/second.
- Retrieval `hit@k`, MRR nếu có query/reference.
- Summarization quality proxy bằng lexical coverage của reference summary trong top retrieved chunks.

Trong production nên bổ sung đánh giá end-to-end:

- ROUGE/BERTScore giữa final summary và reference summary.
- Factual consistency bằng NLI hoặc LLM-as-judge có citation bắt buộc.
- Human audit cho lỗi số liệu, tên riêng, mốc thời gian.
- Regression set gồm PDF scan, PDF nhiều cột, DOCX có bảng, TXT encoding CP1258.
