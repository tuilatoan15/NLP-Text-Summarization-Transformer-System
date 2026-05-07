# 🇻🇳 Hệ thống Tóm tắt Văn bản Tiếng Việt Đa Tài liệu

> Hệ thống NLP hoàn chỉnh cho bài toán tóm tắt văn bản tiếng Việt, sử dụng kết hợp **TextRank (Extractive)** và **ViT5 Transformer (Abstractive)**.

---

## 📁 Cấu trúc dự án

```
NLP-Text-Summarization-Transformer-System/
│
├── data/                      ← Thư mục chứa dataset (CSV/JSON)
├── models/                    ← Thư mục lưu model đã fine-tune
│   └── vit5-finetuned/       ← Được tạo sau khi chạy training
│
├── src/
│   ├── __init__.py
│   ├── crawler.py             ← Thu thập bài báo từ URLs (newspaper3k)
│   ├── preprocess.py          ← Làm sạch văn bản, tách câu tiếng Việt
│   ├── extractive.py          ← Tóm tắt trích xuất (TextRank/sumy)
│   ├── abstractive.py         ← Tóm tắt diễn giải (VietAI/vit5-base)
│   ├── evaluate.py            ← Đánh giá ROUGE-1/2/L
│   ├── selector.py            ← Chọn bản tóm tắt tốt nhất
│   └── utils.py               ← Logging, file I/O, tiện ích
│
├── train/
│   ├── __init__.py
│   ├── dataset_loader.py      ← Tải & chuẩn bị dataset huấn luyện
│   └── train_vit5.py          ← Script fine-tune ViT5
│
├── api/
│   ├── __init__.py
│   └── main.py                ← FastAPI server
│
├── logs/                      ← Tự động tạo khi chạy
├── requirements.txt
└── README.md
```

---

## ⚙️ Yêu cầu hệ thống

- **Python**: 3.10+
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **GPU**: Không bắt buộc (CPU hoạt động được, nhưng chậm hơn ~10-50x)
- **Dung lượng**: ~5GB (model ViT5 ~1.2GB + dataset + dependencies)

---

## 🚀 Cài đặt môi trường

### Bước 1: Clone repository & tạo virtual environment

```bash
# Clone hoặc cd vào thư mục project
cd NLP-Text-Summarization-Transformer-System

# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> ⚠️ **Lưu ý cho Windows**: Nếu gặp lỗi với `newspaper3k`, cài thêm:
> ```bash
> pip install newspaper3k lxml_html_clean
> ```

### Bước 3: Tải NLTK data (cần cho sumy)

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 🧠 Fine-tune Model (Training)

### Option 1: Dùng dataset VnExpress từ Hugging Face Hub (tự động tải)

Mặc định script dùng dataset:

```python
from datasets import load_dataset
ds = load_dataset("thanhnew2001/vnexpress")
```

Dataset này có cột `content/title`; loader sẽ tự map thành `article/title` để fine-tune ViT5.

```bash
# Huấn luyện với 5000 samples, 3 epochs, batch size 2 (phù hợp CPU/máy yếu)
python -m train.train_vit5 --max_samples 5000 --epochs 3 --batch_size 2

# Test nhanh với 100 samples (để kiểm tra pipeline hoạt động)
python -m train.train_vit5 --max_samples 100 --epochs 1 --batch_size 2

# Chỉ định rõ dataset nếu cần
python -m train.train_vit5 --dataset_name thanhnew2001/vnexpress --max_samples 5000 --epochs 3 --batch_size 2
```

### Option 2: Dùng dataset CSV nội bộ

Chuẩn bị file CSV với 2 cột: `article` và `title`:

```csv
article,title
"Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp...","Hội đồng Bảo an họp về Trung Đông"
"Việt Nam ghi nhận tăng trưởng GDP 6.5%...","GDP Việt Nam tăng 6.5% năm 2024"
```

```bash
# Đặt file vào thư mục data/
python -m train.train_vit5 --local_data data/your_dataset.csv --max_samples 5000 --epochs 2
```

### Tham số training

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--max_samples` | 5000 | Số sample tối đa dùng để train |
| `--epochs` | 3 | Số epoch huấn luyện |
| `--batch_size` | 2 | Batch size (2-4 cho CPU, 8-16 cho GPU) |
| `--lr` | 5e-5 | Learning rate |
| `--output_dir` | `./models/vit5-finetuned` | Thư mục lưu model |

Model sẽ được lưu vào `./models/vit5-finetuned/` sau khi hoàn tất.
Kết quả đánh giá sau train được lưu tại `./models/vit5-finetuned/eval_results.json`.

---

## 🌐 Chạy API Server

```bash
# Cách 1: Chạy trực tiếp
python -m api.main

# Cách 2: Dùng uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Cách 3: Production (không reload)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Sau khi khởi động, truy cập:
- 📖 **Swagger UI**: http://localhost:8000/docs
- 📚 **ReDoc**: http://localhost:8000/redoc
- 💚 **Health check**: http://localhost:8000/health

> 💡 **Lần đầu chạy** sẽ cần tải model ViT5 từ Hugging Face (~1.2GB). Đảm bảo có kết nối internet tốt.

---

## 🖥️ Chạy Frontend React

```bash
cd frontend
npm install
npm run dev
```

Frontend mặc định gọi API tại `http://localhost:8000`. Nếu backend chạy port khác:

```bash
set VITE_API_BASE=http://localhost:8001
npm run dev
```

Màn hình chính hỗ trợ:
- Upload nhiều file `.txt`, `.pdf`, `.docx`
- Chọn độ dài `20%`, `50%`, `100 từ`, `200 từ`
- Chọn model `ViT5/T5` hoặc `BART`
- Xem summary tổng hợp, summary từng tài liệu
- Highlight câu nguồn được chọn và lý do chọn câu
- Highlight câu summary nghi vấn theo consistency checker

---

## 📁 Upload nhiều file

Endpoint: `POST /summarize/files`

Form fields:
- `files`: danh sách file TXT/PDF/DOCX
- `length_control`: `auto`, `20_percent`, `50_percent`, `100_words`, `200_words`
- `model_name`: `vit5`, `t5`, `bart`, hoặc tên model Hugging Face
- `save_result`: lưu kết quả vào `storage/results`, và lưu MongoDB nếu có `MONGO_URI`

Ví dụ:

```bash
curl -X POST "http://localhost:8000/summarize/files" \
  -F "files=@data/report.txt" \
  -F "files=@data/article.pdf" \
  -F "length_control=100_words" \
  -F "model_name=vit5"
```

---

## 🔎 Consistency Score & Explainability

Response mới có thêm:
- `consistency.consistency_score`: điểm nhất quán 0-1
- `consistency.suspicious_spans`: các câu nghi vấn cần review
- `consistency.checks`: bằng chứng gần nhất từ văn bản gốc cho từng câu summary
- `explainability.highlights`: câu nguồn được chọn, keyword và lý do chọn
- `documents`: kết quả summary/fact-check riêng cho từng file
- `storage`: đường dẫn JSON đã lưu và `_id` MongoDB nếu cấu hình

---

## 📡 Sử dụng API

### Endpoint: `POST /summarize`

#### Ví dụ 1: Tóm tắt văn bản trực tiếp

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập tức và mở hành lang nhân đạo cho người dân vùng chiến sự. Đại diện Mỹ phát biểu rằng Washington ủng hộ giải pháp hai nhà nước. Cuộc khủng hoảng nhân đạo ngày càng nghiêm trọng khi hàng nghìn thường dân phải di tản.",
    "extractive_sentences": 2,
    "max_abstractive_length": 80
  }'
```

#### Ví dụ 2: Tóm tắt từ URLs

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://vnexpress.net/some-article-url",
      "https://tuoitre.vn/some-article-url"
    ],
    "extractive_sentences": 5
  }'
```

#### Ví dụ 3: Kết hợp text + URLs với reference

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Đoạn văn bản bổ sung...",
    "urls": ["https://vnexpress.net/article"],
    "reference": "Đây là bản tóm tắt mẫu để so sánh ROUGE.",
    "extractive_sentences": 3,
    "max_abstractive_length": 100
  }'
```

#### Response mẫu

```json
{
  "extractive": "Hội đồng Bảo an họp khẩn về Trung Đông. Nhiều nước kêu gọi ngừng bắn và hành lang nhân đạo.",
  "abstractive": "Liên Hợp Quốc tổ chức họp khẩn để giải quyết khủng hoảng nhân đạo tại Trung Đông.",
  "best": "Hội đồng Bảo an họp khẩn về Trung Đông. Nhiều nước kêu gọi ngừng bắn và hành lang nhân đạo.",
  "best_type": "extractive",
  "scores": {
    "extractive": {
      "rouge1": 0.6154,
      "rouge2": 0.4286,
      "rougeL": 0.5385,
      "rougeLsum": 0.5385,
      "length_score": 0.9,
      "combined_score": 0.5523
    },
    "abstractive": {
      "rouge1": 0.5238,
      "rouge2": 0.3636,
      "rougeL": 0.4762,
      "rougeLsum": 0.4762,
      "length_score": 1.0,
      "combined_score": 0.4934
    }
  },
  "word_count": {
    "input": 67,
    "extractive": 19,
    "abstractive": 16,
    "best": 19
  },
  "processing_time_seconds": 3.42
}
```

---

## 📊 Đánh giá ROUGE

Công thức điểm tổng hợp:

```
Combined Score = 0.4 × ROUGE-L + 0.3 × ROUGE-1 + 0.2 × ROUGE-2 + 0.1 × Length Score
```

**Length Score** (0.0 - 1.0):
- Lý tưởng: 30 - 150 từ → điểm 1.0
- Quá ngắn (< 30 từ) hoặc quá dài (> 150 từ) → phạt tuyến tính

---

## 🧩 Sử dụng từng module riêng lẻ

```python
# Crawl bài báo
from src.crawler import crawl_articles
texts = crawl_articles(["https://vnexpress.net/article-url"])

# Tiền xử lý
from src.preprocess import preprocess
result = preprocess(texts[0])
clean_text = result["cleaned"]

# Tóm tắt trích xuất
from src.extractive import extractive_summarize
extractive = extractive_summarize(clean_text, sentence_count=3)

# Tóm tắt diễn giải
from src.abstractive import abstractive_summarize
abstractive = abstractive_summarize(clean_text, max_output_length=100)

# Đánh giá ROUGE
from src.evaluate import compute_rouge
scores = compute_rouge(extractive, clean_text)

# Chọn bản tốt nhất
from src.selector import select_best_summary
result = select_best_summary(extractive, abstractive, reference=clean_text)
print(result["best_summary"])
```

---

## 🐛 Xử lý lỗi thường gặp

### Lỗi `ModuleNotFoundError: No module named 'underthesea'`
```bash
pip install underthesea
```

### Lỗi `newspaper3k` không tải được bài
```bash
pip install newspaper3k lxml_html_clean
```

### Model load chậm / hết RAM
- Giảm `max_samples` trong training
- Dùng CPU thay vì cố dùng GPU không đủ VRAM
- Tắt các ứng dụng nặng khác trước khi chạy

### Lỗi `CUDA out of memory`
```bash
# Chạy trên CPU hoàn toàn
set CUDA_VISIBLE_DEVICES=-1
python -m train.train_vit5 --batch_size 2
```

---

## 📚 Tài liệu tham khảo

- [VietAI/vit5-base](https://huggingface.co/VietAI/vit5-base) — Mô hình ViT5 tiếng Việt
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [sumy](https://github.com/miso-belica/sumy) — TextRank Summarization
- [underthesea](https://github.com/undertheseanlp/underthesea) — NLP Tiếng Việt
- [FastAPI](https://fastapi.tiangolo.com/) — Web Framework

---

## 👨‍💻 Tác giả

Hệ thống được xây dựng cho đồ án tốt nghiệp về NLP và xử lý ngôn ngữ tự nhiên tiếng Việt.
