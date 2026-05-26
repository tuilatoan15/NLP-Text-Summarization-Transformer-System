# 🚀 Quick Start Guide - Comparison Platform

## Prerequisites
- Python 3.10+
- Node.js 18+
- GPU (optional, but recommended for Transformer models)

---

## ⚡ 5-Minute Setup

### Step 1: Start Backend (Terminal 1)

```bash
# Navigate to project root
cd path/to/NLP-Text-Summarization-Transformer-System

# Install Python dependencies (if not done)
pip install -r requirements.txt

# Start FastAPI backend on port 8000
python -m backend.app.main
```

**Expected output**:
```
INFO: Started server process [12345]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 2: Start Frontend (Terminal 2)

```bash
# Navigate to frontend directory
cd path/to/frontend

# Install dependencies (first time only)
npm install

# Start Vite dev server on port 5173
npm run dev
```

**Expected output**:
```
  ➜  Local:   http://localhost:5173/
  ➜  press h to show help
```

### Step 3: Open in Browser

```
http://localhost:5173
```

---

## 🎯 Navigate to Comparison Page

1. **Sidebar Navigation**: Click on `🔬 Comparison` in the left sidebar
2. **Or Direct URL**: http://localhost:5173/comparison

---

## 📝 Try the System

### Input Sample Text

Copy and paste this Vietnamese text:

```vietnamese
Trí tuệ nhân tạo (AI) đang cách mạng hóa ngành y tế hiện đại. Các ứng dụng của AI bao gồm chẩn đoán bệnh, dự đoán tiến triển bệnh, phát hiện ung thư sớm, và tối ưu hóa liệu pháp điều trị. Mô hình học sâu đã chứng minh khả năng phân tích hình ảnh y tế với độ chính xác vượt quá bác sĩ con người. Các bệnh viện lớn trên thế giới đã triển khai hệ thống AI để giúp bác sĩ chẩn đoán chính xác hơn, nhanh hơn. Tuy nhiên, các thách thức như thiếu dữ liệu huấn luyện, vấn đề quyền riêng tư bệnh nhân, và sự tin tưởng của nhân viên y tế vẫn cần được giải quyết. Các nhà khoa học đang làm việc để phát triển các mô hình AI giải thích được, cho phép bác sĩ hiểu tại sao AI đưa ra quyết định nhất định.
```

### Optional: Add Reference Summary

```vietnamese
AI đang thay đổi y tế thông qua chẩn đoán bệnh, dự đoán tiến triển, và tối ưu hóa điều trị. Mô hình học sâu vượt quá bác sĩ con người trong phân tích hình ảnh y tế. Các thách thức bao gồm thiếu dữ liệu, quyền riêng tư, và tin tưởng của bác sĩ.
```

### Click "▶ Chạy So Sánh" Button

Wait for results to load (first time may take 10-15 seconds if models need to load).

---

## 📊 Explore Results

### Comparison Tab (⚖️)
- **LEFT**: Extractive methods (TextRank, LexRank, LSA)
- **RIGHT**: Abstractive methods (ViT5, BARTPho, mT5)
- Click cards to expand and see details

### Charts Tab (📊)
Select different charts:
- **ROUGE Comparison** - Compare metrics across models
- **Semantic Similarity** - BERTScore results
- **Inference Time** - Speed comparison
- **Compression Ratio** - Text compression efficiency
- **Radar Chart** - Multi-dimensional comparison

### Explanation Tab (📚)
- Overview table of all models
- Detailed explanations for each model
- Mathematical principles
- Advantages/disadvantages
- Use cases and references

### Report Tab (📄)
- **Export HTML** - Full formatted report
- **Export Markdown** - Publication-ready format

---

## 🔌 API Access

### View API Documentation

Open http://localhost:8000/docs in browser

### Main Endpoints

#### 1. Detailed Comparison
```bash
curl -X POST http://localhost:8000/research/compare/detailed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Vietnamese text here...",
    "extractive_sentences": 5,
    "max_abstractive_length": 150
  }'
```

#### 2. Get Model Information
```bash
curl http://localhost:8000/research/models/info
```

#### 3. Get Metrics Explanation
```bash
curl http://localhost:8000/research/metrics/explanation
```

#### 4. Get Benchmark Data
```bash
curl http://localhost:8000/research/benchmark/data
```

---

## 🛠️ Troubleshooting

### Backend Issues

**Error: Port 8000 already in use**
```bash
# Use different port
python -m backend.app.main --port 9000
```

**Error: CUDA out of memory**
- Models will fallback to CPU automatically
- Or reduce `max_abstractive_length` parameter

**Error: Models not loading**
- First run takes time to download/load models
- Check internet connection
- See logs for progress

### Frontend Issues

**Error: Port 5173 already in use**
```bash
# Use different port
npm run dev -- --port 4173
```

**Error: API not found**
- Ensure backend is running on correct port
- Check browser console for network errors
- Verify CORS headers in terminal

### GPU Issues

**Check if GPU is available**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

---

## 📈 Expected Performance

### First Run
- **Backend startup**: 30-45 seconds (loading Transformer models)
- **First comparison**: 10-15 seconds (warmup time)

### Subsequent Runs
- **Comparison time**: 2-3 seconds for demo documents

### Hardware Impact
- **GPU**: ~6-8 seconds per abstractive model
- **CPU Only**: ~30-45 seconds per abstractive model
- **Extractive models**: Always <100ms (no GPU needed)

---

## 🎓 Demo Scenario for Thesis Defense

### Recommended Flow

1. **Load Platform** (30 seconds)
2. **Input Document** (2 minutes)
   - Paste sample Vietnamese text
   - Can add reference summary (optional)
3. **Run Comparison** (10 seconds)
   - Click "▶ Chạy So Sánh"
   - Show loading progress
4. **Present Results** (5 minutes)
   - Show comparison side-by-side
   - Highlight differences in output quality
5. **Show Metrics** (3 minutes)
   - Switch to Charts tab
   - Discuss ROUGE, BERTScore, timing
6. **Explain Models** (5 minutes)
   - Switch to Explanation tab
   - Walk through each model's principle
7. **Export Report** (1 minute)
   - Generate HTML/Markdown report
   - Show what gets exported

**Total Time**: ~15-20 minutes

---

## 📚 Key Points for Presentation

### Extractive Summarization
```
"Extractive methods identify and extract important sentences 
directly from the original document without modification."

Examples: TextRank, LexRank, LSA
Speed: ~50ms average
Quality: ~0.45 ROUGE-1
```

### Abstractive Summarization
```
"Abstractive methods use AI to understand the text and 
generate a paraphrased summary, potentially reorganizing 
and simplifying the content."

Examples: ViT5, BARTPho, mT5
Speed: ~7s average
Quality: ~0.56 ROUGE-1
```

### Trade-offs
```
"Extractive is 100x faster but extractive quality is limited.
Abstractive is slower but achieves higher semantic quality 
and natural language generation."
```

---

## 🔍 Verifying Installation

Run this Python script to verify everything is set up:

```python
# verify_setup.py
import sys
import importlib

print("🔍 Verifying Installation...\n")

# Check Python version
print(f"✓ Python {sys.version.split()[0]}")

# Check critical packages
packages = [
    'fastapi', 'torch', 'transformers', 
    'rouge_score', 'bert_score', 'sentence_transformers'
]

for package in packages:
    try:
        importlib.import_module(package.replace('_', '-'))
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} (missing)")

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ GPU: Not available (CPU mode)")
except:
    print("✗ PyTorch GPU check failed")

print("\n✅ Setup verification complete!")
```

Run it:
```bash
python verify_setup.py
```

---

## 📞 Quick Links

- **Frontend**: http://localhost:5173
- **Backend Docs**: http://localhost:8000/docs
- **Comparison Page**: http://localhost:5173/comparison

---

## ✅ You're Ready!

Everything should be working now. Enjoy exploring the comparison platform! 🎉

For detailed documentation, see [RESEARCH_PLATFORM_README.md](./RESEARCH_PLATFORM_README.md)
