"""
GPU vs CPU Benchmark for Vietnamese NLP summarization.
Measures inference time for a typical paragraph across:
  - CPU (baseline)
  - GPU fp32
  - GPU fp16 (AMP autocast)
"""
import sys, time, gc
sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH = "models/vit5-finetuned"
TEXT = (
    "summarize: Trí tuệ nhân tạo đang thay đổi cách con người làm việc và học tập. "
    "Các mô hình ngôn ngữ lớn đã chứng minh khả năng xử lý ngôn ngữ tự nhiên ở mức độ "
    "gần với con người. Tại Việt Nam, nghiên cứu về NLP tiếng Việt đang phát triển mạnh "
    "mẽ với nhiều bộ dữ liệu và mô hình mới ra đời. Việc tóm tắt văn bản tự động giúp "
    "người dùng nắm bắt thông tin nhanh hơn và hiệu quả hơn trong thời đại thông tin bùng nổ."
)
GEN_KWARGS = dict(max_new_tokens=80, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)
WARMUP_RUNS = 1
BENCH_RUNS  = 3

print("=" * 60)
print("  Vietnamese NLP — GPU vs CPU Inference Benchmark")
print("=" * 60)
print(f"torch  : {torch.__version__}")
print(f"CUDA   : {torch.version.cuda}")

if not torch.cuda.is_available():
    print("\n❌ No CUDA GPU detected. Install torch+cu124 first.\n")
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
print(f"GPU    : {props.name}")
print(f"VRAM   : {props.total_memory // 1024**2} MB")
print(f"CC     : {props.major}.{props.minor}")
print()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
encoded_cpu = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=512)

results = []

# ── 1. CPU fp32 ─────────────────────────────────────────────────────────────
print("Loading model → CPU (fp32)...")
model_cpu = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model_cpu.eval()
inputs_cpu = {k: v.to("cpu") for k, v in encoded_cpu.items()}

for _ in range(WARMUP_RUNS):
    with torch.inference_mode():
        _ = model_cpu.generate(**inputs_cpu, **GEN_KWARGS)

times = []
for _ in range(BENCH_RUNS):
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model_cpu.generate(**inputs_cpu, **GEN_KWARGS)
    times.append(time.perf_counter() - t0)
summary_cpu = tokenizer.decode(out[0], skip_special_tokens=True)
avg_cpu = sum(times) / len(times)
results.append(("CPU fp32", avg_cpu, summary_cpu))
print(f"  CPU fp32  avg: {avg_cpu:.3f} s")

del model_cpu; gc.collect()

# ── 2. GPU fp32 ─────────────────────────────────────────────────────────────
print("Loading model → CUDA:0 (fp32)...")
model_gpu = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to("cuda:0")
model_gpu.eval()
inputs_gpu = {k: v.to("cuda:0") for k, v in encoded_cpu.items()}
torch.cuda.synchronize()

for _ in range(WARMUP_RUNS):
    with torch.inference_mode():
        _ = model_gpu.generate(**inputs_gpu, **GEN_KWARGS)
torch.cuda.synchronize()

times = []
for _ in range(BENCH_RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model_gpu.generate(**inputs_gpu, **GEN_KWARGS)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
summary_gpu_fp32 = tokenizer.decode(out[0], skip_special_tokens=True)
avg_fp32 = sum(times) / len(times)
results.append(("GPU fp32", avg_fp32, summary_gpu_fp32))
print(f"  GPU fp32  avg: {avg_fp32:.3f} s")

# ── 3. GPU fp16 (autocast) ──────────────────────────────────────────────────
print("Benchmarking GPU fp16 (autocast)...")
times = []
for _ in range(BENCH_RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model_gpu.generate(**inputs_gpu, **GEN_KWARGS)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
summary_gpu_fp16 = tokenizer.decode(out[0], skip_special_tokens=True)
avg_fp16 = sum(times) / len(times)
results.append(("GPU fp16", avg_fp16, summary_gpu_fp16))
print(f"  GPU fp16  avg: {avg_fp16:.3f} s")

del model_gpu; torch.cuda.empty_cache(); gc.collect()

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  BENCHMARK RESULTS")
print("=" * 60)
print(f"{'Mode':<12}  {'Avg Time':>10}  {'vs CPU':>10}")
print("-" * 36)
for name, t, _ in results:
    vs_cpu = f"{avg_cpu/t:.1f}x faster" if t < avg_cpu else "baseline"
    print(f"{name:<12}  {t:>8.3f} s  {vs_cpu:>14}")
print()
print("Summaries generated:")
for name, _, s in results:
    print(f"  [{name}]: {s[:100]}...")

vram_used = torch.cuda.max_memory_allocated(0) // 1024**2
print(f"\nPeak VRAM used during bench: {vram_used} MB")
print("=" * 60)
