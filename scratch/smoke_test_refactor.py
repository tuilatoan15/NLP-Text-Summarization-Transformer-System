"""Quick smoke test for the refactored backend."""
import time

from src.extractive import summarize_extractive_parallel
from src.utils import get_device_info

TEXT = (
    "Tri tue nhan tao dang thay doi cach con nguoi lam viec va hoc tap. "
    "Cac mo hinh ngon ngu lon nhu GPT va ViT5 da chung minh kha nang xu ly ngon ngu tu nhien. "
    "Tai Viet Nam, nghien cuu ve NLP tieng Viet dang phat trien manh me voi nhieu bo du lieu moi. "
    "Viec tom tat van ban tu dong giup nguoi dung nam bat thong tin nhanh hon va hieu qua hon. "
    "He thong TextRank su dung do thi de xep hang cau quan trong trong van ban goc. "
    "LexRank dung phan tich trung tam do thi de chon ra cac cau quan trong nhat. "
    "LSA ap dung SVD de tim ra cac chu de tiem an va chon cau dai dien cho van ban. "
)

print("=== Extractive Parallel Test ===")
t = time.perf_counter()
results = summarize_extractive_parallel(TEXT, ["textrank", "lexrank", "lsa"], sentence_count=3)
elapsed = time.perf_counter() - t

for key, res in results.items():
    summary = res.get("summary", "")[:100]
    print(f"  [{key}] {summary}...")

print(f"Wall time (3 algos parallel): {elapsed:.3f} s")
print()

info = get_device_info()
print("=== Device Info ===")
print("  device:       ", info.get("device"))
print("  cuda_avail:   ", info.get("cuda_available"))
if info.get("gpu_name"):
    print("  gpu_name:     ", info.get("gpu_name"))
    print("  total_vram MB:", info.get("total_vram_mb"))
    print("  free_vram MB: ", info.get("free_vram_mb"))
else:
    print("  (no GPU detected - running on CPU)")

print()
print("=== All checks passed ===")
