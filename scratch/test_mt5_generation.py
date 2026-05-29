import sys
from pathlib import Path

# Thêm project root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_loader import get_loaded_model
from summarizers.abstractive.abstractive_summarizer import abstractive_summarize_key
from src import config
import json

# Đọc text
text = Path("scratch/text_1007.txt").read_text(encoding="utf-8")

# Test 1: Mặc định (sampling)
print("Running default sampling...")
try:
    summary_sample = abstractive_summarize_key("mt5", text)
    print("Default sampling summary:", repr(summary_sample))
except Exception as e:
    print("Default sampling failed:", e)
    summary_sample = str(e)

# Thay đổi cấu hình mt5 sang greedy để kiểm tra
print("\nTemporary modifying config to greedy...")
config.GENERATION_CONFIGS["mt5"] = dict(
    max_new_tokens=120,
    min_new_tokens=15,
    num_beams=1,
    no_repeat_ngram_size=5,
    repetition_penalty=2.0,
    length_penalty=1.0,
    early_stopping=False,
    do_sample=False,  # greedy!
)

try:
    summary_greedy = abstractive_summarize_key("mt5", text)
    print("Greedy summary:", repr(summary_greedy))
except Exception as e:
    print("Greedy failed:", e)
    summary_greedy = str(e)

# Test 3: Beam search
print("\nTemporary modifying config to beam search...")
config.GENERATION_CONFIGS["mt5"] = dict(
    max_new_tokens=120,
    min_new_tokens=15,
    num_beams=4,      # beam search
    no_repeat_ngram_size=5,
    repetition_penalty=2.0,
    length_penalty=1.0,
    early_stopping=True,
    do_sample=False,
)

try:
    summary_beam = abstractive_summarize_key("mt5", text)
    print("Beam search summary:", repr(summary_beam))
except Exception as e:
    print("Beam search failed:", e)
    summary_beam = str(e)

# Lưu kết quả
out = {
    "sampling": summary_sample,
    "greedy": summary_greedy,
    "beam": summary_beam,
}
Path("scratch/mt5_debug_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nSaved scratch/mt5_debug_results.json")
