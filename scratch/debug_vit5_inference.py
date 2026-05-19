import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import unicodedata

def debug():
    # 1. Local Model
    local_path = "models/vit5-finetuned"
    print("\n================ LOCAL MODEL ================")
    try:
        tokenizer_local = AutoTokenizer.from_pretrained(local_path, use_fast=False)
        model_local = AutoModelForSeq2SeqLM.from_pretrained(local_path)
        
        text = "summarize: Ngày 19/5, OpenAI chính thức ra mắt GPT-4o, mô hình ngôn ngữ lớn thế hệ mới nhất của hãng với tốc độ xử lý nhanh gấp đôi và chi phí rẻ hơn một nửa so với GPT-4 Turbo. Điểm nhấn lớn nhất của GPT-4o nằm ở khả năng tương tác tự nhiên thời gian thực bằng cả giọng nói, văn bản và hình ảnh. Giới công nghệ nhận định GPT-4o mở ra kỷ nguyên mới của trợ lý ảo AI thế hệ tiếp theo."
        encoded = tokenizer_local(text, return_tensors="pt")
        
        with torch.no_grad():
            output_ids = model_local.generate(**encoded, max_new_tokens=64, num_beams=2, no_repeat_ngram_size=3)
        decoded = tokenizer_local.decode(output_ids[0], skip_special_tokens=True)
        print("Local Decoded:", decoded)
    except Exception as e:
        print("Local failed:", e)
        
    # 2. Hub Model
    hub_path = "VietAI/vit5-base"
    print("\n================ HUB MODEL ================")
    try:
        tokenizer_hub = AutoTokenizer.from_pretrained(hub_path, use_fast=False)
        model_hub = AutoModelForSeq2SeqLM.from_pretrained(hub_path)
        
        encoded_hub = tokenizer_hub(text, return_tensors="pt")
        with torch.no_grad():
            output_ids_hub = model_hub.generate(**encoded_hub, max_new_tokens=64, num_beams=2, no_repeat_ngram_size=3)
        decoded_hub = tokenizer_hub.decode(output_ids_hub[0], skip_special_tokens=True)
        print("Hub Decoded:", decoded_hub)
    except Exception as e:
        print("Hub failed:", e)

if __name__ == "__main__":
    debug()
