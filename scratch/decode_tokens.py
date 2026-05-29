import sys
from transformers import AutoTokenizer

model_path = "models/vit5-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)

ids = [0, 3, 2, 35792, 35998, 2, 35792, 1181, 247, 1625, 260, 889, 2155, 258, 86, 1433, 1789, 440, 180, 1407, 953, 183, 456, 401, 2078, 1580, 110, 292, 480, 35792, 755, 233, 1187, 675, 320, 100, 408, 2078, 1580, 2809, 2381, 2078, 1580, 198, 1553, 1227, 785, 35792, 320, 39, 292, 258, 999, 271, 466, 1407, 843, 649, 35792, 292, 258, 39, 320, 999, 271, 843, 649, 35790, 100, 408, 945, 1044, 399, 1498, 2407, 701, 35792, 320, 35790, 680]

print("=== DECODING TOKENS WITH UTF-8 ===")
sys.stdout.reconfigure(encoding='utf-8')
summary = tokenizer.decode(ids, skip_special_tokens=False)
print("RAW:", summary)
summary_clean = tokenizer.decode(ids, skip_special_tokens=True)
print("CLEAN:", summary_clean)
