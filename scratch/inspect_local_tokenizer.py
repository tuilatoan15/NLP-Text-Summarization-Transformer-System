from transformers import AutoTokenizer
import sys

sys.stdout.reconfigure(encoding='utf-8')
tokenizer = AutoTokenizer.from_pretrained("models/vit5-finetuned")

ids = [0, 4, 18765, 36087, 18765, 12693, 18765, 15003, 18765, 2, 18765, 18765, 18765, 18765, 18765, 18765, 18765, 18765, 18765, 2309, 18765, 18765, 18765, 18765, 18765, 18765, 18765, 18765, 18765, 1]

print("=== LOCAL TOKENIZER IN models/vit5-finetuned ===")
print("vocab size:", len(tokenizer))
for idx in ids:
    val = tokenizer.decode([idx])
    print(f"ID {idx} -> {repr(val)}")
