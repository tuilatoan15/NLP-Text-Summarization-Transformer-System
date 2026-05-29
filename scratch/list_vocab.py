import json

p = 'models/vit5-finetuned/tokenizer.json'
with open(p, 'r', encoding='utf-8') as f:
    d = json.load(f)

vocab = d.get("model", {}).get("vocab", {})
print("Vocab type:", type(vocab))
print("Vocab size:", len(vocab))

# Print first 20 tokens in vocabulary
if isinstance(vocab, dict):
    first_20 = list(vocab.items())[:20]
    print("First 20 tokens:")
    for k, v in first_20:
        print(f"  {repr(k)}: {v}")
    
    # Search if 'Hội đồng' is in vocabulary keys
    matches = [k for k in vocab.keys() if 'Hội đồng' in k]
    print(f"Matches for 'Hội đồng': {matches}")
elif isinstance(vocab, list):
    first_20 = vocab[:20]
    print("First 20 tokens:")
    for item in first_20:
        print(f"  {repr(item)}")
    
    # Search if 'Hội đồng' is in vocabulary keys
    matches = [item for item in vocab if isinstance(item, str) and 'Hội đồng' in item]
    print(f"Matches for 'Hội đồng': {matches}")
