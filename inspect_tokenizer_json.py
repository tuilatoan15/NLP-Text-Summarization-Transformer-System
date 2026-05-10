import json
p='models/vit5-finetuned/tokenizer.json'
with open(p,'r',encoding='utf-8') as f:
    d=json.load(f)
print('root keys:', list(d.keys()))
model = d.get('model')
print('model type:', type(model))
if model:
    print('model keys:', list(model.keys()))
    vocab = model.get('vocab')
    print('vocab type:', type(vocab))
    if isinstance(vocab, dict):
        print('vocab len:', len(vocab))
    else:
        try:
            print('vocab sample:', list(vocab)[:5])
        except Exception as e:
            print('vocab sample error', e)
