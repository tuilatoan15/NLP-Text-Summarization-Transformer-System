from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

dest = Path('models/vit5-finetuned')
dest.mkdir(parents=True, exist_ok=True)

filenames = [
    'spiece.model',
    'sentencepiece.bpe.model',
    'sentencepiece.model',
    'tokenizer.model',
    'tokenizer.json',
    'vocab.json',
    'vocab.txt',
    'merges.txt',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'tokenizer_config.json',
]

for f in filenames:
    try:
        print('Trying', f)
        p = hf_hub_download(repo_id='VietAI/vit5-base', filename=f)
        print('FOUND', f, '->', p)
        shutil.copy(p, dest / f)
    except Exception as e:
        print('NOT FOUND', f, '->', str(e))

print('Done')
