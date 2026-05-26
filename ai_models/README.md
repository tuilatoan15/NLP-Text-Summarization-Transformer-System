# AI Models Directory

Place fine-tuned checkpoints here:

- `vit5-finetuned/` — Vietnamese ViT5 seq2seq
- `bartpho-finetuned/` — BARTPho syllable model
- `mt5-finetuned/` — mT5 baseline

Download base weights via Hugging Face on first run, or set `LOCAL_VIT5_DIR` in `.env`.

Benchmark embeddings:

```bash
python -m embeddings.benchmark --models hash,BAAI/bge-m3
```
