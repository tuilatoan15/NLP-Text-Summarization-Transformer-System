from summarizers.abstractive.abstractive_summarizer import (
    abstractive_summarize_key, get_summarizer,
    _generate_chunks_parallel, _build_generation_preset,
    _sanitize_gen_preset, _chunk_text,
)
from src import config

print("MAX_INPUT_TOKENS =", config.MAX_INPUT_TOKENS)
print("MAX_OUTPUT_LENGTH =", config.MAX_OUTPUT_LENGTH)
print("ABSTRACTIVE_CHUNK_WORKERS =", config.ABSTRACTIVE_CHUNK_WORKERS)
print("ABSTRACTIVE_MAX_CHUNKS =", config.ABSTRACTIVE_MAX_CHUNKS)
print()
print("BARTPho config:", config.GENERATION_CONFIGS.get("bartpho"))
print()
print("mT5 config:", config.GENERATION_CONFIGS.get("mt5"))
print()
print("ViT5 config:", config.GENERATION_CONFIGS.get("vit5"))
print()

# Test sanitize function with mT5 (do_sample=True)
preset = dict(config.GENERATION_CONFIGS["mt5"])
sanitized = _sanitize_gen_preset("mt5", preset)
assert "early_stopping" not in sanitized, "early_stopping must be removed for do_sample=True"
assert sanitized.get("num_beams") == 1, "num_beams must be 1 for do_sample=True"
print("_sanitize_gen_preset OK for mT5 (do_sample=True)")

# Test chunk splitting
sample = " ".join(["word"] * 2000)
chunks = _chunk_text(sample, 300)
print(f"_chunk_text(2000 words, max=300) -> {len(chunks)} chunks (max 16)")
assert len(chunks) <= 16

# Test preset build for bartpho
bartpho_preset = _build_generation_preset("bartpho", word_budget=120)
print("BARTPho preset (budget=120):", bartpho_preset)

print()
print("ALL IMPORTS AND CHECKS PASSED OK")
