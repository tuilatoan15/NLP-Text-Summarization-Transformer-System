from src.length_control import (
    allocate_chunk_word_budgets,
    compute_length_targets,
    min_new_tokens_for_budget,
    trim_summary_to_word_budget,
    words_to_max_new_tokens,
)


def test_compute_length_targets_half():
    text = " ".join(["câu"] * 100)
    t = compute_length_targets(text, 50)
    assert t["target_words"] == 50
    assert t["target_length_ratio"] == 50


def test_trim_summary_respects_budget():
    summary = "Một. Hai. Ba. Bốn. Năm."
    out = trim_summary_to_word_budget(summary, 3)
    assert len(out.split()) <= 3


def test_words_to_max_new_tokens_scales_with_budget():
    assert words_to_max_new_tokens(272) >= 300
    assert words_to_max_new_tokens(40) >= 24


def test_chunk_budgets_sum_to_target():
    chunks = [" ".join(["a"] * 200), " ".join(["b"] * 300)]
    budgets = allocate_chunk_word_budgets(chunks, 270)
    assert sum(budgets) == 270
    assert budgets[1] > budgets[0]


def test_min_new_tokens_for_long_budget():
    assert min_new_tokens_for_budget(272) >= 80
