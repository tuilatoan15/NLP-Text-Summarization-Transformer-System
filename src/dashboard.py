"""
dashboard.py — Orchestrator cho multi-algorithm summarization và streaming progress.

Chức năng chính:
 - Chạy đồng thời nhiều thuật toán (TextRank, LSA, BART, T5, Pegasus, ...)
 - Tính các chỉ số (ROUGE, BLEU, semantic similarity)
 - Cung cấp hàm trả về kết quả đầy đủ và generator SSE cho tiến trình realtime
 - Cache kết quả theo hash của input + options
"""

import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import logger, ensure_dir, count_words
from src.extractive import extractive_summarize, extractive_summarize_with_details, lexrank_summarize
from src.abstractive import get_summarizer, resolve_model_name
from src.evaluate import compute_rouge, compute_bleu, compute_semantic_similarity, compute_bertscore
from src.preprocess import split_sentences


CACHE_DIR = ensure_dir("cache/dashboard")


def _cache_path(key: str) -> Path:
    return Path(CACHE_DIR) / f"{key}.json"


def _make_cache_key(text: str, algorithms: List[str], options: Dict) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update("|".join(sorted(algorithms)).encode("utf-8"))
    h.update(json.dumps(options, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _cache_set(key: str, data: dict) -> None:
    p = _cache_path(key)
    try:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Không thể lưu cache: {e}")


def _run_textrank(text: str, sentence_count: int = 5) -> dict:
    return extractive_summarize_with_details(text, sentence_count=sentence_count)


def _run_lsa(text: str, sentence_count: int = 5) -> dict:
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, sentence_count)
        sentences = [str(s) for s in summary_sentences]
        summary = " ".join(sentences)
        # Mocking details for LSA for consistency in return format
        return {"summary": summary, "selected_sentences": []}
    except Exception as e:
        logger.warning(f"LSA summarization failed: {e}")
        return _run_textrank(text, sentence_count=sentence_count)


def _run_lexrank(text: str, sentence_count: int = 5) -> dict:
    try:
        from src.extractive import lexrank_summarize
        summary = lexrank_summarize(text, sentence_count=sentence_count)
        return {"summary": summary, "selected_sentences": []} # LexRank with details could be added later
    except Exception as e:
        logger.warning(f"LexRank summarization failed: {e}")
        return _run_textrank(text, sentence_count=sentence_count)


def _run_abstractive(text: str, model_key: str, max_output_length: int = 120) -> str:
    resolved = resolve_model_name(model_key)
    summarizer = get_summarizer(model_name=resolved)
    try:
        return summarizer.summarize(text, max_output_length=max_output_length)
    except Exception as e:
        logger.error(f"Abstractive model {resolved} failed: {e}")
        return ""


def _measure_and_eval(func, name: str, text: str, reference: str, **kwargs) -> dict:
    start = time.time()
    try:
        res = func(text, **kwargs)
        if isinstance(res, dict):
            summary = res.get("summary", "")
            details = res
        else:
            summary = str(res)
            details = {"summary": summary}
    except Exception as e:
        logger.error(f"Algorithm {name} error: {e}")
        summary = ""
        details = {"summary": "", "error": str(e)}
    
    duration = round(time.time() - start, 3)
    length = count_words(summary)

    # Evaluate (use reference if provided, otherwise use original text)
    ref = reference or text
    rouge = compute_rouge(summary, ref)
    bleu = compute_bleu(summary, ref)
    semantic = compute_semantic_similarity(summary, ref)

    return {
        "algorithm": name,
        "summary": summary,
        "details": details,
        "time_seconds": duration,
        "length_words": length,
        "rouge": rouge,
        "bleu": bleu,
        "semantic_similarity": semantic,
        "bertscore": compute_bertscore(summary, ref),
        "source_sentences": split_sentences(text)[:100] # Limit for UI performance
    }


def summarize_all(
    text: str,
    reference: Optional[str] = None,
    algorithms: Optional[List[str]] = None,
    sentence_count: int = 5,
    max_output_length: int = 120,
    use_cache: bool = True,
) -> dict:
    """
    Chạy tuần tự/đồng thời các thuật toán tóm tắt và trả về kết quả tổng hợp.
    """
    if algorithms is None:
        algorithms = ["textrank", "lsa", "lexrank", "vit5", "t5", "bart", "pegasus"]

    key = _make_cache_key(text, algorithms, {"sentence_count": sentence_count, "max_output_length": max_output_length})
    if use_cache:
        cached = _cache_get(key)
        if cached:
            logger.info("Load result từ cache")
            return cached

    tasks = {}
    results = []

    with ThreadPoolExecutor(max_workers=min(6, len(algorithms))) as ex:
        futures = {}
        for alg in algorithms:
            name = alg.lower()
            if name == "textrank":
                futures[ex.submit(_measure_and_eval, _run_textrank, "TextRank", text, reference, sentence_count=sentence_count)] = "TextRank"
            elif name == "lsa":
                futures[ex.submit(_measure_and_eval, _run_lsa, "LSA", text, reference, sentence_count=sentence_count)] = "LSA"
            elif name == "lexrank":
                futures[ex.submit(_measure_and_eval, _run_lexrank, "LexRank", text, reference, sentence_count=sentence_count)] = "LexRank"
            else:
                # treat as abstractive model key
                futures[ex.submit(_measure_and_eval, _run_abstractive, name.upper(), text, reference, model_key=name, max_output_length=max_output_length)] = name

        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                logger.error(f"Task failed: {e}")
                continue
            results.append(res)

    # Compute pairwise similarity between generated summaries (optional)
    summaries = [r["summary"] for r in results if r.get("summary")]
    pairwise_sim = []
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            sim = compute_semantic_similarity(summaries[i], summaries[j])
            pairwise_sim.append({"pair": f"{results[i]['algorithm']}__{results[j]['algorithm']}", "sim": sim})

    final = {
        "algorithms": algorithms,
        "results": results,
        "pairwise_similarity": pairwise_sim,
        "meta": {
            "input_words": count_words(text),
            "reference_provided": bool(reference),
        },
    }

    if use_cache:
        _cache_set(key, final)

    return final


def stream_compare(text: str, reference: Optional[str], algorithms: Optional[List[str]] = None, sentence_count: int = 5, max_output_length: int = 120):
    """
    Generator trả về các event theo định dạng SSE (text/event-stream).
    Mỗi event là JSON có 2 keys: 'event' ('progress'|'done'|'error') và 'data'.
    """
    if algorithms is None:
        algorithms = ["textrank", "lsa", "lexrank", "vit5", "t5", "bart", "pegasus"]

    # streaming: chạy tuần tự nhưng yield progress từng algorithm
    yield f"data: {json.dumps({'event': 'start', 'algorithms': algorithms}, ensure_ascii=False)}\n\n"

    results = []
    for alg in algorithms:
        name = alg.lower()
        yield f"data: {json.dumps({'event': 'running', 'algorithm': name}, ensure_ascii=False)}\n\n"
        try:
            if name == "textrank":
                res = _measure_and_eval(_run_textrank, "TextRank", text, reference, sentence_count=sentence_count)
            elif name == "lsa":
                res = _measure_and_eval(_run_lsa, "LSA", text, reference, sentence_count=sentence_count)
            elif name == "lexrank":
                res = _measure_and_eval(_run_lexrank, "LexRank", text, reference, sentence_count=sentence_count)
            else:
                res = _measure_and_eval(_run_abstractive, name.upper(), text, reference, model_key=name, max_output_length=max_output_length)

            results.append(res)
            yield f"data: {json.dumps({'event': 'done', 'algorithm': name, 'result': res}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Error running {name}: {e}")
            yield f"data: {json.dumps({'event': 'error', 'algorithm': name, 'error': str(e)}, ensure_ascii=False)}\n\n"

    # Final aggregated event
    final = {
        'results': results,
        'meta': {'input_words': count_words(text), 'reference_provided': bool(reference)}
    }
    yield f"data: {json.dumps({'event': 'finished', 'data': final}, ensure_ascii=False)}\n\n"
