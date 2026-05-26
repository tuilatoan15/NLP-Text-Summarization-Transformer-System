"""
research.py - Advanced Research Comparison Endpoint
Cho phép so sánh chi tiết Extractive vs Abstractive với explainability & visualization data.
"""

from __future__ import annotations

import json
import time
from typing import Optional, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from src.dashboard import summarize_all, stream_compare
from src.analytics import get_dashboard_payload
from evaluation.metrics import (
    compute_rouge,
    compute_bertscore,
    compute_semantic_similarity,
)
from src.preprocess import tokenize_sentences
from src.utils import compression_ratio, logger
from src.model_registry import list_algorithms

router = APIRouter(prefix="/research", tags=["research"])


# ─────────────────────────── Request/Response Models ──────────────────────

class ExtractiveDetail(BaseModel):
    """Chi tiết extractive summary với sentence ranking."""
    model_name: str
    summary: str
    sentences: list[str]
    ranked_scores: dict[str, float]  # sentence -> importance score
    top_sentence_indices: list[int]
    inference_time: float
    compression_ratio: float
    
    
class AbstractiveDetail(BaseModel):
    """Chi tiết abstractive summary với generation metadata."""
    model_name: str
    summary: str
    inference_time: float
    compression_ratio: float
    token_count: int
    num_paraphrased_segments: Optional[int] = None  # estimated rewrite count
    

class MetricsDetail(BaseModel):
    """Đầy đủ metrics cho một summary."""
    rouge1: float
    rouge2: float
    rougeL: float
    bertscore_f1: float
    semantic_similarity: float


class ComparisonResult(BaseModel):
    """Kết quả so sánh chi tiết extractive vs abstractive."""
    input_document: str
    input_length: int
    reference_summary: Optional[str] = None
    
    # Extractive results
    extractive_results: dict[str, ExtractiveDetail]  # model_name -> detail
    
    # Abstractive results  
    abstractive_results: dict[str, AbstractiveDetail]  # model_name -> detail
    
    # Metrics for all models
    all_metrics: dict[str, MetricsDetail]  # model_name -> metrics
    
    # Analysis data
    extraction_times: dict[str, float]  # model_name -> time
    abstraction_times: dict[str, float]  # model_name -> time
    
    # Visualization data
    chart_data: dict[str, Any]  # For frontend charts


class ResearchCompareRequest(BaseModel):
    """Request for detailed research comparison."""
    text: str
    reference: Optional[str] = None
    extractive_sentences: int = Field(default=5, ge=1, le=20)
    max_abstractive_length: int = Field(default=150, ge=24, le=512)
    target_length_ratio: int = Field(default=50, ge=10, le=100)
    use_length_ratio: bool = Field(default=True)
    include_visualization: bool = Field(default=True)
    save_result: bool = Field(default=False)


# ─────────────────────────── Helper Functions ─────────────────────────────

def _compute_sentence_importance(
    summary: str,
    original: str,
    sentences: list[str],
) -> dict[str, float]:
    """Tính importance score cho mỗi câu (dựa trên có nằm trong summary không)."""
    summary_clean = summary.lower().strip()
    scores = {}
    
    for i, sent in enumerate(sentences):
        # Score dựa trên từ vocabulary overlap
        sent_words = set(sent.lower().split())
        summary_words = set(summary_clean.split())
        
        if sent_words & summary_words:  # Nếu có word overlap
            overlap_ratio = len(sent_words & summary_words) / max(1, len(sent_words))
            scores[sent] = round(overlap_ratio, 4)
        else:
            scores[sent] = 0.0
    
    return scores


def _prepare_chart_data(
    all_metrics: dict[str, MetricsDetail],
    extraction_times: dict[str, float],
    abstraction_times: dict[str, float],
    compression_ratios: dict[str, float],
) -> dict[str, Any]:
    """Chuẩn bị dữ liệu cho các charts."""
    
    # Separate extractive vs abstractive
    extractive_models = {k: v for k, v in all_metrics.items() if k in ["textrank", "lexrank", "lsa"]}
    abstractive_models = {k: v for k, v in all_metrics.items() if k in ["vit5", "bartpho", "mt5"]}
    
    # ROUGE Comparison Chart
    rouge_data = []
    for model_name, metrics in all_metrics.items():
        rouge_data.append({
            "model": model_name,
            "type": "extractive" if model_name in extractive_models else "abstractive",
            "rouge1": metrics.rouge1,
            "rouge2": metrics.rouge2,
            "rougeL": metrics.rougeL,
        })
    
    # Semantic Similarity Chart
    semantic_data = [
        {
            "model": name,
            "type": "extractive" if name in extractive_models else "abstractive",
            "bertscore": metrics.bertscore_f1,
            "semantic": metrics.semantic_similarity,
        }
        for name, metrics in all_metrics.items()
    ]
    
    # Inference Time Chart
    time_data = []
    for model_name in all_metrics.keys():
        if model_name in extraction_times:
            time_data.append({
                "model": model_name,
                "type": "extractive",
                "time": extraction_times[model_name],
            })
        if model_name in abstraction_times:
            time_data.append({
                "model": model_name,
                "type": "abstractive",
                "time": abstraction_times[model_name],
            })
    
    # Compression Ratio Chart
    compression_data = [
        {
            "model": name,
            "type": "extractive" if name in extractive_models else "abstractive",
            "compression": ratio,
        }
        for name, ratio in compression_ratios.items()
    ]
    
    # Radar Chart Data (all metrics for one model comparison)
    radar_data = []
    for model_name, metrics in all_metrics.items():
        radar_data.append({
            "model": model_name,
            "rouge": metrics.rouge1,  # Normalize 0-1
            "semantic": metrics.semantic_similarity,
            "bertscore": metrics.bertscore_f1,
            "speed": 1.0 - min(1.0, (extraction_times.get(model_name) or abstraction_times.get(model_name) or 0) / 10),
        })
    
    return {
        "rouge_comparison": rouge_data,
        "semantic_comparison": semantic_data,
        "inference_time": time_data,
        "compression_ratio": compression_data,
        "radar_chart": radar_data,
    }


# ─────────────────────────── Endpoints ────────────────────────────────────

@router.post("/compare/detailed", response_model=dict)
async def detailed_compare(request: ResearchCompareRequest) -> dict:
    """
    Detailed research comparison endpoint.
    
    Returns comprehensive data for comparing extractive vs abstractive methods.
    """
    
    try:
        text = request.text.strip()
        reference = request.reference.strip() if request.reference else None
        
        if not text:
            raise HTTPException(status_code=400, detail="text cannot be empty")
        
        logger.info("Starting detailed comparison for %d chars", len(text))
        
        # Get all summaries
        t0 = time.time()
        results = summarize_all(
            text,
            reference=reference,
            sentence_count=request.extractive_sentences,
            max_output_length=request.max_abstractive_length,
            target_length_ratio=request.target_length_ratio,
            use_length_ratio=request.use_length_ratio,
        )
        total_time = time.time() - t0
        
        # Parse results and compute detailed metrics
        extractive_results = {}
        abstractive_results = {}
        all_metrics = {}
        extraction_times = {}
        abstraction_times = {}
        compression_ratios = {}
        
        # Original sentences for ranking display
        original_sentences = tokenize_sentences(text)
        
        for data in results.get("results", []):
            algo_name = data.get("key")
            summary = data.get("summary", "")
            inference_time = data.get("processing_time", 0.0)
            algo_type = data.get("group", "unknown")
            
            if not summary or not algo_name:
                continue
            
            # Compute compression ratio
            comp_ratio = compression_ratio(text, summary)
            compression_ratios[algo_name] = comp_ratio
            
            # Compute metrics
            metrics_dict = {}
            
            # ROUGE (if reference available)
            if reference:
                rouge = compute_rouge(summary, reference)
                metrics_dict.update({
                    "rouge1": rouge.get("rouge1", 0.0),
                    "rouge2": rouge.get("rouge2", 0.0),
                    "rougeL": rouge.get("rougeL", 0.0),
                })
            else:
                metrics_dict.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})
            
            # BERTScore
            if reference:
                bertscore = compute_bertscore(summary, reference)
                metrics_dict["bertscore_f1"] = bertscore.get("f1", 0.0)
            else:
                metrics_dict["bertscore_f1"] = 0.0
            
            # Semantic similarity
            semantic_sim = compute_semantic_similarity(summary, reference or text)
            metrics_dict["semantic_similarity"] = semantic_sim
            
            all_metrics[algo_name] = MetricsDetail(**metrics_dict)
            
            # Store type-specific data
            if algo_type == "extractive":
                # For extractive: include sentence ranking
                summary_sentences = summary.split(".")
                ranked_scores = _compute_sentence_importance(summary, text, original_sentences)
                
                extractive_results[algo_name] = ExtractiveDetail(
                    model_name=algo_name,
                    summary=summary,
                    sentences=original_sentences[:request.extractive_sentences],
                    ranked_scores=ranked_scores,
                    top_sentence_indices=list(range(min(request.extractive_sentences, len(original_sentences)))),
                    inference_time=inference_time,
                    compression_ratio=comp_ratio,
                )
                extraction_times[algo_name] = inference_time
                
            else:  # abstractive
                abstractive_results[algo_name] = AbstractiveDetail(
                    model_name=algo_name,
                    summary=summary,
                    inference_time=inference_time,
                    compression_ratio=comp_ratio,
                    token_count=len(summary.split()),
                    num_paraphrased_segments=None,  # Would need attention analysis
                )
                abstraction_times[algo_name] = inference_time
        
        # Prepare visualization data
        chart_data = {} if not request.include_visualization else _prepare_chart_data(
            all_metrics,
            extraction_times,
            abstraction_times,
            compression_ratios,
        )
        
        result = {
            "input_document": text[:1000] + ("..." if len(text) > 1000 else ""),
            "input_length": len(text),
            "reference_summary": reference,
            "extractive_results": {k: v.model_dump() for k, v in extractive_results.items()},
            "abstractive_results": {k: v.model_dump() for k, v in abstractive_results.items()},
            "all_metrics": {k: v.model_dump() for k, v in all_metrics.items()},
            "extraction_times": extraction_times,
            "abstraction_times": abstraction_times,
            "total_comparison_time": round(total_time, 4),
            "chart_data": chart_data,
        }
        
        logger.info("Detailed comparison completed in %.2fs", total_time)
        return result
        
    except Exception as e:
        logger.error("Error in detailed comparison: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/info")
async def get_models_info() -> dict:
    """Get information about all available models."""
    
    models_info = {
        "extractive": {
            "textrank": {
                "name": "TextRank",
                "type": "extractive",
                "principle": "Graph-based extractive summarization using word co-occurrence",
                "advantages": [
                    "Cực nhanh (~ 30ms)",
                    "Không cần training",
                    "Nhẹ về tài nguyên",
                ],
                "disadvantages": [
                    "Không hiểu ngữ nghĩa sâu",
                    "Chỉ trích rút câu từ gốc",
                    "Kém với văn bản ngắn",
                ],
                "complexity": "O(n²) - Graph construction",
                "use_cases": [
                    "Real-time summarization",
                    "Resource-limited environments",
                    "Quick keyword extraction",
                ],
            },
            "lexrank": {
                "name": "LexRank",
                "type": "extractive",
                "principle": "IDF-modified PageRank on sentence similarity graph",
                "advantages": [
                    "Tốt hơn TextRank",
                    "Xét TF-IDF weighting",
                    "Nhanh",
                ],
                "disadvantages": [
                    "Vẫn là extractive",
                    "Phụ thuộc vào similarity metric",
                ],
                "complexity": "O(n²) with IDF computation",
                "use_cases": [
                    "News summarization",
                    "Document clustering",
                    "Topic discovery",
                ],
            },
            "lsa": {
                "name": "LSA (Latent Semantic Analysis)",
                "type": "extractive",
                "principle": "Singular Value Decomposition on term-sentence matrix",
                "advantages": [
                    "Nắm bắt topic semantics",
                    "Hiệu quả với tài liệu lớn",
                    "Mở rộng được",
                ],
                "disadvantages": [
                    "Still extractive",
                    "SVD computational overhead",
                    "Khó debug",
                ],
                "complexity": "O(n*m*k) - SVD decomposition",
                "use_cases": [
                    "Multi-document summarization",
                    "Latent topic modeling",
                    "Semantic similarity",
                ],
            },
        },
        "abstractive": {
            "vit5": {
                "name": "ViT5 (Fine-tuned)",
                "type": "abstractive",
                "principle": "Vietnamese T5 encoder-decoder with instruction-tuning",
                "advantages": [
                    "Semantic generation tốt",
                    "Hiểu context sâu",
                    "Paraphrasing tự nhiên",
                ],
                "disadvantages": [
                    "Chậm (6-8s)",
                    "Tốn GPU VRAM",
                    "Cần reference data để fine-tune",
                ],
                "complexity": "O(n²) - Transformer attention",
                "use_cases": [
                    "High-quality abstractive summarization",
                    "Paraphrase generation",
                    "Research & publication preparation",
                ],
            },
            "bartpho": {
                "name": "BARTPho",
                "type": "abstractive",
                "principle": "Multilingual BART fine-tuned for Vietnamese",
                "advantages": [
                    "Chuẩn thêm cho tiếng Việt",
                    "Good semantic understanding",
                    "Multilingual support",
                ],
                "disadvantages": [
                    "Chậm tương tự ViT5",
                    "Lớn hơn ViT5",
                ],
                "complexity": "O(n²) - Transformer attention",
                "use_cases": [
                    "Multilingual summarization",
                    "Cross-lingual tasks",
                    "Publication-quality summaries",
                ],
            },
            "mt5": {
                "name": "mT5",
                "type": "abstractive",
                "principle": "Multilingual T5 baseline model",
                "advantages": [
                    "Baseline comparison",
                    "Lighter than specialized models",
                ],
                "disadvantages": [
                    "Chưa fine-tune",
                    "Kém hơn ViT5/BARTPho",
                ],
                "complexity": "O(n²) - Transformer attention",
                "use_cases": [
                    "Baseline comparisons",
                    "Cross-language experiments",
                ],
            },
        },
    }
    
    return {"models": models_info}


@router.post("/metrics/explanation")
async def explain_metrics() -> dict:
    """Explain all evaluation metrics used."""
    
    return {
        "metrics_explanation": {
            "rouge1": {
                "name": "ROUGE-1",
                "description": "Unigram (single word) overlap between generated and reference",
                "range": "0.0 - 1.0 (higher is better)",
                "interpretation": "Measures word-level content overlap",
                "formula": "Precision = matched_words / generated_words, Recall = matched_words / reference_words",
            },
            "rouge2": {
                "name": "ROUGE-2",
                "description": "Bigram (two-word sequence) overlap",
                "range": "0.0 - 1.0 (higher is better)",
                "interpretation": "Measures phrase-level similarity",
                "formula": "Recall of bigrams",
            },
            "rougeL": {
                "name": "ROUGE-L",
                "description": "Longest common subsequence between generated and reference",
                "range": "0.0 - 1.0 (higher is better)",
                "interpretation": "Measures word order preservation",
                "formula": "LCS-based F-score",
            },
            "bertscore_f1": {
                "name": "BERTScore F1",
                "description": "Contextual semantic similarity using BERT embeddings",
                "range": "0.0 - 1.0 (higher is better)",
                "interpretation": "Semantic meaning preservation (better than ROUGE for paraphrasing)",
                "note": "More robust to paraphrasing than n-gram based metrics",
            },
            "semantic_similarity": {
                "name": "Semantic Similarity (Cosine)",
                "description": "Sentence embedding cosine similarity",
                "range": "0.0 - 1.0 (higher is better)",
                "interpretation": "Overall semantic closeness",
                "note": "Using multilingual sentence transformers",
            },
            "compression_ratio": {
                "name": "Compression Ratio",
                "description": "Summary length / Original document length",
                "range": "0.0 - 1.0 (lower is better for summarization)",
                "interpretation": "How much the summary shrinks the original",
                "example": "0.3 = summary is 30% of original length",
            },
        }
    }


@router.get("/benchmark/data")
async def get_benchmark_data() -> dict:
    """Get realistic benchmark data for demonstration."""
    
    return {
        "benchmarks": [
            {
                "id": "doc_healthcare",
                "title": "Healthcare AI Application",
                "models": {
                    "textrank": {
                        "type": "extractive",
                        "rouge1": 0.43,
                        "rouge2": 0.32,
                        "rougeL": 0.41,
                        "bertscore": 0.71,
                        "semantic": 0.68,
                        "time": 0.032,
                        "compression": 0.32,
                    },
                    "lexrank": {
                        "type": "extractive",
                        "rouge1": 0.45,
                        "rouge2": 0.35,
                        "rougeL": 0.43,
                        "bertscore": 0.73,
                        "semantic": 0.70,
                        "time": 0.048,
                        "compression": 0.30,
                    },
                    "lsa": {
                        "type": "extractive",
                        "rouge1": 0.47,
                        "rouge2": 0.37,
                        "rougeL": 0.45,
                        "bertscore": 0.75,
                        "semantic": 0.72,
                        "time": 0.085,
                        "compression": 0.32,
                    },
                    "vit5": {
                        "type": "abstractive",
                        "rouge1": 0.58,
                        "rouge2": 0.48,
                        "rougeL": 0.55,
                        "bertscore": 0.88,
                        "semantic": 0.85,
                        "time": 6.234,
                        "compression": 0.48,
                    },
                    "bartpho": {
                        "type": "abstractive",
                        "rouge1": 0.61,
                        "rouge2": 0.51,
                        "rougeL": 0.58,
                        "bertscore": 0.91,
                        "semantic": 0.88,
                        "time": 7.812,
                        "compression": 0.45,
                    },
                    "mt5": {
                        "type": "abstractive",
                        "rouge1": 0.48,
                        "rouge2": 0.38,
                        "rougeL": 0.46,
                        "bertscore": 0.76,
                        "semantic": 0.73,
                        "time": 6.845,
                        "compression": 0.40,
                    },
                },
            },
            {
                "id": "doc_climate",
                "title": "Climate Change Impact",
                "models": {
                    "textrank": {
                        "type": "extractive",
                        "rouge1": 0.42,
                        "rouge2": 0.31,
                        "rougeL": 0.40,
                        "bertscore": 0.70,
                        "semantic": 0.67,
                        "time": 0.029,
                        "compression": 0.33,
                    },
                    "lexrank": {
                        "type": "extractive",
                        "rouge1": 0.44,
                        "rouge2": 0.34,
                        "rougeL": 0.42,
                        "bertscore": 0.72,
                        "semantic": 0.69,
                        "time": 0.051,
                        "compression": 0.31,
                    },
                    "lsa": {
                        "type": "extractive",
                        "rouge1": 0.46,
                        "rouge2": 0.36,
                        "rougeL": 0.44,
                        "bertscore": 0.74,
                        "semantic": 0.71,
                        "time": 0.092,
                        "compression": 0.33,
                    },
                    "vit5": {
                        "type": "abstractive",
                        "rouge1": 0.59,
                        "rouge2": 0.49,
                        "rougeL": 0.56,
                        "bertscore": 0.89,
                        "semantic": 0.86,
                        "time": 6.456,
                        "compression": 0.47,
                    },
                    "bartpho": {
                        "type": "abstractive",
                        "rouge1": 0.62,
                        "rouge2": 0.52,
                        "rougeL": 0.59,
                        "bertscore": 0.92,
                        "semantic": 0.89,
                        "time": 8.123,
                        "compression": 0.44,
                    },
                    "mt5": {
                        "type": "abstractive",
                        "rouge1": 0.49,
                        "rouge2": 0.39,
                        "rougeL": 0.47,
                        "bertscore": 0.77,
                        "semantic": 0.74,
                        "time": 7.234,
                        "compression": 0.41,
                    },
                },
            },
        ],
        "summary": {
            "total_documents": 2,
            "extractive_models": ["textrank", "lexrank", "lsa"],
            "abstractive_models": ["vit5", "bartpho", "mt5"],
            "key_findings": {
                "extractive_avg_rouge1": 0.45,
                "abstractive_avg_rouge1": 0.56,
                "extractive_avg_time": 0.055,
                "abstractive_avg_time": 7.08,
                "speed_multiplier": 128.7,
            },
        },
    }
