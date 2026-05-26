"""Model and algorithm registry used by API, CLI scripts, and frontend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src import config


AlgorithmGroup = Literal["extractive", "abstractive"]


@dataclass(frozen=True)
class AlgorithmConfig:
    key: str
    name: str
    group: AlgorithmGroup
    description: str
    model_name: str | None = None
    local_dir: str | None = None
    requires_finetuning: bool = False
    recommended_for_vietnamese: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


EXTRACTIVE_ALGORITHMS: dict[str, AlgorithmConfig] = {
    "textrank": AlgorithmConfig(
        key="textrank",
        name="TextRank",
        group="extractive",
        description="Graph-based sentence ranking; strong baseline for news summarization.",
        recommended_for_vietnamese=True,
    ),
    "lexrank": AlgorithmConfig(
        key="lexrank",
        name="LexRank",
        group="extractive",
        description="Centroid/graph centrality with thresholded sentence similarity.",
    ),
    "lsa": AlgorithmConfig(
        key="lsa",
        name="LSA Summarizer",
        group="extractive",
        description="Latent semantic analysis over the term-sentence matrix.",
    ),
    "tfidf": AlgorithmConfig(
        key="tfidf",
        name="TF-IDF Ranking",
        group="extractive",
        description="Sentence ranking by aggregate TF-IDF term weights; strong lexical baseline.",
        recommended_for_vietnamese=True,
    ),
}


ABSTRACTIVE_ALGORITHMS: dict[str, AlgorithmConfig] = {
    "vit5": AlgorithmConfig(
        key="vit5",
        name="ViT5",
        group="abstractive",
        model_name="VietAI/vit5-base",
        local_dir=str(config.LOCAL_VIT5_DIR),
        description="Vietnamese T5 model; best default when fine-tuned on VNExpress.",
        requires_finetuning=True,
        recommended_for_vietnamese=True,
    ),
    "mt5": AlgorithmConfig(
        key="mt5",
        name="mT5",
        group="abstractive",
        model_name="google/mt5-small",
        local_dir=str(config.MODEL_DIR / "mt5-finetuned"),
        description="Multilingual T5 baseline for cross-lingual comparison.",
        requires_finetuning=True,
    ),
    "bartpho": AlgorithmConfig(
        key="bartpho",
        name="BARTPho",
        group="abstractive",
        model_name="vinai/bartpho-syllable",
        local_dir=str(config.MODEL_DIR / "bartpho-finetuned"),
        description="Vietnamese BART-style seq2seq model from VinAI.",
        requires_finetuning=True,
        recommended_for_vietnamese=True,
    ),
}


ALGORITHMS: dict[str, AlgorithmConfig] = {
    **EXTRACTIVE_ALGORITHMS,
    **ABSTRACTIVE_ALGORITHMS,
}

DEFAULT_ALGORITHMS = ["textrank", "lexrank", "lsa", "tfidf", "vit5", "mt5", "bartpho"]


def normalize_algorithm_key(key: str) -> str:
    return (key or "").strip().lower().replace("_", "-").replace(" ", "-")


def resolve_algorithm(key: str) -> AlgorithmConfig:
    normalized = normalize_algorithm_key(key)
    aliases = {
        "text-rank": "textrank",
        "lex-rank": "lexrank",
        "lsa-summarizer": "lsa",
        "tf-idf": "tfidf",
        "tf_idf": "tfidf",
        "bart": "bartpho",
        "phobart": "bartpho",
        "pho-bart": "bartpho",
        "vit5-base": "vit5",
        "mt5-small": "mt5",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ALGORITHMS:
        raise KeyError(f"Unsupported algorithm: {key}")
    return ALGORITHMS[normalized]


def list_algorithms() -> list[dict]:
    return [ALGORITHMS[key].to_dict() for key in DEFAULT_ALGORITHMS]


def resolve_model_path(algorithm: AlgorithmConfig, prefer_local: bool = True) -> str:
    if algorithm.group != "abstractive":
        raise ValueError(f"{algorithm.key} is not an abstractive model")
    local_dir = Path(algorithm.local_dir or "")
    if prefer_local and local_dir.exists() and any(local_dir.iterdir()):
        return str(local_dir)
    if not algorithm.model_name:
        raise ValueError(f"No HuggingFace model configured for {algorithm.key}")
    return algorithm.model_name
