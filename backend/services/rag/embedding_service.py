from __future__ import annotations

from pipeline.schema import EmbeddingConfig
from embeddings.embedder import SentenceTransformerEmbedder, EmbeddingModelRegistry


class EmbeddingService:
    def list_models(self) -> dict:
        return EmbeddingModelRegistry.list_models()

    def embed_documents(self, texts: list[str], model_name: str) -> list[list[float]]:
        config = EmbeddingConfig(model_name=model_name)
        result = SentenceTransformerEmbedder(config).embed_documents(texts)
        return result.embeddings.tolist()

    def embed_query(self, text: str, model_name: str) -> list[float]:
        config = EmbeddingConfig(model_name=model_name)
        vector = SentenceTransformerEmbedder(config).embed_query(text)
        return vector.tolist()

