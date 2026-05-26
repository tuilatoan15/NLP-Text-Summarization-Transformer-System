"""2D embedding projections and similarity heatmaps."""

from __future__ import annotations

from typing import Any

import numpy as np


def embedding_map_2d(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]] | None,
    *,
    method: str = "pca",
) -> list[dict[str, Any]]:
    if not embeddings:
        return []
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return []

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    coords: np.ndarray
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
            coords = reducer.fit_transform(centered)
        except Exception:
            coords = _pca_2d(centered)
    else:
        coords = _pca_2d(centered)

    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    max_abs = np.max(np.abs(coords)) or 1.0
    coords = coords / max_abs

    return [
        {
            "chunk_id": chunks[idx].get("chunk_id") if idx < len(chunks) else str(idx),
            "x": round(float(coords[idx, 0]), 4),
            "y": round(float(coords[idx, 1]), 4),
            "section": " / ".join((chunks[idx].get("section_path") if idx < len(chunks) else []) or []),
        }
        for idx in range(coords.shape[0])
    ]


def similarity_heatmap(embeddings: list[list[float]] | None, max_items: int = 24) -> list[list[float]]:
    if not embeddings:
        return []
    vectors = np.asarray(embeddings[:max_items], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return []
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sims = (vectors / norms) @ (vectors / norms).T
    return np.round(sims, 4).tolist()


def _pca_2d(centered: np.ndarray) -> np.ndarray:
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
    except Exception:
        coords = centered[:, :2] if centered.shape[1] >= 2 else np.pad(centered, ((0, 0), (0, 1)))
    if coords.ndim != 2:
        coords = coords.reshape(-1, 1)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return coords
