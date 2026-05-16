"""
model_loader.py — Singleton preloader for ALL models used by the system.

This module solves the #1 performance bottleneck: loading heavy Transformer
models (ViT5, mT5, BARTPho) on EVERY request.  Instead, all models are loaded
ONCE at server startup and kept in memory for the lifetime of the process.

Design
──────
• ModelRegistry: one global instance that holds every loaded model/tokenizer.
• preload_all_models(): called from api/main.py lifespan startup hook.
• get_model(key): O(1) retrieval — never triggers a disk read after preload.
• GPU-aware: auto-selects CUDA / CPU, applies fp16, and optionally
  torch.compile() for PyTorch ≥ 2.
• Thread-safe: a threading.Lock guards initial load of each model so parallel
  requests during warm-up cannot cause a double-load race condition.

Singleton pattern used
──────────────────────
_registry: ModelRegistry is a module-level singleton.
All public helpers (get_model, get_tokenizer, …) delegate to it.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src import config
from src.model_registry import ABSTRACTIVE_ALGORITHMS, AlgorithmConfig, resolve_model_path
from src.utils import clear_gpu_cache, log_device_info, log_vram_usage, logger


# ─────────────────────────── Device resolution ─────────────────────────────

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _should_use_fp16(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    mode = config.USE_FP16
    if mode == "1":
        return True
    if mode == "0":
        return False
    # "auto" — check that the GPU supports it
    try:
        props = torch.cuda.get_device_properties(device)
        # fp16 is safe on Ampere (cc 8.x), Turing (7.5), Volta (7.0)+
        return props.major >= 7
    except Exception:
        return False


# ─────────────────────────── Data classes ──────────────────────────────────

@dataclass
class LoadedModel:
    key: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    fp16: bool
    model_path: str
    load_time_s: float
    algorithm: AlgorithmConfig


# ─────────────────────────── Registry (singleton) ──────────────────────────

class ModelRegistry:
    """Thread-safe singleton registry for all loaded abstractive models."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model_locks: dict[str, threading.Lock] = {}
        self._loaded: dict[str, LoadedModel] = {}
        self.device: torch.device = _resolve_device()
        self.fp16: bool = _should_use_fp16(self.device)
        self._preloaded = False

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_model_lock(self, key: str) -> threading.Lock:
        with self._lock:
            if key not in self._model_locks:
                self._model_locks[key] = threading.Lock()
            return self._model_locks[key]

    def _load_tokenizer(self, algorithm: AlgorithmConfig, model_path: str) -> PreTrainedTokenizerBase:
        """Load tokenizer, falling back to HuggingFace Hub if local fails."""
        kwargs: dict[str, Any] = {"use_fast": False}
        try:
            return AutoTokenizer.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            if model_path != algorithm.model_name and algorithm.model_name:
                logger.warning(
                    "[%s] Local tokenizer failed (%s) — falling back to hub: %s",
                    algorithm.key, exc, algorithm.model_name,
                )
                return AutoTokenizer.from_pretrained(algorithm.model_name, **kwargs)
            raise

    def _repair_vocab_mismatch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        algorithm: AlgorithmConfig,
        model_path: str,
    ) -> PreTrainedTokenizerBase:
        """Detect and repair tokenizer/model vocabulary size mismatch."""
        model_vocab = int(model.get_input_embeddings().weight.shape[0])
        tok_vocab = len(tokenizer)

        if model_vocab == tok_vocab:
            return tokenizer

        logger.warning(
            "[%s] Vocab mismatch — tokenizer=%d model=%d",
            algorithm.key, tok_vocab, model_vocab,
        )

        # Strategy 1: try the hub tokenizer for fine-tuned checkpoints
        if model_path != algorithm.model_name and algorithm.model_name:
            try:
                base_tok = AutoTokenizer.from_pretrained(algorithm.model_name, use_fast=False)
                if len(base_tok) == model_vocab:
                    logger.info("[%s] Repaired: using hub tokenizer %s", algorithm.key, algorithm.model_name)
                    return base_tok
            except Exception as exc:
                logger.warning("[%s] Hub tokenizer repair failed: %s", algorithm.key, exc)

        # Strategy 2: resize embeddings if difference is small (< 256 tokens)
        diff = abs(model_vocab - tok_vocab)
        if diff <= 256:
            model.resize_token_embeddings(tok_vocab)
            logger.warning("[%s] Resized token embeddings to %d", algorithm.key, tok_vocab)
            return tokenizer

        raise RuntimeError(
            f"[{algorithm.key}] Unsafe vocab mismatch: tokenizer={tok_vocab}, model={model_vocab}. "
            "Retrain with the correct tokenizer."
        )

    def _patch_generation_config(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Ensure generation tokens are set on the model config."""
        cfg = model.config
        if cfg.decoder_start_token_id is None and tokenizer.pad_token_id is not None:
            cfg.decoder_start_token_id = tokenizer.pad_token_id
        if cfg.eos_token_id is None and tokenizer.eos_token_id is not None:
            cfg.eos_token_id = tokenizer.eos_token_id
        if cfg.pad_token_id is None and tokenizer.pad_token_id is not None:
            cfg.pad_token_id = tokenizer.pad_token_id

    def _maybe_compile(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply torch.compile() on PyTorch >= 2 if enabled via config."""
        if not config.USE_TORCH_COMPILE:
            return model
        try:
            torch_ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if torch_ver >= (2, 0):
                logger.info("⚡ torch.compile() enabled — first inference will be slower (JIT warm-up)")
                return torch.compile(model)  # type: ignore[return-value]
        except Exception as exc:
            logger.warning("torch.compile() skipped: %s", exc)
        return model

    def _load_single(self, algorithm: AlgorithmConfig) -> LoadedModel:
        """Load one abstractive model + tokenizer. NOT thread-safe (caller holds lock)."""
        t0 = time.perf_counter()
        model_path = resolve_model_path(algorithm, prefer_local=True)
        logger.info("⏳ Loading [%s] from %s on %s …", algorithm.key, model_path, self.device)
        log_vram_usage(f"before_load_{algorithm.key}")

        tokenizer = self._load_tokenizer(algorithm, model_path)

        # Load model — use float16 from disk if GPU fp16 is enabled (saves VRAM)
        load_kwargs: dict[str, Any] = {}
        if self.fp16:
            load_kwargs["torch_dtype"] = torch.float16
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_path, **load_kwargs)

        tokenizer = self._repair_vocab_mismatch(model, tokenizer, algorithm, model_path)
        self._patch_generation_config(model, tokenizer)

        model = model.to(self.device)
        model.eval()

        model = self._maybe_compile(model)

        elapsed = time.perf_counter() - t0
        log_vram_usage(f"after_load_{algorithm.key}")
        logger.info("✅ [%s] loaded in %.2f s  fp16=%s", algorithm.key, elapsed, self.fp16)

        return LoadedModel(
            key=algorithm.key,
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            fp16=self.fp16,
            model_path=model_path,
            load_time_s=round(elapsed, 3),
            algorithm=algorithm,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def ensure_loaded(self, key: str) -> LoadedModel:
        """Return a LoadedModel, loading it on first access (lazy + thread-safe)."""
        if key in self._loaded:
            return self._loaded[key]

        lock = self._get_model_lock(key)
        with lock:
            # Double-checked locking
            if key in self._loaded:
                return self._loaded[key]

            if key not in ABSTRACTIVE_ALGORITHMS:
                raise KeyError(f"Unknown abstractive algorithm: {key!r}")
            algorithm = ABSTRACTIVE_ALGORITHMS[key]
            loaded = self._load_single(algorithm)
            self._loaded[key] = loaded
            return loaded

    def preload_all(self) -> None:
        """Eagerly load every abstractive model. Called once at server startup."""
        if self._preloaded:
            return
        log_device_info()
        logger.info("🔄 Preloading %d abstractive models …", len(ABSTRACTIVE_ALGORITHMS))
        t_total = time.perf_counter()

        for key in ABSTRACTIVE_ALGORITHMS:
            try:
                self.ensure_loaded(key)
                # Release CUDA fragmentation after each model
                clear_gpu_cache()
            except Exception as exc:
                logger.error("❌ Failed to preload [%s]: %s", key, exc, exc_info=True)

        elapsed = time.perf_counter() - t_total
        logger.info("🏁 All models preloaded in %.2f s", elapsed)
        self._preloaded = True

    def get(self, key: str) -> LoadedModel:
        return self.ensure_loaded(key)

    def is_loaded(self, key: str) -> bool:
        return key in self._loaded

    def status(self) -> dict:
        """Return a status dict suitable for the /health endpoint."""
        from src.utils import get_device_info
        return {
            "device": str(self.device),
            "fp16": self.fp16,
            "torch_compile": config.USE_TORCH_COMPILE,
            "preloaded": self._preloaded,
            "models": {
                key: {
                    "loaded": True,
                    "load_time_s": m.load_time_s,
                    "model_path": m.model_path,
                }
                for key, m in self._loaded.items()
            },
            "gpu_info": get_device_info(),
        }


# ─────────────────────────── Module-level singleton ────────────────────────

_registry = ModelRegistry()


def preload_all_models() -> None:
    """Eagerly preload all abstractive models. Call once from API startup."""
    _registry.preload_all()


def get_loaded_model(key: str) -> LoadedModel:
    """Return the LoadedModel for *key*, loading it if not yet cached."""
    return _registry.get(key)


def get_model(key: str) -> PreTrainedModel:
    return _registry.get(key).model


def get_tokenizer(key: str) -> PreTrainedTokenizerBase:
    return _registry.get(key).tokenizer


def get_device() -> torch.device:
    return _registry.device


def is_fp16() -> bool:
    return _registry.fp16


def registry_status() -> dict:
    return _registry.status()
