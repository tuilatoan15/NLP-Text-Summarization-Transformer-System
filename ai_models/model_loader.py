"""
model_loader.py — Singleton preloader for ALL models used by the system.
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

from src import config
from src.model_registry import ABSTRACTIVE_ALGORITHMS, AlgorithmConfig, resolve_model_path
from src.utils import MODEL_LOAD_LOCK, clear_gpu_cache, cuda_is_usable, log_device_info, log_vram_usage, logger


def _resolve_device() -> torch.device:
    if cuda_is_usable():
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
    try:
        props = torch.cuda.get_device_properties(device)
        return props.major >= 7
    except Exception:
        return False


@dataclass
class LoadedModel:
    key: str
    model: Any
    tokenizer: Any
    device: torch.device
    fp16: bool
    model_path: str
    load_time_s: float
    algorithm: AlgorithmConfig


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model_locks: dict[str, threading.Lock] = {}
        self._loaded: dict[str, LoadedModel] = {}
        self.device: torch.device = _resolve_device()
        self.fp16: bool = _should_use_fp16(self.device)
        self._preloaded = False

    def _get_model_lock(self, key: str) -> threading.Lock:
        with self._lock:
            if key not in self._model_locks:
                self._model_locks[key] = threading.Lock()
            return self._model_locks[key]

    def _load_tokenizer(self, algorithm: AlgorithmConfig, model_path: str) -> Any:
        from transformers import AutoTokenizer

        # mT5 tokenizer_config.json có trường 'backend' không tương thích với slow tokenizer
        # → bắt buộc dùng use_fast=True cho mT5 để tránh lỗi "not a string"
        # vit5 & bartpho dùng SentencePiece nên cần use_fast=False (slow tokenizer ổn định hơn)
        use_fast = algorithm.key == "mt5"
        kwargs: dict[str, Any] = {"use_fast": use_fast}

        try:
            tok = AutoTokenizer.from_pretrained(model_path, **kwargs)
            logger.info("[%s] Tokenizer loaded from local path (use_fast=%s): %s", algorithm.key, use_fast, type(tok).__name__)
            return tok
        except Exception as exc:
            logger.warning(
                "[%s] Local tokenizer failed (%s) — trying hub fallback: %s",
                algorithm.key, exc, algorithm.model_name,
            )
            if model_path != algorithm.model_name and algorithm.model_name:
                try:
                    tok = AutoTokenizer.from_pretrained(algorithm.model_name, **kwargs)
                    logger.info("[%s] Tokenizer loaded from hub (use_fast=%s)", algorithm.key, use_fast)
                    return tok
                except Exception as exc2:
                    logger.error("[%s] Hub tokenizer also failed: %s", algorithm.key, exc2)
            raise

    def _repair_vocab_mismatch(
        self,
        model: Any,
        tokenizer: Any,
        algorithm: AlgorithmConfig,
        model_path: str,
    ) -> Any:
        model_vocab = int(model.get_input_embeddings().weight.shape[0])
        tok_vocab = len(tokenizer)

        if algorithm.key == "vit5":
            tokenizer.clean_up_tokenization_spaces = True

        if model_vocab == tok_vocab and len(tokenizer) == model.config.vocab_size:
            return tokenizer

        logger.warning(
            "[%s] Vocab mismatch — tokenizer=%d model=%d config_vocab=%d",
            algorithm.key, tok_vocab, model_vocab, model.config.vocab_size,
        )

        loaded_from_hub = model_path == algorithm.model_name

        if algorithm.key in {"mt5", "bartpho"} and not loaded_from_hub:
            diff = abs(model_vocab - tok_vocab)
            if diff <= 256 and model_vocab != tok_vocab:
                if model_vocab > tok_vocab:
                    logger.warning(
                        "[%s] Tokenizer smaller than model (%d vs %d) — keeping model embeddings",
                        algorithm.key, tok_vocab, model_vocab,
                    )
                else:
                    model.resize_token_embeddings(tok_vocab)
                    logger.info("[%s] Aligned embeddings to local tokenizer size %d", algorithm.key, tok_vocab)
            return tokenizer

        if algorithm.key == "vit5" and loaded_from_hub:
            if len(tokenizer) != model.config.vocab_size or model_vocab != tok_vocab:
                logger.warning(
                    "[%s] tokenizer/model vocab mismatch (len=%d, config=%d, embed=%d)",
                    algorithm.key,
                    len(tokenizer),
                    model.config.vocab_size,
                    model_vocab,
                )
                try:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        algorithm.model_name,
                        use_fast=False,
                        legacy=True,
                    )
                    tokenizer.clean_up_tokenization_spaces = True
                    return tokenizer
                except Exception as exc:
                    logger.error("[%s] Failed to load fallback hub tokenizer: %s", algorithm.key, exc)

        if algorithm.key != "vit5":
            diff = abs(model_vocab - tok_vocab)
            if diff <= 256 and model_vocab != tok_vocab:
                model.resize_token_embeddings(tok_vocab)
                logger.warning("[%s] Resized token embeddings to %d", algorithm.key, tok_vocab)
                return tokenizer

        return tokenizer

    def _patch_generation_config(
        self,
        model: Any,
        tokenizer: Any,
    ) -> None:
        cfg = model.config
        if cfg.decoder_start_token_id is None and tokenizer.pad_token_id is not None:
            cfg.decoder_start_token_id = tokenizer.pad_token_id
        if cfg.eos_token_id is None and tokenizer.eos_token_id is not None:
            cfg.eos_token_id = tokenizer.eos_token_id
        if cfg.pad_token_id is None and tokenizer.pad_token_id is not None:
            cfg.pad_token_id = tokenizer.pad_token_id

    def _maybe_compile(self, model: Any) -> Any:
        if not config.USE_TORCH_COMPILE:
            return model
        try:
            torch_ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if torch_ver >= (2, 0):
                logger.info("⚡ torch.compile() enabled — first inference will be slower (JIT warm-up)")
                return torch.compile(model)
        except Exception as exc:
            logger.warning("torch.compile() skipped: %s", exc)
        return model

    def _offload_other_models(self, keep_key: str) -> None:
        """Move all other loaded models to CPU and empty GPU cache to free up VRAM."""
        # Quantized models cannot be moved to CPU via .to("cpu").
        # If quantization is active, we don't offload since they are already small enough to fit.
        if getattr(config, "USE_8BIT", False) or getattr(config, "USE_4BIT", False):
            return

        with self._lock:
            for k, loaded in list(self._loaded.items()):
                if k != keep_key:
                    try:
                        # Check if it is on GPU currently
                        current_device = next(loaded.model.parameters()).device
                        if current_device.type == "cuda":
                            logger.info("⏳ Offloading [%s] to CPU to free up VRAM...", k)
                            loaded.model = loaded.model.to("cpu")
                            clear_gpu_cache()
                            logger.info("✅ [%s] successfully offloaded to CPU.", k)
                    except Exception as exc:
                        logger.warning("Failed to offload [%s] to CPU: %s", k, exc)

    def _load_single(self, algorithm: AlgorithmConfig) -> LoadedModel:
        t0 = time.perf_counter()
        model_path = resolve_model_path(algorithm, prefer_local=True)
        logger.info("⏳ Loading [%s] from %s on %s …", algorithm.key, model_path, self.device)
        log_vram_usage(f"before_load_{algorithm.key}")

        with MODEL_LOAD_LOCK:
            tokenizer = self._load_tokenizer(algorithm, model_path)

            load_kwargs: dict[str, Any] = {}
            use_8bit = getattr(config, "USE_8BIT", False)
            use_4bit = getattr(config, "USE_4BIT", False)

            if use_4bit:
                load_kwargs["low_cpu_mem_usage"] = True
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16 if self.fp16 else torch.float32,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    # device_map is required for quantization
                    device_idx = self.device.index if self.device.type == "cuda" else 0
                    load_kwargs["device_map"] = {"": device_idx}
                    logger.info("[%s] Configured for 4-bit quantization (nf4)", algorithm.key)
                except Exception as exc:
                    logger.warning("[%s] Failed to configure 4-bit quantization: %s. Falling back to default.", algorithm.key, exc)
                    if self.fp16:
                        load_kwargs["torch_dtype"] = torch.float16
            elif use_8bit:
                load_kwargs["low_cpu_mem_usage"] = True
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    # device_map is required for quantization
                    device_idx = self.device.index if self.device.type == "cuda" else 0
                    load_kwargs["device_map"] = {"": device_idx}
                    logger.info("[%s] Configured for 8-bit quantization", algorithm.key)
                except Exception as exc:
                    logger.warning("[%s] Failed to configure 8-bit quantization: %s. Falling back to default.", algorithm.key, exc)
                    if self.fp16:
                        load_kwargs["torch_dtype"] = torch.float16
            else:
                if self.fp16:
                    load_kwargs["torch_dtype"] = torch.float16

            from transformers import AutoModelForSeq2SeqLM
            
            is_peft = False
            adapter_config_file = Path(model_path) / "adapter_config.json"
            if adapter_config_file.exists():
                is_peft = True

            if is_peft:
                from peft import PeftModel
                import json
                logger.info("[%s] PEFT adapter config found at %s. Loading base model...", algorithm.key, adapter_config_file)
                with open(adapter_config_file, "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                base_model_name = adapter_cfg.get("base_model_name_or_path")
                if not base_model_name:
                    base_model_name = algorithm.model_name
                logger.info("[%s] Loading base model: %s", algorithm.key, base_model_name)
                base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, **load_kwargs)
                model = PeftModel.from_pretrained(base_model, model_path)
                if not (use_8bit or use_4bit):
                    model = model.merge_and_unload()
                    logger.info("[%s] PEFT adapter merged successfully into base model", algorithm.key)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **load_kwargs)

            tokenizer = self._repair_vocab_mismatch(model, tokenizer, algorithm, model_path)
            self._patch_generation_config(model, tokenizer)

            if not (use_8bit or use_4bit):
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

    def ensure_loaded(self, key: str) -> LoadedModel:
        if key in self._loaded:
            loaded = self._loaded[key]
            # If the model is currently offloaded to CPU, move it back to GPU!
            try:
                current_device = next(loaded.model.parameters()).device
                if loaded.device.type == "cuda" and current_device.type == "cpu":
                    logger.info("⏳ Moving [%s] back to GPU VRAM...", key)
                    self._offload_other_models(key)
                    loaded.model = loaded.model.to(self.device)
                    clear_gpu_cache()
                    logger.info("✅ [%s] is now active on GPU.", key)
            except Exception as exc:
                logger.warning("Failed to reactivate model on GPU: %s", exc)
            return loaded

        lock = self._get_model_lock(key)
        with lock:
            if key in self._loaded:
                return self._loaded[key]

            if key not in ABSTRACTIVE_ALGORITHMS:
                raise KeyError(f"Unknown abstractive algorithm: {key!r}")
            
            # Offload other models before loading the new one to prevent VRAM spikes
            self._offload_other_models(key)
            
            algorithm = ABSTRACTIVE_ALGORITHMS[key]
            loaded = self._load_single(algorithm)
            self._loaded[key] = loaded
            return loaded

    def preload_all(self, model_keys: list[str] | None = None) -> None:
        if self._preloaded:
            return
        log_device_info()
        if model_keys:
            keys = [k for k in model_keys if k in ABSTRACTIVE_ALGORITHMS]
            missing = [k for k in model_keys if k not in ABSTRACTIVE_ALGORITHMS]
            if missing:
                logger.warning("Bỏ qua preload — model không tồn tại: %s", missing)
        else:
            keys = list(ABSTRACTIVE_ALGORITHMS)
        logger.info("🔄 Preloading %d abstractive model(s) …", len(keys))
        t_total = time.perf_counter()

        for key in keys:
            try:
                self.ensure_loaded(key)
                clear_gpu_cache()
            except Exception as exc:
                logger.error("❌ Failed to preload [%s]: %s", key, exc, exc_info=True)

        elapsed = time.perf_counter() - t_total
        logger.info("🏁 Model preload finished in %.2f s (%d model(s))", elapsed, len(keys))
        self._preloaded = True

    def get(self, key: str) -> LoadedModel:
        return self.ensure_loaded(key)

    def is_loaded(self, key: str) -> bool:
        return key in self._loaded

    def status(self) -> dict:
        from src.utils import get_device_info, get_model_device_str
        models_status = {}
        for key, m in self._loaded.items():
            try:
                actual_dev = get_model_device_str(m.model)
            except Exception:
                actual_dev = "unknown"
            models_status[key] = {
                "loaded": True,
                "device": actual_dev,
                "fp16": m.fp16,
                "load_time_s": m.load_time_s,
                "model_path": m.model_path,
            }
        return {
            "device": str(self.device),
            "fp16": self.fp16,
            "torch_compile": config.USE_TORCH_COMPILE,
            "preloaded": self._preloaded,
            "models": models_status,
            "gpu_info": get_device_info(),
        }


_registry = ModelRegistry()


def preload_all_models(model_keys: list[str] | None = None) -> None:
    keys = model_keys
    if keys is None and config.PRELOAD_MODELS_LIST:
        keys = config.PRELOAD_MODELS_LIST
    _registry.preload_all(keys)


def get_loaded_model(key: str) -> LoadedModel:
    return _registry.get(key)


def get_model(key: str) -> Any:
    return _registry.get(key).model


def get_tokenizer(key: str) -> Any:
    return _registry.get(key).tokenizer


def get_device() -> torch.device:
    return _registry.device


def is_fp16() -> bool:
    return _registry.fp16


def registry_status() -> dict:
    return _registry.status()
