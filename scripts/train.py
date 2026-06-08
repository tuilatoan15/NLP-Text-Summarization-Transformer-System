"""CLI: fine-tune ViT5, mT5, or BARTPho on VietNews (nam194/vietnews)."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src import config
from src.evaluate import compute_rouge_batch
from src.model_registry import ABSTRACTIVE_ALGORITHMS, resolve_algorithm
from src.preprocess import clean_generated_summary, augment_text
from src.utils import logger, save_json
from train.dataset_loader import load_vnexpress_dataset


def _prefix(model_key: str, text: str) -> str:
    return f"summarize: {text}" if model_key in {"vit5", "mt5"} else text


def tokenize_batch(examples, tokenizer, model_key: str, max_input_len: int, max_target_len: int, use_augmentation: bool = False):
    if use_augmentation:
        inputs = [_prefix(model_key, augment_text(str(article))) for article in examples["article"]]
    else:
        inputs = [_prefix(model_key, str(article)) for article in examples["article"]]
    # VietNews: target column is 'abstract' (not 'title')
    target_col = "abstract" if "abstract" in examples else "title"
    targets = [str(summary) for summary in examples[target_col]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_len,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_compute_metrics(tokenizer):
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        predictions = np.where(predictions != -100, predictions, pad_token_id)
        predictions = np.clip(np.asarray(predictions), 0, vocab_size - 1)
        labels = np.where(labels != -100, labels, pad_token_id)
        labels = np.clip(np.asarray(labels), 0, vocab_size - 1)
        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_preds = [clean_generated_summary(text) for text in decoded_preds]
        decoded_labels = [clean_generated_summary(text) for text in decoded_labels]
        return compute_rouge_batch(decoded_preds, decoded_labels)

    return compute_metrics


def _training_schedule(args, train_size: int) -> dict:
    """Pick eval/save cadence from dataset size (tuned for ~2000 VNExpress samples)."""
    steps_per_epoch = max(
        1,
        train_size // max(1, args.batch_size * args.gradient_accumulation_steps),
    )
    eval_steps = args.eval_steps
    save_steps = args.save_steps
    if args.auto_schedule:
        eval_steps = max(50, min(250, steps_per_epoch // 3))
        save_steps = eval_steps
    return {"eval_steps": eval_steps, "save_steps": save_steps, "steps_per_epoch": steps_per_epoch}


def train_model(args) -> dict:
    algorithm = resolve_algorithm(args.model)
    if algorithm.key not in ABSTRACTIVE_ALGORITHMS:
        raise ValueError("--model must be one of: vit5, mt5, bartpho")

    output_dir = Path(args.output_dir or algorithm.local_dir or config.MODEL_DIR / f"{algorithm.key}-finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_source = algorithm.model_name
    logger.info("Training %s from %s", algorithm.name, model_source)
    # mT5 tokenizer_config.json có 'backend' field không tương thích với slow tokenizer
    use_fast_tok = algorithm.key == "mt5"
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=use_fast_tok)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source)

    if args.use_lora:
        logger.info("Applying LoRA (Low-Rank Adaptation) configuration...")
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Target modules cho từng loại mô hình
        if algorithm.key in {"vit5", "mt5"}:
            # T5 architectures sử dụng q, v
            target_modules = ["q", "v"]
        elif algorithm.key == "bartpho":
            # BART architectures
            target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = ["q", "v"]

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset = load_vnexpress_dataset(
        local_csv_path=args.local_data,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        use_cache=not args.no_cache,
    )
    schedule = _training_schedule(args, len(dataset["train"]))
    logger.info(
        "Training schedule: train=%s val=%s steps/epoch≈%s eval/save every %s steps",
        len(dataset["train"]),
        len(dataset["validation"]),
        schedule["steps_per_epoch"],
        schedule["eval_steps"],
    )

    tokenized = dataset.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
            algorithm.key,
            args.max_input_tokens,
            args.max_target_tokens,
            use_augmentation=args.use_augmentation,
        ),
        batched=True,
        batch_size=32,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    save_strategy = "no" if args.no_save else "steps"

    # Check BF16 capability and set optimizer/precision for stability
    bf16_supported = False
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            bf16_supported = True

    is_t5 = algorithm.key in {"vit5", "mt5"}
    if is_t5:
        if bf16_supported:
            use_fp16 = False
            use_bf16 = True
            optim_name = "adamw_torch"
            logger.info("T5 model on BF16-capable GPU. Using BF16 mixed precision.")
        else:
            # T5 is unstable with FP16 on T4/P100 (causes NaN loss). Fallback to FP32 + Adafactor.
            use_fp16 = False
            use_bf16 = False
            optim_name = "adafactor"
            logger.info("T5 model on T4/P100 (no BF16). Using FP32 with Adafactor optimizer to avoid NaN/OOM.")
    else:
        # BART is stable on FP16
        if bf16_supported:
            use_fp16 = False
            use_bf16 = True
            optim_name = "adamw_torch"
            logger.info("BART model on BF16-capable GPU. Using BF16 mixed precision.")
        else:
            use_fp16 = torch.cuda.is_available()
            use_bf16 = False
            optim_name = "adamw_torch"
            logger.info("BART model on T4/P100. Using FP16 mixed precision.")

    if args.gradient_checkpointing:
        if args.use_lora:
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        label_smoothing_factor=0.1,
        eval_strategy="steps",
        eval_steps=schedule["eval_steps"],
        save_strategy=save_strategy,
        save_steps=schedule["save_steps"],
        save_total_limit=3,  # Giu 3 checkpoint gan nhat de an toan
        load_best_model_at_end=not args.no_save,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=not args.skip_rouge_eval,
        generation_max_length=args.max_target_tokens,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=optim_name,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        group_by_length=True,
        logging_steps=args.logging_steps,
        report_to="none",
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    from transformers import TrainerCallback
    class KaggleOptimizationCallback(TrainerCallback):
        def __init__(self, check_output_dir, min_free_gb=3.0):
            self.check_output_dir = Path(check_output_dir)
            self.min_free_bytes = min_free_gb * 1024 * 1024 * 1024
            
        def _print_gpu_memory(self, step_info=""):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                logger.info("[VRAM Monitoring] %s -> Allocated: %.2f MB, Reserved: %.2f MB", step_info, allocated, reserved)
                
        def _check_disk_space(self):
            import shutil
            try:
                total, used, free = shutil.disk_usage(self.check_output_dir)
                free_gb = free / (1024 ** 3)
                logger.info("[Disk Monitoring] Free space: %.2f GB", free_gb)
                if free < self.min_free_bytes:
                    logger.warning("[DISK WARNING] Low disk space (< %.2f GB). Cleaning HF cache...", free_gb)
                    cache_dir = Path(os.path.expanduser("~/.cache/huggingface"))
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir, ignore_errors=True)
                        logger.info("Cleaned HuggingFace cache directory.")
            except Exception as e:
                logger.error("Failed to check disk space: %s", e)
                
        def on_log(self, args, state, control, **kwargs):
            self._print_gpu_memory(f"Step {state.global_step} Log")
            
        def on_evaluate(self, args, state, control, **kwargs):
            self._print_gpu_memory(f"Step {state.global_step} Eval")
            
        def on_save(self, args, state, control, **kwargs):
            self._print_gpu_memory(f"Step {state.global_step} Save")
            self._check_disk_space()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        def on_epoch_end(self, args, state, control, **kwargs):
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Epoch ended. GPU Cache cleared.")
            self._check_disk_space()

    compute_metrics = None if args.skip_rouge_eval else build_compute_metrics(tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            KaggleOptimizationCallback(training_args.output_dir)
        ],
    )

    # De quy quet va lay checkpoint hop le cuoi cung su dung get_last_checkpoint
    from transformers.trainer_utils import get_last_checkpoint
    
    def get_valid_resume_checkpoint(checkpoint_dir_path: Path) -> str | None:
        if not checkpoint_dir_path.exists():
            return None
            
        last_checkpoint = get_last_checkpoint(str(checkpoint_dir_path))
        if not last_checkpoint:
            return None
            
        # Kiem tra tinh toàn ven cua checkpoint gan nhat
        d = Path(last_checkpoint)
        has_state = (d / "trainer_state.json").exists()
        has_weights = (
            (d / "pytorch_model.bin").exists() or 
            (d / "model.safetensors").exists() or
            (d / "adapter_model.safetensors").exists() or
            (d / "adapter_model.bin").exists()
        )
        has_config = (d / "config.json").exists() or (d / "adapter_config.json").exists()
        
        if has_state and has_weights and has_config:
            logger.info("Found valid last checkpoint: %s", last_checkpoint)
            return last_checkpoint
        else:
            # Xoa checkpoint bi loi va tu dong tim lai checkpoint lien truoc no
            logger.warning("Checkpoint %s is corrupted/incomplete. Deleting to fallback...", last_checkpoint)
            import shutil
            try:
                shutil.rmtree(d)
            except Exception as e:
                logger.error("Failed to delete corrupted checkpoint %s: %s", d, e)
                
            # De quy de tim checkpoint phia truoc
            return get_valid_resume_checkpoint(checkpoint_dir_path)

    # Backup checkpoint tot nhat tranh bi ghi de hoac mat mat
    def backup_best_checkpoint(checkpoint_dir_path: Path, backup_dir_path: Path):
        if not checkpoint_dir_path.exists():
            return
            
        import json
        import shutil
        best_checkpoint_path = None
        best_metric = float("inf")
        
        for d in checkpoint_dir_path.iterdir():
            if d.is_dir() and d.name.startswith("checkpoint-"):
                state_file = d / "trainer_state.json"
                if state_file.exists():
                    try:
                        with open(state_file, "r") as f:
                            state = json.load(f)
                        metric = state.get("best_metric")
                        if metric is not None and metric < best_metric:
                            best_metric = metric
                            best_checkpoint_path = d
                    except Exception:
                        pass
                        
        if best_checkpoint_path:
            logger.info("[Backup Best Model] Copying best checkpoint (%s) to %s", best_checkpoint_path.name, backup_dir_path)
            if backup_dir_path.exists():
                shutil.rmtree(backup_dir_path)
            shutil.copytree(best_checkpoint_path, backup_dir_path)
        else:
            last_cp = get_last_checkpoint(str(checkpoint_dir_path))
            if last_cp:
                logger.info("[Backup Best Model] Copying last checkpoint (%s) to %s", Path(last_cp).name, backup_dir_path)
                if backup_dir_path.exists():
                    shutil.rmtree(backup_dir_path)
                shutil.copytree(last_cp, backup_dir_path)

    resume_from_checkpoint = None
    if not getattr(args, "no_resume", False):
        checkpoint_dir = Path(training_args.output_dir)
        resume_from_checkpoint = get_valid_resume_checkpoint(checkpoint_dir)
        if resume_from_checkpoint:
            logger.info("Resuming training from valid checkpoint: %s", resume_from_checkpoint)
        else:
            logger.info("No valid checkpoint found. Starting training from scratch.")

    import gc
    def cleanup():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Bat dau huan luyen...")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        try:
            backup_best_checkpoint(checkpoint_dir, output_dir / "best-checkpoint-backup")
        except Exception as e:
            logger.error("Failed to backup best checkpoint: %s", e)
    except KeyboardInterrupt:
        logger.warning("Training bi ngat boi nguoi dung! Dang luu checkpoint tam thoi...")
        trainer.save_model(str(Path(training_args.output_dir) / "interrupted-checkpoint"))
        tokenizer.save_pretrained(str(Path(training_args.output_dir) / "interrupted-checkpoint"))
        cleanup()
        raise
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        if is_oom:
            logger.error("⚠️ [CUDA OUT OF MEMORY] HET BO NHO GPU TRONG KHI HUAN LUYEN!")
            logger.error("Chi tiet loi: %s", e)
            logger.info("Dang giai phong VRAM va luu emergency checkpoint...")
            cleanup()
            try:
                emergency_dir = str(Path(training_args.output_dir) / "oom-emergency-checkpoint")
                trainer.save_model(emergency_dir)
                tokenizer.save_pretrained(emergency_dir)
                logger.info("[Emergency Save] Da luu model cap cuu tai: %s", emergency_dir)
            except Exception as save_err:
                logger.error("Luu model cap cuu that bai: %s", save_err)
        else:
            logger.error("Gap loi khi dang huan luyen: %s", e)
            try:
                emergency_dir = str(Path(training_args.output_dir) / "error-checkpoint")
                trainer.save_model(emergency_dir)
                tokenizer.save_pretrained(emergency_dir)
            except Exception:
                pass
        try:
            backup_best_checkpoint(checkpoint_dir, output_dir / "best-checkpoint-backup")
        except Exception:
            pass
        cleanup()
        raise
    except Exception as e:
        logger.error("Gap loi chung khi huan luyen: %s", e)
        try:
            emergency_dir = str(Path(training_args.output_dir) / "error-checkpoint")
            trainer.save_model(emergency_dir)
            tokenizer.save_pretrained(emergency_dir)
        except Exception:
            pass
        try:
            backup_best_checkpoint(checkpoint_dir, output_dir / "best-checkpoint-backup")
        except Exception:
            pass
        cleanup()
        raise

    logger.info("Danh gia validation...")
    metrics = trainer.evaluate()
    if not args.no_save:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    save_json(
        {
            "algorithm": algorithm.key,
            "base_model": model_source,
            "dataset": args.dataset_name,
            "max_samples": args.max_samples,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": metrics,
        },
        output_dir / "training_report.json",
    )
    cleanup()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Vietnamese summarization model.")
    parser.add_argument("--model", default="vit5", choices=["vit5", "mt5", "bartpho"])
    parser.add_argument("--dataset_name", default=config.DATASET_NAME)
    parser.add_argument("--local_data", default=None)
    parser.add_argument("--max_samples", type=int, default=config.MAX_TRAIN_SAMPLES)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--eval_batch_size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=config.WARMUP_STEPS)
    parser.add_argument("--max_input_tokens", type=int, default=config.MAX_INPUT_TOKENS)
    parser.add_argument("--max_target_tokens", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument(
        "--auto_schedule",
        action="store_true",
        help="Derive eval/save steps from train set size (recommended for 2000-sample runs).",
    )
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Skip checkpoint and final model saves (smoke tests when disk is low).",
    )
    parser.add_argument(
        "--skip_rouge_eval",
        action="store_true",
        help="Use eval_loss only (faster on CPU; skips generate+ROUGE during training).",
    )
    # LoRA / PEFT parameters
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA parameter-efficient training.")
    parser.add_argument("--lora_r", type=int, default=8, help="Rank size for LoRA adapters.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor (alpha) for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.")
    # Performance & training optimization
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM.",
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        help="Enable data augmentation for Vietnamese training text.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume training from existing checkpoints (force training from scratch).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
