import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_notebook(model_name: str, prefix: str, model_id: str, display_name: str):
    output_dir = f"./{model_id}-colab-checkpoints"
    final_dir = f"./{model_id}-colab-finetuned"
    zip_name = f"{model_id}-colab-finetuned.zip"
    drive_dir = f"/content/drive/MyDrive/{model_id}-training"

    cells = []

    # Cell 1: Introduction Markdown
    intro_md = """# [DISPLAY_NAME] Vietnamese Summarization Training - Colab Standard

File này dùng để fine-tune `[MODEL_NAME]` cho bài toán tóm tắt tiếng Việt trên bộ dữ liệu `nam194/vietnews`.
Kết quả sau training sẽ được lưu thành `[ZIP_NAME]`.

Cách đưa model về dự án:
1. Tải `[ZIP_NAME]` từ Colab hoặc Google Drive.
2. Đặt file vào `data/vietnamese-summarization-dataset-0001/` (hoặc giải nén trực tiếp vào `models/[MODEL_ID]-finetuned/`).
3. Chạy trong repo: `python scripts/extract_colab_models.py` (nếu tải zip) hoặc đặt trực tiếp vào thư mục.
4. Restart backend để hệ thống dùng model mới tại `models/[MODEL_ID]-finetuned/`.
"""
    intro_md = intro_md.replace("[DISPLAY_NAME]", display_name)
    intro_md = intro_md.replace("[MODEL_NAME]", model_name)
    intro_md = intro_md.replace("[ZIP_NAME]", zip_name)
    intro_md = intro_md.replace("[MODEL_ID]", model_id)
    cells.append({"cell_type": "markdown", "metadata": {}, "source": intro_md.splitlines(keepends=True)})

    # Cell 2: Markdown block
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Cell 1 - Cài đặt môi trường và cấu hình\n"]})

    # Cell 3: Setup Code
    setup_code = """!pip install -q -U transformers datasets evaluate rouge-score bert-score accelerate sentencepiece safetensors

import inspect
import os
import random
import shutil
import warnings
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from IPython.display import display
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 160)
plt.rcParams.update({"figure.dpi": 110, "font.size": 11})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__}")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

CFG = {
    "model_name": "[MODEL_NAME]",
    "dataset_name": "nam194/vietnews",
    "source_col": "article",
    "target_col": "abstract",
    "prefix": "[PREFIX]",
    "max_src_len": 512,
    "max_tgt_len": 128,
    "epochs": 3,
    "batch_size": 4,
    "eval_batch_size": 4,
    "grad_acc_steps": 4,
    "lr": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "seed": 42,
    "train_sample_size": 20000,
    "val_sample_size": 2000,
    "test_sample_size": 1000,
    "output_dir": "[OUTPUT_DIR]",
    "final_dir": "[FINAL_DIR]",
    "zip_name": "[ZIP_NAME]",
    "save_to_drive": True,
    "drive_dir": "[DRIVE_DIR]",
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])
print("Config OK:", CFG)
"""
    setup_code = setup_code.replace("[MODEL_NAME]", model_name)
    setup_code = setup_code.replace("[PREFIX]", prefix)
    setup_code = setup_code.replace("[OUTPUT_DIR]", output_dir)
    setup_code = setup_code.replace("[FINAL_DIR]", final_dir)
    setup_code = setup_code.replace("[ZIP_NAME]", zip_name)
    setup_code = setup_code.replace("[DRIVE_DIR]", drive_dir)
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": setup_code.splitlines(keepends=True)})

    # Cell 4: Markdown block
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Cell 2 - Tải dữ liệu VietNews\n"]})

    # Cell 5: Load Dataset Code
    load_code = """print("[Data] Dang tai dataset...")
dataset = load_dataset(CFG["dataset_name"])
print(dataset)

def pick_split(name: str, fallback: str):
    if name in dataset:
        return dataset[name]
    return dataset[fallback]

train_full = pick_split("train", list(dataset.keys())[0])
val_full = pick_split("validation", "train")
test_full = pick_split("test", "validation" if "validation" in dataset else "train")

def safe_select(ds, n: int):
    n = min(n, len(ds))
    return ds.shuffle(seed=CFG["seed"]).select(range(n))

train_data = safe_select(train_full, CFG["train_sample_size"])
val_data = safe_select(val_full, CFG["val_sample_size"])
test_data = safe_select(test_full, CFG["test_sample_size"])

print("\\n=== Kich thuoc du lieu ===")
print(f"Train: {len(train_data):,}")
print(f"Val  : {len(val_data):,}")
print(f"Test : {len(test_data):,}")
print("\\nColumns:", train_data.column_names)

sample_df = pd.DataFrame(train_data[:3])
display(sample_df[[CFG["source_col"], CFG["target_col"]]])
"""
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": load_code.splitlines(keepends=True)})

    # Cell 6: Markdown block
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Cell 3 - Khám phá phân phối token\n"]})

    # Cell 7: Token distribution Code
    token_dist_code = """tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"], use_fast=False)

def plot_token_distribution(dataset_sample, num_samples=1000):
    sample = dataset_sample.select(range(min(num_samples, len(dataset_sample))))
    src_lens = [
        len(tokenizer.encode(str(x), truncation=False))
        for x in sample[CFG["source_col"]]
    ]
    tgt_lens = [
        len(tokenizer.encode(str(x), truncation=False))
        for x in sample[CFG["target_col"]]
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    sns.histplot(src_lens, bins=40, color="#1e88e5", ax=ax1, kde=True)
    ax1.axvline(np.mean(src_lens), color="red", ls="--", label=f"Mean: {np.mean(src_lens):.0f}")
    ax1.axvline(CFG["max_src_len"], color="black", ls="-", label=f"Cut-off: {CFG['max_src_len']}")
    ax1.set_title("Do dai bai viet (tokens)")
    ax1.legend()

    sns.histplot(tgt_lens, bins=40, color="#43a047", ax=ax2, kde=True)
    ax2.axvline(np.mean(tgt_lens), color="red", ls="--", label=f"Mean: {np.mean(tgt_lens):.0f}")
    ax2.axvline(CFG["max_tgt_len"], color="black", ls="-", label=f"Cut-off: {CFG['max_tgt_len']}")
    ax2.set_title("Do dai tom tat (tokens)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

plot_token_distribution(train_data)
"""
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": token_dist_code.splitlines(keepends=True)})

    # Cell 8: Markdown block
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Cell 4 - Tiền xử lý và tokenize\n"]})

    # Cell 9: Tokenize Code
    tokenize_code = """def preprocess_function(examples):
    inputs = [
        CFG["prefix"] + str(doc).strip()
        for doc in examples[CFG["source_col"]]
    ]
    targets = [
        str(summary).strip()
        for summary in examples[CFG["target_col"]]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=CFG["max_src_len"],
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=CFG["max_tgt_len"],
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

remove_cols = train_data.column_names
print("[Process] Dang tokenize...")
tokenized_train = train_data.map(
    preprocess_function,
    batched=True,
    batch_size=64,
    remove_columns=remove_cols,
    desc="Tokenizing train",
)
tokenized_val = val_data.map(
    preprocess_function,
    batched=True,
    batch_size=64,
    remove_columns=remove_cols,
    desc="Tokenizing validation",
)
tokenized_test = test_data.map(
    preprocess_function,
    batched=True,
    batch_size=64,
    remove_columns=remove_cols,
    desc="Tokenizing test",
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=CFG["model_name"],
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
)
print("Tien xu ly hoan tat.")
"""
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": tokenize_code.splitlines(keepends=True)})

    # Cell 10: Markdown block
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Cell 5 - Khởi tạo model và training\n"]})

    # Cell 11: Train Code
    train_code = """model = AutoModelForSeq2SeqLM.from_pretrained(CFG["model_name"])
model.config.use_cache = False
model.to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
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

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=False,
    )
    return {k: round(v * 100, 4) for k, v in result.items()}

def build_training_args():
    kwargs = {
        "output_dir": CFG["output_dir"],
        "save_strategy": "epoch",
        "learning_rate": CFG["lr"],
        "per_device_train_batch_size": CFG["batch_size"],
        "per_device_eval_batch_size": CFG["eval_batch_size"],
        "gradient_accumulation_steps": CFG["grad_acc_steps"],
        "weight_decay": CFG["weight_decay"],
        "save_total_limit": 2,
        "num_train_epochs": CFG["epochs"],
        "predict_with_generate": True,
        "generation_max_length": CFG["max_tgt_len"],
        "logging_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "warmup_ratio": CFG["warmup_ratio"],
        "dataloader_num_workers": 0,
        "gradient_checkpointing": True,
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    }

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return Seq2SeqTrainingArguments(**kwargs)

training_args = build_training_args()

from transformers import EarlyStoppingCallback
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("[Trainer] Bat dau huan luyen...")
trainer.train()

print("[Trainer] Danh gia validation...")
metrics = trainer.evaluate()
print(metrics)

final_dir = Path(CFG["final_dir"])
if final_dir.exists():
    shutil.rmtree(final_dir)
final_dir.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))

metadata = {
    "base_model": CFG["model_name"],
    "dataset": CFG["dataset_name"],
    "source_col": CFG["source_col"],
    "target_col": CFG["target_col"],
    "max_src_len": CFG["max_src_len"],
    "max_tgt_len": CFG["max_tgt_len"],
    "epochs": CFG["epochs"],
    "train_sample_size": len(train_data),
    "val_sample_size": len(val_data),
    "metrics": metrics,
}
pd.Series(metadata, dtype="object").to_json(final_dir / "training_report.json", force_ascii=False, indent=2)
print(f"Da luu model + tokenizer tai: {final_dir}")
"""
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": train_code.splitlines(keepends=True)})

    # Cell 12: Markdown block
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Cell 6 - Kiểm thử sinh tóm tắt, BERTScore và đóng gói zip\n"]})

    # Cell 13: Inference & Save Code
    inf_code = """def generate_summary(text: str) -> str:
    prefixed = CFG["prefix"] + str(text).strip()
    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        max_length=CFG["max_src_len"],
        truncation=True,
    ).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        min_new_tokens=20,
        num_beams=2,
        no_repeat_ngram_size=5,
        repetition_penalty=2.5,
        length_penalty=1.05,
        early_stopping=True,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

test_samples = test_data.select(range(min(3, len(test_data))))
results = []

model_key = CFG["model_name"].split('/')[-1] + " prediction"

for i, example in enumerate(test_samples):
    original = example[CFG["source_col"]]
    reference = example[CFG["target_col"]]
    prediction = generate_summary(original)
    results.append({
        "ID": i + 1,
        "Bai viet goc": str(original)[:260] + "...",
        "Tom tat chuan": reference,
        model_key: prediction,
    })

bertscore = evaluate.load("bertscore")
preds = [r[model_key] for r in results]
refs = [r["Tom tat chuan"] for r in results]
b_score = bertscore.compute(predictions=preds, references=refs, lang="vi")

for i in range(len(results)):
    results[i]["BERTScore F1"] = round(float(b_score["f1"][i]), 4)

df_results = pd.DataFrame(results)
display(df_results.style.set_properties(**{"text-align": "left", "vertical-align": "top"}))

zip_path = Path(CFG["zip_name"])
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(CFG["zip_name"].replace(".zip", ""), "zip", root_dir=".", base_dir=CFG["final_dir"].lstrip("./"))
print(f"Da dong goi: {zip_path.resolve()}")

if CFG["save_to_drive"]:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        drive_dir = Path(CFG["drive_dir"])
        drive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, drive_dir / zip_path.name)
        print(f"Da copy len Google Drive: {drive_dir / zip_path.name}")
    except Exception as exc:
        print(f"Khong copy duoc len Drive: {exc}")

try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as exc:
    print(f"Khong tu dong download duoc, hay tai file thu cong trong Colab: {exc}")
"""
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": inf_code.splitlines(keepends=True)})

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.x"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook


def main():
    configs = [
        {
            "model_name": "VietAI/vit5-base",
            "prefix": "summarize: ",
            "model_id": "vit5",
            "display_name": "ViT5",
            "filename": "Colab_ViT5_Training_Standard.ipynb"
        },
        {
            "model_name": "vinai/bartpho-syllable",
            "prefix": "",
            "model_id": "bartpho",
            "display_name": "BARTPho",
            "filename": "Colab_BARTPho_Training_Standard.ipynb"
        },
        {
            "model_name": "google/mt5-small",
            "prefix": "summarize: ",
            "model_id": "mt5",
            "display_name": "mT5",
            "filename": "Colab_mT5_Training_Standard.ipynb"
        }
    ]

    for config in configs:
        notebook = build_notebook(
            model_name=config["model_name"],
            prefix=config["prefix"],
            model_id=config["model_id"],
            display_name=config["display_name"]
        )
        path = ROOT / config["filename"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, ensure_ascii=False, indent=2)
        print(f"Generated: {config['filename']}")


if __name__ == "__main__":
    main()
