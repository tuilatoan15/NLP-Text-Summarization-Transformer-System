import json
from pathlib import Path

# Paths
notebook_path = Path("Colab_ViT5_VietNews_30k_3Epochs.ipynb")
output_path = Path("Colab_ViT5_VietNews_30k_3Epochs_Checkpoint2000.ipynb")

if not notebook_path.exists():
    print(f"Error: {notebook_path} not found.")
    exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells.")

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        
        # 1. Update CFG cell to add "checkpoint_path"
        if 'CFG = {' in source_str and '"model_name":' in source_str:
            print(f"Found CFG cell at index {idx}")
            target = '"model_name": "VietAI/vit5-base",'
            replacement = '"model_name": "VietAI/vit5-base",\n    "checkpoint_path": "/content/drive/MyDrive/checkpoint-2000",  # Đường dẫn lối tắt của checkpoint-2000 trong My Drive'
            source_str = source_str.replace(target, replacement)
            cell["source"] = [line + "\n" for line in source_str.split("\n")]
            if cell["source"][-1] == "\n":
                cell["source"].pop()
            
        # 2. Update Tokenizer cell
        elif 'tokenizer = AutoTokenizer.from_pretrained(' in source_str:
            print(f"Found Tokenizer cell at index {idx}")
            new_source = (
                "# Kích hoạt Google Drive sớm để đọc checkpoint từ lối tắt\n"
                "try:\n"
                "    from google.colab import drive\n"
                "    import os\n"
                "    print('[Drive] Đang kết nối tới Google Drive...')\n"
                "    drive.mount('/content/drive')\n"
                "except Exception as exc:\n"
                "    print(f'[Drive] Không thể kết nối Google Drive: {exc}')\n\n"
                "# Kiểm tra đường dẫn checkpoint\n"
                "model_load_path = CFG.get(\"checkpoint_path\", CFG[\"model_name\"])\n"
                "if model_load_path.startswith(\"/content/drive\") and not os.path.exists(model_load_path):\n"
                "    print(f'[Warn] Không tìm thấy checkpoint tại {model_load_path}. Sẽ sử dụng model gốc mặc định: {CFG[\"model_name\"]}')\n"
                "    model_load_path = CFG[\"model_name\"]\n"
                "else:\n"
                "    print(f'[Load] Sẽ tải mô hình và tokenizer từ: {model_load_path}')\n\n"
                "tokenizer = AutoTokenizer.from_pretrained(model_load_path, use_fast=CFG.get(\"use_fast\", False))\n"
            )
            cell["source"] = [line + "\n" for line in new_source.split("\n")]
            if cell["source"][-1] == "\n":
                cell["source"].pop()

        # 3. Update Model cell
        elif 'model = AutoModelForSeq2SeqLM.from_pretrained(CFG["model_name"])' in source_str:
            print(f"Found Model cell at index {idx}")
            
            # Replace model loading line (0 leading spaces)
            source_str = source_str.replace(
                'model = AutoModelForSeq2SeqLM.from_pretrained(CFG["model_name"])',
                '# Load model từ checkpoint_path nếu hợp lệ (đã gán vào model_load_path ở bước Tokenizer)\nmodel = AutoModelForSeq2SeqLM.from_pretrained(model_load_path)'
            )
            
            # Replace the resume_from_checkpoint block completely (0 leading spaces for root block statements)
            target_resume_block = (
                "# Tiếp tục huấn luyện từ checkpoint cũ nếu có\n"
                "resume_from_checkpoint = None\n"
                "if not CFG.get(\"no_resume\", False):\n"
                "    checkpoint_dir = Path(training_output_dir)\n"
                "    if checkpoint_dir.exists():\n"
                "        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith(\"checkpoint-\")]\n"
                "        if checkpoints:\n"
                "            resume_from_checkpoint = True\n"
                "            print(f\"[Trainer] Phát hiện checkpoint cũ tại {training_output_dir}. Sẽ tiếp tục huấn luyện từ checkpoint mới nhất...\")\n"
                "        else:\n"
                "            print(f\"[Trainer] Không tìm thấy checkpoint cũ trong {training_output_dir}. Huấn luyện từ đầu.\")\n"
                "    else:\n"
                "        print(f\"[Trainer] Chưa có thư mục checkpoints. Huấn luyện từ đầu.\")"
            )
            
            replacement_resume_block = (
                "# Tiếp tục huấn luyện từ checkpoint cũ nếu có\n"
                "resume_from_checkpoint = None\n"
                "if model_load_path.startswith(\"/content/drive\") and os.path.exists(model_load_path):\n"
                "    resume_from_checkpoint = model_load_path\n"
                "    print(f\"[Trainer] Phát hiện checkpoint từ Drive. Sẽ tiếp tục huấn luyện từ: {resume_from_checkpoint}\")\n"
                "\n"
                "if resume_from_checkpoint is None and not CFG.get(\"no_resume\", False):\n"
                "    checkpoint_dir = Path(training_output_dir)\n"
                "    if checkpoint_dir.exists():\n"
                "        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith(\"checkpoint-\")]\n"
                "        if checkpoints:\n"
                "            resume_from_checkpoint = True\n"
                "            print(f\"[Trainer] Phát hiện checkpoint cũ tại {training_output_dir}. Sẽ tiếp tục huấn luyện từ checkpoint mới nhất...\")\n"
                "        else:\n"
                "            print(f\"[Trainer] Không tìm thấy checkpoint cũ trong {training_output_dir}. Huấn luyện từ đầu.\")\n"
                "    else:\n"
                "        print(f\"[Trainer] Chưa có thư mục checkpoints. Huấn luyện từ đầu.\")"
            )
            
            source_str = source_str.replace(target_resume_block, replacement_resume_block)
            
            cell["source"] = [line + "\n" for line in source_str.split("\n")]
            if cell["source"][-1] == "\n":
                cell["source"].pop()

# Save the updated notebook
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"Successfully created: {output_path}")
