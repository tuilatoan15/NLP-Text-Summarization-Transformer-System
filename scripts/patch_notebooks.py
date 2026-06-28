"""Patch BARTPho and mT5 notebooks: update dataset + fix column detection for VietNews (abstract col)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OLD_DATASET = "8Opt/vietnamese-summarization-dataset-0001"
NEW_DATASET = "nam194/vietnews"
OLD_PARQUET_DIR = "/content/vietnamese-summarization-dataset-0001"
NEW_PARQUET_DIR = "/content/vietnews"

# The notebooks detect target_col but miss 'abstract' - add it to candidates
OLD_TARGET_DETECT = (
    'target_col = next((col for col in ["summary", "title", "headline"] if col in available_cols), None)'
)
NEW_TARGET_DETECT = (
    'target_col = next((col for col in ["abstract", "summary", "title", "headline"] if col in available_cols), None)'
)

OLD_INPUT_DETECT = (
    'input_col = next((col for col in ["document", "article", "text", "content"] if col in available_cols), None)'
)
NEW_INPUT_DETECT = (
    'input_col = next((col for col in ["article", "document", "text", "content"] if col in available_cols), None)'
)

NOTEBOOKS = [
    "Colab_BARTPho_Training_Playbook.ipynb",
    "Colab_mT5_Training_Playbook.ipynb",
    "Colab_Training_Playbook.ipynb",
]

for nb_name in NOTEBOOKS:
    nb_path = ROOT / "notebooks" / nb_name
    if not nb_path.exists():
        print(f"[SKIP] {nb_name} - not found")
        continue

    with open(nb_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    original = content

    # 1. Update dataset name
    content = content.replace(OLD_DATASET, NEW_DATASET)
    content = content.replace(OLD_PARQUET_DIR, NEW_PARQUET_DIR)
    # Fix escaped version in JSON
    content = content.replace(
        "8Opt\\/vietnamese-summarization-dataset-0001",
        "nam194\\/vietnews"
    )

    # 2. Fix column detection — add 'abstract' as priority
    content = content.replace(OLD_TARGET_DETECT, NEW_TARGET_DETECT)
    content = content.replace(OLD_INPUT_DETECT, NEW_INPUT_DETECT)

    # 3. Fix the "Bộ dữ liệu mục tiêu" description in markdown
    content = content.replace(
        "`8Opt/vietnamese-summarization-dataset-0001` (Gồm 19,525 mẫu dữ liệu tóm tắt tiếng Việt sạch)",
        "`nam194/vietnews` (VietNews — 143,000+ mẫu tin tức tiếng Việt từ báo Tuổi Trẻ, VnExpress, Người Đưa Tin)"
    )

    if content != original:
        with open(nb_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[UPDATED] {nb_name}")
    else:
        print(f"[no change] {nb_name}")

print("Patch complete.")
