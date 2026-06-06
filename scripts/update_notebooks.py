"""Update all Colab notebooks to use nam194/vietnews dataset with correct column mapping."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_OLD = ["thanhnew2001/vnexpress", "ThanhChinhBK/vietnews"]
DATASET_NEW = "nam194/vietnews"


def update_notebook(path: Path) -> bool:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    original = content

    # 1. Fix dataset name
    for old in DATASET_OLD:
        content = content.replace(old, DATASET_NEW)

    # 2. Fix column mappings in CFG dict (VietNews uses article + abstract)
    content = re.sub(r'("text_col"\s*:\s*)"text"', r'\1"article"', content)
    content = re.sub(r'("summary_col"\s*:\s*)"title"', r'\1"abstract"', content)
    content = re.sub(r'("article_col"\s*:\s*)"text"', r'\1"article"', content)
    content = re.sub(r'("summary_col"\s*:\s*)"summary"', r'\1"abstract"', content)

    if content != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


if __name__ == "__main__":
    notebooks = list(ROOT.glob("*.ipynb"))
    print(f"Found {len(notebooks)} notebooks")
    for nb in notebooks:
        changed = update_notebook(nb)
        status = "[UPDATED]" if changed else "[no change]"
        print(f"  {status}: {nb.name}")
    print("Done!")
