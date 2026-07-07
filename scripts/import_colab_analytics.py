#!/usr/bin/env python
"""
Import VietNews dataset analytics exported from Google Colab.

Usage:
    python scripts/import_colab_analytics.py vietnews_analytics_colab.zip
    python scripts/import_colab_analytics.py --zip path/to/vietnews_analytics_colab.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

REQUIRED_JSON = (
    "metadata.json",
    "dataset_overview.json",
    "dataset_statistics.json",
    "vocabulary.json",
    "compression_statistics.json",
    "correlation.json",
    "word_frequency.json",
    "length_distribution.json",
    "dataset_quality.json",
    "training_statistics.json",
    "rouge_baseline.json",
    "charts_index.json",
    "dataset_analytics_bundle.json",
)

OPTIONAL_JSON = (
    "token_statistics.json",
    "category_stats.json",
    "extractive_metrics.json",
)


def _fmt_count(value) -> str:
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    return str(value) if value is not None else "—"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Colab dataset analytics zip")
    parser.add_argument("zip_path", nargs="?", help="Path to vietnews_analytics_colab.zip")
    parser.add_argument("--zip", dest="zip_opt", help="Alternative: --zip path/to/file.zip")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, do not copy files")
    return parser.parse_args()


def _find_root(extract_dir: Path) -> tuple[Path, Path]:
    """Locate analytics/ and charts/ inside extracted zip."""
    candidates = [
        (extract_dir / "analytics", extract_dir / "charts"),
        (extract_dir / "storage" / "analytics", extract_dir / "storage" / "charts"),
    ]
    for analytics_dir, charts_dir in candidates:
        if analytics_dir.is_dir():
            return analytics_dir, charts_dir if charts_dir.is_dir() else extract_dir / "charts"

    # Flat layout: JSON files at extract root
    if (extract_dir / "metadata.json").exists():
        return extract_dir, extract_dir / "charts"
    raise FileNotFoundError("Không tìm thấy thư mục analytics/ trong file zip")


def validate_metadata(meta: dict) -> list[str]:
    errors: list[str] = []
    if meta.get("source") != "colab":
        errors.append(f"metadata.source phải là 'colab' (nhận: {meta.get('source')!r})")
    if meta.get("full_dataset") is not True:
        errors.append("metadata.full_dataset phải là true (phân tích full dataset)")
    if not meta.get("generated_at"):
        errors.append("metadata.generated_at thiếu")
    if not meta.get("dataset_name"):
        errors.append("metadata.dataset_name thiếu")
    return errors


def validate_analytics_dir(analytics_dir: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    for name in REQUIRED_JSON:
        if not (analytics_dir / name).exists():
            errors.append(f"Thiếu file bắt buộc: {name}")

    meta: dict = {}
    meta_path = analytics_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"metadata.json không hợp lệ: {exc}")
        else:
            errors.extend(validate_metadata(meta))

    return meta, errors


def copy_tree(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
            count += sum(1 for f in target.rglob("*") if f.is_file())
        else:
            shutil.copy2(item, target)
            count += 1
    return count


def import_zip(zip_path: Path, *, dry_run: bool = False) -> int:
    if not zip_path.exists():
        print(f"ERROR: Không tìm thấy file: {zip_path}")
        return 1

    analytics_dst = config.ANALYTICS_DIR
    charts_dst = config.CHARTS_DIR

    with tempfile.TemporaryDirectory(prefix="colab_analytics_") as tmp:
        extract_dir = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        analytics_src, charts_src = _find_root(extract_dir)
        meta, errors = validate_analytics_dir(analytics_src)
        if errors:
            print("VALIDATION FAILED:")
            for err in errors:
                print(f"  - {err}")
            return 1

        png_count = len(list(charts_src.glob("*.png"))) if charts_src.is_dir() else 0

        if dry_run:
            print("DRY RUN — validation OK, không ghi file.")
        else:
            analytics_dst.mkdir(parents=True, exist_ok=True)
            charts_dst.mkdir(parents=True, exist_ok=True)

            # Clear old analytics JSON (keep progress.json if any)
            for f in analytics_dst.glob("*.json"):
                f.unlink()

            json_count = copy_tree(analytics_src, analytics_dst)
            if charts_src.is_dir():
                for png in charts_src.glob("*.png"):
                    shutil.copy2(png, charts_dst / png.name)
            else:
                png_count = 0
            print(f"Đã import {json_count} file JSON → {analytics_dst}")
            print(f"Đã import {png_count} biểu đồ PNG → {charts_dst}")

        print()
        print("=" * 60)
        print("  COLAB ANALYTICS IMPORT SUMMARY")
        print("=" * 60)
        print(f"  Dataset       : {meta.get('dataset_name')}")
        print(f"  Source        : {meta.get('source')}")
        print(f"  Full dataset  : {meta.get('full_dataset')}")
        print(f"  Records       : {_fmt_count(meta.get('record_count', meta.get('total_samples')))}")
        print(f"  Generated at  : {meta.get('generated_at')}")
        print(f"  Duration (s)  : {meta.get('analysis_duration_sec', '—')}")
        print(f"  Charts        : {meta.get('chart_count', png_count)}")
        print(f"  Target        : {analytics_dst}")
        print("=" * 60)
        if not dry_run:
            print("Khởi động lại backend và mở Dashboard → Dataset Analytics.")
        return 0


def main() -> int:
    args = parse_args()
    zip_path = args.zip_opt or args.zip_path
    if not zip_path:
        print("Usage: python scripts/import_colab_analytics.py vietnews_analytics_colab.zip")
        return 1
    return import_zip(Path(zip_path).resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
