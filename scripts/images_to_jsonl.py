from __future__ import annotations

import argparse
import base64
import csv
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.manifest_io import write_jsonl  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_labels_csv(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("labels_csv must include headers")
        fieldnames = {name.lower() for name in reader.fieldnames}
        if "image_id" in fieldnames:
            key_field = next(name for name in reader.fieldnames if name.lower() == "image_id")
        elif "filename" in fieldnames:
            key_field = next(name for name in reader.fieldnames if name.lower() == "filename")
        else:
            raise ValueError("labels_csv must include 'image_id' or 'filename' column")
        label_field = next((name for name in reader.fieldnames if name.lower() == "label"), None)
        if label_field is None:
            raise ValueError("labels_csv must include 'label' column")
        for row in reader:
            key = Path(row[key_field]).stem
            mapping[key] = row[label_field]
    return mapping


def derive_label(path: Path, input_dir: Path) -> str:
    rel = path.relative_to(input_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return "unknown"


def build_records(
    input_dir: Path,
    labels_csv: Path | None,
    embed_base64: bool,
    skip_unlabeled: bool,
):
    mapping = load_labels_csv(labels_csv) if labels_csv else None
    paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    paths.sort()
    for path in paths:
        image_id = path.stem
        if mapping is not None:
            label = mapping.get(image_id)
            if label is None:
                if skip_unlabeled:
                    continue
                raise ValueError(f"Missing label for image_id '{image_id}'")
        else:
            label = derive_label(path, input_dir)
        record = {
            "image_id": image_id,
            "label": label,
            "path": str(path.resolve()),
        }
        if embed_base64:
            data = path.read_bytes()
            record["image_base64"] = base64.b64encode(data).decode("utf-8")
        yield record


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert image dataset to JSONL manifest.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--labels_csv", type=Path, default=None)
    parser.add_argument("--embed_base64", action="store_true")
    parser.add_argument("--skip_unlabeled", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")

    records = list(build_records(args.input_dir, args.labels_csv, args.embed_base64, args.skip_unlabeled))
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
