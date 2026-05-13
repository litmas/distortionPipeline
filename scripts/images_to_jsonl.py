from __future__ import annotations

import argparse
import base64
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.deepfakebench import build_frame_record, is_image_path  # noqa: E402


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


def iter_candidate_paths(input_dir: Path) -> Iterator[Path]:
    frame_dirs = sorted(
        path for path in input_dir.rglob("frames") if path.is_dir() and ".venv" not in path.parts
    )
    if frame_dirs:
        for frame_dir in frame_dirs:
            for path in sorted(frame_dir.rglob("*")):
                if path.is_file() and is_image_path(path):
                    yield path
        return

    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and is_image_path(path):
            yield path


def build_records(
    input_dir: Path,
    labels_csv: Path | None,
    embed_base64: bool,
    skip_unlabeled: bool,
) -> Iterable[dict]:
    mapping = load_labels_csv(labels_csv) if labels_csv else None
    for path in iter_candidate_paths(input_dir):
        record = build_frame_record(path, input_dir)
        image_id = str(record.get("sample_id") or path.stem)
        if mapping is not None:
            label = mapping.get(image_id)
            if label is None:
                label = mapping.get(path.stem)
            if label is None:
                if skip_unlabeled:
                    continue
                raise ValueError(f"Missing label for image_id '{image_id}'")
            record["label"] = label
        else:
            record["label"] = record.get("label") or derive_label(path, input_dir)
        record["image_id"] = image_id
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

    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in build_records(args.input_dir, args.labels_csv, args.embed_base64, args.skip_unlabeled):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 5000 == 0:
                print(f"Indexed {count} images...", file=sys.stderr)

    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
