from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "val", "test", "validation"}
FAKE_HINTS = ("fake", "synthesis", "swap", "manip", "deepfake", "forged")
REAL_HINTS = ("real", "original", "youtube-real", "authentic")


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS and path.name != ".DS_Store"


def find_frames_index(parts: tuple[str, ...]) -> int | None:
    try:
        return parts.index("frames")
    except ValueError:
        return None


def infer_split(parts: tuple[str, ...], frames_index: int) -> str | None:
    for idx in range(frames_index - 1, -1, -1):
        token = parts[idx].lower()
        if token in SPLIT_NAMES:
            return "val" if token == "validation" else token
    return None


def infer_label(parts: tuple[str, ...], frames_index: int) -> str:
    for idx in range(frames_index - 1, -1, -1):
        token = parts[idx]
        lowered = token.lower()
        if lowered in SPLIT_NAMES:
            continue
        if lowered.startswith("method_"):
            return "fake"
        if any(hint in lowered for hint in REAL_HINTS):
            return "real"
        if any(hint in lowered for hint in FAKE_HINTS):
            return "fake"
    return "unknown"


def sanitize_sample_id(relative_path: Path) -> str:
    token = str(relative_path.with_suffix(""))
    return re.sub(r"[^A-Za-z0-9._-]+", "__", token)


def resolve_dataset_name(input_dir: Path, parts: tuple[str, ...]) -> str:
    if input_dir.name == "datasets" and parts:
        return parts[0]
    return input_dir.name


def publish_relative_path(input_dir: Path, dataset_name: str, path: Path) -> Path:
    if input_dir.name == "datasets":
        return path
    return Path(dataset_name) / path


def normalize_pair_video_id(video_id: str) -> str:
    normalized = video_id
    if normalized.endswith("_fake"):
        normalized = normalized[: -len("_fake")]
    celeb_pair_match = re.match(r"^(id\d+)_id\d+_(\d+)$", normalized)
    if celeb_pair_match:
        source_id, clip_id = celeb_pair_match.groups()
        return f"{source_id}_{clip_id}"
    return normalized


def replace_frames_component(relative_path: Path, replacement: str, suffix: str | None = None) -> Path:
    parts = list(relative_path.parts)
    frames_index = find_frames_index(tuple(parts))
    if frames_index is None:
        raise ValueError(f"relative path does not contain 'frames': {relative_path}")
    parts[frames_index] = replacement
    new_path = Path(*parts)
    if suffix is not None:
        new_path = new_path.with_suffix(suffix)
    return new_path


def build_frame_record(path: Path, input_dir: Path) -> Dict[str, Any]:
    rel = path.relative_to(input_dir)
    parts = rel.parts
    frames_index = find_frames_index(parts)
    record: Dict[str, Any] = {
        "image_id": path.stem,
        "path": str(path.resolve()),
    }

    if frames_index is None or frames_index + 2 >= len(parts):
        record["label"] = parts[0] if len(parts) > 1 else "unknown"
        dataset_name = resolve_dataset_name(input_dir, parts)
        published_rel = publish_relative_path(input_dir, dataset_name, rel)
        record["dataset_name"] = dataset_name
        record["sample_id"] = sanitize_sample_id(published_rel)
        record["relative_path"] = str(published_rel)
        return record

    dataset_name = resolve_dataset_name(input_dir, parts)
    split = infer_split(parts, frames_index)
    label = infer_label(parts, frames_index)
    video_id = parts[frames_index + 1]
    storage_frame_rel = Path(*parts)
    frame_rel = publish_relative_path(input_dir, dataset_name, storage_frame_rel)
    frame_id = path.stem
    video_key = str(frame_rel.parent)
    paired_video_key = f"{dataset_name}/{normalize_pair_video_id(video_id)}"

    record.update(
        {
            "dataset_name": dataset_name,
            "split": split,
            "label": label,
            "video_id": video_id,
            "frame_id": frame_id,
            "video_key": video_key,
            "paired_video_key": paired_video_key,
            "relative_path": str(frame_rel),
            "sample_id": sanitize_sample_id(frame_rel),
        }
    )

    landmark_storage_rel = replace_frames_component(storage_frame_rel, "landmarks", ".npy")
    landmark_path = input_dir / landmark_storage_rel
    if landmark_path.exists():
        record["landmark_path"] = str(landmark_path.resolve())
        record["landmark_relative_path"] = str(
            publish_relative_path(input_dir, dataset_name, landmark_storage_rel)
        )

    mask_storage_rel = replace_frames_component(storage_frame_rel, "masks", path.suffix)
    mask_path = input_dir / mask_storage_rel
    if mask_path.exists():
        record["mask_path"] = str(mask_path.resolve())
        record["mask_relative_path"] = str(
            publish_relative_path(input_dir, dataset_name, mask_storage_rel)
        )

    return record


def job_identity(record: Dict[str, Any], seed_scope: str) -> str:
    scope = seed_scope.lower().strip()
    if scope == "video":
        return str(record.get("video_key") or record.get("video_id") or record.get("sample_id") or record.get("image_id"))
    if scope != "frame":
        raise ValueError("seed_scope must be 'frame' or 'video'")
    return str(record.get("sample_id") or record.get("relative_path") or record.get("image_id"))


def build_distorted_relative_path(record: Dict[str, Any]) -> str | None:
    relative_path = record.get("relative_path")
    if not relative_path:
        return None
    return str(Path(relative_path))


def copy_if_missing(src: str | None, dest: Path) -> None:
    if not src:
        return
    source = Path(src)
    if not source.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(source, dest)


def collect_deepfakebench_entries(records: Iterable[Dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    frames_by_key: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for record in records:
        distorted_path = record.get("distorted_path")
        video_key = record.get("video_key")
        if not distorted_path or not video_key:
            continue

        frame_sort_key = str(record.get("relative_path") or record.get("frame_id") or record.get("image_id"))
        frames_by_key[video_key].append((frame_sort_key, distorted_path))

        if video_key not in grouped:
            grouped[video_key] = {
                "dataset_name": record.get("dataset_name"),
                "label": record.get("label"),
                "set": record.get("split"),
                "video_id": record.get("video_id"),
                "video_key": video_key,
            }

    for video_key, items in frames_by_key.items():
        items.sort(key=lambda item: item[0])
        grouped[video_key]["frames"] = [path for _, path in items]

    return grouped


def write_deepfakebench_json(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    payload = collect_deepfakebench_entries(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_missing_frame_paths(json_path: Path) -> Iterator[str]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DeepfakeBench JSON must be a mapping keyed by video id")
    for _, video_info in payload.items():
        if not isinstance(video_info, dict):
            continue
        frames = video_info.get("frames", [])
        if not isinstance(frames, list):
            raise ValueError("Each video entry must contain a list 'frames'")
        for frame_path in frames:
            if not Path(frame_path).exists():
                yield str(frame_path)
