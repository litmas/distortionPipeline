from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.deepfakebench import build_distorted_relative_path, job_identity  # noqa: E402
from src.pipeline.manifest_io import read_jsonl, write_jsonl  # noqa: E402


def short_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def is_rgb_list(values: Any) -> bool:
    return (
        isinstance(values, list)
        and len(values) == 3
        and all(isinstance(v, (int, float)) for v in values)
        and all(0 <= v <= 255 for v in values)
    )


def is_rgb_list_series(values: Any) -> bool:
    return isinstance(values, list) and len(values) > 1 and all(is_rgb_list(v) for v in values)


def is_grid_parameter(key: str, value: Any) -> bool:
    if key in {"word_pool", "styles"}:
        return False
    return isinstance(value, list) and not is_rgb_list(value) and not is_rgb_list_series(value)


def expand_steps(steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grid_params: List[tuple[int, str, List[Any]]] = []
    for idx, step in enumerate(steps):
        params = step.get("params", {})
        for key, value in params.items():
            if is_grid_parameter(key, value):
                grid_params.append((idx, key, value))

    if not grid_params:
        return [steps]

    combos = product(*[values for (_, _, values) in grid_params])
    expanded: List[List[Dict[str, Any]]] = []
    for combo in combos:
        new_steps = deepcopy(steps)
        for (idx, key, _), val in zip(grid_params, combo):
            new_steps[idx].setdefault("params", {})
            new_steps[idx]["params"][key] = val
        expanded.append(new_steps)
    return expanded


def load_experiment(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("experiment YAML must be a mapping")
    return data


def _collect_video_keys_from_node(node: Any, output: Set[str]) -> None:
    if not isinstance(node, dict):
        return
    frames = node.get("frames")
    if isinstance(frames, list) and frames:
        first_frame = str(frames[0]).replace("\\", "/")
        output.add(str(Path(first_frame).parent))
        return
    for value in node.values():
        _collect_video_keys_from_node(value, output)


def load_video_keys_from_deepfakebench_json(path: Path, include_splits: Set[str] | None = None) -> Set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    video_keys: Set[str] = set()
    if include_splits:
        for dataset_body in payload.values():
            if not isinstance(dataset_body, dict):
                continue
            for label_body in dataset_body.values():
                if not isinstance(label_body, dict):
                    continue
                for split_name, split_payload in label_body.items():
                    if split_name in include_splits:
                        _collect_video_keys_from_node(split_payload, video_keys)
    else:
        _collect_video_keys_from_node(payload, video_keys)
    return video_keys


def _record_sort_key(record: Dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("dataset_name", "")),
        str(record.get("label", "unknown")),
        str(record.get("video_key") or record.get("sample_id") or record.get("image_id", "")),
        str(record.get("relative_path") or record.get("frame_id") or record.get("image_id", "")),
    )


def _dataset_label_key(record: Dict[str, Any]) -> tuple[str, str]:
    return str(record.get("dataset_name", "")), str(record.get("label", "unknown"))


def _video_group_key(record: Dict[str, Any]) -> str:
    return str(record.get("video_key") or record.get("sample_id") or record.get("image_id"))


def filter_images(
    records: Iterable[Dict[str, Any]],
    include_labels: List[str] | None,
    max_per_label: int | None,
    include_datasets: List[str] | None = None,
    include_splits: List[str] | None = None,
    max_videos_per_label: int | None = None,
    include_video_keys: Set[str] | None = None,
):
    records = list(records)

    if include_datasets:
        allowed_datasets = set(include_datasets)
        records = [r for r in records if r.get("dataset_name") in allowed_datasets]

    if include_video_keys is not None:
        records = [r for r in records if str(r.get("video_key") or "") in include_video_keys]

    if include_splits:
        allowed_splits = set(include_splits)
        records = [r for r in records if r.get("split") in allowed_splits]

    if include_labels:
        allowed_labels = set(include_labels)
        records = [r for r in records if r.get("label") in allowed_labels]

    records.sort(key=_record_sort_key)

    if max_videos_per_label is not None:
        grouped_videos: Dict[tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}
        for record in records:
            label_key = _dataset_label_key(record)
            grouped_videos.setdefault(label_key, {}).setdefault(_video_group_key(record), []).append(record)

        filtered_by_video: List[Dict[str, Any]] = []
        for label_key, videos in grouped_videos.items():
            del label_key
            for video_key in sorted(videos)[:max_videos_per_label]:
                filtered_by_video.extend(videos[video_key])
        records = sorted(filtered_by_video, key=_record_sort_key)

    if max_per_label is None:
        return records

    grouped_records: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        grouped_records.setdefault(_dataset_label_key(record), []).append(record)

    filtered: List[Dict[str, Any]] = []
    for _, items in sorted(grouped_records.items()):
        filtered.extend(items[:max_per_label])
    return filtered


def build_jobs(
    dataset_records: List[Dict[str, Any]],
    recipes: List[Dict[str, Any]],
    global_seed: int,
    variants: int,
    seed_scope: str,
):
    jobs: List[Dict[str, Any]] = []
    for record in dataset_records:
        image_id = record.get("image_id")
        label = record.get("label")
        src_path = record.get("path")
        image_base64 = record.get("image_base64")

        for recipe in recipes:
            recipe_id = recipe["recipe_id"]
            recipe_label = recipe.get("label")
            recipe_seed_key = recipe.get("job_seed_key")
            if recipe_seed_key:
                identity = str(record.get(recipe_seed_key) or record.get("sample_id") or image_id)
            else:
                identity = job_identity(record, seed_scope)
            base_steps = recipe["steps"]
            for steps in expand_steps(base_steps):
                recipe_hash = short_hash(steps)
                recipe_instance_id = f"{recipe_id}__{recipe_hash}"
                for variant in range(variants):
                    job_payload = {
                        "sample_id": identity,
                        "recipe_instance_id": recipe_instance_id,
                        "variant": variant,
                    }
                    job_id = short_hash(job_payload)
                    job = {
                        "job_id": job_id,
                        "image_id": image_id,
                        "sample_id": identity,
                        "label": label,
                        "src_path": src_path,
                        "recipe_id": recipe_id,
                        "recipe_instance_id": recipe_instance_id,
                        "recipe_label": recipe_label,
                        "steps": steps,
                        "variant": variant,
                        "global_seed": global_seed,
                    }
                    for key in (
                        "dataset_name",
                        "split",
                        "video_id",
                        "video_key",
                        "paired_video_key",
                        "frame_id",
                        "relative_path",
                        "landmark_path",
                        "landmark_relative_path",
                        "mask_path",
                        "mask_relative_path",
                    ):
                        if key in record:
                            job[key] = record[key]
                    distorted_relative_path = build_distorted_relative_path(record)
                    if distorted_relative_path:
                        job["distorted_relative_path"] = distorted_relative_path
                    if image_base64:
                        job["image_base64"] = image_base64
                    jobs.append(job)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate distortion jobs manifest JSONL.")
    parser.add_argument("--dataset_jsonl", required=True, type=Path)
    parser.add_argument("--recipes_dir", required=True, type=Path)
    parser.add_argument("--experiment_yaml", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    dataset_records = list(read_jsonl(args.dataset_jsonl))
    experiment = load_experiment(args.experiment_yaml)

    global_seed = int(experiment.get("global_seed", 0))
    variants = int(experiment.get("variants", 1))
    seed_scope = str(experiment.get("seed_scope", "frame"))
    recipe_specs = experiment.get("recipes", [])
    if not recipe_specs:
        raise ValueError("experiment must include recipes")

    recipes: List[Dict[str, Any]] = []
    for spec in recipe_specs:
        recipe_id = spec.get("recipe_id")
        if not recipe_id:
            raise ValueError("each recipe entry must include recipe_id")
        recipe_path = args.recipes_dir / f"{recipe_id}.json"
        if not recipe_path.exists():
            raise FileNotFoundError(f"recipe not found: {recipe_path}")
        with recipe_path.open("r", encoding="utf-8") as handle:
            recipe = json.load(handle)
        recipes.append(recipe)

    images_cfg = experiment.get("images", {})
    include_labels = images_cfg.get("include_labels")
    include_datasets = images_cfg.get("include_datasets")
    include_splits = images_cfg.get("include_splits")
    max_images_per_label = images_cfg.get("max_images_per_label")
    max_videos_per_label = images_cfg.get("max_videos_per_label")
    include_video_keys_json = images_cfg.get("include_video_keys_json")
    include_video_keys_splits = images_cfg.get("include_video_keys_splits")

    include_video_keys = None
    if include_video_keys_json:
        include_video_keys_path = Path(include_video_keys_json)
        if not include_video_keys_path.exists():
            raise FileNotFoundError(f"include_video_keys_json not found: {include_video_keys_path}")
        include_video_keys = load_video_keys_from_deepfakebench_json(
            include_video_keys_path,
            include_splits=set(include_video_keys_splits) if include_video_keys_splits else None,
        )

    filtered = filter_images(
        dataset_records,
        include_labels,
        max_images_per_label,
        include_datasets=include_datasets,
        include_splits=include_splits,
        max_videos_per_label=max_videos_per_label,
        include_video_keys=include_video_keys,
    )
    jobs = build_jobs(filtered, recipes, global_seed, variants, seed_scope)
    write_jsonl(args.out, jobs)
    print(f"Wrote {len(jobs)} jobs to {args.out}")


if __name__ == "__main__":
    main()
