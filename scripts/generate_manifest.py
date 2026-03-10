from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def expand_steps(steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grid_params: List[tuple[int, str, List[Any]]] = []
    for idx, step in enumerate(steps):
        params = step.get("params", {})
        for key, value in params.items():
            if (
                isinstance(value, list)
                and not is_rgb_list(value)
                and not is_rgb_list_series(value)
            ):
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


def filter_images(records: Iterable[Dict[str, Any]], include_labels: List[str] | None, max_per_label: int | None):
    if include_labels:
        records = [r for r in records if r.get("label") in include_labels]
    else:
        records = list(records)

    if max_per_label is None:
        return records

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record.get("label", "unknown"), []).append(record)

    filtered: List[Dict[str, Any]] = []
    for label, items in grouped.items():
        items.sort(key=lambda r: r.get("image_id", ""))
        filtered.extend(items[:max_per_label])
    return filtered


def build_jobs(dataset_records: List[Dict[str, Any]], recipes: List[Dict[str, Any]], global_seed: int, variants: int):
    jobs: List[Dict[str, Any]] = []
    for record in dataset_records:
        image_id = record.get("image_id")
        label = record.get("label")
        src_path = record.get("path")
        image_base64 = record.get("image_base64")

        for recipe in recipes:
            recipe_id = recipe["recipe_id"]
            recipe_label = recipe.get("label")
            base_steps = recipe["steps"]
            for steps in expand_steps(base_steps):
                recipe_hash = short_hash(steps)
                recipe_instance_id = f"{recipe_id}__{recipe_hash}"
                for variant in range(variants):
                    job_payload = {
                        "image_id": image_id,
                        "recipe_instance_id": recipe_instance_id,
                        "variant": variant,
                    }
                    job_id = short_hash(job_payload)
                    job = {
                        "job_id": job_id,
                        "image_id": image_id,
                        "label": label,
                        "src_path": src_path,
                        "recipe_id": recipe_id,
                        "recipe_instance_id": recipe_instance_id,
                        "recipe_label": recipe_label,
                        "steps": steps,
                        "variant": variant,
                        "global_seed": global_seed,
                    }
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
    max_images_per_label = images_cfg.get("max_images_per_label")

    filtered = filter_images(dataset_records, include_labels, max_images_per_label)
    jobs = build_jobs(filtered, recipes, global_seed, variants)
    write_jsonl(args.out, jobs)
    print(f"Wrote {len(jobs)} jobs to {args.out}")


if __name__ == "__main__":
    main()
