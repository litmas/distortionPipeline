from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.apply_distortions import apply_steps  # noqa: E402
from src.pipeline.caching import cache_path, compute_cache_key  # noqa: E402
from src.pipeline.deepfakebench import copy_if_missing, write_deepfakebench_json  # noqa: E402
from src.pipeline.manifest_io import read_jsonl  # noqa: E402
from src.pipeline.seeding import compute_seed  # noqa: E402


def load_image(job: Dict[str, Any]) -> Image.Image:
    if job.get("image_base64"):
        data = base64.b64decode(job["image_base64"])
        return Image.open(BytesIO(data))
    src_path = job.get("src_path") or job.get("path")
    if not src_path:
        raise ValueError("job missing src_path/path and image_base64")
    path = Path(src_path)
    if not path.exists():
        raise FileNotFoundError(f"source image not found: {path}")
    return Image.open(path)


def build_output_path(
    out_dir: Path,
    recipe_id: str,
    label: str,
    image_id: str,
    recipe_instance_id: str,
    variant: int,
    relative_path: str | None = None,
    ext: str = ".png",
) -> Path:
    if relative_path:
        path = out_dir / recipe_instance_id / f"variant_{variant}" / Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    safe_label = label or "unknown"
    folder = out_dir / recipe_id / safe_label
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{image_id}__{recipe_instance_id}__v{variant}{ext}"
    return folder / filename


def build_auxiliary_output_path(
    out_dir: Path,
    recipe_instance_id: str,
    variant: int,
    relative_path: str | None,
) -> Path | None:
    if not relative_path:
        return None
    path = out_dir / recipe_instance_id / f"variant_{variant}" / Path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def publish_output(source: Path, destination: Path, publish_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination == source:
        return "source"

    if destination.exists():
        try:
            if destination.samefile(source):
                return "existing"
        except OSError:
            pass
        destination.unlink()

    if publish_mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copyfile(source, destination)
            return "copy_fallback"

    shutil.copyfile(source, destination)
    return "copy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distortion pipeline.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--write_augmented_manifest", required=True, type=Path)
    parser.add_argument("--write_deepfakebench_json_dir", type=Path, default=None)
    parser.add_argument("--publish_mode", choices=("copy", "hardlink"), default="hardlink")
    parser.add_argument("--skip_auxiliary_assets", action="store_true")
    parser.add_argument("--skip_errors", action="store_true")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.write_augmented_manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.write_deepfakebench_json_dir is not None:
        args.write_deepfakebench_json_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    detector_records: dict[tuple[str, int], list[Dict[str, Any]]] = defaultdict(list)

    with args.write_augmented_manifest.open("w", encoding="utf-8") as out_handle:
        for job in read_jsonl(args.manifest):
            total += 1
            try:
                image_id = job.get("image_id")
                label = job.get("label", "unknown")
                recipe_id = job.get("recipe_id")
                recipe_instance_id = job.get("recipe_instance_id", recipe_id)
                steps = job.get("steps", [])
                variant = int(job.get("variant", 0))
                global_seed = int(job.get("global_seed", 0))
                seed_identity = str(job.get("sample_id") or image_id)
                relative_path = job.get("distorted_relative_path") or job.get("relative_path")

                seed = compute_seed(global_seed, seed_identity, recipe_id, variant)
                src_key = job.get("src_path") or job.get("path") or image_id
                cache_key = compute_cache_key(src_key, steps, seed, variant)
                cache_file = cache_path(args.cache_dir, cache_key)
                cache_hit = cache_file.exists()

                if not cache_hit:
                    image = load_image(job)
                    rng = np.random.default_rng(seed)
                    output = apply_steps(image, steps, rng)
                    output.save(cache_file)

                out_path = build_output_path(
                    args.out_dir,
                    recipe_id,
                    label,
                    image_id,
                    recipe_instance_id,
                    variant,
                    relative_path=relative_path,
                )
                publish_status = publish_output(cache_file, out_path, args.publish_mode)

                if not args.skip_auxiliary_assets:
                    landmark_out = build_auxiliary_output_path(
                        args.out_dir,
                        recipe_instance_id,
                        variant,
                        job.get("landmark_relative_path"),
                    )
                    mask_out = build_auxiliary_output_path(
                        args.out_dir,
                        recipe_instance_id,
                        variant,
                        job.get("mask_relative_path"),
                    )
                    if landmark_out is not None:
                        copy_if_missing(job.get("landmark_path"), landmark_out)
                    if mask_out is not None:
                        copy_if_missing(job.get("mask_path"), mask_out)

                augmented = dict(job)
                augmented.update(
                    {
                        "seed": seed,
                        "cache_key": cache_key,
                        "cache_hit": cache_hit,
                        "distorted_path": str(out_path),
                        "publish_status": publish_status,
                        "steps": steps,
                    }
                )
                out_handle.write(json_line(augmented))
                if args.write_deepfakebench_json_dir is not None:
                    detector_records[(recipe_instance_id, variant)].append(augmented)
                written += 1
            except Exception as exc:  # noqa: BLE001
                message = f"Error processing job {job.get('job_id')}: {exc}"
                if args.skip_errors:
                    print(message, file=sys.stderr)
                    continue
                raise RuntimeError(message) from exc

    if args.write_deepfakebench_json_dir is not None:
        for (recipe_instance_id, variant), records in detector_records.items():
            json_path = args.write_deepfakebench_json_dir / f"{recipe_instance_id}__v{variant}.json"
            write_deepfakebench_json(json_path, records)

    print(f"Processed {total} jobs, wrote {written} records.")


def json_line(record: Dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    main()
