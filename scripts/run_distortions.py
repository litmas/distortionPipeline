from __future__ import annotations

import argparse
import base64
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.apply_distortions import apply_steps  # noqa: E402
from src.pipeline.caching import cache_path, compute_cache_key  # noqa: E402
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


def build_output_path(out_dir: Path, recipe_id: str, label: str, image_id: str, recipe_instance_id: str, variant: int, ext: str = ".png") -> Path:
    safe_label = label or "unknown"
    folder = out_dir / recipe_id / safe_label
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{image_id}__{recipe_instance_id}__v{variant}{ext}"
    return folder / filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distortion pipeline.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--write_augmented_manifest", required=True, type=Path)
    parser.add_argument("--skip_errors", action="store_true")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.write_augmented_manifest.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0

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

                seed = compute_seed(global_seed, image_id, recipe_id, variant)
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
                    args.out_dir, recipe_id, label, image_id, recipe_instance_id, variant
                )
                if out_path != cache_file:
                    shutil.copyfile(cache_file, out_path)

                augmented = dict(job)
                augmented.update(
                    {
                        "seed": seed,
                        "cache_key": cache_key,
                        "cache_hit": cache_hit,
                        "distorted_path": str(out_path),
                        "steps": steps,
                    }
                )
                out_handle.write(json_line(augmented))
                written += 1
            except Exception as exc:  # noqa: BLE001
                message = f"Error processing job {job.get('job_id')}: {exc}"
                if args.skip_errors:
                    print(message, file=sys.stderr)
                    continue
                raise RuntimeError(message) from exc

    print(f"Processed {total} jobs, wrote {written} records.")


def json_line(record: Dict[str, Any]) -> str:
    import json

    return json.dumps(record, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    main()
