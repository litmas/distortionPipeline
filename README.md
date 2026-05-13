# Distortion Pipeline

This project is a lightweight, deterministic image-distortion pipeline designed for experiment workflows.
It converts an image dataset into JSONL, expands recipe definitions into job manifests, applies
reproducible augmentations, and writes outputs with caching.

The code is intentionally small and editable: recipes and experiment settings are plain JSON/YAML files.

The pipeline is driven by a single experiment config (`configs/experiments/exp.yaml`): add/remove recipes there and regenerate the jobs manifest.
For DeepfakeBench-style datasets, the pipeline now preserves `frames/<video>/<frame>.png` layout, keeps frame ordering stable, and can emit detector-ready JSON files for each distorted experiment branch.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```
Use:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
or your preferred virtual environment flow.

3. Convert dataset to JSONL:

```bash
./.venv/bin/python -m scripts.images_to_jsonl \
  --input_dir datasets \
  --output manifests/dataset.jsonl
```

4. Generate jobs:

```bash
./.venv/bin/python -m scripts.generate_manifest \
  --dataset_jsonl manifests/dataset.jsonl \
  --recipes_dir configs/recipes \
  --experiment_yaml configs/experiments/exp.yaml \
  --out manifests/jobs.jsonl
```

5. Run distortion pipeline:

```bash
./.venv/bin/python -m scripts.run_distortions \
  --manifest manifests/jobs.jsonl \
  --cache_dir data/cache/distorted \
  --out_dir data/outputs/distorted \
  --write_augmented_manifest manifests/jobs_with_paths.jsonl \
  --write_deepfakebench_json_dir manifests/deepfakebench
```

6. Validate one detector JSON before using it in DeepfakeBench:

```bash
./.venv/bin/python -m scripts.validate_deepfakebench_json \
  --json_path manifests/deepfakebench/<recipe_instance_id>__v0.json
```

## Restart from a clean slate (recommended after config/recipe edits)

```bash
# Remove generated manifests
rm -f manifests/jobs.jsonl manifests/jobs_with_paths.jsonl
rm -f manifests/jobs_debug.jsonl manifests/jobs_with_paths_debug.jsonl
rm -f manifests/jobs_sample.jsonl manifests/jobs_with_paths_sample.jsonl

# Remove old outputs for selected recipe classes (optional, replace recipe id as needed)
rm -rf data/outputs/distorted/tiktok_hybrid_v1
rm -rf data/outputs/distorted/snapchat_text_overlay_v1
rm -rf data/outputs/distorted/resize_256_v1
rm -rf data/outputs/distorted/gaussian_blur_v1
rm -rf data/outputs/distorted/noise_v1
rm -rf data/outputs/distorted/lut_warm_v1
rm -rf data/outputs/distorted/jpeg_compress_v1

# Optional full cache clear
rm -rf data/cache/distorted
mkdir -p data/cache/distorted
```

Then regenerate and run again with the three core commands above.

### Optional: clear one old recipe completely

If you remove `instagram_reels_ui_v1` from `configs/experiments/exp.yaml`, you should also remove old files to avoid confusion:

```bash
rm -rf data/outputs/distorted/instagram_reels_ui_v1
rm -rf data/cache/distorted/instagram_reels_ui_v1
```

Then regenerate jobs and run.

## Project layout

- `configs/`
  - `recipes/`: recipe JSON files (what to run)
  - `experiments/`: experiment YAML files (which recipes + run settings)
- `manifests/`: intermediate dataset/job manifests (`.jsonl`) created by the scripts
- `src/distortions/`: all distortion modules and base interface
  - `ui_overlay/`: overlay distortions
- `src/pipeline/`: registry, seeds, cache key logic, manifest IO, and apply orchestration
- `scripts/`: CLI entry points
- `tests/`: unit tests (registry, seeding, blur)
- `data/outputs/`: generated output images
- `data/cache/`: reusable cache files keyed by source + steps + seed + variant

## Core flow

1. `scripts.images_to_jsonl` scans `--input_dir` and writes records:
   - flat-image fields: `image_id`, `label`, `path`
   - DeepfakeBench frame fields when a path contains `frames/`: `dataset_name`, `video_id`, `frame_id`, `relative_path`, `sample_id`
   - auxiliary paths when present: `landmark_path`, `mask_path`
   - optional `image_base64` when `--embed_base64` is used
2. `scripts.generate_manifest` reads:
   - dataset JSONL (`manifests/dataset.jsonl`)
   - all recipe JSON files in `configs/recipes/`
   - experiment YAML (`configs/experiments/exp.yaml`)

   It expands param grids in recipe steps and creates one job per frame/recipe/variant.
   For video datasets it also carries `relative_path`, `video_key`, and a stable `sample_id` into the jobs manifest.
3. `scripts.run_distortions` reads each job, computes deterministic seed + cache key,
   applies distortion steps via `src.pipeline.apply_distortions.apply_steps`, and writes:
   - distorted frames under `--out_dir/<recipe_instance_id>/variant_<n>/<original relative frame path>`
   - copied `landmarks/` and `masks/` into the same mirrored tree when those files exist
   - augmented job line to `--write_augmented_manifest`
   - optional DeepfakeBench detector JSONs to `--write_deepfakebench_json_dir`

## Distortion modules currently available

- `jpeg`: JPEG re-compress with `quality` (1-100)
- `resize`: width/height or `scale`
- `gaussian_blur`: blur with `sigma`
- `noise`: Gaussian noise with `std`
- `lut_filter`: lightweight color transforms (`warm_01`, `cool_01`, `vintage_01`)
- `text_overlay`: draw configurable text
- `emoji_overlay`: place sticker PNG with optional random position/scale/rotation (future extension ready)
- `tiktok_ui_overlay`: available when included in active experiment recipes.

All distortions implement the same interface in `src/distortions/base.py`:

- `name`: module key used in recipe `steps`
- `validate_params(...)`: per-distortion schema checks
- `apply(image, rng, params)`: transform function

The registry at `src/pipeline/registry.py` maps string names to classes used by the pipeline.

## Recipe format

Each recipe is a JSON file in `configs/recipes/<recipe_id>.json`:

```json
{
  "recipe_id": "example_v1",
  "label": "example",
  "steps": [
    {"name": "lut_filter", "params": {"preset": "warm_01", "strength": 0.7}},
    {"name": "jpeg", "params": {"quality": 60}}
  ]
}
```

Lists in params are treated as a grid and expanded.

Example grid in `params`:

```json
{"name": "jpeg", "params": {"quality": [95, 75, 60]}}
```

With this recipe plus `variants: 2`, each image can yield multiple outputs.

## Experiment format

`configs/experiments/exp.yaml` controls what is generated:

```yaml
global_seed: 12345
seed_scope: video
variants: 1
recipes:
  - recipe_id: tiktok_hybrid_v1
  - recipe_id: snapchat_text_overlay_v1
images:
  include_labels: ["real", "fake"]
  max_images_per_label: 50
```

- `global_seed`: seed root used with image id + recipe id + variant
- `seed_scope`: `frame` or `video`
- `variants`: per-job replicate count
- `recipes`: list of recipe ids to include
- `images.include_labels`: optional label filter
- `images.max_images_per_label`: optional cap per label

## Determinism and caching

- Seed formula: `stable_hash(global_seed, sample_identity, recipe_id, variant)` in `src/pipeline/seeding.py`
- For DeepfakeBench-style data, set `seed_scope: video` to keep random placements consistent across all frames in the same video.
- Cache key: hash of `src_path`, normalized steps, `seed`, and `variant`
- If a cache file exists, it is reused and `cache_hit=true` is written into the augmented manifest.

## Settings menu (easy editing)

Use this instead of manually opening every JSON/YAML file:

```bash
./.venv/bin/python -m scripts.settings_menu
```

Optional:

```bash
./.venv/bin/python -m scripts.settings_menu \
  --experiment configs/experiments/exp.yaml \
  --recipes-dir configs/recipes
```

Menu options:

- Edit experiment values (`global_seed`, `variants`, labels, active recipes)
- Edit recipe JSON files (labels and step params)
- Save or exit without saving

This is the easiest way to switch which recipes run and tune parameters before re-running
`generate_manifest`.

## Command reference

- Convert dataset:

```bash
./.venv/bin/python -m scripts.images_to_jsonl --input_dir <dataset_dir> --output manifests/dataset.jsonl
```

- Generate jobs:

```bash
./.venv/bin/python -m scripts.generate_manifest --dataset_jsonl manifests/dataset.jsonl --recipes_dir configs/recipes --experiment_yaml <exp_yaml> --out manifests/jobs.jsonl
```

- Run distortions:

```bash
./.venv/bin/python -m scripts.run_distortions --manifest manifests/jobs.jsonl --cache_dir data/cache/distorted --out_dir data/outputs/distorted --write_augmented_manifest manifests/jobs_with_paths.jsonl --write_deepfakebench_json_dir manifests/deepfakebench
```

- Validate detector JSON:

```bash
./.venv/bin/python -m scripts.validate_deepfakebench_json --json_path manifests/deepfakebench/<recipe_instance_id>__v0.json
```

## Settings menu (easy config editing)

Run:
```bash
./.venv/bin/python -m scripts.settings_menu --experiment configs/experiments/exp.yaml --recipes-dir configs/recipes
```

What you can change from the menu:
- `global_seed`
- `variants`
- `images.include_labels`
- `images.max_images_per_label`
- active recipes list (select by file number)
- recipe labels and step parameters for any recipe JSON

After saving in the menu:
1) regenerate `manifests/jobs.jsonl`, then
2) run `scripts.run_distortions`.

This is the preferred workflow for turning pipelines on/off and tuning parameters without manual file edits.

### Remove a recipe from the active pipeline

1) Remove the recipe entry from `configs/experiments/exp.yaml`.
2) Clear affected outputs/manifests (see restart section).
3) Regenerate jobs.
4) Verify it is absent:

```bash
./.venv/bin/python - <<'PY'
import json
with open("manifests/jobs.jsonl", "r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            rec = json.loads(line)
            if rec.get("recipe_id") == "instagram_reels_ui_v1":
                raise SystemExit("Still present!")
print("No instagram_reels_ui_v1 jobs.")
PY
```

## Troubleshooting

- Missing files: verify paths passed to scripts are relative to repository root or absolute.
- Empty jobs manifest: check `configs/experiments/exp.yaml` recipe IDs and label filter.
- `text_overlay` `word_pool` was serialized as a scalar string in old manifests: regenerate manifests from scratch; the new generator keeps `word_pool` as list.
- `python: command not found`: activate your virtualenv and run `./.venv/bin/python`.
- If outputs look poor from UI overlays, adjust overlay templates and keep overlays same resolution as input.
- For DeepfakeBench frame datasets, avoid using bare frame stems as identity. The pipeline now uses `sample_id` and `video_key` automatically to prevent collisions like `000.png` appearing in many videos.
- For unlabeled test-only datasets such as DFDC test, clear `images.include_labels` in `configs/experiments/exp.yaml` so `unknown` labels are not filtered out.

## Notes

- JSONL is line-delimited plain JSON and can be streamed/edited safely.
- Outputs are written as PNG to avoid added JPEG artifacts from intermediate saves.
- Tests are intentionally minimal; current tests verify seed logic, registry loading and blur behavior:

```bash
pytest
```
