# Distortion Pipeline (Thesis-Ready)

A clean, deterministic image distortion pipeline built with Pillow and NumPy. It converts datasets into JSONL, expands distortion recipes into job manifests, and runs a reproducible pipeline with optional caching.

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `src/distortions/` – distortion modules (jpeg, resize, blur, noise, LUT-style, overlays)
- `src/pipeline/` – registry, seeding, caching, manifest IO, apply pipeline
- `scripts/` – CLI tools
- `configs/recipes/` – recipe JSON files
- `configs/experiments/` – experiment YAML files
- `manifests/` – JSONL manifests and outputs
- `data/cache/distorted/` – cache outputs (hashed)
- `data/outputs/distorted/` – organized outputs by recipe/label

## 1) Prepare Dataset JSONL

Convert a folder of images into a JSONL manifest.

```bash
python -m scripts.images_to_jsonl \
  --input_dir /path/to/dataset \
  --output manifests/dataset.jsonl
```

Labeling options:

- Default: label is the subfolder name (e.g., `/dataset/real/*.jpg` → `label=real`)
- CSV mapping:

```bash
python -m scripts.images_to_jsonl \
  --input_dir /path/to/dataset \
  --output manifests/dataset.jsonl \
  --labels_csv /path/to/labels.csv
```

The CSV must include columns `image_id` or `filename`, and `label`.

Optional base64 embedding:

```bash
python -m scripts.images_to_jsonl \
  --input_dir /path/to/dataset \
  --output manifests/dataset.jsonl \
  --embed_base64
```

JSONL format (one line per image):

```json
{"image_id":"abc123","label":"real","path":"/abs/path/abc123.jpg"}
```

If base64 is enabled, `image_base64` is added.

## 2) Define Distortion Recipes

Each recipe lives in `configs/recipes/<recipe_id>.json`.

Example:

```json
{
  "recipe_id": "tiktok_hybrid_v1",
  "label": "hybrid",
  "steps": [
    {"name":"lut_filter","params":{"preset":"warm_01","strength":0.7}},
{"name":"tiktok_ui_overlay","params":{"template_path":"src/distortions/ui_overlay/assets/insta_ui_overlay_transp_template.png","opacity":0.9}},
    {"name":"jpeg","params":{"quality":60}}
  ]
}
```

## 3) Create Experiment YAML

Experiments select recipes and define image filters and variants.

```yaml
global_seed: 12345
variants: 3
recipes:
  - recipe_id: tiktok_hybrid_v1
  - recipe_id: snapchat_text_overlay_v1
images:
  include_labels: ["real", "fake"]
  max_images_per_label: 200
```

Parameter grids are supported by listing parameter values:

```json
{"name":"jpeg","params":{"quality":[95,75,60]}}
```

The cartesian product of list parameters is expanded into separate recipe instances.

## 4) Generate a Jobs Manifest

```bash
python -m scripts.generate_manifest \
  --dataset_jsonl manifests/dataset.jsonl \
  --recipes_dir configs/recipes \
  --experiment_yaml configs/experiments/exp.yaml \
  --out manifests/jobs.jsonl
```

## 5) Run Distortions

```bash
python -m scripts.run_distortions \
  --manifest manifests/jobs.jsonl \
  --cache_dir data/cache/distorted \
  --out_dir data/outputs/distorted \
  --write_augmented_manifest manifests/jobs_with_paths.jsonl
```

The augmented manifest includes:

- `distorted_path`
- `cache_hit`
- `cache_key`
- `seed`
- `steps`

## Determinism and Caching

- Deterministic RNG seed:
  - `seed = stable_hash(global_seed, image_id, recipe_id, variant)`
- Cache key:
  - `cache_key = hash(src_path + steps + seed + variant)`
- Cache outputs are written to `data/cache/distorted/` and reused if present.

## Distortions Implemented

- `jpeg` (quality 1–100)
- `resize` (width/height or scale)
- `gaussian_blur` (sigma)
- `noise` (Gaussian std)
- `lut_filter` (presets: warm_01, cool_01, vintage_01)
- `text_overlay` (position, font size, stroke, opacity)
- `emoji_overlay` (PNG sticker with random position/scale/rotation)
- `tiktok_ui_overlay` (PNG template with alpha)
- `instagram_reels_ui_overlay` (PNG template with alpha)

## Notes

- UI overlay assets in `src/distortions/ui_overlay/assets/` are placeholder PNGs. Replace them with real templates for production.
- Outputs are saved as `.png` to preserve distortion effects without further compression.

## Tests

Run:

```bash
pytest
```
