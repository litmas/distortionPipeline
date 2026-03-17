from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPES_DIR = ROOT / "configs" / "recipes"
DEFAULT_EXPERIMENT_PATH = ROOT / "configs" / "experiments" / "exp.yaml"


def input_with_default(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else (default or "")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_experiment(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "global_seed": 12345,
            "variants": 1,
            "recipes": [],
            "images": {"include_labels": [], "max_images_per_label": 50},
        }
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Experiment file must be a YAML mapping")
    return data


def write_experiment(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def list_recipes(recipes_dir: Path) -> List[Path]:
    return sorted(recipes_dir.glob("*.json"), key=lambda path: path.stem)


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    return list(OrderedDict((value, None) for value in values if value).keys())


def parse_index_expression(raw: str, max_value: int) -> List[int]:
    expanded: List[int] = []
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    for part in parts:
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                print(f"  invalid range token '{part}'")
                continue
            step = 1 if start <= end else -1
            expanded.extend(range(start, end + step, step))
        else:
            try:
                expanded.append(int(part))
            except ValueError:
                print(f"  invalid index token '{part}'")
                continue

    return [idx for idx in expanded if 1 <= idx <= max_value]


def show_experiment(exp: Dict[str, Any]) -> None:
    print("\nCurrent experiment:")
    print(f"  global_seed: {exp.get('global_seed', 0)}")
    print(f"  variants: {exp.get('variants', 1)}")
    images = exp.get("images", {})
    print(f"  include_labels: {images.get('include_labels', [])}")
    print(f"  max_images_per_label: {images.get('max_images_per_label')}")
    print("  recipes:")
    for recipe in exp.get("recipes", []):
        print(f"    - {recipe.get('recipe_id')}")
    print()


def show_recipes(recipes_dir: Path) -> List[Path]:
    paths = list_recipes(recipes_dir)
    print("Available recipe files:")
    for idx, path in enumerate(paths, start=1):
        data = read_json(path)
        recipe_id = data.get("recipe_id", path.stem)
        label = data.get("label", "")
        print(f"  {idx}. {recipe_id} ({label}) -> {path.name}")
    return paths


def choose_recipes(current: List[str], recipes_dir: Path) -> List[str]:
    paths = show_recipes(recipes_dir)
    if not paths:
        print("No recipes found.")
        return current

    print("\nChoose recipes by number (comma-separated).")
    print("Examples: 1,3,5 or 2-4,6.")
    print("Press Enter to keep current selection.")
    raw = input_with_default("Recipe selection", ",".join(current)).strip()
    if not raw:
        return current

    selected: List[str] = []
    for idx in parse_index_expression(raw, len(paths)):
        payload = read_json(paths[idx - 1])
        selected.append(payload.get("recipe_id", paths[idx - 1].stem))
    return _dedupe_preserve_order(selected)


def edit_recipe(path: Path) -> None:
    data = read_json(path)
    print(f"\nEditing {path.name}")
    print(json.dumps(data, indent=2))
    print()

    current_label = data.get("label", "")
    new_label = input_with_default("New recipe label (blank to keep)", str(current_label))
    if new_label:
        data["label"] = new_label

    steps = data.get("steps", [])
    if not isinstance(steps, list):
        print("  steps should be a list; aborting edit")
        return

    while True:
        print("\nRecipe steps:")
        for idx, step in enumerate(steps):
            print(f"  {idx}. {step.get('name', '<missing>')}")
        print("  (empty input finishes editing)")
        raw = input("Step index to edit (Enter to finish): ").strip()
        if not raw:
            break
        try:
            step_idx = int(raw)
        except ValueError:
            print("  invalid step index")
            continue
        if not (0 <= step_idx < len(steps)):
            print("  index out of range")
            continue

        step = steps[step_idx]
        if not isinstance(step, dict):
            print("  malformed step")
            continue
        name = step.get("name", "<missing>")
        params = step.get("params", {})
        print(f"\nStep {step_idx}: {name}")
        print(f"Current params: {json.dumps(params, indent=2)}")
        raw = input_with_default(
            "Replace step params with JSON object (blank to keep)",
            "",
        )
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"  invalid JSON ({exc})")
            continue
        if not isinstance(parsed, dict):
            print("  params must be a JSON object")
            continue
        step["params"] = parsed

    data["steps"] = steps
    write_json(path, data)
    print(f"Saved {path}")


def edit_experiment(exp: Dict[str, Any], recipes_dir: Path) -> Dict[str, Any]:
    print("\nEdit experiment settings")

    while True:
        try:
            exp["global_seed"] = int(input_with_default("Global seed", str(exp.get("global_seed", 12345))))
            break
        except ValueError:
            print("  global_seed must be an integer")

    while True:
        try:
            exp["variants"] = int(input_with_default("Variants", str(exp.get("variants", 1))))
            if exp["variants"] < 1:
                raise ValueError("variants must be >= 1")
            break
        except ValueError:
            print("  variants must be a positive integer")

    image_cfg = exp.get("images", {})
    include_labels_raw = input_with_default(
        "include_labels (comma-separated, blank for no filter)",
        ",".join(image_cfg.get("include_labels", []) or []),
    )
    if include_labels_raw.strip():
        image_cfg["include_labels"] = [value.strip() for value in include_labels_raw.split(",") if value.strip()]
    else:
        image_cfg["include_labels"] = []

    max_images_value = image_cfg.get("max_images_per_label")
    max_images_raw = input_with_default(
        "max_images_per_label", "" if max_images_value is None else str(max_images_value)
    )
    if max_images_raw is None or max_images_raw == "":
        image_cfg["max_images_per_label"] = None
    else:
        while True:
            try:
                image_cfg["max_images_per_label"] = int(max_images_raw)
                if image_cfg["max_images_per_label"] < 1:
                    raise ValueError
                break
            except ValueError:
                max_images_raw = input("  must be a positive integer (blank for no limit): ").strip()
                if max_images_raw == "":
                    image_cfg["max_images_per_label"] = None
                    break

    current_recipes = [item.get("recipe_id") for item in exp.get("recipes", []) if isinstance(item, dict)]
    selected = choose_recipes(current_recipes, recipes_dir)
    exp["recipes"] = [{"recipe_id": recipe_id} for recipe_id in selected]
    exp["images"] = image_cfg
    return exp


def edit_recipe_by_index(recipes_dir: Path) -> None:
    paths = show_recipes(recipes_dir)
    if not paths:
        print("No recipes found.")
        return
    raw = input_with_default("Recipe number to edit")
    if not raw:
        return
    try:
        idx = int(raw)
    except ValueError:
        print("Invalid recipe index.")
        return
    if not (1 <= idx <= len(paths)):
        print("Index out of range.")
        return
    edit_recipe(paths[idx - 1])


def main_menu(exp: Dict[str, Any], exp_path: Path, recipes_dir: Path) -> None:
    while True:
        show_experiment(exp)
        print("Menu:")
        print("  1) Edit experiment settings")
        print("  2) Edit a recipe JSON")
        print("  3) Save and exit")
        print("  4) Exit without saving")
        choice = input("Select option: ").strip()

        if choice == "1":
            exp = edit_experiment(exp, recipes_dir)
        elif choice == "2":
            edit_recipe_by_index(recipes_dir)
        elif choice == "3":
            write_experiment(exp_path, exp)
            print(f"Saved settings to {exp_path}")
            return
        elif choice == "4":
            print("Changes discarded.")
            return
        else:
            print("Invalid option.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive distortion settings menu")
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT_PATH,
        type=Path,
        help="Experiment YAML file to edit",
    )
    parser.add_argument(
        "--recipes-dir",
        default=DEFAULT_RECIPES_DIR,
        type=Path,
        help="Directory with recipe JSON files",
    )
    args = parser.parse_args()

    if not args.recipes_dir.exists():
        raise FileNotFoundError(f"recipes-dir not found: {args.recipes_dir}")

    experiment = read_experiment(args.experiment)
    main_menu(experiment, args.experiment, args.recipes_dir)


if __name__ == "__main__":
    main()
