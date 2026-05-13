from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def is_video_mapping(node: Any) -> bool:
    if not isinstance(node, dict) or not node:
        return False
    return all(isinstance(value, dict) and "frames" in value for value in node.values())


def trim_split_tree(node: Any, max_videos: int) -> Any:
    if not isinstance(node, dict):
        return node
    if is_video_mapping(node):
        selected_keys = sorted(node)[:max_videos]
        return {key: node[key] for key in selected_keys}
    return {key: trim_split_tree(value, max_videos) for key, value in node.items()}


def count_videos(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    if is_video_mapping(node):
        return len(node)
    return sum(count_videos(value) for value in node.values())


def build_subset(
    payload: dict[str, Any],
    max_videos_per_label: int,
    include_labels: set[str] | None,
    include_splits: set[str] | None,
) -> dict[str, Any]:
    dataset_name = next(iter(payload))
    dataset_body = payload[dataset_name]
    subset_body: dict[str, Any] = {}

    for label, split_mapping in dataset_body.items():
        if include_labels and label not in include_labels:
            continue
        if not isinstance(split_mapping, dict):
            subset_body[label] = split_mapping
            continue

        next_split_mapping: dict[str, Any] = {}
        for split_name, split_payload in split_mapping.items():
            if include_splits and split_name in include_splits:
                next_split_mapping[split_name] = trim_split_tree(split_payload, max_videos_per_label)
            else:
                next_split_mapping[split_name] = split_payload
        subset_body[label] = next_split_mapping

    return {dataset_name: subset_body}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a smaller DeepfakeBench dataset_json file by trimming videos per label."
    )
    parser.add_argument("--input_json", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--max_videos_per_label", required=True, type=int)
    parser.add_argument("--include_labels", nargs="*", default=None)
    parser.add_argument("--include_splits", nargs="*", default=["test"])
    args = parser.parse_args()

    if args.max_videos_per_label < 1:
        raise ValueError("--max_videos_per_label must be >= 1")

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("input_json must contain exactly one top-level dataset mapping")

    before = count_videos(payload)
    subset = build_subset(
        payload,
        max_videos_per_label=args.max_videos_per_label,
        include_labels=set(args.include_labels) if args.include_labels else None,
        include_splits=set(args.include_splits) if args.include_splits else None,
    )
    after = count_videos(subset)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote subset JSON to {args.output_json} "
        f"(videos before={before}, after={after}, max_videos_per_label={args.max_videos_per_label})"
    )


if __name__ == "__main__":
    main()
