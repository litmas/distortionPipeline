from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.deepfakebench import iter_missing_frame_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate detector-ready DeepfakeBench JSON frame paths.")
    parser.add_argument("--json_path", required=True, type=Path)
    args = parser.parse_args()

    missing = list(iter_missing_frame_paths(args.json_path))
    if missing:
        print(f"Validation failed: {len(missing)} missing frame paths in {args.json_path}")
        for path in missing[:20]:
            print(f"  {path}")
        raise SystemExit(1)

    print(f"Validation passed: all frame paths exist in {args.json_path}")


if __name__ == "__main__":
    main()
