from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def compute_cache_key(
    src_path: str,
    steps: list[dict[str, Any]],
    seed: int,
    variant: int,
) -> str:
    payload = {
        "src_path": str(src_path),
        "steps": steps,
        "seed": seed,
        "variant": variant,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_path(cache_dir: str | Path, cache_key: str, ext: str = ".png") -> Path:
    path = Path(cache_dir) / f"{cache_key}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
