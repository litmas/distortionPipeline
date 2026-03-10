from __future__ import annotations

import hashlib


def stable_hash_int(payload: str, mod: int = 2**32) -> int:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value % mod


def compute_seed(global_seed: int, image_id: str, recipe_id: str, variant: int) -> int:
    payload = f"{global_seed}|{image_id}|{recipe_id}|{variant}"
    return stable_hash_int(payload)
