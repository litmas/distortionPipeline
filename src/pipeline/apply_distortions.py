from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
from PIL import Image

from src.pipeline.registry import get_distortion


def apply_steps(
    image: Image.Image,
    steps: Iterable[Dict[str, Any]],
    rng: np.random.Generator,
) -> Image.Image:
    current = image
    for step in steps:
        name = step.get("name")
        if not name:
            raise ValueError("Each step must include a 'name'")
        params = step.get("params", {})
        distortion = get_distortion(name)
        current = distortion.apply(current, rng, params)
    return current
