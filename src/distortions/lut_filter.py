from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image, ImageEnhance

from .base import Distortion

_PRESETS = {
    "warm_01": {"brightness": 1.05, "contrast": 1.1, "saturation": 1.15},
    "cool_01": {"brightness": 0.98, "contrast": 1.05, "saturation": 0.9},
    "vintage_01": {"brightness": 1.02, "contrast": 0.95, "saturation": 0.85},
}


class LutFilterDistortion(Distortion):
    name = "lut_filter"

    def validate_params(self, params: Dict[str, Any]) -> None:
        preset = params.get("preset")
        if preset is None:
            raise ValueError("lut_filter requires 'preset' param")
        if preset not in _PRESETS:
            raise ValueError(f"lut_filter preset '{preset}' is not supported")
        strength = params.get("strength", 1.0)
        if not isinstance(strength, (int, float)):
            raise ValueError("lut_filter 'strength' must be a number")
        if strength < 0 or strength > 1:
            raise ValueError("lut_filter 'strength' must be in [0, 1]")

    def apply(
        self, image: Image.Image, rng: np.random.Generator, params: Dict[str, Any]
    ) -> Image.Image:
        self.validate_params(params)
        preset = params["preset"]
        strength = float(params.get("strength", 1.0))
        factors = _PRESETS[preset]
        alpha = None
        if image.mode == "RGBA":
            alpha = image.split()[-1]
            base = image.convert("RGB")
        else:
            base = image.convert("RGB")
        filtered = ImageEnhance.Brightness(base).enhance(factors["brightness"])
        filtered = ImageEnhance.Contrast(filtered).enhance(factors["contrast"])
        filtered = ImageEnhance.Color(filtered).enhance(factors["saturation"])
        blended = Image.blend(base, filtered, strength)
        if alpha is not None:
            blended = blended.convert("RGBA")
            blended.putalpha(alpha)
        return blended
