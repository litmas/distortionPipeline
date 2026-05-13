from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image, ImageEnhance

from .base import Distortion

_STYLES = {
    "clarendon": {
        "brightness": 1.08,
        "contrast": 1.24,
        "saturation": 1.18,
        "sharpness": 1.08,
        "tint": (110, 155, 225),
        "tint_strength": 0.08,
    },
    "juno": {
        "brightness": 1.06,
        "contrast": 1.18,
        "saturation": 1.24,
        "sharpness": 1.02,
        "tint": (236, 170, 110),
        "tint_strength": 0.10,
    },
    "valencia": {
        "brightness": 1.09,
        "contrast": 0.94,
        "saturation": 0.92,
        "sharpness": 1.00,
        "tint": (245, 210, 165),
        "tint_strength": 0.16,
    },
    "aden": {
        "brightness": 1.10,
        "contrast": 0.88,
        "saturation": 0.82,
        "sharpness": 0.98,
        "tint": (225, 215, 240),
        "tint_strength": 0.18,
    },
}


class InstagramStyleFilterDistortion(Distortion):
    name = "instagram_style_filter"

    def validate_params(self, params: Dict[str, Any]) -> None:
        style = params.get("style", "random")
        if not isinstance(style, str):
            raise ValueError("instagram_style_filter 'style' must be a string")
        if style != "random" and style not in _STYLES:
            raise ValueError(
                f"instagram_style_filter style '{style}' is not supported"
            )

        styles = params.get("styles")
        if styles is not None:
            if not isinstance(styles, list) or not styles:
                raise ValueError("instagram_style_filter 'styles' must be a non-empty list")
            for item in styles:
                if item not in _STYLES:
                    raise ValueError(
                        f"instagram_style_filter style '{item}' is not supported"
                    )

        strength = params.get("strength", 1.0)
        if not isinstance(strength, (int, float)):
            raise ValueError("instagram_style_filter 'strength' must be a number")
        if not (0 <= float(strength) <= 1):
            raise ValueError("instagram_style_filter 'strength' must be in [0, 1]")

    def _choose_style(self, rng: np.random.Generator, params: Dict[str, Any]) -> str:
        if isinstance(params.get("styles"), list) and params["styles"]:
            styles = [str(item) for item in params["styles"]]
            index = int(rng.integers(0, len(styles)))
            return styles[index]

        style = str(params.get("style", "random"))
        if style != "random":
            return style

        available = sorted(_STYLES.keys())
        index = int(rng.integers(0, len(available)))
        return available[index]

    def apply(
        self,
        image: Image.Image,
        rng: np.random.Generator,
        params: Dict[str, Any],
    ) -> Image.Image:
        self.validate_params(params)

        style = self._choose_style(rng, params)
        params["applied_style"] = style
        factors = _STYLES[style]
        strength = float(params.get("strength", 1.0))

        alpha = None
        if image.mode == "RGBA":
            alpha = image.getchannel("A")
            base = image.convert("RGB")
        else:
            base = image.convert("RGB")

        filtered = ImageEnhance.Brightness(base).enhance(factors["brightness"])
        filtered = ImageEnhance.Contrast(filtered).enhance(factors["contrast"])
        filtered = ImageEnhance.Color(filtered).enhance(factors["saturation"])
        filtered = ImageEnhance.Sharpness(filtered).enhance(factors["sharpness"])

        tint = Image.new("RGB", base.size, factors["tint"])
        tinted = Image.blend(filtered, tint, factors["tint_strength"])
        blended = Image.blend(base, tinted, strength)

        if alpha is not None:
            blended = blended.convert("RGBA")
            blended.putalpha(alpha)

        return blended
