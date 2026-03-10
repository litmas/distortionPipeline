from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image, ImageFilter

from .base import Distortion


class GaussianBlurDistortion(Distortion):
    name = "gaussian_blur"

    def validate_params(self, params: Dict[str, Any]) -> None:
        sigma = params.get("sigma")
        if sigma is None:
            raise ValueError("gaussian_blur requires 'sigma' param")
        if not isinstance(sigma, (int, float)):
            raise ValueError("gaussian_blur 'sigma' must be a number")
        if sigma < 0:
            raise ValueError("gaussian_blur 'sigma' must be >= 0")

    def apply(
        self, image: Image.Image, rng: np.random.Generator, params: Dict[str, Any]
    ) -> Image.Image:
        self.validate_params(params)
        sigma = float(params["sigma"])
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
