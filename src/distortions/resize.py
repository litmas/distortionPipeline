from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from .base import Distortion


class ResizeDistortion(Distortion):
    name = "resize"

    def validate_params(self, params: Dict[str, Any]) -> None:
        width = params.get("width")
        height = params.get("height")
        scale = params.get("scale")
        if width is None and height is None and scale is None:
            raise ValueError("resize requires width/height or scale")
        if scale is not None:
            if not isinstance(scale, (int, float)):
                raise ValueError("resize 'scale' must be a number")
            if scale <= 0:
                raise ValueError("resize 'scale' must be > 0")
        if width is not None and not isinstance(width, int):
            raise ValueError("resize 'width' must be int")
        if height is not None and not isinstance(height, int):
            raise ValueError("resize 'height' must be int")

    def _compute_size(self, image: Image.Image, params: Dict[str, Any]) -> Tuple[int, int]:
        width = params.get("width")
        height = params.get("height")
        scale = params.get("scale")
        if scale is not None:
            return int(image.width * float(scale)), int(image.height * float(scale))
        if width is not None and height is not None:
            return int(width), int(height)
        if width is not None:
            new_h = int(image.height * (int(width) / image.width))
            return int(width), new_h
        if height is not None:
            new_w = int(image.width * (int(height) / image.height))
            return new_w, int(height)
        return image.size

    def apply(
        self, image: Image.Image, rng: np.random.Generator, params: Dict[str, Any]
    ) -> Image.Image:
        self.validate_params(params)
        size = self._compute_size(image, params)
        return image.resize(size, resample=Image.LANCZOS)
