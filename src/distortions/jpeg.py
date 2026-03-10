from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import numpy as np
from PIL import Image

from .base import Distortion


class JpegDistortion(Distortion):
    name = "jpeg"

    def validate_params(self, params: Dict[str, Any]) -> None:
        quality = params.get("quality")
        if quality is None:
            raise ValueError("jpeg requires 'quality' param")
        if not isinstance(quality, int):
            raise ValueError("jpeg 'quality' must be int")
        if quality < 1 or quality > 100:
            raise ValueError("jpeg 'quality' must be in [1, 100]")

    def apply(
        self, image: Image.Image, rng: np.random.Generator, params: Dict[str, Any]
    ) -> Image.Image:
        self.validate_params(params)
        quality = int(params["quality"])
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
