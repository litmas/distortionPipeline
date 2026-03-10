from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from .base import Distortion


class NoiseDistortion(Distortion):
    name = "noise"

    def validate_params(self, params: Dict[str, Any]) -> None:
        std = params.get("std")
        if std is None:
            raise ValueError("noise requires 'std' param")
        if not isinstance(std, (int, float)):
            raise ValueError("noise 'std' must be a number")
        if std < 0:
            raise ValueError("noise 'std' must be >= 0")

    def _split_alpha(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray | None]:
        if image.mode == "RGBA":
            rgba = np.array(image)
            return rgba[:, :, :3], rgba[:, :, 3]
        return np.array(image.convert("RGB")), None

    def apply(
        self, image: Image.Image, rng: np.random.Generator, params: Dict[str, Any]
    ) -> Image.Image:
        self.validate_params(params)
        std = float(params["std"])
        rgb, alpha = self._split_alpha(image)
        noise = rng.normal(0.0, std, size=rgb.shape).astype(np.float32)
        noisy = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if alpha is None:
            return Image.fromarray(noisy, mode="RGB")
        rgba = np.dstack([noisy, alpha])
        return Image.fromarray(rgba, mode="RGBA")
