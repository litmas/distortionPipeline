from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image

from .base import Distortion


class EmojiOverlayDistortion(Distortion):
    name = "emoji_overlay"

    def validate_params(self, params: Dict[str, Any]) -> None:
        sticker_path = params.get("sticker_path")
        if not sticker_path:
            raise ValueError("emoji_overlay requires 'sticker_path' param")
        scale_range = params.get("scale_range", [0.2, 0.4])
        if not isinstance(scale_range, (list, tuple)) or len(scale_range) != 2:
            raise ValueError("emoji_overlay 'scale_range' must be [min, max]")
        rotation_range = params.get("rotation_range", [-15, 15])
        if not isinstance(rotation_range, (list, tuple)) or len(rotation_range) != 2:
            raise ValueError("emoji_overlay 'rotation_range' must be [min, max]")
        opacity = params.get("opacity", 1.0)
        if not isinstance(opacity, (int, float)):
            raise ValueError("emoji_overlay 'opacity' must be a number")
        if opacity < 0 or opacity > 1:
            raise ValueError("emoji_overlay 'opacity' must be in [0, 1]")

    def apply(
        self, image: Image.Image, rng: np.random.Generator, params: Dict[str, Any]
    ) -> Image.Image:
        self.validate_params(params)
        sticker_path = params["sticker_path"]
        scale_range = params.get("scale_range", [0.2, 0.4])
        rotation_range = params.get("rotation_range", [-15, 15])
        opacity = float(params.get("opacity", 1.0))
        position = params.get("position", "random")

        base = image.convert("RGBA")
        sticker = Image.open(sticker_path).convert("RGBA")
        scale = float(rng.uniform(scale_range[0], scale_range[1]))
        new_w = max(1, int(sticker.width * scale))
        new_h = max(1, int(sticker.height * scale))
        sticker = sticker.resize((new_w, new_h), resample=Image.LANCZOS)
        rotation = float(rng.uniform(rotation_range[0], rotation_range[1]))
        sticker = sticker.rotate(rotation, expand=True, resample=Image.BICUBIC)

        if opacity < 1.0:
            alpha = sticker.split()[-1]
            alpha = alpha.point(lambda p: int(p * opacity))
            sticker.putalpha(alpha)

        if position == "center":
            x = (base.width - sticker.width) // 2
            y = (base.height - sticker.height) // 2
        else:
            max_x = max(0, base.width - sticker.width)
            max_y = max(0, base.height - sticker.height)
            x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
            y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        overlay.paste(sticker, (x, y), sticker)
        return Image.alpha_composite(base, overlay)
