from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from PIL import Image, ImageColor

from ..base import Distortion


class TikTokUIOverlayDistortion(Distortion):
    name = "tiktok_ui_overlay"

    def _parse_color(self, color: Any) -> Tuple[int, int, int]:
        if isinstance(color, str):
            rgb = ImageColor.getrgb(color)
        elif isinstance(color, (list, tuple)) and len(color) == 3:
            rgb = tuple(int(c) for c in color)
        else:
            raise ValueError("transparent_colors entries must be [R,G,B] or CSS color")
        if any(c < 0 or c > 255 for c in rgb):
            raise ValueError("transparent_colors values must be in [0, 255]")
        return rgb

    def _parse_colors(self, colors: Iterable[Iterable[int]]) -> list[Tuple[int, int, int]]:
        parsed: list[Tuple[int, int, int]] = []
        for color in colors:
            parsed.append(self._parse_color(color))
        return parsed

    def validate_params(self, params: Dict[str, Any]) -> None:
        template_path = params.get("template_path")
        if not template_path:
            raise ValueError("tiktok_ui_overlay requires 'template_path'")
        if not isinstance(template_path, str) or not template_path.strip():
            raise ValueError("template_path must be a non-empty string")

        opacity = params.get("opacity", 1.0)
        if not isinstance(opacity, (int, float)):
            raise ValueError("tiktok_ui_overlay 'opacity' must be a number")
        if opacity < 0 or opacity > 1:
            raise ValueError("tiktok_ui_overlay 'opacity' must be in [0, 1]")

        transparent_colors = params.get("transparent_colors")
        if transparent_colors is not None:
            if not isinstance(transparent_colors, (list, tuple)):
                raise ValueError("transparent_colors must be a list of [R,G,B] values")
            self._parse_colors(transparent_colors)

        tolerance = params.get("tolerance", 0)
        if not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError("tolerance must be a non-negative number")

    def _apply_color_key(
        self, overlay: Image.Image, colors: list[Tuple[int, int, int]], tolerance: float
    ) -> Image.Image:
        rgba = np.array(overlay)
        rgb = rgba[:, :, :3].astype(np.int16)
        alpha = rgba[:, :, 3]
        mask = np.zeros(alpha.shape, dtype=bool)
        tol = int(tolerance)
        for color in colors:
            target = np.array(color, dtype=np.int16)
            diff = np.abs(rgb - target)
            within = (diff[:, :, 0] <= tol) & (diff[:, :, 1] <= tol) & (diff[:, :, 2] <= tol)
            mask |= within
        alpha[mask] = 0
        rgba[:, :, 3] = alpha
        return Image.fromarray(rgba, mode="RGBA")

    def _load_overlay(self, template_path: str, size: tuple[int, int], opacity: float) -> Image.Image:
        source = Path(template_path)
        if not source.exists():
            raise FileNotFoundError(f"tiktok_ui_overlay template not found: {template_path}")
        overlay = Image.open(source).convert("RGBA")
        if overlay.size != size:
            overlay = overlay.resize(size, resample=Image.LANCZOS)
        if opacity < 1.0:
            alpha = overlay.split()[-1]
            alpha = alpha.point(lambda p: int(p * opacity))
            overlay.putalpha(alpha)
        return overlay

    def apply(
        self, image: Image.Image, rng: np.random.Generator, params: Dict[str, Any]
    ) -> Image.Image:
        self.validate_params(params)
        base = image.convert("RGBA")
        template_path = params["template_path"]
        opacity = float(params.get("opacity", 1.0))
        transparent_colors = params.get("transparent_colors")
        tolerance = float(params.get("tolerance", 0))

        overlay = self._load_overlay(template_path, base.size, opacity)
        if transparent_colors:
            colors = self._parse_colors(transparent_colors)
            overlay = self._apply_color_key(overlay, colors, tolerance)
        return Image.alpha_composite(base, overlay)
