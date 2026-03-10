from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .base import Distortion


class TextOverlayDistortion(Distortion):
    name = "text_overlay"

    def validate_params(self, params: Dict[str, Any]) -> None:
        text = params.get("text")
        if text is None:
            raise ValueError("text_overlay requires 'text' param")
        position = params.get("position", "top")
        if position not in {"top", "bottom", "random"}:
            raise ValueError("text_overlay 'position' must be top/bottom/random")
        font_size = params.get("font_size", 32)
        if not isinstance(font_size, int):
            raise ValueError("text_overlay 'font_size' must be int")
        opacity = params.get("opacity", 1.0)
        if not isinstance(opacity, (int, float)):
            raise ValueError("text_overlay 'opacity' must be a number")
        if opacity < 0 or opacity > 1:
            raise ValueError("text_overlay 'opacity' must be in [0, 1]")
        text_background = bool(params.get("text_background", False))
        if text_background:
            text_background_color = params.get("text_background_color", [128, 128, 128])
            bg_rgb = self._parse_rgb(text_background_color)
            if bg_rgb is None:
                raise ValueError(
                    "text_overlay 'text_background_color' must be [R, G, B] or a CSS color"
                )
            text_background_opacity = params.get("text_background_opacity", 0.5)
            if not isinstance(text_background_opacity, (int, float)):
                raise ValueError("text_overlay 'text_background_opacity' must be a number")
            if text_background_opacity < 0 or text_background_opacity > 1:
                raise ValueError(
                    "text_overlay 'text_background_opacity' must be in [0, 1]"
                )
            text_background_padding = params.get("text_background_padding", 12)
            if not isinstance(text_background_padding, int) or text_background_padding < 0:
                raise ValueError("text_overlay 'text_background_padding' must be a non-negative int")
            text_background_radius = params.get("text_background_radius", 12)
            if not isinstance(text_background_radius, int) or text_background_radius < 0:
                raise ValueError("text_overlay 'text_background_radius' must be a non-negative int")

    def _load_font(self, font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if font_path:
            return ImageFont.truetype(font_path, font_size)
        return ImageFont.load_default()

    def _parse_rgb(self, color: Any) -> tuple[int, int, int] | None:
        if isinstance(color, str):
            try:
                return tuple(ImageColor.getrgb(color))
            except ValueError:
                return None
        if isinstance(color, (tuple, list)) and len(color) == 3:
            try:
                rgb = tuple(int(channel) for channel in color)
            except (TypeError, ValueError):
                return None
            if any(c < 0 or c > 255 for c in rgb):
                return None
            return rgb
        return None

    def apply(
        self, image: Image.Image, rng: np.random.Generator, params: Dict[str, Any]
    ) -> Image.Image:
        self.validate_params(params)
        text = str(params["text"])
        position = params.get("position", "top")
        font_size = int(params.get("font_size", 32))
        opacity = float(params.get("opacity", 1.0))
        stroke_width = int(params.get("stroke_width", 2))
        fill = params.get("fill", "white")
        stroke_fill = params.get("stroke_fill", "black")
        margin = int(params.get("margin", 10))
        font_path = params.get("font_path")
        text_background = bool(params.get("text_background", False))
        text_background_color = self._parse_rgb(params.get("text_background_color", [128, 128, 128]))
        text_background_opacity = float(params.get("text_background_opacity", 0.5))
        text_background_padding = int(params.get("text_background_padding", 12))
        text_background_radius = int(params.get("text_background_radius", 12))

        font = self._load_font(font_path, font_size)
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        if position == "top":
            x = (base.width - text_w) // 2
            y = margin
        elif position == "bottom":
            x = (base.width - text_w) // 2
            y = max(margin, base.height - text_h - margin)
        else:
            max_x = max(margin, base.width - text_w - margin)
            max_y = max(margin, base.height - text_h - margin)
            x = int(rng.integers(margin, max_x + 1)) if max_x >= margin else margin
            y = int(rng.integers(margin, max_y + 1)) if max_y >= margin else margin

        if text_background and text_background_color is not None:
            bar_left = max(0, x - text_background_padding)
            bar_top = max(0, y - text_background_padding)
            bar_right = min(base.width, x + text_w + text_background_padding)
            bar_bottom = min(base.height, y + text_h + text_background_padding)
            bar_fill = text_background_color + (int(255 * text_background_opacity),)
            draw.rounded_rectangle(
                [bar_left, bar_top, bar_right, bar_bottom],
                radius=text_background_radius,
                fill=bar_fill,
            )

        draw.text(
            (x, y),
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

        if opacity < 1.0:
            alpha = overlay.split()[-1]
            alpha = alpha.point(lambda p: int(p * opacity))
            overlay.putalpha(alpha)

        composed = Image.alpha_composite(base, overlay)
        return composed
