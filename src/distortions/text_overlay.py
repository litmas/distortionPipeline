from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .base import Distortion


class TextOverlayDistortion(Distortion):
    name = "text_overlay"

    _VALID_POSITIONS = {
        "top",
        "bottom",
        "random",
        "center",
        "upper_third",
        "lower_third",
        "middle",
    }

    @staticmethod
    def _parse_rgb(color: Any) -> tuple[int, int, int] | None:
        if isinstance(color, str):
            try:
                return tuple(ImageColor.getrgb(color))
            except ValueError:
                return None
        if isinstance(color, (tuple, list)) and len(color) == 3:
            try:
                rgb = tuple(int(v) for v in color)
            except (TypeError, ValueError):
                return None
            if any(v < 0 or v > 255 for v in rgb):
                return None
            return rgb
        return None

    def _normalize_position(self, params: Dict[str, Any]) -> str:
        position = params.get("position")
        if position is None:
            position = params.get("anchor_y")
        if position is None:
            return "top"
        if not isinstance(position, str):
            raise ValueError("text_overlay 'position' must be a string")

        normalized = position.strip().lower().replace("-", "_")
        if normalized in {"mid", "middle", "center", "centre"}:
            return "center"
        if normalized in {"upper", "uppermiddle"}:
            return "upper_third"
        if normalized in {"lower", "lowermiddle"}:
            return "lower_third"
        if normalized in self._VALID_POSITIONS:
            return normalized
        raise ValueError(
            "text_overlay 'position' must be top/bottom/random/center/upper_third/lower_third "
            "(or legacy anchor_y: top/middle/bottom)"
        )

    @staticmethod
    def _parse_anchor(raw: Any, axis: str, canvas: int, item: int) -> int | None:
        if raw is None:
            return None
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return int(raw)
        if not isinstance(raw, str):
            return None
        value = raw.lower().strip()
        if axis == "x":
            if value in {"left", "start"}:
                return 0
            if value in {"center", "middle", "centre"}:
                return max(0, (canvas - item) // 2)
            if value in {"right", "end"}:
                return max(0, canvas - item)
        else:
            if value in {"top", "start"}:
                return 0
            if value in {"middle", "center", "centre"}:
                return max(0, (canvas - item) // 2)
            if value in {"bottom", "end"}:
                return max(0, canvas - item)
        if value.endswith("%"):
            try:
                pct = float(value[:-1]) / 100.0
            except ValueError:
                return None
            return max(0, min(canvas - item, int((canvas - item) * pct)))
        return None

    def _parse_background_width(self, value: Any, canvas_w: int) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text == "fit":
                return None
            if text.endswith("%"):
                try:
                    pct = float(text[:-1]) / 100.0
                except ValueError as exc:
                    raise ValueError(
                        "text_overlay 'text_background_width' must be 'fit' or percentage string"
                    ) from exc
                if not (0 < pct <= 1):
                    raise ValueError(
                        "text_overlay 'text_background_width' percentage must be in (0,100]"
                    )
                return int(canvas_w * pct)
            raise ValueError(
                "text_overlay 'text_background_width' must be a number or percentage string"
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value <= 0:
                raise ValueError(
                    "text_overlay 'text_background_width' must be a positive number"
                )
            return int(value)
        raise ValueError("text_overlay 'text_background_width' must be a number or percentage string")

    @staticmethod
    def _parse_word_pool(raw: Any) -> list[str]:
        if isinstance(raw, str):
            raw = raw.replace(",", " ").split()
            if not raw:
                raise ValueError("text_overlay 'word_pool' must be a non-empty list of strings")

        if not isinstance(raw, list) or not raw:
            raise ValueError("text_overlay 'word_pool' must be a non-empty list of strings")
        words: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("text_overlay 'word_pool' must contain non-empty strings")
            words.append(item.strip())
        if not words:
            raise ValueError("text_overlay 'word_pool' must be a non-empty list of strings")
        return words

    def _load_font(self, font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if font_path:
            return ImageFont.truetype(font_path, size)
        return ImageFont.load_default()

    def _position_xy(
        self,
        w: int,
        h: int,
        tw: int,
        th: int,
        rng: np.random.Generator,
        params: Dict[str, Any],
    ) -> Tuple[int, int]:
        position = params["position"]
        margin = max(0, int(params.get("margin", 10)))

        if position == "top":
            x = (w - tw) // 2
            y = margin
        elif position == "bottom":
            x = (w - tw) // 2
            y = max(margin, h - th - margin)
        elif position == "upper_third":
            x = (w - tw) // 2
            y = max(margin, int(h * 0.2))
        elif position == "lower_third":
            x = (w - tw) // 2
            y = max(margin, int(h * 0.75) - th)
        elif position in {"center", "middle"}:
            x = (w - tw) // 2
            y = max(margin, (h - th) // 2)
        else:
            max_x = max(margin, w - tw - margin)
            max_y = max(margin, h - th - margin)
            x = int(rng.integers(margin, max_x + 1)) if max_x >= margin else margin
            y = int(rng.integers(margin, max_y + 1)) if max_y >= margin else margin

        anchor_x = self._parse_anchor(params.get("anchor_x"), "x", w, tw)
        anchor_y = self._parse_anchor(params.get("anchor_y"), "y", h, th)
        if anchor_x is not None:
            x = anchor_x
        if anchor_y is not None:
            y = anchor_y

        text_align = str(params.get("text_align", "left")).lower()
        if text_align == "center":
            x = (w - tw) // 2
        elif text_align == "right":
            x = max(0, w - tw - margin)
        elif text_align == "left":
            x = margin

        x = max(margin, min(w - tw - margin, x))
        y = max(margin, min(h - th - margin, y))

        if bool(params.get("random_vertical", False)):
            max_y = max(margin, h - th - margin)
            y = int(rng.integers(margin, max_y + 1)) if max_y >= margin else margin

        return x, y

    def _background_box(
        self,
        w: int,
        h: int,
        x: int,
        y: int,
        tw: int,
        th: int,
        params: Dict[str, Any],
    ) -> Tuple[int, int, int, int]:
        background_mode = str(params.get("background_mode", "fit")).lower()
        p_v = max(0, int(params.get("text_background_padding_vertical", params.get("text_background_padding", 12))))
        p_h = max(0, int(params.get("text_background_padding_horizontal", params.get("text_background_padding", 12))))

        if background_mode == "full_width":
            width = self._parse_background_width(
                params.get("text_background_width", "100%"),
                w,
            )
            if width is None:
                left = 0
                right = w
            else:
                width = max(1, min(width, w))
                left = max(0, (w - width) // 2)
                right = left + width
            top = max(0, y - p_v)
            bottom = min(h, y + th + p_v)
            return left, top, right, bottom

        left = max(0, x - p_h)
        right = min(w, x + tw + p_h)
        top = max(0, y - p_v)
        bottom = min(h, y + th + p_v)
        return left, top, right, bottom

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not params.get("random_words", False) and "text" not in params:
            raise ValueError("text_overlay requires 'text' when random_words is false")
        if params.get("random_words", False):
            self._parse_word_pool(params.get("word_pool"))
            min_words = params.get("min_words", 3)
            max_words = params.get("max_words", 10)
            if not isinstance(min_words, int) or not isinstance(max_words, int):
                raise ValueError("text_overlay 'min_words' and 'max_words' must be integers")
            if min_words < 1 or max_words < min_words:
                raise ValueError(
                    "text_overlay 'min_words' must be >= 1 and <= max_words"
                )
            if min_words < 3 or max_words > 10:
                raise ValueError("text_overlay 'min_words'/'max_words' should be within 3-10")
        self._normalize_position(params)

        font_size = params.get("font_size", 32)
        if not isinstance(font_size, int):
            raise ValueError("text_overlay 'font_size' must be int")
        opacity = params.get("opacity", 1.0)
        if not isinstance(opacity, (int, float)):
            raise ValueError("text_overlay 'opacity' must be a number")
        if not (0 <= opacity <= 1):
            raise ValueError("text_overlay 'opacity' must be in [0, 1]")

        if bool(params.get("text_background", False)):
            color = params.get("text_background_color", [128, 128, 128])
            if self._parse_rgb(color) is None:
                raise ValueError(
                    "text_overlay 'text_background_color' must be [R, G, B] or css color"
                )
            tbo = params.get("text_background_opacity", 0.5)
            if not isinstance(tbo, (int, float)) or not (0 <= tbo <= 1):
                raise ValueError(
                    "text_overlay 'text_background_opacity' must be a number in [0, 1]"
                )
            mode = str(params.get("background_mode", "fit")).lower()
            if mode not in {"fit", "full_width"}:
                raise ValueError("text_overlay 'background_mode' must be fit/full_width")
            if params.get("text_background_padding") is not None and not isinstance(params.get("text_background_padding"), int):
                raise ValueError("text_overlay 'text_background_padding' must be int")
            if params.get("text_background_padding_horizontal") is not None and not isinstance(
                params.get("text_background_padding_horizontal"), int
            ):
                raise ValueError("text_overlay 'text_background_padding_horizontal' must be int")
            if params.get("text_background_padding_vertical") is not None and not isinstance(
                params.get("text_background_padding_vertical"), int
            ):
                raise ValueError("text_overlay 'text_background_padding_vertical' must be int")
            if params.get("text_background_radius") is not None and not isinstance(
                params.get("text_background_radius"), int
            ):
                raise ValueError("text_overlay 'text_background_radius' must be int")
            bg_width = params.get("text_background_width")
            if bg_width is not None:
                self._parse_background_width(bg_width, 1000)

            txt_align = str(params.get("text_align", "left")).lower()
            if txt_align not in {"left", "center", "right"}:
                raise ValueError("text_overlay 'text_align' must be left/center/right")

    def apply(
        self, image: Image.Image, rng: np.random.Generator, params: Dict[str, Any]
    ) -> Image.Image:
        self.validate_params(params)

        # keep normalized position in params for downstream use
        params["position"] = self._normalize_position(params)

        if bool(params.get("random_words", False)):
            words = self._parse_word_pool(params.get("word_pool"))
            min_words = int(params.get("min_words", 3))
            max_words = int(params.get("max_words", 10))
            count = int(rng.integers(min_words, max_words + 1))
            count = min(len(words), max(1, count))
            if len(words) >= count:
                idx = rng.choice(len(words), size=count, replace=False)
            else:
                idx = rng.integers(0, len(words), size=count)
            text = " ".join(words[i] for i in idx)
        else:
            text = str(params["text"])
        font_size = int(params.get("font_size", 32))
        opacity = float(params.get("opacity", 1.0))
        stroke_width = int(params.get("stroke_width", 2))
        fill = params.get("fill", "white")
        stroke_fill = params.get("stroke_fill", "black")
        font = self._load_font(params.get("font_path"), font_size)

        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        x, y = self._position_xy(base.width, base.height, tw, th, rng, params)

        if bool(params.get("text_background", False)):
            left, top, right, bottom = self._background_box(
                base.width,
                base.height,
                x,
                y,
                tw,
                th,
                params,
            )
            p_color = self._parse_rgb(params.get("text_background_color", [128, 128, 128]))
            p_alpha = int(255 * float(params.get("text_background_opacity", 0.5)))
            p_alpha = max(0, min(255, p_alpha))
            radius = int(params.get("text_background_radius", 12))
            draw.rounded_rectangle(
                [left, top, right, bottom],
                radius=radius,
                fill=p_color + (p_alpha,),
            )

        draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)

        if opacity < 1.0:
            a = overlay.split()[-1]
            overlay.putalpha(a.point(lambda p: int(p * opacity)))

        return Image.alpha_composite(base, overlay)
