from __future__ import annotations

from typing import Dict, List, Type

from src.distortions.base import Distortion
from src.distortions.gaussian_blur import GaussianBlurDistortion
from src.distortions.jpeg import JpegDistortion
from src.distortions.lut_filter import LutFilterDistortion
from src.distortions.noise import NoiseDistortion
from src.distortions.resize import ResizeDistortion
from src.distortions.text_overlay import TextOverlayDistortion
from src.distortions.emoji_overlay import EmojiOverlayDistortion
from src.distortions.ui_overlay.instagram_reels_ui_overlay import (
    InstagramReelsUIOverlayDistortion,
)
from src.distortions.ui_overlay.tiktok_ui_overlay import TikTokUIOverlayDistortion


def _build_registry() -> Dict[str, Type[Distortion]]:
    distortions = [
        JpegDistortion,
        ResizeDistortion,
        GaussianBlurDistortion,
        NoiseDistortion,
        LutFilterDistortion,
        TextOverlayDistortion,
        EmojiOverlayDistortion,
        TikTokUIOverlayDistortion,
        InstagramReelsUIOverlayDistortion,
    ]
    return {dist.name: dist for dist in distortions}


_REGISTRY: Dict[str, Type[Distortion]] = _build_registry()


def list_distortions() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_distortion(name: str) -> Distortion:
    if name not in _REGISTRY:
        available = ", ".join(list_distortions())
        raise KeyError(f"Unknown distortion '{name}'. Available: {available}")
    return _REGISTRY[name]()
