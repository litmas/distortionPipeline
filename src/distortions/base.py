from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image


class Distortion:
    """Base interface for all distortions."""

    name: str = "base"

    def validate_params(self, params: Dict[str, Any]) -> None:
        """Validate params in-place. Subclasses may raise ValueError."""
        return None

    def apply(
        self,
        image: Image.Image,
        rng: np.random.Generator,
        params: Dict[str, Any],
    ) -> Image.Image:
        raise NotImplementedError
