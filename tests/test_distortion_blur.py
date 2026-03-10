import numpy as np
from PIL import Image, ImageDraw

from src.distortions.gaussian_blur import GaussianBlurDistortion


def test_gaussian_blur_changes_image():
    base = Image.new("RGB", (64, 64), "black")
    draw = ImageDraw.Draw(base)
    draw.rectangle([0, 0, 32, 64], fill="white")

    distortion = GaussianBlurDistortion()
    blurred = distortion.apply(base, np.random.default_rng(0), {"sigma": 2.0})

    assert blurred.size == base.size
    assert np.any(np.array(blurred) != np.array(base))
