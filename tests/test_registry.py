from src.pipeline.registry import list_distortions


def test_registry_loads_all():
    expected = {
        "jpeg",
        "resize",
        "gaussian_blur",
        "noise",
        "lut_filter",
        "text_overlay",
        "emoji_overlay",
        "tiktok_ui_overlay",
        "instagram_reels_ui_overlay",
    }
    available = set(list_distortions())
    assert expected.issubset(available)
