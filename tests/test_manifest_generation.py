import json
from pathlib import Path

from scripts.build_deepfakebench_subset import build_subset
from scripts.generate_manifest import build_jobs, filter_images, load_video_keys_from_deepfakebench_json


def test_instagram_styles_are_not_expanded_as_grid():
    records = [
        {
            "image_id": "img001",
            "label": "real",
            "path": "/tmp/img001.png",
            "sample_id": "img001",
        }
    ]
    recipe = json.loads(Path("configs/recipes/instagram_style_v1.json").read_text())

    jobs = build_jobs(records, [recipe], 12345, 1, "frame")

    assert len(jobs) == 1
    assert jobs[0]["recipe_id"] == "instagram_style_v1"


def test_snapchat_recipe_uses_paired_video_seed_identity():
    records = [
        {
            "image_id": "real_frame",
            "label": "real",
            "path": "/tmp/real.png",
            "sample_id": "real_frame",
            "paired_video_key": "UADFV/0000",
        },
        {
            "image_id": "fake_frame",
            "label": "fake",
            "path": "/tmp/fake.png",
            "sample_id": "fake_frame",
            "paired_video_key": "UADFV/0000",
        },
    ]
    recipe = json.loads(Path("configs/recipes/snapchat_text_overlay_v1.json").read_text())

    jobs = build_jobs(records, [recipe], 12345, 1, "frame")

    assert len(jobs) == 2
    assert jobs[0]["sample_id"] == "UADFV/0000"
    assert jobs[1]["sample_id"] == "UADFV/0000"


def test_filter_images_can_limit_videos_per_dataset_label():
    records = [
        {
            "dataset_name": "Celeb-DF-v1",
            "label": "real",
            "video_key": "Celeb-DF-v1/Celeb-real/frames/0001",
            "relative_path": "Celeb-DF-v1/Celeb-real/frames/0001/000.png",
            "image_id": "000",
        },
        {
            "dataset_name": "Celeb-DF-v1",
            "label": "real",
            "video_key": "Celeb-DF-v1/Celeb-real/frames/0001",
            "relative_path": "Celeb-DF-v1/Celeb-real/frames/0001/001.png",
            "image_id": "001",
        },
        {
            "dataset_name": "Celeb-DF-v1",
            "label": "real",
            "video_key": "Celeb-DF-v1/Celeb-real/frames/0002",
            "relative_path": "Celeb-DF-v1/Celeb-real/frames/0002/000.png",
            "image_id": "000",
        },
        {
            "dataset_name": "Celeb-DF-v1",
            "label": "fake",
            "video_key": "Celeb-DF-v1/Celeb-synthesis/frames/0003",
            "relative_path": "Celeb-DF-v1/Celeb-synthesis/frames/0003/000.png",
            "image_id": "000",
        },
    ]

    filtered = filter_images(
        records,
        include_labels=None,
        max_per_label=None,
        include_datasets=["Celeb-DF-v1"],
        include_splits=None,
        max_videos_per_label=1,
    )

    assert len(filtered) == 3
    assert {record["video_key"] for record in filtered} == {
        "Celeb-DF-v1/Celeb-real/frames/0001",
        "Celeb-DF-v1/Celeb-synthesis/frames/0003",
    }


def test_build_subset_trims_test_split_videos_only():
    payload = {
        "Celeb-DF-v1": {
            "CelebDFv1_real": {
                "train": {
                    "0002": {"label": "CelebDFv1_real", "frames": ["a"]},
                    "0001": {"label": "CelebDFv1_real", "frames": ["b"]},
                },
                "test": {
                    "0002": {"label": "CelebDFv1_real", "frames": ["a"]},
                    "0001": {"label": "CelebDFv1_real", "frames": ["b"]},
                },
            }
        }
    }

    subset = build_subset(
        payload,
        max_videos_per_label=1,
        include_labels=None,
        include_splits={"test"},
    )

    assert set(subset["Celeb-DF-v1"]["CelebDFv1_real"]["train"]) == {"0001", "0002"}
    assert set(subset["Celeb-DF-v1"]["CelebDFv1_real"]["test"]) == {"0001"}


def test_load_video_keys_from_deepfakebench_json_collects_frame_parents(tmp_path):
    path = tmp_path / "Celeb-DF-v1.json"
    path.write_text(
        json.dumps(
            {
                "Celeb-DF-v1": {
                    "CelebDFv1_real": {
                        "test": {
                            "00138": {
                                "label": "CelebDFv1_real",
                                "frames": ["Celeb-DF-v1\\YouTube-real\\frames\\00138\\015.png"],
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    video_keys = load_video_keys_from_deepfakebench_json(path)

    assert video_keys == {"Celeb-DF-v1/YouTube-real/frames/00138"}


def test_load_video_keys_from_deepfakebench_json_can_filter_splits(tmp_path):
    path = tmp_path / "Celeb-DF-v1.json"
    path.write_text(
        json.dumps(
            {
                "Celeb-DF-v1": {
                    "CelebDFv1_real": {
                        "train": {
                            "00001": {
                                "label": "CelebDFv1_real",
                                "frames": ["Celeb-DF-v1\\YouTube-real\\frames\\00001\\000.png"],
                            }
                        },
                        "test": {
                            "00138": {
                                "label": "CelebDFv1_real",
                                "frames": ["Celeb-DF-v1\\YouTube-real\\frames\\00138\\015.png"],
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    video_keys = load_video_keys_from_deepfakebench_json(path, include_splits={"test"})

    assert video_keys == {"Celeb-DF-v1/YouTube-real/frames/00138"}
