from pathlib import Path

from src.pipeline.deepfakebench import (
    build_frame_record,
    collect_deepfakebench_entries,
    job_identity,
)


def test_build_frame_record_uses_video_aware_identity(tmp_path):
    root = tmp_path / "datasets"
    frame_a = root / "UADFV" / "real" / "frames" / "0001" / "000.png"
    frame_b = root / "UADFV" / "real" / "frames" / "0002" / "000.png"
    landmark_a = root / "UADFV" / "real" / "landmarks" / "0001" / "000.npy"

    frame_a.parent.mkdir(parents=True, exist_ok=True)
    frame_b.parent.mkdir(parents=True, exist_ok=True)
    landmark_a.parent.mkdir(parents=True, exist_ok=True)
    frame_a.write_bytes(b"frame-a")
    frame_b.write_bytes(b"frame-b")
    landmark_a.write_bytes(b"landmark-a")

    record_a = build_frame_record(frame_a, root)
    record_b = build_frame_record(frame_b, root)

    assert record_a["label"] == "real"
    assert record_a["video_id"] == "0001"
    assert record_a["landmark_relative_path"] == "UADFV/real/landmarks/0001/000.npy"
    assert record_a["sample_id"] != record_b["sample_id"]
    assert job_identity(record_a, "frame") != job_identity(record_b, "frame")
    assert job_identity(record_a, "video") == "UADFV/real/frames/0001"
    assert record_a["paired_video_key"] == "UADFV/0001"


def test_build_frame_record_keeps_dataset_prefix_for_dataset_specific_runs(tmp_path):
    root = tmp_path / "UADFV"
    frame = root / "fake" / "frames" / "0001_fake" / "000.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"frame")

    record = build_frame_record(frame, root)

    assert record["dataset_name"] == "UADFV"
    assert record["relative_path"] == "UADFV/fake/frames/0001_fake/000.png"
    assert record["video_key"] == "UADFV/fake/frames/0001_fake"
    assert record["paired_video_key"] == "UADFV/0001"


def test_build_frame_record_marks_dfdcp_methods_as_fake(tmp_path):
    root = tmp_path / "datasets"
    frame = root / "DFDCP" / "method_A" / "frames" / "clip_001" / "000.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"frame")

    record = build_frame_record(frame, root)

    assert record["dataset_name"] == "DFDCP"
    assert record["label"] == "fake"


def test_build_frame_record_pairs_celebdf_real_and_synthesis_videos(tmp_path):
    root = tmp_path / "datasets"
    real_frame = root / "Celeb-DF-v1" / "Celeb-real" / "frames" / "id0_0000" / "000.png"
    fake_frame = root / "Celeb-DF-v1" / "Celeb-synthesis" / "frames" / "id0_id16_0000" / "000.png"
    real_frame.parent.mkdir(parents=True, exist_ok=True)
    fake_frame.parent.mkdir(parents=True, exist_ok=True)
    real_frame.write_bytes(b"real")
    fake_frame.write_bytes(b"fake")

    real_record = build_frame_record(real_frame, root)
    fake_record = build_frame_record(fake_frame, root)

    assert real_record["paired_video_key"] == "Celeb-DF-v1/id0_0000"
    assert fake_record["paired_video_key"] == "Celeb-DF-v1/id0_0000"


def test_collect_deepfakebench_entries_groups_and_sorts_frames():
    records = [
        {
            "dataset_name": "UADFV",
            "label": "real",
            "split": "test",
            "video_id": "0001",
            "video_key": "UADFV/real/frames/0001",
            "relative_path": "UADFV/real/frames/0001/010.png",
            "distorted_path": "/tmp/010.png",
        },
        {
            "dataset_name": "UADFV",
            "label": "real",
            "split": "test",
            "video_id": "0001",
            "video_key": "UADFV/real/frames/0001",
            "relative_path": "UADFV/real/frames/0001/002.png",
            "distorted_path": "/tmp/002.png",
        },
    ]

    payload = collect_deepfakebench_entries(records)
    entry = payload["UADFV/real/frames/0001"]

    assert entry["frames"] == ["/tmp/002.png", "/tmp/010.png"]
    assert entry["label"] == "real"
    assert entry["set"] == "test"
