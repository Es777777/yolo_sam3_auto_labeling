from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_label_pipeline import (
    LabelPrompt,
    main,
    mask_to_yolo_polygon,
    relative_output_name,
)


def test_mask_to_yolo_polygon_accepts_single_channel_3d_mask():
    mask = np.zeros((1, 8, 10), dtype=bool)
    mask[0, 2:6, 3:8] = True

    polygon = mask_to_yolo_polygon(mask, max_points=32)

    assert polygon is not None
    assert len(polygon) >= 6
    assert all(0.0 <= value <= 1.0 for value in polygon)


def test_relative_output_name_handles_extracted_video_frames():
    extracted_root = Path("/tmp/run/extracted_frames")
    frame_path = extracted_root / "videos__bedroom" / "000001.jpg"

    output_name = relative_output_name(extracted_root, frame_path)

    assert output_name == "videos__bedroom__000001.jpg"


def test_main_smoke_runs_with_mocked_components(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    images_dir = src_dir / "images"
    videos_dir = src_dir / "videos"
    images_dir.mkdir(parents=True)
    videos_dir.mkdir(parents=True)
    (src_dir / "prompts.yaml").write_text(
        "classes:\n  - name: person\n    prompt: person\n",
        encoding="utf-8",
    )
    (images_dir / "sample.jpg").write_bytes(b"fake-image")
    (videos_dir / "clip.mp4").write_bytes(b"fake-video")

    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        "auto_label_pipeline.build_model",
        lambda device, checkpoint, threshold: object(),
    )
    monkeypatch.setattr(
        "auto_label_pipeline.resolve_checkpoint_path",
        lambda checkpoint, checkpoint_url, checkpoint_cache_dir: "mock.pt",
    )

    def fake_extract_video_frames(video_path, frames_dir, fps):
        frame_path = frames_dir / "000001.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"fake-frame")
        return [frame_path]

    monkeypatch.setattr(
        "auto_label_pipeline.extract_video_frames",
        fake_extract_video_frames,
    )

    def fake_infer_instances(
        processor, image_path, prompts, min_mask_area, min_box_size, max_polygon_points
    ):
        if image_path.name == "sample.jpg":
            return []
        return [
            {
                "class_id": 0,
                "class_name": prompts[0].class_name,
                "prompt": prompts[0].prompt,
                "score": 0.9,
                "mask": np.ones((8, 8), dtype=bool),
                "box_xyxy": [0.0, 0.0, 7.0, 7.0],
                "polygon": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            }
        ]

    monkeypatch.setattr("auto_label_pipeline.infer_instances", fake_infer_instances)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "auto_label_pipeline.py",
            "--src-dir",
            str(src_dir),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
    )

    exit_code = main()

    assert exit_code == 0
    assert (output_dir / "dataset.yaml").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "dataset" / "images" / "images__sample.jpg").exists()
    assert (output_dir / "dataset" / "images" / "videos__clip__000001.jpg").exists()
    assert (output_dir / "dataset" / "labels" / "videos__clip__000001.txt").exists()
