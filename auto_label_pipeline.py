#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}
DEFAULT_MODELSCOPE_CKPT_URL = (
    "https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt"
)


@dataclass
class LabelPrompt:
    class_id: int
    class_name: str
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SAM3 to auto-generate YOLO segmentation labels for images and videos."
    )
    parser.add_argument(
        "--src-dir",
        default="src",
        help="Resource directory. Images/videos/prompts should be prepared here.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Prompt YAML path. Defaults to <src-dir>/prompts.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/auto_label",
        help="Directory for extracted frames, copied images, labels, and metadata.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional SAM3 checkpoint path. If omitted, SAM3 will try Hugging Face download.",
    )
    parser.add_argument(
        "--checkpoint-url",
        default=DEFAULT_MODELSCOPE_CKPT_URL,
        help="Fallback checkpoint download URL, defaulting to the ModelScope mirror.",
    )
    parser.add_argument(
        "--checkpoint-cache-dir",
        default="checkpoints",
        help="Directory used to store downloaded checkpoint files.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=64,
        help="Minimum mask area in pixels to keep.",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=4,
        help="Minimum width/height in pixels for valid instances.",
    )
    parser.add_argument(
        "--max-polygon-points",
        type=int,
        default=200,
        help="Upper bound for polygon points per instance after contour simplification.",
    )
    parser.add_argument(
        "--frame-fps",
        type=float,
        default=None,
        help="Optional FPS for video extraction. If omitted, extract all frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove previous output directory before running.",
    )
    return parser.parse_args()


def ensure_layout(src_dir: Path) -> None:
    for relative in ["images", "videos", "labels", "frames"]:
        (src_dir / relative).mkdir(parents=True, exist_ok=True)


def load_prompts(prompts_path: Path) -> List[LabelPrompt]:
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompts_path}. Please edit src/prompts.yaml first."
        )

    with prompts_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    raw_classes = data.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("prompts.yaml must contain a non-empty 'classes' list.")

    prompts: List[LabelPrompt] = []
    for class_id, item in enumerate(raw_classes):
        if isinstance(item, str):
            class_name = item.strip()
            prompt = class_name
        elif isinstance(item, dict):
            class_name = str(item.get("name", "")).strip()
            prompt = str(item.get("prompt", class_name)).strip()
        else:
            raise ValueError(f"Invalid class entry in prompts.yaml: {item!r}")

        if not class_name or not prompt:
            raise ValueError(f"Invalid class entry in prompts.yaml: {item!r}")
        prompts.append(
            LabelPrompt(class_id=class_id, class_name=class_name, prompt=prompt)
        )

    return prompts


def collect_files(directory: Path, suffixes: Sequence[str]) -> List[Path]:
    suffixes_lower = {suffix.lower() for suffix in suffixes}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes_lower
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def extract_video_frames(
    video_path: Path, frames_dir: Path, fps: Optional[float]
) -> List[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frames_dir / "%06d.jpg"
    command = ["ffmpeg", "-y", "-i", str(video_path)]
    if fps is not None:
        command.extend(["-vf", f"fps={fps}"])
    command.extend(["-q:v", "2", str(output_pattern)])
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to extract frames from {video_path}: {exc.stderr.strip()}"
        ) from exc

    return collect_files(frames_dir, [".jpg", ".jpeg", ".png"])


def fit_contour_points(contour: np.ndarray, max_points: int) -> np.ndarray:
    if len(contour) <= max_points:
        return contour
    epsilon = 1.0
    simplified = contour
    while len(simplified) > max_points and epsilon < 50.0:
        simplified = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
        epsilon *= 1.5
    return simplified


def mask_to_yolo_polygon(mask: np.ndarray, max_points: int) -> Optional[List[float]]:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")

    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = fit_contour_points(contour, max_points)
    contour = contour.reshape(-1, 2)
    if contour.shape[0] < 3:
        return None

    height, width = mask.shape
    polygon: List[float] = []
    for x, y in contour:
        polygon.extend(
            [float(np.clip(x / width, 0.0, 1.0)), float(np.clip(y / height, 0.0, 1.0))]
        )
    return polygon


def boxes_intersect(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter_w > 0.0 and inter_h > 0.0


def suppress_cross_class_duplicates(
    records: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    kept: List[Dict[str, object]] = []
    for record in sorted(records, key=lambda item: float(item["score"]), reverse=True):
        box = record["box_xyxy"]
        duplicate = False
        for existing in kept:
            if boxes_intersect(box, existing["box_xyxy"]):
                mask_a = record["mask"]
                mask_b = existing["mask"]
                intersection = float(np.logical_and(mask_a, mask_b).sum())
                union = float(np.logical_or(mask_a, mask_b).sum())
                iou = intersection / union if union > 0 else 0.0
                if iou > 0.85:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(record)
    return kept


def write_yolo_label_file(
    label_path: Path, instances: Iterable[Dict[str, object]]
) -> None:
    lines: List[str] = []
    for item in instances:
        polygon = item["polygon"]
        class_id = item["class_id"]
        values = [str(class_id)] + [f"{value:.6f}" for value in polygon]
        lines.append(" ".join(values))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def download_with_requests(url: str, target_path: Path) -> Path:
    import requests

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(target_path)
    return target_path


def resolve_checkpoint_path(
    checkpoint: Optional[str], checkpoint_url: str, checkpoint_cache_dir: str
) -> Optional[str]:
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return str(checkpoint_path)

    cache_dir = Path(checkpoint_cache_dir)
    local_checkpoint = cache_dir / "sam3.pt"
    if local_checkpoint.exists():
        return str(local_checkpoint)

    print(f"Checkpoint not found locally, downloading from {checkpoint_url} ...")
    return str(download_with_requests(checkpoint_url, local_checkpoint))


def build_model(
    device: str, checkpoint: Optional[str], threshold: float
) -> Sam3Processor:
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint,
        load_from_HF=checkpoint is None,
    )
    return Sam3Processor(model, device=device, confidence_threshold=threshold)


def infer_instances(
    processor: Sam3Processor,
    image_path: Path,
    prompts: Sequence[LabelPrompt],
    min_mask_area: int,
    min_box_size: int,
    max_polygon_points: int,
) -> List[Dict[str, object]]:
    image = Image.open(image_path).convert("RGB")
    state = processor.set_image(image)

    records: List[Dict[str, object]] = []
    for prompt in prompts:
        state = processor.set_text_prompt(prompt.prompt, state)
        masks = state["masks"].detach().cpu().numpy().astype(bool)
        boxes = state["boxes"].detach().cpu().numpy()
        scores = state["scores"].detach().cpu().numpy()

        for mask, box, score in zip(masks, boxes, scores):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            width = x2 - x1
            height = y2 - y1
            area = int(mask.sum())
            if area < min_mask_area:
                continue
            if width < min_box_size or height < min_box_size:
                continue
            polygon = mask_to_yolo_polygon(mask, max_polygon_points)
            if polygon is None:
                continue
            records.append(
                {
                    "class_id": prompt.class_id,
                    "class_name": prompt.class_name,
                    "prompt": prompt.prompt,
                    "score": float(score),
                    "mask": mask,
                    "box_xyxy": [x1, y1, x2, y2],
                    "polygon": polygon,
                }
            )

        processor.reset_all_prompts(state)

    return suppress_cross_class_duplicates(records)


def relative_output_name(src_root: Path, file_path: Path) -> str:
    return str(file_path.relative_to(src_root)).replace(os.sep, "__")


def process_images(
    processor: Sam3Processor,
    prompts: Sequence[LabelPrompt],
    src_dir: Path,
    images: Sequence[Path],
    output_dir: Path,
    min_mask_area: int,
    min_box_size: int,
    max_polygon_points: int,
) -> List[Dict[str, object]]:
    dataset_images_dir = output_dir / "dataset" / "images"
    dataset_labels_dir = output_dir / "dataset" / "labels"
    manifest: List[Dict[str, object]] = []

    for image_path in images:
        output_stem = Path(relative_output_name(src_dir, image_path)).with_suffix("")
        out_image_path = dataset_images_dir / f"{output_stem}.jpg"
        out_label_path = dataset_labels_dir / f"{output_stem}.txt"
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, out_image_path)

        instances = infer_instances(
            processor=processor,
            image_path=image_path,
            prompts=prompts,
            min_mask_area=min_mask_area,
            min_box_size=min_box_size,
            max_polygon_points=max_polygon_points,
        )
        write_yolo_label_file(out_label_path, instances)
        manifest.append(
            {
                "source": str(image_path),
                "image": str(out_image_path),
                "label": str(out_label_path),
                "num_instances": len(instances),
            }
        )
        print(f"[image] {image_path.name}: {len(instances)} instances")

    return manifest


def process_videos(
    processor: Sam3Processor,
    prompts: Sequence[LabelPrompt],
    src_dir: Path,
    videos: Sequence[Path],
    output_dir: Path,
    min_mask_area: int,
    min_box_size: int,
    max_polygon_points: int,
    frame_fps: Optional[float],
) -> List[Dict[str, object]]:
    manifest: List[Dict[str, object]] = []
    extracted_root = output_dir / "extracted_frames"

    for video_path in videos:
        video_tag = Path(relative_output_name(src_dir, video_path)).with_suffix("")
        frame_dir = extracted_root / video_tag
        frames = extract_video_frames(video_path, frame_dir, frame_fps)
        print(f"[video] {video_path.name}: extracted {len(frames)} frames")
        manifest.extend(
            process_images(
                processor=processor,
                prompts=prompts,
                src_dir=extracted_root,
                images=frames,
                output_dir=output_dir,
                min_mask_area=min_mask_area,
                min_box_size=min_box_size,
                max_polygon_points=max_polygon_points,
            )
        )

    return manifest


def write_dataset_yaml(output_dir: Path, prompts: Sequence[LabelPrompt]) -> Path:
    dataset_yaml = output_dir / "dataset.yaml"
    content = {
        "path": str(output_dir.resolve() / "dataset"),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": {prompt.class_id: prompt.class_name for prompt in prompts},
    }
    dataset_yaml.write_text(
        yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return dataset_yaml


def write_manifest(
    output_dir: Path, prompts: Sequence[LabelPrompt], items: Sequence[Dict[str, object]]
) -> None:
    serializable_items = []
    for item in items:
        serializable_items.append(item)
    manifest = {
        "classes": [prompt.__dict__ for prompt in prompts],
        "items": serializable_items,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    src_dir = Path(args.src_dir).resolve()
    prompts_path = (
        Path(args.prompts_file).resolve()
        if args.prompts_file
        else src_dir / "prompts.yaml"
    )
    output_dir = Path(args.output_dir).resolve()

    ensure_layout(src_dir)
    prepare_output_dir(output_dir, args.overwrite)

    prompts = load_prompts(prompts_path)
    images = collect_files(src_dir / "images", IMAGE_EXTS)
    videos = collect_files(src_dir / "videos", VIDEO_EXTS)

    if not images and not videos:
        raise RuntimeError(
            f"No inputs found. Put images into {src_dir / 'images'} and/or videos into {src_dir / 'videos'}."
        )

    print(f"Loading SAM3 on {args.device} ...")
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        checkpoint_url=args.checkpoint_url,
        checkpoint_cache_dir=args.checkpoint_cache_dir,
    )
    processor = build_model(
        device=args.device, checkpoint=checkpoint_path, threshold=args.threshold
    )

    all_items: List[Dict[str, object]] = []
    if images:
        all_items.extend(
            process_images(
                processor=processor,
                prompts=prompts,
                src_dir=src_dir,
                images=images,
                output_dir=output_dir,
                min_mask_area=args.min_mask_area,
                min_box_size=args.min_box_size,
                max_polygon_points=args.max_polygon_points,
            )
        )
    if videos:
        all_items.extend(
            process_videos(
                processor=processor,
                prompts=prompts,
                src_dir=src_dir,
                videos=videos,
                output_dir=output_dir,
                min_mask_area=args.min_mask_area,
                min_box_size=args.min_box_size,
                max_polygon_points=args.max_polygon_points,
                frame_fps=args.frame_fps,
            )
        )

    dataset_yaml = write_dataset_yaml(output_dir, prompts)
    write_manifest(output_dir, prompts, all_items)
    print(f"Done. Wrote {len(all_items)} labeled images/frames.")
    print(f"Dataset yaml: {dataset_yaml}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LocalEntryNotFoundError as exc:
        print(
            "ERROR: SAM3 checkpoint is not available locally and Hugging Face download failed. "
            "Please run `hf auth login` and/or pass `--checkpoint /path/to/sam3.pt`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
