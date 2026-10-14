"""Annotate dataset images with detections and transparent geometric depth estimates."""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from .calibrate import load_calibrations
from .distance import estimate_depth

PROJECT = Path(__file__).resolve().parents[1]
COLORS = [(30, 155, 245), (245, 145, 30), (60, 60, 235)]
LABELS = ["Cone", "Barrier", "Stop Sign"]


def annotate(image, result, camera, settings, priors):
    canvas = image.copy()
    detections = []
    height, width = image.shape[:2]
    label_regions = []
    for box, category, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        category = int(category)
        depth = estimate_depth(category, box, camera, image.shape, settings, priors)
        detections.append({"class_id": category, "class_name": ["cone", "barrier", "stop_sign"][category],
                           "confidence": float(confidence), "bbox_xyxy": box, "distance": depth})
        color = COLORS[category]
        x1, y1, x2, y2 = map(lambda value: int(round(value)), box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        marker = "*" if depth["calibration_source"] == "assumed_fov" else ""
        value = f"~{depth['depth_m']:.1f} m{marker}" if depth["depth_m"] is not None else "N/A"
        text = f"{LABELS[category]}, {value}"
        font_size = max(0.42, min(0.7, width / 2200))
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
        left = min(max(0, x1), max(0, width - tw - 8))
        top = max(0, y1 - th - baseline - 8)
        label_width, label_height = tw + 8, th + baseline + 8
        original_left, original_top = left, top
        candidates = []
        for offset in range(8):
            for shift in (0, -label_width, label_width):
                candidate_x = min(max(0, left + shift), max(0, width - label_width))
                candidate_y = top - offset * (label_height + 3)
                if candidate_y >= 0 and candidate_y + label_height < height - 30:
                    candidates.append((candidate_x, candidate_y))
        candidates.append((left, min(y2 + 4, height - label_height - 31)))
        for candidate_x, candidate_y in candidates:
            region = (candidate_x, candidate_y, candidate_x + label_width, candidate_y + label_height)
            overlaps = any(not (region[2]+2 <= other[0] or region[0] >= other[2]+2 or
                                region[3]+2 <= other[1] or region[1] >= other[3]+2) for other in label_regions)
            if not overlaps:
                left, top = candidate_x, candidate_y
                break
        label_regions.append((left, top, left + label_width, top + label_height))
        if (left, top) != (original_left, original_top):
            cv2.line(canvas, (max(0,x1), max(0,y1)), (left + label_width//2, top + label_height), color, 1)
        cv2.rectangle(canvas, (left, top), (left + tw + 8, top + th + baseline + 8), color, -1)
        cv2.putText(canvas, text, (left + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, height - 30), (width, height), (24, 24, 24), -1)
    legend = "Camera-forward depth estimates | * assumed camera calibration | N/A: unreliable geometry"
    cv2.putText(canvas, legend, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, min(0.55, width/2300)), (235,235,235), 1, cv2.LINE_AA)
    return canvas, detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", default=str(PROJECT / "models/baseline.pt"))
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/demo")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.source is None:
        data = yaml.safe_load((PROJECT / "configs/data.yaml").read_text(encoding="utf-8"))
        args.source = Path(data["path"]) / data["test"]
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [args.source] if args.source.is_file() else sorted(path for path in args.source.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not sources:
        parser.error("No input images found")
    # Include both dataset sources in the gallery, without selecting on measured model accuracy.
    bdd = [path for path in sources if path.name.startswith("bdd_")]
    other = [path for path in sources if not path.name.startswith("bdd_")]
    rng = random.Random(42)
    rng.shuffle(bdd)
    rng.shuffle(other)
    bdd_count = min(4, len(bdd), args.max_images)
    sources = bdd[:bdd_count] + other[:max(0, args.max_images - bdd_count)]
    settings = json.loads((PROJECT / "configs/camera.json").read_text())
    priors = yaml.safe_load((PROJECT / "configs/object_size.yaml").read_text())
    cameras = load_calibrations()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs/ultralytics"))
    from ultralytics import YOLO
    model = YOLO(args.weights)
    records = []
    tiles = []
    video_path = args.output / "dataset_slideshow.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (960, 540))
    if not writer.isOpened():
        parser.error("Could not create demo video")
    for source in sources:
        image = cv2.imread(str(source))
        result = model.predict(image, imgsz=640, rect=False, device=args.device, conf=args.conf, max_det=50, verbose=False)[0]
        canvas, detections = annotate(image, result, cameras.get(source.name), settings, priors)
        cv2.imwrite(str(args.output / source.name), canvas)
        records.append({"filename": source.name, "detections": detections})
        tile = cv2.resize(canvas, (480, 270))
        tiles.append(tile)
        frame = cv2.resize(canvas, (960, 540))
        for _ in range(4):
            writer.write(frame)
    writer.release()
    rows = (len(tiles) + 2) // 3
    gallery = np.full((rows * 270, 3 * 480, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        x, y = index % 3 * 480, index // 3 * 270
        gallery[y:y+270, x:x+480] = tile
    cv2.imwrite(str(args.output / "gallery.jpg"), gallery)
    (args.output / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(records), "detections": sum(len(row["detections"]) for row in records), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
