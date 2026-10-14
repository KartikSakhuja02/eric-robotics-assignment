"""Validate three-class YOLO labels and save annotated inspection images."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ["cone", "barrier", "stop_sign"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [(0, 180, 255), (255, 160, 0), (0, 60, 255)]


def read_labels(path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 values for a detection box")
        category = int(parts[0])
        box = list(map(float, parts[1:]))
        if category not in range(3) or not all(math.isfinite(value) for value in box):
            raise ValueError(f"{path}:{line_number}: invalid class or non-finite coordinates")
        x, y, width, height = box
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: coordinates must be normalized; dimensions must be positive")
        if x - width / 2 < -1e-5 or y - height / 2 < -1e-5 or x + width / 2 > 1 + 1e-5 or y + height / 2 > 1 + 1e-5:
            raise ValueError(f"{path}:{line_number}: box extends beyond image boundaries")
        rows.append((category, box))
    return rows


def inspect(root, output, sample_count=12, seed=42):
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    errors = []
    seen_stems = set()
    rng = random.Random(seed)
    for split in ("train", "val", "test"):
        images = sorted(p for p in (root / "images" / split).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        selected = set(rng.sample(images, min(sample_count, len(images))))
        counts = [0, 0, 0]
        negatives = 0
        for image_path in images:
            if image_path.stem in seen_stems:
                errors.append(f"Duplicate image stem across dataset: {image_path.stem}")
            seen_stems.add(image_path.stem)
            image = cv2.imread(str(image_path))
            if image is None:
                errors.append(f"Cannot decode image: {image_path}")
                continue
            label = root / "labels" / split / (image_path.stem + ".txt")
            if not label.is_file():
                errors.append(f"Missing label: {label}")
                continue
            try:
                rows = read_labels(label)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            negatives += not rows
            height, width = image.shape[:2]
            for category, (x, y, bw, bh) in rows:
                counts[category] += 1
                if image_path in selected:
                    x1, y1 = round((x - bw / 2) * width), round((y - bh / 2) * height)
                    x2, y2 = round((x + bw / 2) * width), round((y + bh / 2) * height)
                    cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[category], 2)
                    cv2.putText(image, CLASSES[category], (max(0, x1), max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[category], 2)
            if image_path in selected and not cv2.imwrite(str(output / f"{split}_{image_path.name}"), image):
                errors.append(f"Failed to save preview for {image_path}")
        missing_classes = [name for index, name in enumerate(CLASSES) if not counts[index]]
        summary[split] = {"images": len(images), "negative_images": negatives,
                          "instances": dict(zip(CLASSES, counts)), "missing_classes": missing_classes}
        if not images:
            errors.append(f"Empty {split} split")
        if missing_classes:
            errors.append(f"{split} has no labeled instances of: {', '.join(missing_classes)}")
        expected = {image.stem for image in images}
        for label in (root / "labels" / split).glob("*.txt"):
            if label.stem not in expected:
                errors.append(f"Orphan label: {label}")
    manifest = root / "manifest.csv"
    if manifest.is_file():
        assigned = {}
        manifest_names = set()
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["filename"] in manifest_names:
                    errors.append(f"Duplicate manifest image: {row['filename']}")
                manifest_names.add(row["filename"])
                if row["group"] in assigned and assigned[row["group"]] != row["split"]:
                    errors.append(f"Recording/session group leaks across splits: {row['group']}")
                assigned[row["group"]] = row["split"]
    report = {"splits": summary, "errors": errors}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Inspection images: {output.resolve()}")
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "data/processed")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/dataset_check")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    if not inspect(args.root, args.output, args.samples):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
