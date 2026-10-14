"""Convert nuImages cone/barrier annotations and preserve per-image calibration.

Output labels require review for unannotated stop signs before three-class training.
This converter reads JSON tables directly; no nuScenes SDK installation is needed.
"""

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from .prepare_data import CLASSES, iter_json_array
from .io_utils import link_or_copy_image

CATEGORY_IDS = {"movable_object.trafficcone": 0, "movable_object.barrier": 1}


def table_records(metadata, table):
    with (metadata / f"{table}.json").open(encoding="utf-8-sig") as handle:
        yield from iter_json_array(handle)


def token_table(metadata, table):
    return {record["token"]: record for record in table_records(metadata, table)}


def normalized_box(box, width, height):
    if len(box) != 4 or width <= 0 or height <= 0:
        raise ValueError("Invalid box or image dimensions")
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("Non-finite box coordinate")
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def convert(args):
    root = args.root.resolve()
    metadata = root / args.version
    required = ("category", "sample", "sensor", "calibrated_sensor", "object_ann", "sample_data")
    for table in required:
        if not (metadata / f"{table}.json").is_file():
            raise ValueError(f"Missing {metadata / (table + '.json')}; download metadata for this version")
    output = args.output.resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Input dataset and converted output must be separate sibling folders")
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise ValueError("Output contains files; choose a new --output folder to preserve edits")
    categories = token_table(metadata, "category")
    samples = token_table(metadata, "sample")
    sensors = token_table(metadata, "sensor")
    calibrations = token_table(metadata, "calibrated_sensor")
    plan_path = root / "selection_plan.json"
    selected_tokens = None
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if args.version in plan:
            selected_tokens = {row["token"] for row in plan[args.version]}
    boxes = defaultdict(list)
    for annotation in table_records(metadata, "object_ann"):
        name = categories[annotation["category_token"]]["name"]
        if name in CATEGORY_IDS:
            boxes[annotation["sample_data_token"]].append((CATEGORY_IDS[name], annotation["bbox"]))
    if not boxes:
        raise ValueError("No cone/barrier object annotations found in this version")
    selected = []
    missing_files = 0
    for image in table_records(metadata, "sample_data"):
        if selected_tokens is not None and image["token"] not in selected_tokens:
            continue
        if image["token"] not in boxes or not image.get("is_key_frame"):
            continue
        calibration = calibrations[image["calibrated_sensor_token"]]
        sensor = sensors[calibration["sensor_token"]]
        if sensor["channel"] != args.camera:
            continue
        source = (root / image["filename"]).resolve()
        if root not in source.parents:
            raise ValueError("An image filename points outside the dataset root")
        if not source.is_file():
            missing_files += 1
            continue
        width, height = image["width"], image["height"]
        labels = []
        for category, raw_box in boxes[image["token"]]:
            box = normalized_box(raw_box, width, height)
            if box:
                labels.append((category, box))
        if not labels:
            continue
        if any(char in image["token"] for char in ("/", "\\")):
            raise ValueError("Invalid sample-data token")
        selected.append((image, calibration, source, labels))
        if len(selected) >= args.max_images:
            break
    if not selected:
        raise ValueError(f"No eligible {args.camera} images found; check downloaded samples and version")
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images/train"
    label_dir = output / "labels/train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    counts = [0, 0, 0]
    with (output / "camera_index.jsonl").open("w", encoding="utf-8") as cameras, \
            (output / "groups.csv").open("w", newline="", encoding="utf-8") as groups:
        writer = csv.DictWriter(groups, fieldnames=["filename", "group", "split"])
        writer.writeheader()
        for image, calibration, source, labels in selected:
            filename = f"nuim_{image['token']}{source.suffix.lower()}"
            link_or_copy_image(source, image_dir / filename)
            label_text = ""
            for category, box in labels:
                counts[category] += 1
                label_text += str(category) + " " + " ".join(f"{value:.8f}" for value in box) + "\n"
            (label_dir / Path(filename).with_suffix(".txt")).write_text(label_text, encoding="utf-8")
            log_token = samples[image["sample_token"]]["log_token"]
            # The temporary images/train folder is an export layout, not a final split.
            # Allocate groups after reviewing/merging; original dataset version is retained.
            writer.writerow({"filename": filename, "group": f"nuim_{log_token}", "split": ""})
            cameras.write(json.dumps({
                "filename": filename, "dataset": "nuImages", "version": args.version,
                "original_filename": image["filename"], "sample_data_token": image["token"],
                "group": f"nuim_{log_token}", "camera_channel": args.camera,
                "image_width": image["width"], "image_height": image["height"],
                "camera_intrinsic": calibration["camera_intrinsic"],
                "camera_distortion": calibration["camera_distortion"],
                "camera_to_ego_rotation_wxyz": calibration["rotation"],
                "camera_to_ego_translation_m": calibration["translation"],
                "stop_sign_review_required": True,
            }) + "\n")
    (output / "data.yaml").write_text(yaml.safe_dump({
        "path": output.as_posix(), "train": "images/train",
        "names": dict(enumerate(CLASSES)),
    }, sort_keys=False), encoding="utf-8")
    report = {"images": len(selected), "instances": dict(zip(CLASSES, counts)),
              "missing_eligible_image_files_encountered": missing_files,
              "camera": args.camera, "version": args.version,
              "annotation_status": "Cone/barrier boxes converted; review all images for stop signs before training",
              "distance_ground_truth_available": False}
    (output / "conversion_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved annotations, recording groups and camera parameters to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", choices=["v1.0-mini", "v1.0-train", "v1.0-val"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--camera", default="CAM_FRONT")
    args = parser.parse_args()
    if args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.camera == "CAM_BACK":
        parser.error("Use a front/side camera for this baseline; CAM_BACK has a different distortion model")
    try:
        convert(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
