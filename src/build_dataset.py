"""Build a grouped three-class dataset with reviewed BDD stop signs."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

from .io_utils import link_or_copy_image
from .prepare_data import CLASSES, split_yolo

PROJECT = Path(__file__).resolve().parents[1]


def reviewed_image_path(record):
    original = Path(record["source_path"])
    if not original.is_absolute():
        original = PROJECT / original
    if original.is_file():
        return original
    # Review manifests are portable across checkouts after candidate extraction.
    for pool in ("bdd_pool", "bdd_val_pool", "bdd_candidates"):
        candidate = PROJECT / "data/raw" / pool / "images" / record["source_split"] / record["filename"]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Reviewed BDD image unavailable: {record['filename']}; prepare BDD candidate pools first")


def stable_holdout(group):
    return "val" if int(hashlib.sha256(group.encode()).hexdigest(), 16) % 2 == 0 else "test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuimages", type=Path, nargs="+", default=[PROJECT / "data/nuimages_train", PROJECT / "data/nuimages_val"])
    parser.add_argument("--reviews", type=Path, default=PROJECT / "outputs/label_review/verified_stops.json")
    parser.add_argument("--proposals", type=Path, default=PROJECT / "outputs/label_review/proposals.jsonl")
    parser.add_argument("--staging", type=Path, default=PROJECT / "data/annotated")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed")
    parser.add_argument("--stop-repeat", type=int, default=8)
    parser.add_argument("--nuimages-stop-reviews", type=Path, default=PROJECT / "outputs/nuimages_stop_review/verified_stops.json")
    parser.add_argument("--nuimages-stop-proposals", type=Path, default=PROJECT / "outputs/nuimages_stop_review/proposals.jsonl")
    parser.add_argument("--extra-review", type=Path)
    parser.add_argument("--extra-proposals", type=Path)
    args = parser.parse_args()
    if args.stop_repeat < 1:
        parser.error("--stop-repeat must be positive")
    staging = args.staging.resolve()
    if staging.exists() and any(path.is_file() for path in staging.rglob("*")):
        parser.error("Staging contains files; select a new --staging path")
    (staging / "images").mkdir(parents=True, exist_ok=True)
    (staging / "labels").mkdir(parents=True, exist_ok=True)
    group_rows = []
    cameras = []
    source_counts = Counter()
    nuimage_stops = defaultdict(list)
    if args.nuimages_stop_reviews.is_file():
        extra_review = json.loads(args.nuimages_stop_reviews.read_text(encoding="utf-8"))
        extra_ids = set(extra_review["verified_ids"])
        with args.nuimages_stop_proposals.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["candidate_id"] in extra_ids:
                    nuimage_stops[row["filename"]].append(row["bbox_xyxy"])
                    extra_ids.remove(row["candidate_id"])
        if extra_ids:
            parser.error(f"Unmatched nuImages review IDs: {sorted(extra_ids)}")
    for source in args.nuimages:
        data = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
        if [data["names"][index] for index in range(3)] != CLASSES:
            parser.error(f"Unexpected category order in {source}")
        camera_rows = [json.loads(line) for line in (source / "camera_index.jsonl").read_text(encoding="utf-8").splitlines()]
        for camera in camera_rows:
            filename = camera["filename"]
            link_or_copy_image(source / "images/train" / filename, staging / "images" / filename)
            label_text = (source / "labels/train" / Path(filename).with_suffix(".txt")).read_text(encoding="utf-8")
            for x1, y1, x2, y2 in nuimage_stops.get(filename, []):
                width, height = camera["image_width"], camera["image_height"]
                label_text += f"2 {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} {(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}\n"
            (staging / "labels" / Path(filename).with_suffix(".txt")).write_text(label_text, encoding="utf-8")
            split = "train" if camera["version"] == "v1.0-train" else stable_holdout(camera["group"])
            group_rows.append({"filename": filename, "group": camera["group"], "split": split})
            cameras.append(camera)
            source_counts[camera["version"]] += 1
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    verified = set(reviews["verified_ids"])
    proposal_files = [args.proposals]
    if args.extra_review:
        if not args.extra_proposals:
            parser.error("--extra-review requires --extra-proposals")
        extra = json.loads(args.extra_review.read_text(encoding="utf-8"))
        verified.update(extra["verified_ids"])
        reviews["additional_review"] = extra
        proposal_files.append(args.extra_proposals)
    matches = defaultdict(list)
    for proposal_file in proposal_files:
        with proposal_file.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["candidate_id"] in verified:
                    matches[row["filename"]].append(row)
                    verified.remove(row["candidate_id"])
    if verified:
        parser.error(f"Verified IDs not found in proposal file: {sorted(verified)}")
    unique_stop_instances = 0
    stop_counts = Counter()
    for filename, records in matches.items():
        first = records[0]
        reviewed_path = reviewed_image_path(first)
        image = cv2.imread(str(reviewed_path))
        if image is None:
            parser.error(f"Cannot read reviewed image: {filename}")
        height, width = image.shape[:2]
        text = ""
        for record in records:
            x1, y1, x2, y2 = record["bbox_xyxy"]
            text += f"2 {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} {(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}\n"
        unique_stop_instances += len(records)
        split = "train" if first["source_split"] == "train" else stable_holdout(first["group"])
        repeats = args.stop_repeat if split == "train" else 1
        for repeat in range(repeats):
            name = filename if repeat == 0 else f"{Path(filename).stem}_repeat{repeat:02d}{Path(filename).suffix}"
            link_or_copy_image(reviewed_path, staging / "images" / name)
            (staging / "labels" / Path(name).with_suffix(".txt")).write_text(text, encoding="utf-8")
            group_rows.append({"filename": name, "group": first["group"], "split": split})
            stop_counts[split] += len(records)
            cameras.append({"filename": name, "dataset": "BDD100K", "group": first["group"],
                            "image_width": width, "image_height": height,
                            "calibration_source": "unavailable", "annotation_source": "visually reviewed traffic-sign boxes"})
        source_counts["bdd_unique_images"] += 1
    groups_path = staging / "groups.csv"
    with groups_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "group", "split"])
        writer.writeheader()
        writer.writerows(group_rows)
    (staging / "data.yaml").write_text(yaml.safe_dump({"names": dict(enumerate(CLASSES))}), encoding="utf-8")
    split_yolo(argparse.Namespace(input=staging, groups=groups_path, output=args.output,
                                 config=PROJECT / "configs/data.yaml", seed=42))
    (PROJECT / "configs/dataset_manifest.csv").write_text((args.output / "manifest.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (PROJECT / "configs/image_calibrations.jsonl").write_text("".join(json.dumps(camera) + "\n" for camera in cameras), encoding="utf-8")
    report = {"source_counts": dict(source_counts), "unique_stop_instances": unique_stop_instances,
              "visually_verified_nuimages_stop_instances": sum(map(len, nuimage_stops.values())),
              "stop_instances_after_training_oversampling": dict(stop_counts), "training_stop_repeat": args.stop_repeat,
              "holdout_images_repeated": False, "annotation_review": reviews,
              "stop_sign_coverage": "Model-assisted selection and visual verification; limited sample coverage",
              "nuimages_stop_sign_coverage": "Model-assisted incidental-stop audit; positive crops visually reviewed, possible missed signs remain"}
    path = PROJECT / "outputs/metrics/dataset_build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
