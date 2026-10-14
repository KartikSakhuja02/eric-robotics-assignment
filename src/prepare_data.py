"""Select BDD candidates and split manually verified YOLO annotations."""

import argparse
import csv
import io
import json
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from .io_utils import link_or_copy_image

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ["cone", "barrier", "stop_sign"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_json_array(stream, chunk_size=65536):
    """Read one array item at a time without loading the entire annotation file."""
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False
    started = False
    after_item = False
    first_item = True
    def finish_array(remainder):
        if remainder.strip():
            raise ValueError("Unexpected content after JSON array")
        for chunk in iter(lambda: stream.read(chunk_size), ""):
            if chunk.strip():
                raise ValueError("Unexpected content after JSON array")
    while True:
        buffer = buffer.lstrip()
        if not buffer:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError("Unexpected end of JSON array")
            buffer += chunk
            continue
        if not started:
            if buffer[0] != "[":
                raise ValueError("Expected a top-level JSON array")
            buffer = buffer[1:]
            started = True
            continue
        if after_item:
            if buffer[0] == "]":
                finish_array(buffer[1:])
                return
            if buffer[0] != ",":
                raise ValueError("Expected a comma between JSON items")
            buffer = buffer[1:]
            after_item = False
            continue
        if buffer[0] == "]":
            if first_item:
                finish_array(buffer[1:])
                return
            raise ValueError("Trailing comma in JSON array")
        # raw_decode accepts a partial exponent (e.g. '1.25e' as 1.25).
        # Wait for a numeric separator before decoding a top-level number.
        if buffer[0] in "-0123456789" and not re.search(r"[\s,\]]", buffer) and not eof:
            chunk = stream.read(chunk_size)
            eof = not chunk
            buffer += chunk
            if len(buffer) > 16 * 1024 * 1024:
                raise ValueError("A JSON item exceeded the 16 MiB safety limit")
            continue
        try:
            item, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError as exc:
            if eof:
                raise ValueError("Incomplete or invalid JSON item") from exc
            chunk = stream.read(chunk_size)
            eof = not chunk
            buffer += chunk
            if len(buffer) > 16 * 1024 * 1024:
                raise ValueError("A JSON item exceeded the 16 MiB safety limit")
            continue
        # Numbers can end at a chunk boundary; wait for their delimiter.
        if end == len(buffer) and not eof:
            chunk = stream.read(chunk_size)
            eof = not chunk
            buffer += chunk
            continue
        buffer = buffer[end:]
        first_item = False
        after_item = True
        yield item


def bdd_candidates(args):
    output = args.output.resolve()
    if (output / "candidates.jsonl").exists():
        raise ValueError("Candidate output already exists; select a new --output folder")
    output.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(args.archive) as archive, \
            (output / "candidates.jsonl").open("w", encoding="utf-8") as metadata, \
            (output / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=["filename", "group", "original_split"])
        writer.writeheader()
        for split, limit in [("train", args.train_limit), ("val", args.val_limit)]:
            if limit == 0:
                continue
            suffix = f"bdd100k_labels_images_{split}.json"
            matches = [name for name in archive.namelist() if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {suffix}, found {len(matches)}")
            image_members = {
                Path(name).name: name for name in archive.namelist()
                if f"/images/100k/{split}/" in name and name.lower().endswith(".jpg")
            }
            split_dir = output / "images" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            selected = 0
            seen = set()
            with archive.open(matches[0]) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as text:
                for frame in iter_json_array(text):
                    if args.daytime_only and frame.get("attributes", {}).get("timeofday") != "daytime":
                        continue
                    signs = []
                    for label in frame.get("labels", []):
                        box = label.get("box2d")
                        flags = label.get("attributes", {})
                        if label.get("category") != "traffic sign" or not box:
                            continue
                        if flags.get("occluded") or flags.get("truncated"):
                            continue
                        if box["y2"] - box["y1"] < args.min_height or box["x2"] <= box["x1"]:
                            continue
                        signs.append(box)
                    original_name = Path(frame.get("name", "")).name
                    if not signs or original_name not in image_members or original_name in seen:
                        continue
                    filename = f"bdd_{split}_{original_name}"
                    reuse_root = getattr(args, "reuse_root", None)
                    reuse = reuse_root / "images" / split / filename if reuse_root else None
                    if reuse and reuse.is_file() and reuse.stat().st_size == archive.getinfo(image_members[original_name]).file_size:
                        link_or_copy_image(reuse, split_dir / filename)
                    else:
                        with archive.open(image_members[original_name]) as source, (split_dir / filename).open("wb") as target:
                            shutil.copyfileobj(source, target)
                    seen.add(original_name)
                    writer.writerow({"filename": filename, "group": f"bdd_{Path(original_name).stem}", "original_split": split})
                    metadata.write(json.dumps({
                        "filename": filename, "original_split": split,
                        "attributes": frame.get("attributes", {}),
                        "traffic_sign_candidates": signs,
                        "verified_stop_sign": False,
                    }) + "\n")
                    selected += 1
                    if selected >= limit:
                        break
            total += selected
            print(f"Selected {selected} {split} images", flush=True)
    print(f"Saved {total} candidates to {output}")
    print("Review and label these images manually: traffic sign is not a verified stop sign.")


def split_yolo(args):
    import yaml

    root = args.input.resolve()
    images_root = root / "images"
    labels_root = root / "labels"
    images = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No images found under {images_root}")
    # Check exported category IDs before accepting a CVAT/other YOLO export.
    category_file = root / "data.yaml"
    if not category_file.is_file():
        raise ValueError("Keep the annotation export's data.yaml in --input so class IDs can be checked")
    category_data = yaml.safe_load(category_file.read_text(encoding="utf-8-sig")) or {}
    names = category_data.get("names", [])
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if names != CLASSES:
        raise ValueError(f"Export class order must be {CLASSES}; found {names}. Remap labels before merging sources.")
    stems = [p.stem for p in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Image stems must be unique across all source folders")
    with args.groups.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"filename", "group"}.issubset(reader.fieldnames or []):
            raise ValueError("Groups CSV needs filename,group and optional split columns")
        rows = list(reader)
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("Duplicate filenames in groups CSV")
    groups_by_image = {row["filename"]: row for row in rows}
    grouped = defaultdict(list)
    fixed = {}
    labels = {}
    for image in images:
        row = groups_by_image.get(image.name)
        if not row or not row["group"].strip():
            raise ValueError(f"Missing group metadata for {image.name}")
        group = row["group"].strip()
        assigned = (row.get("split") or "").strip()
        if assigned and assigned not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {image.name}: {assigned}")
        if assigned and group in fixed and fixed[group] != assigned:
            raise ValueError(f"Group {group} was assigned to multiple splits")
        if assigned:
            fixed[group] = assigned
        grouped[group].append(image)
        relative = image.relative_to(images_root).with_suffix(".txt")
        label = labels_root / relative
        if not label.is_file():
            raise ValueError(f"Missing label for {image.name}; verified negative images need an empty .txt file")
        labels[image] = label
    output = args.output.resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Input and output folders must be separate, without nesting")
    if output.exists() and any(p.is_file() for p in output.rglob("*")):
        raise ValueError("Output contains files; use a new --output folder")
    remaining = sorted(set(grouped) - set(fixed))
    random.Random(args.seed).shuffle(remaining)
    # Ratios apply to freely assigned groups; explicit split assignments are respected.
    train_count = round(len(remaining) * 0.7)
    val_count = round(len(remaining) * 0.15)
    if len(remaining) >= 3:
        train_count = min(max(1, train_count), len(remaining) - 2)
        val_count = min(max(1, val_count), len(remaining) - train_count - 1)
    assignments = dict(fixed)
    for index, group in enumerate(remaining):
        assignments[group] = "train" if index < train_count else "val" if index < train_count + val_count else "test"
    if set(assignments.values()) != {"train", "val", "test"}:
        raise ValueError("Need nonempty train, val and test splits; assign more independent groups")
    output.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "group", "split"])
        writer.writeheader()
        for group, members in sorted(grouped.items()):
            split = assignments[group]
            for image in members:
                target_image = output / "images" / split / image.name
                target_label = output / "labels" / split / (image.stem + ".txt")
                target_image.parent.mkdir(parents=True, exist_ok=True)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy_image(image, target_image)
                shutil.copy2(labels[image], target_label)
                writer.writerow({"filename": image.name, "group": group, "split": split})
                counts[split] += 1
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(yaml.safe_dump({
        "path": output.as_posix(), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": dict(enumerate(CLASSES)),
    }, sort_keys=False), encoding="utf-8")
    print(json.dumps({"image_counts": dict(counts), "config": str(args.config.resolve())}, indent=2))


def groups_template(args):
    images = sorted(p for p in (args.input / "images").rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError("No exported images found under --input/images")
    if args.output.exists():
        raise ValueError("Groups CSV already exists; select a new --output filename to preserve edits")
    source_groups = {}
    for source_csv in getattr(args, "source_groups", None) or []:
        with source_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if not row.get("filename") or not row.get("group"):
                    raise ValueError(f"Missing filename/group in {source_csv}")
                if row["filename"] in source_groups and source_groups[row["filename"]]["group"] != row["group"]:
                    raise ValueError(f"Conflicting source groups for {row['filename']}")
                source_groups[row["filename"]] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    unresolved = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "group", "split"])
        writer.writeheader()
        for image in images:
            group, split = "", ""
            if image.name in source_groups:
                row = source_groups[image.name]
                group, split = row["group"], row.get("split") or ""
            elif image.stem.startswith("bdd_train_"):
                group, split = "bdd_" + image.stem.removeprefix("bdd_train_"), "train"
            elif image.stem.startswith("bdd_val_"):
                group = "bdd_" + image.stem.removeprefix("bdd_val_")
                # Split the officially labeled validation images into val/test groups.
                import hashlib
                split = "val" if int(hashlib.sha256(group.encode()).hexdigest(), 16) % 2 == 0 else "test"
            else:
                camera = re.match(r"^(cam_s\d+)_", image.stem)
                if camera:
                    group = camera.group(1)
                else:
                    unresolved += 1
            writer.writerow({"filename": image.name, "group": group, "split": split})
    print(f"Wrote {len(images)} rows to {args.output.resolve()}")
    print(f"Fill recording/log IDs for {unresolved} unresolved rows; review all groups before splitting.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bdd = commands.add_parser("bdd-candidates", help="Select unlabeled traffic-sign candidate images from a ZIP")
    bdd.add_argument("--archive", type=Path, required=True)
    bdd.add_argument("--output", type=Path, default=PROJECT / "data/raw/bdd_candidates")
    bdd.add_argument("--train-limit", type=int, default=400)
    bdd.add_argument("--val-limit", type=int, default=100)
    bdd.add_argument("--min-height", type=float, default=24)
    bdd.add_argument("--daytime-only", action="store_true")
    bdd.add_argument("--reuse-root", type=Path, help="Reuse matching images from an existing candidate subset")
    bdd.set_defaults(run=bdd_candidates)
    split = commands.add_parser("split", help="Split verified YOLO labels by recording/session group")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--groups", type=Path, required=True)
    split.add_argument("--output", type=Path, default=PROJECT / "data/processed")
    split.add_argument("--config", type=Path, default=PROJECT / "configs/data.yaml")
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(run=split_yolo)
    groups = commands.add_parser("groups-template", help="Create a recording/session manifest for an annotation export")
    groups.add_argument("--input", type=Path, required=True)
    groups.add_argument("--output", type=Path, default=PROJECT / "data/groups.csv")
    groups.add_argument("--source-groups", type=Path, nargs="+", help="Group manifests from dataset converters")
    groups.set_defaults(run=groups_template)
    args = parser.parse_args()
    if args.command == "bdd-candidates" and (args.train_limit < 0 or args.val_limit < 0 or args.min_height <= 0):
        parser.error("Limits must be nonnegative and min-height must be positive")
    try:
        args.run(args)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
