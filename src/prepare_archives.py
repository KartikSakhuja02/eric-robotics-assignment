"""Prepare a bounded nuImages subset directly from downloaded archives."""

import argparse
import json
import random
import shutil
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from .nuimages_to_yolo import CATEGORY_IDS, normalized_box, table_records, token_table

PROJECT = Path(__file__).resolve().parents[1]
TABLES = ("category", "sample", "sensor", "calibrated_sensor", "object_ann", "sample_data", "attribute", "log")
GIB = 1024 ** 3


def normalized_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    return path.as_posix()


def ensure_space(root, required, reserve_gb):
    free = shutil.disk_usage(root).free
    if free - required < reserve_gb * GIB:
        raise ValueError(f"Insufficient disk space: {free / GIB:.2f} GiB free, need {required / GIB:.2f} GiB plus {reserve_gb} GiB reserve")


def copy_member(archive, member, root, reserve_gb):
    destination = (root / normalized_member(member.name)).resolve()
    if root.resolve() not in destination.parents:
        raise ValueError("Archive member escapes the intended extraction directory")
    if not member.isfile():
        raise ValueError("Only ordinary files may be extracted")
    if destination.exists() and destination.stat().st_size == member.size:
        return 0
    ensure_space(root, member.size, reserve_gb)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with archive.extractfile(member) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if temporary.stat().st_size != member.size:
        raise ValueError(f"Incomplete extraction: {member.name}")
    temporary.replace(destination)
    return member.size


def extract_metadata(archive_path, root, reserve_gb):
    root.mkdir(parents=True, exist_ok=True)
    wanted = {f"{version}/{table}.json" for version in ("v1.0-train", "v1.0-val") for table in TABLES}
    wanted.add("LICENSE")
    extracted = 0
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            name = normalized_member(member.name)
            if name in wanted:
                extracted += copy_member(archive, member, root, reserve_gb)
                wanted.remove(name)
                print(f"Metadata ready: {name} ({member.size / 1024 ** 2:.1f} MiB)", flush=True)
    if wanted:
        raise ValueError(f"Metadata archive is missing: {sorted(wanted)}")
    return extracted


def plan_version(root, version, limit, seed, max_per_log):
    metadata = root / version
    categories = token_table(metadata, "category")
    samples = token_table(metadata, "sample")
    calibrations = token_table(metadata, "calibrated_sensor")
    sensors = token_table(metadata, "sensor")
    objects = defaultdict(list)
    for annotation in table_records(metadata, "object_ann"):
        name = categories[annotation["category_token"]]["name"]
        if name in CATEGORY_IDS:
            objects[annotation["sample_data_token"]].append((CATEGORY_IDS[name], annotation["bbox"]))
    candidates = {}
    pools = [[], []]
    for image in table_records(metadata, "sample_data"):
        if image["token"] not in objects or not image.get("is_key_frame"):
            continue
        calibration = calibrations[image["calibrated_sensor_token"]]
        if sensors[calibration["sensor_token"]]["channel"] != "CAM_FRONT":
            continue
        classes = set()
        for category, raw_box in objects[image["token"]]:
            box = normalized_box(raw_box, image["width"], image["height"])
            if box and box[2] * image["width"] >= 8 and box[3] * image["height"] >= 12:
                classes.add(category)
        if not classes:
            continue
        record = {"token": image["token"], "filename": normalized_member(image["filename"]),
                  "log_token": samples[image["sample_token"]]["log_token"], "classes": sorted(classes)}
        candidates[image["token"]] = record
        for category in classes:
            pools[category].append(image["token"])
    rng = random.Random(seed)
    for pool in pools:
        rng.shuffle(pool)
    positions = [0, 0]
    chosen = {}
    log_counts = Counter()
    while len(chosen) < limit:
        progressed = False
        for category in (0, 1):
            while positions[category] < len(pools[category]):
                token = pools[category][positions[category]]
                positions[category] += 1
                record = candidates[token]
                if token in chosen or log_counts[record["log_token"]] >= max_per_log:
                    continue
                chosen[token] = record
                log_counts[record["log_token"]] += 1
                progressed = True
                break
            if len(chosen) >= limit:
                break
        if not progressed:
            break
    if not chosen:
        raise ValueError(f"No eligible images in {version}")
    report = {"version": version, "candidate_images": len(candidates), "selected_images": len(chosen),
              "recordings": len(log_counts), "max_images_per_recording": max(log_counts.values()),
              "selected_frames_by_class": {"cone": sum(0 in row["classes"] for row in chosen.values()),
                                           "barrier": sum(1 in row["classes"] for row in chosen.values())}}
    print(json.dumps(report), flush=True)
    return list(chosen.values()), report


def extract_samples(archive_path, root, requested, reserve_gb):
    remaining = set(requested)
    sizes = {}
    extracted = 0
    archive_size = archive_path.stat().st_size
    last_update = time.monotonic()
    with archive_path.open("rb") as raw, tarfile.open(fileobj=raw, mode="r|gz") as archive:
        for member in archive:
            name = normalized_member(member.name)
            if name in remaining:
                extracted += copy_member(archive, member, root, reserve_gb)
                sizes[name] = member.size
                remaining.remove(name)
                if not remaining:
                    break
            if time.monotonic() - last_update >= 15:
                print(f"Selected images ready: {len(requested) - len(remaining)}/{len(requested)}; compressed archive scanned: {raw.tell() / archive_size:.0%}", flush=True)
                last_update = time.monotonic()
    if remaining:
        raise ValueError(f"Samples archive is missing {len(remaining)} requested images; examples: {sorted(remaining)[:3]}")
    return extracted, sum(sizes.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-archive", type=Path, default=PROJECT / "nuimages-v1.0-all-metadata.tgz")
    parser.add_argument("--samples-archive", type=Path, default=PROJECT / "nuimages-v1.0-all-samples.tgz")
    parser.add_argument("--root", type=Path, default=PROJECT / "data/raw/nuimages")
    parser.add_argument("--train-images", type=int, default=1000)
    parser.add_argument("--val-images", type=int, default=300)
    parser.add_argument("--max-per-log", type=int, default=20)
    parser.add_argument("--reserve-gb", type=float, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_images, args.val_images, args.max_per_log) <= 0 or args.reserve_gb < 0:
        parser.error("Image limits must be positive and disk reserve nonnegative")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    starting_free = shutil.disk_usage(root).free
    try:
        metadata_bytes = extract_metadata(args.metadata_archive, root, args.reserve_gb)
        plan, reports = {}, []
        for version, count in (("v1.0-train", args.train_images), ("v1.0-val", args.val_images)):
            rows, report = plan_version(root, version, count, args.seed, args.max_per_log)
            plan[version] = rows
            reports.append(report)
        plan_path = root / "selection_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        requested = {row["filename"] for rows in plan.values() for row in rows}
        image_bytes, subset_bytes = extract_samples(args.samples_archive, root, requested, args.reserve_gb)
        report = {"selection": reports, "new_metadata_bytes": metadata_bytes, "new_image_bytes": image_bytes,
                  "selected_image_bytes": subset_bytes, "starting_free_gib": starting_free / GIB,
                  "remaining_free_gib": shutil.disk_usage(root).free / GIB,
                  "archives_preserved": True, "selection_plan": str(plan_path)}
        report_path = PROJECT / "outputs/archive_preparation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    except (ValueError, OSError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
