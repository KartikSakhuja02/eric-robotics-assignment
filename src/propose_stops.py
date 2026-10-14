"""Create stop-sign proposals for visual review; never treat proposals as verified labels."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def make_sheets(rows, output):
    tile_width, tile_height = 192, 224
    for start in range(0, len(rows), 40):
        page = rows[start:start + 40]
        sheet = np.full((8 * tile_height, 5 * tile_width, 3), 245, dtype=np.uint8)
        for index, row in enumerate(page):
            image = cv2.imread(row["crop_path"])
            if image is None:
                continue
            scale = min(184 / image.shape[1], 182 / image.shape[0])
            image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
            x = (index % 5) * tile_width
            y = (index // 5) * tile_height
            xoff, yoff = (tile_width - image.shape[1]) // 2, (192 - image.shape[0]) // 2
            sheet[y + yoff:y + yoff + image.shape[0], x + xoff:x + xoff + image.shape[1]] = image
            color = (0, 100, 0) if row["proposed_stop"] else (70, 70, 70)
            cv2.putText(sheet, row["candidate_id"], (x + 8, y + 205), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            cv2.putText(sheet, f"proposal {row['proposal_confidence']:.2f}", (x + 8, y + 221), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        cv2.imwrite(str(output / f"sheet_{start // 40:03d}.jpg"), sheet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/label_review")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--id-prefix", default="B")
    parser.add_argument("--exclude-root", type=Path, help="Skip image filenames already scanned in another folder")
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "proposals.jsonl").exists():
        parser.error("Review output exists; choose a new --output folder")
    output.mkdir(parents=True, exist_ok=True)
    (output / "crops").mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs/ultralytics"))
    from ultralytics import YOLO
    model = YOLO(args.weights)
    stop_id = next(key for key, name in model.names.items() if name == "stop sign")
    metadata = {}
    with (args.bdd_root / "candidates.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata[row["filename"]] = row
    rows = []
    processed = 0
    positive_images = 0
    with (output / "proposals.jsonl").open("w", encoding="utf-8") as handle:
        for split in ("train", "val"):
            folder = args.bdd_root / "images" / split
            if not folder.is_dir():
                continue
            source = str(folder)
            if args.exclude_root:
                excluded = {path.name for path in args.exclude_root.rglob("*.jpg")}
                filenames = [path.resolve() for path in sorted(folder.glob("*.jpg")) if path.name not in excluded]
                if not filenames:
                    continue
                source_file = output / f"{split}_sources.txt"
                source_file.write_text("\n".join(map(str, filenames)), encoding="utf-8")
                source = str(source_file)
            results = model.predict(source=source, stream=True, classes=[stop_id], conf=args.conf,
                                    imgsz=640, rect=False, batch=8, device=0, quantize=16, verbose=False)
            for result in results:
                processed += 1
                filename = Path(result.path).name
                frame = metadata[filename]
                detections = list(zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()))
                if not detections:
                    if processed % 100 == 0:
                        print(f"Reviewed by proposal model: {processed}; candidate images: {positive_images}", flush=True)
                    continue
                signs = [[box["x1"], box["y1"], box["x2"], box["y2"]] for box in frame["traffic_sign_candidates"]]
                confidence = [max((score for box, score in detections if box_iou(sign, box) >= 0.35), default=0.0) for sign in signs]
                if not any(score >= args.conf for score in confidence):
                    continue
                positive_images += 1
                image = result.orig_img
                height, width = image.shape[:2]
                for box_index, (box, score) in enumerate(zip(signs, confidence)):
                    candidate_id = f"{args.id_prefix}{positive_images:04d}_{box_index:02d}"
                    x1, y1, x2, y2 = box
                    margin_x, margin_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
                    crop = image[max(0, int(y1 - margin_y)):min(height, int(y2 + margin_y) + 1),
                                 max(0, int(x1 - margin_x)):min(width, int(x2 + margin_x) + 1)]
                    if not crop.size:
                        continue
                    crop_path = output / "crops" / f"{candidate_id}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    row = {"candidate_id": candidate_id, "filename": filename,
                           "source_path": str(Path(result.path).resolve()), "source_split": split,
                           "group": "bdd_" + filename.removeprefix(f"bdd_{split}_").removesuffix(".jpg"),
                           "bbox_xyxy": box, "proposal_confidence": score,
                           "proposed_stop": score >= args.conf, "crop_path": str(crop_path),
                           "annotation_status": "unverified"}
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                if processed % 100 == 0:
                    print(f"Reviewed by proposal model: {processed}; candidate images: {positive_images}", flush=True)
    make_sheets(rows, output)
    summary = {"images_scanned": processed, "images_with_proposals": positive_images,
               "traffic_sign_crops": len(rows), "stop_proposals": sum(row["proposed_stop"] for row in rows),
               "labels_verified": False, "proposal_model": args.weights}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
