"""Audit incidental stop signs omitted from the nuImages category taxonomy."""

import argparse
import json
import os
from pathlib import Path

import cv2

from .propose_stops import PROJECT, make_sheets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=[PROJECT / "data/nuimages_train", PROJECT / "data/nuimages_val"])
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/nuimages_stop_review")
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "proposals.jsonl").exists():
        parser.error("Output already exists")
    (output / "crops").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs/ultralytics"))
    from ultralytics import YOLO
    model = YOLO(str(PROJECT / "yolo11n.pt"))
    stop_id = next(key for key, name in model.names.items() if name == "stop sign")
    rows = []
    scanned = 0
    with (output / "proposals.jsonl").open("w", encoding="utf-8") as handle:
        for source in args.inputs:
            results = model.predict(source=str(source / "images/train"), stream=True, classes=[stop_id], conf=0.2,
                                    imgsz=640, rect=False, batch=8, device=0, quantize=16, verbose=False)
            for result in results:
                scanned += 1
                image = result.orig_img
                height, width = image.shape[:2]
                for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    x1, y1, x2, y2 = box
                    if y2 - y1 < 12 or x2 - x1 < 8:
                        continue
                    candidate_id = f"N{len(rows) + 1:04d}"
                    mx, my = (x2 - x1) * 0.15, (y2 - y1) * 0.15
                    crop = image[max(0, int(y1 - my)):min(height, int(y2 + my) + 1), max(0, int(x1 - mx)):min(width, int(x2 + mx) + 1)]
                    if not crop.size:
                        continue
                    crop_path = output / "crops" / f"{candidate_id}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    row = {"candidate_id": candidate_id, "filename": Path(result.path).name,
                           "source_path": str(Path(result.path).resolve()), "bbox_xyxy": box,
                           "proposal_confidence": score, "proposed_stop": True,
                           "crop_path": str(crop_path), "annotation_status": "unverified"}
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                if scanned % 100 == 0:
                    print(f"nuImages stop audit: {scanned} images; {len(rows)} proposals", flush=True)
    make_sheets(rows, output)
    print(json.dumps({"images_scanned": scanned, "proposals": len(rows), "labels_verified": False}), flush=True)


if __name__ == "__main__":
    main()
