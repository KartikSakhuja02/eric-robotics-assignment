"""Export matched OpenVINO FP32/INT8 models using training-only calibration."""

import argparse
import json
import os
import random
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]


def calibration_data(data_path, count=320):
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    training = root / data["train"]
    images = sorted(path for path in training.glob("*.jpg") if "_repeat" not in path.stem)
    stops = [path for path in images if path.name.startswith("bdd_")]
    others = [path for path in images if not path.name.startswith("bdd_")]
    random.Random(42).shuffle(others)
    selected = stops + others[:max(0, count - len(stops))]
    if not selected:
        raise ValueError("No training images found for quantization calibration")
    listing = PROJECT / "configs/quantization_images.txt"
    listing.write_text("".join(path.resolve().as_posix() + "\n" for path in selected), encoding="utf-8")
    config = PROJECT / "configs/quantization.yaml"
    config.write_text(yaml.safe_dump({"path": root.as_posix(), "train": listing.as_posix(),
                                      "val": listing.as_posix(), "names": data["names"]}), encoding="utf-8")
    report = {"seed": 42, "source_split": "train only", "images": len(selected),
              "bdd_stop_images": len(stops), "oversampled_copies_excluded": True,
              "filenames": [path.name for path in selected]}
    destination = PROJECT / "outputs/metrics/quantization_calibration.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=PROJECT / "models/baseline.pt")
    parser.add_argument("--data", type=Path, default=PROJECT / "configs/data.yaml")
    parser.add_argument("--calibration-images", type=int, default=320)
    args = parser.parse_args()
    if args.calibration_images <= 0:
        parser.error("Calibration count must be positive")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs/ultralytics"))
    from ultralytics import YOLO
    config = calibration_data(args.data, args.calibration_images)
    outputs = {}
    for precision in (32, 8):
        outputs[str(precision)] = str(YOLO(str(args.weights)).export(
            format="openvino", imgsz=640, batch=1, dynamic=False, nms=False,
            quantize=precision, device="cpu", data=str(config), fraction=1.0))
    (PROJECT / "outputs/metrics/exports.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
