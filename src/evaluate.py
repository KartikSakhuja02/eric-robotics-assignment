"""Evaluate held-out detector accuracy; do not invent distance ground truth."""

import argparse
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(PROJECT / "models/baseline.pt"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--precision", type=int, choices=[16, 32], default=32)
    parser.add_argument("--name", default="baseline_test")
    args = parser.parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs/ultralytics"))
    from ultralytics import YOLO
    model = YOLO(args.weights)
    metrics = model.val(data=str(PROJECT / "configs/data.yaml"), split="test", imgsz=640,
                        rect=False, batch=1, workers=0, device=args.device, quantize=args.precision,
                        conf=0.001, iou=0.7, max_det=50,
                        plots=True, project=str(PROJECT / "outputs/evaluation"), name=args.name)
    classes = []
    for index, category in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(index)
        classes.append({"class_id": int(category), "name": model.names[int(category)],
                        "precision": float(precision), "recall": float(recall),
                        "mAP50": float(ap50), "mAP50_95": float(ap)})
    report = {"model": args.weights, "device": args.device, "requested_precision": args.precision,
              "artifact_precision": "INT8 mixed with FP32" if "int8" in args.weights else ("FP16" if args.precision == 16 else "FP32"),
              "evaluation_settings": {"batch": 1, "imgsz": 640, "rect": False, "confidence_floor": 0.001,
                                      "nms_iou": 0.7, "max_detections": 50,
                                      "precision_recall_operating_point": "framework's best mean-F1 confidence, not a fixed 0.25 threshold"},
              "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
              "per_class": classes, "split": "custom grouped test split",
              "distance_accuracy": "not measured: selected source annotations contain no object-distance ground truth"}
    destination = PROJECT / "outputs/metrics" / f"{args.name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
