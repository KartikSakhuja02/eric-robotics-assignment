"""Measure batch-one detector and annotated-pipeline latency on preloaded frames."""

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .calibrate import load_calibrations
from .infer import annotate

PROJECT = Path(__file__).resolve().parents[1]


def processor_name():
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def summarize(samples):
    values = np.asarray(samples, dtype=np.float64) * 1000
    return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "fps": float(1000 / values.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(PROJECT / "models/baseline.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", type=int, choices=[16, 32], default=32)
    parser.add_argument("--name", default="cpu_fp32")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.threads) <= 0 or args.warmup < 0:
        parser.error("Iterations, repeats and threads must be positive; warmup nonnegative")
    if args.device == "cpu" and args.precision == 16:
        parser.error("Use FP32 for CPU .pt inference; OpenVINO precision is determined by its artifact")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs/ultralytics"))
    from ultralytics import YOLO
    data = yaml.safe_load((PROJECT / "configs/data.yaml").read_text(encoding="utf-8"))
    source = Path(data["path"]) / data["test"]
    paths = sorted(source.glob("*.jpg"))
    random.Random(42).shuffle(paths)
    paths = paths[:16]
    frames = [cv2.imread(str(path)) for path in paths]
    if not frames or any(frame is None for frame in frames):
        parser.error("Test images missing or unreadable")
    settings = json.loads((PROJECT / "configs/camera.json").read_text())
    priors = yaml.safe_load((PROJECT / "configs/object_size.yaml").read_text())
    cameras = load_calibrations()
    model = YOLO(args.weights, task="detect")
    gpu = args.device != "cpu" and str(args.weights).endswith(".pt")
    def synchronize():
        if gpu:
            torch.cuda.synchronize()
    def predict(frame):
        return model.predict(frame, imgsz=640, rect=False, batch=1, device=args.device,
                             quantize=args.precision, conf=0.25, iou=0.7,
                             max_det=50, verbose=False)[0]
    # Initialize separately even when --warmup=0, keeping setup out of timing.
    predict(frames[0])
    for index in range(args.warmup):
        predict(frames[index % len(frames)])
    synchronize()
    runtime = {}
    if str(args.weights).endswith(".pt"):
        runtime["model_parameter_dtype"] = str(next(model.predictor.model.backend.model.parameters()).dtype)
        runtime["backend_fp16"] = bool(model.predictor.model.fp16)
    if hasattr(model.predictor.model, "ov_compiled_model"):
        compiled = model.predictor.model.ov_compiled_model
        for property_name in ("EXECUTION_DEVICES", "INFERENCE_NUM_THREADS", "PERFORMANCE_HINT", "INFERENCE_PRECISION_HINT"):
            try:
                runtime[property_name] = str(compiled.get_property(property_name))
            except Exception:
                pass
    all_detector, all_pipeline, runs = [], [], []
    for repeat in range(args.repeats):
        detector, pipeline = [], []
        for index in range(args.iterations):
            position = index % len(frames)
            synchronize()
            started = time.perf_counter()
            result = predict(frames[position])
            synchronize()
            detected = time.perf_counter()
            annotate(frames[position], result, cameras.get(paths[position].name), settings, priors)
            synchronize()
            finished = time.perf_counter()
            detector.append(detected - started)
            pipeline.append(finished - started)
        all_detector.extend(detector)
        all_pipeline.extend(pipeline)
        runs.append({"detector": summarize(detector), "annotated_pipeline": summarize(pipeline)})
        print(f"Repeat {repeat + 1}: {runs[-1]}", flush=True)
    artifact = Path(args.weights)
    size = artifact.stat().st_size if artifact.is_file() else sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file())
    report = {"weights": str(artifact), "device": args.device, "requested_precision": args.precision,
              "artifact_precision": "INT8 mixed with FP32" if "int8" in str(artifact) else ("FP16" if args.precision == 16 else "FP32"),
              "artifact_bytes": size, "imgsz": [640,640], "batch": 1, "conf": 0.25, "iou": 0.7,
              "max_detections": 50, "warmup": args.warmup, "iterations_per_repeat": args.iterations,
              "repeats": args.repeats, "torch_threads": args.threads, "opencv_threads": args.threads,
              "seed": 42, "frames": [path.name for path in paths], "runs": runs,
              "runtime_properties": runtime,
              "detector": summarize(all_detector), "annotated_pipeline": summarize(all_pipeline),
              "timing_includes": "preprocessing, model, NMS; pipeline also adds geometry and drawing",
              "timing_excludes": "disk reading/writing, video decoding/encoding, camera capture, model initialization",
              "hardware": {"platform": platform.platform(), "processor": processor_name(),
                           "logical_cpus": os.cpu_count(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    destination = PROJECT / "outputs/metrics" / f"benchmark_{args.name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
