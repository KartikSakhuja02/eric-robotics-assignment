"""Build a concise results report and chart from completed experiment files."""

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]


def main():
    metrics_root = PROJECT / "outputs/metrics"
    baseline = json.loads((metrics_root / "baseline_test.json").read_text(encoding="utf-8"))
    entries = [("PyTorch CPU FP32", "cpu_fp32", "cpu_fp32_test"),
               ("PyTorch GPU FP32", "gpu_fp32", "baseline_test"),
               ("PyTorch GPU FP16", "gpu_fp16", "gpu_fp16_test"),
               ("OpenVINO CPU FP32", "openvino_fp32", "openvino_fp32_test"),
               ("OpenVINO CPU INT8", "openvino_int8", "openvino_int8_test")]
    lines = ["# Measured Results", "", "Author: Kartik Sakhuja", "",
             "All results below use the custom grouped test split. Distance accuracy is unmeasured because no object-distance ground truth is available.", "",
             "## Detection and optimization", "",
             "| Runtime | mAP50 | mAP50–95 | Detector FPS | Annotated-pipeline FPS | p95 pipeline latency (ms) | Artifact (MiB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    records = []
    for label, timing_name, accuracy_name in entries:
        timing = json.loads((metrics_root / f"benchmark_{timing_name}.json").read_text(encoding="utf-8"))
        accuracy = json.loads((metrics_root / f"{accuracy_name}.json").read_text(encoding="utf-8"))
        overall = accuracy["metrics"]
        lines.append(f"| {label} | {overall['metrics/mAP50(B)']:.4f} | {overall['metrics/mAP50-95(B)']:.4f} | {timing['detector']['fps']:.2f} | {timing['annotated_pipeline']['fps']:.2f} | {timing['annotated_pipeline']['p95_ms']:.2f} | {timing['artifact_bytes'] / 1024**2:.2f} |")
        records.append({"label": label, "timing": timing, "accuracy": accuracy})
    lines.extend(["", "Artifact size includes XML/BIN/metadata for OpenVINO. The PyTorch checkpoint is shared by CPU/GPU and FP32/FP16 inference; serialized checkpoint size does not imply compute precision.", "", "## Baseline per-class accuracy", "",
                  "| Class | Precision | Recall | AP50 | AP50–95 |", "|---|---:|---:|---:|---:|"])
    for row in baseline["per_class"]:
        lines.append(f"| {row['name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['mAP50']:.4f} | {row['mAP50_95']:.4f} |")
    lines.extend(["", "AP evaluation uses confidence floor 0.001, IoU-NMS 0.7, maximum 50 detections and batch 1. Precision/recall above use the framework's best mean-F1 confidence; they are not scores at the fixed 0.25 demo/timing threshold."])
    fp32, int8 = records[3], records[4]
    speedup = int8["timing"]["annotated_pipeline"]["fps"] / fp32["timing"]["annotated_pipeline"]["fps"]
    difference = int8["accuracy"]["metrics"]["metrics/mAP50-95(B)"] - fp32["accuracy"]["metrics"]["metrics/mAP50-95(B)"]
    lines.extend(["", f"OpenVINO INT8 versus OpenVINO FP32: **{speedup:.2f}×** annotated-pipeline speed ratio, with **{difference:+.4f}** absolute mAP50–95 change. Quantization uses 320 training-only images. A speed ratio below 1 means INT8 was slower on this hardware.", "",
                  "## Measurement conditions", "",
                  "Batch 1, 640×640, identical 16 preloaded test frames, confidence 0.25, IoU 0.7, max 50 detections, 20 warmups, 3×100 timed iterations. Detector timing covers preprocessing, network and NMS. Pipeline timing also covers geometry and drawing. CUDA synchronization is included. Loading, disk I/O, image/video decoding and encoding are excluded. Laptop power and thermals affect timings; these are not measurements on a deployed robot or Jetson.", "",
                  "PyTorch/OpenCV use 4 CPU threads. OpenVINO uses its LATENCY-mode runtime thread settings, recorded in each benchmark JSON. Compare OpenVINO FP32/INT8 to isolate quantization within the same runtime; comparing PyTorch/OpenVINO also changes runtime.", "",
                  f"Hardware: GPU **{records[0]['timing']['hardware']['gpu']}**; host **{records[0]['timing']['hardware']['processor']}**; {records[0]['timing']['hardware']['logical_cpus']} logical CPUs.", "",
                  "Stop-sign test data contains just 18 boxes, selected with a pretrained teacher and visually reviewed. This coverage and the lack of wholly negative scenes limit the strength of the accuracy claims. Dataset labels can miss incidental signs. Depth uses supplied calibration/flat-road assumptions or explicit size/FOV priors and is not validated against real distances.", "", "![Measured latency and accuracy](benchmark_comparison.png)", ""])
    training_csv = PROJECT / "runs/baseline/results.csv"
    if training_csv.is_file():
        with training_csv.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        lines.extend([f"Training completed {len(rows)} epochs; final logged elapsed time {float(rows[-1]['time'])/60:.1f} minutes. Best checkpoint was selected using validation fitness.", ""])
    lines.append(f"Free disk space when report generated: {shutil.disk_usage(PROJECT).free / 1024**3:.2f} GiB. Original downloaded archives remain intact.")
    (PROJECT / "outputs/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    labels = [record["label"].replace("PyTorch ", "PT ").replace("OpenVINO ", "OV ") for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    axes[0].barh(labels, [record["timing"]["annotated_pipeline"]["fps"] for record in records], color="#287da8")
    axes[0].set_xlabel("Annotated-pipeline FPS (higher is faster)")
    axes[0].invert_yaxis()
    axes[1].barh(labels, [record["accuracy"]["metrics"]["metrics/mAP50-95(B)"] for record in records], color="#d18839")
    axes[1].set_xlabel("Test mAP50–95 (higher is better)")
    axes[1].set_xlim(0, 1)
    axes[1].invert_yaxis()
    fig.suptitle("YOLO11n: actual laptop measurements, batch 1, 640×640")
    fig.savefig(PROJECT / "outputs/benchmark_comparison.png", dpi=150)
    plt.close(fig)
    (metrics_root / "comparison.json").write_text(json.dumps({"openvino_int8_pipeline_speed_ratio": speedup,
                                                             "openvino_int8_map50_95_delta": difference}, indent=2), encoding="utf-8")
    print("Saved outputs/report.md and outputs/benchmark_comparison.png")


if __name__ == "__main__":
    main()
