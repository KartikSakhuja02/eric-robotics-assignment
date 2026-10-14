"""Fine-tune a pretrained YOLO11n model on the verified three-class dataset."""

import argparse
import os
import shutil
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "configs/data.yaml")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0, help="0 uses less host RAM; original run used 2")
    parser.add_argument("--resume", type=Path, help="Continue from a last.pt checkpoint, including optimizer state")
    parser.add_argument("--time", type=float, default=3.0, help="Maximum training hours")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0 or args.workers < 0 or args.time < 0:
        parser.error("epochs, batch and imgsz must be positive; workers and time nonnegative")
    if not args.data.is_file() or not args.data.read_text(encoding="utf-8").strip():
        parser.error("Dataset YAML is missing or empty; prepare and inspect your labeled dataset first")
    if not args.weights.endswith(".pt"):
        parser.error("Use a pretrained .pt checkpoint for transfer learning")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs/ultralytics"))
    from ultralytics import YOLO

    model = YOLO(str(args.resume) if args.resume else args.weights)
    def stop_at_time_limit(trainer):
        out_of_time = args.time > 0 and time.time() - trainer.train_time_start >= args.time * 3600
        low_disk = shutil.disk_usage(PROJECT).free < 5 * 1024**3
        if out_of_time or low_disk:
            trainer.stop = True
            if low_disk:
                print("Stopping at a checkpoint: less than 5 GiB free disk space", flush=True)
    model.add_callback("on_fit_epoch_end", stop_at_time_limit)
    model.train(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, amp=True, workers=args.workers,
        resume=str(args.resume.resolve()) if args.resume else False,
        patience=8, close_mosaic=min(5, max(0, args.epochs - 1)), seed=42,
        optimizer="AdamW", lr0=0.001, lrf=0.1, cache=False,
        project=str(PROJECT / "runs"), name=args.name,
    )


if __name__ == "__main__":
    main()
