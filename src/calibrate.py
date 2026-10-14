"""Load per-image dataset calibration; no physical camera capture is required."""

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def load_calibrations(path=PROJECT / "configs/image_calibrations.jsonl"):
    if not Path(path).is_file():
        return {}
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["filename"]] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=PROJECT / "configs/image_calibrations.jsonl")
    args = parser.parse_args()
    records = load_calibrations(args.index)
    print(json.dumps({"images": len(records), "supplied_intrinsics": sum("camera_intrinsic" in row for row in records.values()),
                      "unknown_calibration": sum("camera_intrinsic" not in row for row in records.values())}, indent=2))


if __name__ == "__main__":
    main()
