import argparse
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.inspect_data import inspect, read_labels
from src.prepare_data import CLASSES, iter_json_array, split_yolo
from tests.helpers import fixture_root


class PreparationTests(unittest.TestCase):
    def test_chunk_boundaries_and_unicode(self):
        values = [{"name": "café", "labels": [1, 2]}, 12345, 1.25e12, {}, "end"]
        for size in (1, 3, 17, 65536):
            with self.subTest(chunk=size):
                self.assertEqual(list(iter_json_array(io.StringIO(json.dumps(values, ensure_ascii=False)), size)), values)

    def test_invalid_json(self):
        for value in ("[{},]", "[{} {}]", "[{", "{}", "[{}] extra", "[]extra", "[1.25e+]", "[12x]"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                list(iter_json_array(io.StringIO(value), 3))

    def test_invalid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            for value in ("3 .5 .5 .2 .2", "0 .5 .5 -1 .2", "0 .99 .99 .5 .5", "0 nan .5 .2 .2"):
                path.write_text(value)
                with self.subTest(label=value), self.assertRaises(ValueError):
                    read_labels(path)

    def test_group_separation_and_conflicting_split_rejected(self):
        with tempfile.TemporaryDirectory(dir=fixture_root()) as directory:
            base = Path(directory)
            source = base / "source"
            (source / "images").mkdir(parents=True)
            (source / "labels").mkdir()
            rows = []
            for group in range(6):
                for index in range(2):
                    name = f"g{group}_{index}.jpg"
                    cv2.imwrite(str(source / "images" / name), np.full((120, 160, 3), 30 + group*20, np.uint8))
                    (source / "labels" / Path(name).with_suffix(".txt")).write_text("0 .2 .5 .1 .2\n1 .5 .5 .1 .2\n2 .8 .5 .1 .2\n")
                    rows.append({"filename": name, "group": str(group), "split": ""})
            (source / "data.yaml").write_text(yaml.safe_dump({"names": CLASSES}))
            groups = base / "groups.csv"
            def save_rows():
                with groups.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["filename", "group", "split"])
                    writer.writeheader()
                    writer.writerows(rows)
            save_rows()
            args = argparse.Namespace(input=source, groups=groups, output=base / "processed", config=base / "data.yaml", seed=42)
            split_yolo(args)
            self.assertTrue(inspect(args.output, base / "preview", sample_count=0))
            result = json.loads((base / "preview/summary.json").read_text())
            self.assertEqual(result["errors"], [])
            self.assertEqual(sum(value["images"] for value in result["splits"].values()), 12)
            rows[0]["split"], rows[1]["split"] = "train", "test"
            save_rows()
            args.output = base / "rejected"
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                split_yolo(args)


if __name__ == "__main__":
    unittest.main()
