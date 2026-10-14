import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.export import calibration_data
from tests.helpers import fixture_root


class CalibrationSelectionTests(unittest.TestCase):
    def test_calibration_excludes_holdouts_and_repeated_images(self):
        with tempfile.TemporaryDirectory(dir=fixture_root()) as directory:
            project = Path(directory)
            (project / "configs").mkdir()
            root = project / "data"
            for split in ("train", "val", "test"):
                (root / "images" / split).mkdir(parents=True)
                (root / "images" / split / f"nuim_{split}.jpg").touch()
            for name in ("bdd_train_stop.jpg", "bdd_train_stop_repeat01.jpg"):
                (root / "images/train" / name).touch()
            config = project / "configs/data.yaml"
            config.write_text(yaml.safe_dump({"path": root.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": ["cone", "barrier", "stop_sign"]}))
            with patch("src.export.PROJECT", project):
                calibration_data(config, count=320)
            report = json.loads((project / "outputs/metrics/quantization_calibration.json").read_text())
            self.assertEqual(set(report["filenames"]), {"bdd_train_stop.jpg", "nuim_train.jpg"})
            selected = (project / "configs/quantization_images.txt").read_text().splitlines()
            self.assertTrue(all(Path(path).parent.name == "train" for path in selected))


if __name__ == "__main__":
    unittest.main()
