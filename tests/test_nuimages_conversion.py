import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.inspect_data import read_labels
from src.nuimages_to_yolo import convert, normalized_box
from src.prepare_data import CLASSES
from tests.helpers import fixture_root


class ConversionTests(unittest.TestCase):
    def test_category_mapping_clipping_and_calibration(self):
        with tempfile.TemporaryDirectory(dir=fixture_root()) as directory:
            root = Path(directory)
            raw = root / "raw"
            metadata = raw / "v1.0-mini"
            metadata.mkdir(parents=True)
            (raw / "samples/CAM_FRONT").mkdir(parents=True)
            cv2.imwrite(str(raw / "samples/CAM_FRONT/example.jpg"), np.full((100,200,3), 80, np.uint8))
            tables = {
                "category": [{"token": "cone", "name": "movable_object.trafficcone"}, {"token": "barrier", "name": "movable_object.barrier"}],
                "sample": [{"token": "sample", "log_token": "log1"}],
                "sensor": [{"token": "sensor", "channel": "CAM_FRONT"}],
                "calibrated_sensor": [{"token": "calibration", "sensor_token": "sensor",
                                       "camera_intrinsic": [[100,0,100],[0,100,50],[0,0,1]],
                                       "camera_distortion": [0]*5, "translation": [0,0,1.5], "rotation": [1,0,0,0]}],
                "sample_data": [{"token": "frame1", "sample_token": "sample", "calibrated_sensor_token": "calibration",
                                 "filename": "samples/CAM_FRONT/example.jpg", "is_key_frame": True, "width": 200, "height": 100}],
                "object_ann": [{"sample_data_token": "frame1", "category_token": "cone", "bbox": [20,20,60,80]},
                               {"sample_data_token": "frame1", "category_token": "barrier", "bbox": [100,10,250,90]}]}
            for name, values in tables.items():
                (metadata / f"{name}.json").write_text(json.dumps(values))
            output = root / "converted"
            convert(argparse.Namespace(root=raw, version="v1.0-mini", output=output, max_images=10, camera="CAM_FRONT"))
            self.assertEqual(read_labels(output / "labels/train/nuim_frame1.txt"), [(0,[0.2,0.5,0.2,0.6]),(1,[0.75,0.5,0.5,0.8])])
            self.assertEqual(yaml.safe_load((output / "data.yaml").read_text())["names"], dict(enumerate(CLASSES)))
            camera = json.loads((output / "camera_index.jsonl").read_text())
            self.assertEqual(camera["camera_intrinsic"], tables["calibrated_sensor"][0]["camera_intrinsic"])
            self.assertEqual(camera["camera_to_ego_translation_m"], [0,0,1.5])
            self.assertEqual(camera["group"], "nuim_log1")
            self.assertTrue(camera["stop_sign_review_required"])
            self.assertFalse(json.loads((output / "conversion_summary.json").read_text())["distance_ground_truth_available"])
            self.assertIsNone(normalized_box([10,10,5,20], 200,100))


if __name__ == "__main__":
    unittest.main()
