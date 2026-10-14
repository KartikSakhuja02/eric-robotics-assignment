import copy
import math
import unittest

from src.distance import estimate_depth, quaternion_rotation


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"unknown_camera": {"policy": "assumed_fov", "horizontal_fov_deg": 90},
                         "min_box_height_px": 10, "max_depth_m": 100, "ground_plane_z_m": 0}
        self.priors = {"assumed_heights_m": {"cone": 0.5, "barrier": 0.9, "stop_sign": 0.75}}
        self.camera = {"camera_intrinsic": [[800, 0, 400], [0, 800, 300], [0, 0, 1]],
                       "camera_distortion": [0]*5, "image_width": 800, "image_height": 600}

    def estimate(self, cls, box, camera=None, shape=(600, 800, 3)):
        return estimate_depth(cls, box, self.camera if camera is None else camera,
                              shape, self.settings, self.priors)

    def test_known_size_three_metres(self):
        result = self.estimate(2, [350, 200, 450, 400])
        self.assertAlmostEqual(result["depth_m"], 3.0)
        self.assertEqual(result["method"], "size_prior")

    def test_resize_preserves_depth(self):
        full = self.estimate(2, [350, 200, 450, 400])
        small = self.estimate(2, [175, 100, 225, 200], shape=(300, 400, 3))
        self.assertAlmostEqual(full["depth_m"], small["depth_m"])

    def ground_camera(self):
        camera = copy.deepcopy(self.camera)
        # Camera x points ego -y, camera y points ego -z, camera z points ego +x.
        camera["camera_to_ego_rotation_wxyz"] = [0.5, -0.5, 0.5, -0.5]
        camera["camera_to_ego_translation_m"] = [0, 0, 1.5]
        return camera

    def test_flat_ground_five_metres(self):
        result = self.estimate(0, [370, 400, 430, 540], self.ground_camera())
        self.assertAlmostEqual(result["depth_m"], 5)
        self.assertAlmostEqual(result["ego_contact_point_m"][2], 0)

    def test_above_horizon_and_clipped_ground_rejected(self):
        for box in ([370, 100, 430, 250], [370, 400, 430, 599]):
            self.assertIsNone(self.estimate(0, box, self.ground_camera())["depth_m"])

    def test_small_box_and_clipped_sign_rejected(self):
        for box in ([370, 100, 430, 105], [370, 0, 430, 200]):
            self.assertIsNone(self.estimate(2, box)["depth_m"])

    def test_unknown_camera_marked_as_assumed(self):
        result = estimate_depth(2, [350, 200, 450, 400], None, (600, 800, 3), self.settings, self.priors)
        self.assertAlmostEqual(result["depth_m"], 1.5)
        self.assertEqual(result["calibration_source"], "assumed_fov")
        self.settings["unknown_camera"]["policy"] = "unavailable"
        result = estimate_depth(2, [350, 200, 450, 400], None, (600, 800, 3), self.settings, self.priors)
        self.assertIsNone(result["depth_m"])

    def test_bad_calibration_and_quaternion(self):
        self.camera["camera_intrinsic"][1][1] = 0
        self.assertIsNone(self.estimate(2, [350, 200, 450, 400])["depth_m"])
        with self.assertRaises(ValueError):
            quaternion_rotation([0]*4)


if __name__ == "__main__":
    unittest.main()
