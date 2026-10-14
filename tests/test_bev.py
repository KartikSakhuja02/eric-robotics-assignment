import unittest
import numpy as np

from src.bev import road_homography


class HomographyTests(unittest.TestCase):
    def test_known_ground_point_projection_and_inverse(self):
        camera = {"camera_intrinsic": [[800,0,400],[0,800,300],[0,0,1]],
                  "camera_to_ego_rotation_wxyz": [0.5,-0.5,0.5,-0.5],
                  "camera_to_ego_translation_m": [0,0,1.5]}
        homography = road_homography(camera)
        ground = np.array([5,0,1])
        image = homography @ ground
        image /= image[2]
        np.testing.assert_allclose(image, [400,540,1])
        reconstructed = np.linalg.inv(homography) @ image
        reconstructed /= reconstructed[2]
        np.testing.assert_allclose(reconstructed, ground)


if __name__ == "__main__":
    unittest.main()
