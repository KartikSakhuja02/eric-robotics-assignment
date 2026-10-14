"""Optional flat-road bird's-eye projection using supplied nuImages calibration."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .calibrate import load_calibrations
from .distance import quaternion_rotation

PROJECT = Path(__file__).resolve().parents[1]


def road_homography(camera):
    """Map ego ground coordinates [X,Y,1] into the undistorted camera image."""
    matrix = np.asarray(camera["camera_intrinsic"], dtype=np.float64)
    ego_to_camera = quaternion_rotation(camera["camera_to_ego_rotation_wxyz"]).T
    translation = np.asarray(camera["camera_to_ego_translation_m"], dtype=np.float64)
    return matrix @ np.column_stack((ego_to_camera[:,0], ego_to_camera[:,1], -ego_to_camera @ translation))


def warp_road(image, camera, width=600, height=800):
    # Ego X is forward, Y left. The display covers X=2..30 m, Y=-10..10 m.
    matrix = np.asarray(camera["camera_intrinsic"], dtype=np.float64).copy()
    matrix[0,:] *= image.shape[1] / camera["image_width"]
    matrix[1,:] *= image.shape[0] / camera["image_height"]
    resized_camera = {**camera, "camera_intrinsic": matrix.tolist()}
    undistorted = cv2.undistort(image, matrix, np.asarray(camera["camera_distortion"], dtype=np.float64))
    bev_pixel_to_ground = np.array([[0, -28/(height-1), 30], [-20/(width-1), 0, 10], [0,0,1]], dtype=np.float64)
    bev_to_image = road_homography(resized_camera) @ bev_pixel_to_ground
    road = cv2.warpPerspective(undistorted, bev_to_image, (width,height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    # Exclude areas whose plane points project behind the camera.
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    camera_depth = bev_to_image[2,0]*columns + bev_to_image[2,1]*rows + bev_to_image[2,2]
    road[camera_depth <= 0] = 0
    for forward in (5,10,15,20,25,30):
        v = round((30-forward)/28*(height-1))
        cv2.line(road, (0,v), (width-1,v), (100,130,100), 1)
        cv2.putText(road, f"ego X={forward}m", (8,max(15,v-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (240,240,240), 1, cv2.LINE_AA)
    cv2.line(road, (width//2,0), (width//2,height-1), (100,130,100), 1)
    return undistorted, road


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/bev")
    args = parser.parse_args()
    cameras = load_calibrations()
    if args.source is None:
        data = yaml.safe_load((PROJECT / "configs/data.yaml").read_text(encoding="utf-8"))
        args.source = next(iter(sorted((Path(data["path"]) / data["test"]).glob("nuim_*.jpg"))))
    camera = cameras.get(args.source.name)
    if not camera or "camera_to_ego_rotation_wxyz" not in camera:
        parser.error("BEV requires supplied intrinsics, distortion and camera-to-ego extrinsics")
    image = cv2.imread(str(args.source))
    if image is None:
        parser.error("Input image is unreadable")
    args.output.mkdir(parents=True, exist_ok=True)
    undistorted, road = warp_road(image, camera)
    preview = np.full((880,1500,3), 25, np.uint8)
    front = cv2.resize(undistorted, (880,495))
    preview[100:595,10:890] = front
    preview[45:845,900:1500] = road
    cv2.putText(preview, "Undistorted front camera", (15,75), cv2.FONT_HERSHEY_SIMPLEX, .75, (240,240,240), 2, cv2.LINE_AA)
    cv2.putText(preview, "Flat-road BEV", (920,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (240,240,240), 2, cv2.LINE_AA)
    cv2.putText(preview, "Road plane z=0 | elevated objects stretch | projection is not a true overhead photograph", (15,805), cv2.FONT_HERSHEY_SIMPLEX, .6, (240,240,240), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.output / "road_bev.jpg"), preview)
    (args.output / "geometry.json").write_text(json.dumps({"filename": args.source.name,
        "ground_to_undistorted_image_homography": road_homography(camera).tolist(),
        "extent_m": {"ego_x": [2,30], "ego_y": [-10,10]}, "assumed_ground_plane_z_m": 0,
        "limitations": "Only flat-road points obey this projection; slopes and elevated objects cause distortion"}, indent=2), encoding="utf-8")
    print(f"Saved {args.output / 'road_bev.jpg'}")


if __name__ == "__main__":
    main()
