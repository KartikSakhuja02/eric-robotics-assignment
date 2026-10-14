"""Camera-forward depth from calibrated rays or explicit physical-size priors."""

import math

import cv2
import numpy as np

CLASS_NAMES = ["cone", "barrier", "stop_sign"]


def quaternion_rotation(quaternion):
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or np.linalg.norm(values) < 1e-12:
        raise ValueError("Invalid camera rotation quaternion")
    w, x, y, z = values / np.linalg.norm(values)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def assumed_camera(width, height, fov_deg):
    if width <= 0 or height <= 0 or not 1 < fov_deg < 179:
        raise ValueError("Invalid dimensions or assumed horizontal field of view")
    focal = width / (2 * math.tan(math.radians(fov_deg) / 2))
    return {"camera_intrinsic": [[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]],
            "camera_distortion": [0, 0, 0, 0, 0], "image_width": width, "image_height": height,
            "calibration_source": "assumed_fov", "assumed_horizontal_fov_deg": fov_deg}


def normalized_points(points, camera, width, height):
    matrix = np.asarray(camera["camera_intrinsic"], dtype=np.float64).copy()
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsic matrix")
    # Supports pure resizing; cropping requires a separately adjusted calibration.
    if camera["image_width"] <= 0 or camera["image_height"] <= 0:
        raise ValueError("Invalid calibration image dimensions")
    matrix[0, :] *= width / camera["image_width"]
    matrix[1, :] *= height / camera["image_height"]
    distortion = np.asarray(camera.get("camera_distortion", [0]*5), dtype=np.float64)
    if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError("Unsupported distortion convention")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(points, matrix, distortion).reshape(-1, 2)


def estimate_depth(class_id, bbox, camera, image_shape, settings, size_priors):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    name = CLASS_NAMES[class_id]
    source = "dataset"
    if camera is None or "camera_intrinsic" not in camera:
        unknown = settings["unknown_camera"]
        if unknown["policy"] != "assumed_fov":
            return {"depth_m": None, "method": "unavailable", "reason": "camera calibration missing", "calibration_source": "unavailable"}
        camera = assumed_camera(width, height, unknown["horizontal_fov_deg"])
        source = "assumed_fov"
    elif camera.get("calibration_source") == "assumed_fov":
        source = "assumed_fov"
    base = {"depth_m": None, "calibration_source": source, "estimated": True,
            "output_quantity": "camera_forward_depth_m"}
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return {**base, "method": "unavailable", "reason": "invalid bounding box"}
    if y2 - y1 < settings.get("min_box_height_px", 10):
        return {**base, "method": "unavailable", "reason": "box too small for a stable estimate"}
    try:
        if name in ("cone", "barrier") and "camera_to_ego_rotation_wxyz" in camera:
            if y2 >= height - 1:
                return {**base, "method": "ground_plane", "reason": "ground contact is clipped"}
            xn, yn = normalized_points([[(x1+x2)/2, y2]], camera, width, height)[0]
            ray = np.array([xn, yn, 1.0])
            direction = quaternion_rotation(camera["camera_to_ego_rotation_wxyz"]) @ ray
            origin = np.asarray(camera["camera_to_ego_translation_m"], dtype=np.float64)
            plane_z = settings.get("ground_plane_z_m", 0.0)
            if origin.shape != (3,) or not np.isfinite(origin).all() or origin[2] <= plane_z or direction[2] >= -1e-6:
                return {**base, "method": "ground_plane", "reason": "ray does not intersect ground in front of camera"}
            depth = (plane_z - origin[2]) / direction[2]
            result = {**base, "method": "ground_plane", "reference_point": "bbox bottom-centre ground contact",
                      "assumptions": ["locally flat road at ego z=0", "visible object ground contact"],
                      "ego_contact_point_m": (origin + depth * direction).tolist()}
        else:
            if y1 <= 1 or y2 >= height - 1:
                return {**base, "method": "size_prior", "reason": "object height is clipped"}
            points = normalized_points([[(x1+x2)/2, y1], [(x1+x2)/2, y2]], camera, width, height)
            normalized_height = abs(points[1, 1] - points[0, 1])
            physical_height = float(size_priors["assumed_heights_m"][name])
            if normalized_height <= 1e-9 or physical_height <= 0:
                raise ValueError("Invalid height measurement or prior")
            depth = physical_height / normalized_height
            result = {**base, "method": "size_prior", "reference_point": "object depth from projected height",
                      "assumed_object_height_m": physical_height,
                      "assumptions": ["configured physical height", "small foreshortening and depth variation across object"]}
        if not math.isfinite(depth) or depth <= 0 or depth > settings.get("max_depth_m", 100):
            return {**result, "reason": "estimate outside configured valid depth range"}
        return {**result, "depth_m": float(depth)}
    except (ValueError, KeyError, cv2.error) as exc:
        return {**base, "method": "unavailable", "reason": str(exc)}
