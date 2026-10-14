"""Robotics detection assignment; keep runtime caches inside the workspace."""

import os
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
for _variable, _directory in (("MPLCONFIGDIR", "outputs/matplotlib"), ("YOLO_CONFIG_DIR", "outputs/ultralytics")):
    _path = _root / _directory
    _path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_variable, str(_path))
