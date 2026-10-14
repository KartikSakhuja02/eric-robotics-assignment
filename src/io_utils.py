"""File helpers that keep image datasets small on a shared NTFS volume."""

import os
import shutil
from pathlib import Path


def link_or_copy_image(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an image: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"
