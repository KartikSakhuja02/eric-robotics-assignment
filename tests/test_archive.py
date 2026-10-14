import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.prepare_archives import copy_member, ensure_space, normalized_member
from tests.helpers import fixture_root


class ArchiveTests(unittest.TestCase):
    def test_reject_traversal_and_absolute_names(self):
        for name in ("../../outside.txt", "/absolute.txt", "samples/../../outside.txt"):
            with self.assertRaises(ValueError):
                normalized_member(name)

    def test_space_reserve(self):
        with patch("src.prepare_archives.shutil.disk_usage") as usage:
            usage.return_value.free = 20 * 1024**3
            ensure_space(Path.cwd(), 4 * 1024**3, 15)
            with self.assertRaises(ValueError):
                ensure_space(Path.cwd(), 6 * 1024**3, 15)

    def test_bounded_copy_and_idempotent_repeat(self):
        with tempfile.TemporaryDirectory(dir=fixture_root()) as directory:
            root = Path(directory)
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as tar:
                member = tarfile.TarInfo("samples/CAM_FRONT/image.jpg")
                member.size = 3
                tar.addfile(member, io.BytesIO(b"abc"))
                link = tarfile.TarInfo("link")
                link.type = tarfile.SYMTYPE
                link.linkname = "../outside"
                tar.addfile(link)
            buffer.seek(0)
            with tarfile.open(fileobj=buffer) as tar:
                member = tar.getmember("samples/CAM_FRONT/image.jpg")
                self.assertEqual(copy_member(tar, member, root, 0), 3)
                self.assertEqual(copy_member(tar, member, root, 0), 0)
                self.assertEqual((root / member.name).read_bytes(), b"abc")
                with self.assertRaises(ValueError):
                    copy_member(tar, tar.getmember("link"), root, 0)


if __name__ == "__main__":
    unittest.main()
