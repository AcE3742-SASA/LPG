import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "artifacts.py"
SPEC = importlib.util.spec_from_file_location("artifacts", SCRIPT)
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        (self.root / "manifests").mkdir()
        self.rows = []
        self.blobs = {}
        self.chunk_size = 8
        self.add_file("research/first.txt", b"same")
        self.add_file("research/duplicate.txt", b"same")
        self.add_file("research/nested/chunked.bin", b"01234567abcdefghlast")
        self.add_file("research/empty.txt", b"")
        self.add_file("research/git.py", b"pass\n", storage="git", mode=0o755)
        self.rows.append({"path": "research/link", "type": "symlink", "target": "nested",
                          "original_target": "/original/nested"})
        self.write_archive()

    def add_file(self, name, data, storage="release", mode=0o644):
        row = {"path": name, "size": len(data), "sha256": sha(data), "mode": mode,
               "storage": storage}
        chunks = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
        if len(data) > self.chunk_size:
            row["chunks"] = [sha(chunk) for chunk in chunks]
        else:
            chunks = [data]
        for chunk in chunks:
            self.blobs[sha(chunk)] = chunk
        self.rows.append(row)

    def write_archive(self, extra=None):
        pack = self.cache / "research-0001.tar.gz"
        with tarfile.open(pack, "w:gz") as archive:
            for key, data in self.blobs.items():
                member = tarfile.TarInfo("blobs/" + key)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            if extra:
                archive.addfile(extra)
        manifest = self.root / "manifests/files.jsonl.gz"
        with gzip.open(manifest, "wt", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + "\n")
        self.metadata = {
            "format_version": 1, "repository": "AcE3742-SASA/LPG",
            "release_tag": "research-2026-11-13", "chunk_size": self.chunk_size,
            "files_manifest": {"path": "manifests/files.jsonl.gz",
                               "size": manifest.stat().st_size, "sha256": artifacts.digest(manifest)},
            "packs": [{"name": pack.name, "size": pack.stat().st_size,
                       "sha256": artifacts.digest(pack)}], "counts": {}}
        (self.root / "manifests/artifacts.json").write_text(json.dumps(self.metadata))

    def restore(self, prefix=None):
        with contextlib.redirect_stdout(io.StringIO()):
            artifacts.restore(self.root, self.cache, prefix)

    def test_roundtrip_chunking_duplicates_modes_and_idempotence(self):
        self.restore()
        self.assertEqual((self.root / "research/nested/chunked.bin").read_bytes(),
                         b"01234567abcdefghlast")
        first = self.root / "research/first.txt"
        duplicate = self.root / "research/duplicate.txt"
        self.assertNotEqual(first.stat().st_ino, duplicate.stat().st_ino)
        self.assertEqual(first.stat().st_nlink, 1)
        self.assertEqual((self.root / "research/empty.txt").stat().st_size, 0)
        self.assertEqual(os.readlink(self.root / "research/link"), "nested")
        with contextlib.redirect_stdout(io.StringIO()):
            artifacts.verify(self.root)
        self.restore()
        self.assertFalse(list(self.root.glob(".lpg-restore-*")))

    def test_prefix_restores_only_requested_subtree(self):
        self.restore("research/nested/")
        self.assertTrue((self.root / "research/nested/chunked.bin").is_file())
        self.assertFalse((self.root / "research/first.txt").exists())

    def test_existing_git_file_is_preserved(self):
        path = self.root / "research/git.py"
        path.parent.mkdir()
        path.write_bytes(b"pass\n")
        path.chmod(0o755)
        inode = path.stat().st_ino
        self.restore()
        self.assertEqual(path.stat().st_ino, inode)

    def test_refuses_existing_different_file(self):
        path = self.root / "research/first.txt"
        path.parent.mkdir()
        path.write_text("my local edit")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.restore()
        self.assertEqual(path.read_text(), "my local edit")
        self.assertFalse((self.root / "research/duplicate.txt").exists())

    def test_missing_and_corrupt_pack_fail_before_writing(self):
        pack = self.cache / "research-0001.tar.gz"
        original = pack.read_bytes()
        pack.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.restore()
        pack.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.restore()
        self.assertFalse((self.root / "research").exists())

    def test_corrupt_blob_with_valid_pack_checksum_fails(self):
        key = sha(b"same")
        self.blobs[key] = b"evil"
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "Blob SHA-256 mismatch"):
            self.restore()
        self.assertFalse((self.root / "research").exists())

    def test_missing_blob_with_valid_pack_checksum_fails(self):
        del self.blobs[sha(b"same")]
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "Missing .* blobs"):
            self.restore()
        self.assertFalse((self.root / "research").exists())

    def test_manifest_path_traversal_is_rejected(self):
        for path in ("../escape", "/escape", "research/../../escape", "research\\escape"):
            with self.subTest(path=path):
                self.rows[0]["path"] = path
                self.write_archive()
                with self.assertRaises(ValueError):
                    self.restore()

    def test_archive_traversal_and_symlink_members_are_rejected(self):
        for name, kind in (("../escape", tarfile.REGTYPE),
                           ("blobs/" + "a" * 64, tarfile.SYMTYPE)):
            with self.subTest(name=name):
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = "../../escape"
                self.write_archive(extra=member)
                with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                    self.restore()
                self.assertFalse((self.root / "research").exists())

    def test_symlink_target_escape_is_rejected(self):
        for target in ("../../escape", "/outside", "..\\escape"):
            self.rows[-1]["target"] = target
            self.write_archive()
            with self.assertRaises(ValueError):
                self.restore()

    def test_existing_parent_symlink_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "research").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.restore()
        self.assertEqual(list(outside.iterdir()), [])

    def test_insufficient_disk_space_is_rejected(self):
        with patch.object(artifacts.shutil, "disk_usage", return_value=type("Usage", (), {"free": 0})()):
            with self.assertRaisesRegex(ValueError, "Insufficient space"):
                self.restore()

    def test_disk_estimate_rounds_each_file_and_scratch_to_filesystem_blocks(self):
        block_size = 8192
        rows = [{"size": 8193}, {"size": 1}, {"size": 0}]
        allocated = 3 * block_size
        expected = allocated + 2 * block_size + (allocated + 99) // 100 + 128 * 1024 * 1024
        usage = type("Filesystem", (), {"f_frsize": block_size})()
        with patch.object(artifacts.os, "statvfs", return_value=usage, create=True):
            self.assertEqual(artifacts.restore_bytes(self.root, rows, 268435456), expected)
        with patch.object(artifacts.os, "statvfs", side_effect=OSError, create=True):
            self.assertEqual(artifacts.restore_bytes(self.root, [{"size": 4097}], 268435456),
                             16384 + 82 + 128 * 1024 * 1024)

    def test_disk_estimate_exact_boundary(self):
        rows = [row for row in self.rows if row.get("type") != "symlink"]
        required = artifacts.restore_bytes(self.root.resolve(), rows, self.chunk_size, 1)
        usage = type("Usage", (), {"free": required - 1})()
        with patch.object(artifacts.shutil, "disk_usage", return_value=usage):
            with self.assertRaisesRegex(ValueError, "Insufficient space"):
                self.restore()
        self.assertFalse((self.root / "research").exists())
        usage.free = required
        with patch.object(artifacts.shutil, "disk_usage", return_value=usage):
            self.restore()

    def test_verify_catches_content_and_mode_change(self):
        self.restore()
        path = self.root / "research/first.txt"
        if os.name != "nt":
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                artifacts.verify(self.root)
            path.chmod(0o644)
        path.write_bytes(b"edit")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            artifacts.verify(self.root)

    def test_fetch_reuses_verified_pack_without_network(self):
        with patch.object(artifacts.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            artifacts.fetch(self.root, self.cache)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
