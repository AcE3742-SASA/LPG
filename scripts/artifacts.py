#!/usr/bin/env python3
"""Fetch, restore, and verify the LPG research archive (Python 3.10+)."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unicodedata


HEX = re.compile(r"[0-9a-f]{64}\Z")
PACK = re.compile(r"research-[0-9]+\.tar\.gz\Z")
BUFFER = 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER), b""):
            value.update(block)
    return value.hexdigest()


def relative_path(value, research=True):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("Invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in value.split("/")):
        raise ValueError("Path traversal rejected: " + ascii(value))
    if research and path.parts[0] != "research":
        raise ValueError("File path must be inside research/")
    return path


def safe_destination(root, value):
    path = root.joinpath(*relative_path(value).parts)
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Symlink in destination path: " + ascii(str(parent)))
    return path


def check_link(root, row):
    target = row["target"]
    if not isinstance(target, str) or not target or "\\" in target or "\0" in target:
        raise ValueError("Invalid symlink target")
    if PurePosixPath(target).is_absolute():
        raise ValueError("Absolute symlink target rejected")
    combined = posixpath.normpath(str(PurePosixPath(row["path"]).parent / target))
    relative_path(combined)
    resolved = (root / combined).resolve()
    if not resolved.is_relative_to(root / "research"):
        raise ValueError("Symlink escapes research/")


def check_file(path, record):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or unsafe file: " + ascii(str(path)))
    if path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
        raise ValueError("Size or SHA-256 mismatch: " + ascii(str(path)))


def load(root, prefix=None):
    root = root.resolve()
    metadata = json.loads((root / "manifests/artifacts.json").read_text())
    if metadata["format_version"] != 1 or metadata["chunk_size"] <= 0:
        raise ValueError("Unsupported archive format")
    manifest = metadata["files_manifest"]
    manifest_path = root.joinpath(*relative_path(manifest["path"], False).parts)
    if not manifest_path.resolve().is_relative_to(root):
        raise ValueError("Manifest escapes repository")
    check_file(manifest_path, manifest)
    pack_names = set()
    for pack in metadata["packs"]:
        if not PACK.fullmatch(pack["name"]) or pack["name"] in pack_names:
            raise ValueError("Invalid or duplicate pack name")
        if not HEX.fullmatch(pack["sha256"]) or pack["size"] < 0:
            raise ValueError("Invalid pack record")
        pack_names.add(pack["name"])
    if prefix:
        prefix = str(relative_path(unicodedata.normalize("NFC", prefix.rstrip("/"))))
    rows, seen = [], set()
    with gzip.open(manifest_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            name = str(relative_path(row["path"]))
            if name in seen:
                raise ValueError("Duplicate manifest path: " + ascii(name))
            seen.add(name)
            if row.get("type") == "symlink":
                check_link(root, row)
            else:
                if row.get("storage") not in {"git", "release"}:
                    raise ValueError("Invalid storage value")
                if row["size"] < 0 or not HEX.fullmatch(row["sha256"]):
                    raise ValueError("Invalid file record")
                if not isinstance(row["mode"], int) or row["mode"] & ~0o777:
                    raise ValueError("Unsafe file mode")
                chunks = row.get("chunks")
                if chunks is not None:
                    count = (row["size"] + metadata["chunk_size"] - 1) // metadata["chunk_size"]
                    if len(chunks) != count or not all(HEX.fullmatch(c) for c in chunks):
                        raise ValueError("Invalid chunk list")
                elif row["size"] > metadata["chunk_size"]:
                    raise ValueError("Large file is missing chunks")
            if not prefix or name == prefix or name.startswith(prefix + "/"):
                rows.append(row)
    if not rows:
        raise ValueError("No files match the requested path prefix")
    return metadata, rows


def mode_matches(path, row):
    if os.name == "nt":
        return True
    mask = 0o111 if row["storage"] == "git" else 0o777
    return stat.S_IMODE(path.stat().st_mode) & mask == row["mode"] & mask


def restore_bytes(root, rows, chunk_size, link_count=0):
    if not rows and not link_count:
        return 0
    try:
        block_size = os.statvfs(root).f_frsize or 4096
    except (AttributeError, OSError):
        block_size = 4096
    allocated = sum((row["size"] + block_size - 1) // block_size * block_size for row in rows)
    scratch = min(chunk_size, max((row["size"] for row in rows), default=0))
    scratch = (scratch + block_size - 1) // block_size * block_size
    metadata_margin = (allocated + 99) // 100 + 128 * 1024 * 1024
    return allocated + scratch + metadata_margin


def fetch(root, cache):
    root = root.resolve()
    metadata, _ = load(root)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", metadata["repository"]):
        raise ValueError("Invalid repository name")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", metadata["release_tag"]):
        raise ValueError("Invalid release tag")
    cache.mkdir(parents=True, exist_ok=True)
    for index, pack in enumerate(metadata["packs"], 1):
        destination = cache / pack["name"]
        if destination.exists() or destination.is_symlink():
            check_file(destination, pack)
        else:
            with tempfile.TemporaryDirectory(prefix="download-", dir=cache) as temporary:
                subprocess.run(["gh", "release", "download", metadata["release_tag"],
                                "--repo", metadata["repository"], "--pattern", pack["name"],
                                "--dir", temporary], check=True)
                downloaded = Path(temporary) / pack["name"]
                check_file(downloaded, pack)
                os.link(downloaded, destination)
        print("Verified pack %d/%d" % (index, len(metadata["packs"])), flush=True)


def verify(root, prefix=None):
    root = root.resolve()
    _, rows = load(root, prefix)
    for row in rows:
        path = root / row["path"]
        if row.get("type") == "symlink":
            safe_destination(root, str(PurePosixPath(row["path"]).parent))
            check_link(root, row)
            if not path.is_symlink() or os.readlink(path) != row["target"]:
                raise ValueError("Symlink mismatch: " + ascii(row["path"]))
        else:
            safe_destination(root, row["path"])
            check_file(path, row)
            if not mode_matches(path, row):
                raise ValueError("File mode mismatch: " + ascii(row["path"]))
    print("Verified %d manifest paths" % len(rows))


def restore(root, cache, prefix=None):
    root = root.resolve()
    metadata, rows = load(root, prefix)
    pending, links, skipped = [], [], 0
    for row in rows:
        path = root / row["path"]
        if row.get("type") == "symlink":
            safe_destination(root, str(PurePosixPath(row["path"]).parent))
            if path.is_symlink() and os.readlink(path) == row["target"]:
                skipped += 1
            elif path.exists() or path.is_symlink():
                raise ValueError("Refusing to overwrite: " + ascii(row["path"]))
            else:
                links.append(row)
        else:
            safe_destination(root, row["path"])
            if path.exists():
                check_file(path, row)
                if not mode_matches(path, row):
                    raise ValueError("Existing file mode differs: " + ascii(row["path"]))
                skipped += 1
            else:
                pending.append(row)
    required = restore_bytes(root, pending, metadata["chunk_size"], len(links))
    free = shutil.disk_usage(root).free
    print("Restore needs %d bytes; %d bytes available" % (required, free), flush=True)
    if required > free:
        raise ValueError("Insufficient space; use --path-prefix for a smaller restore")
    if pending:
        for pack in metadata["packs"]:
            check_file(cache / pack["name"], pack)
    routes, lengths = {}, {}
    for index, row in enumerate(pending):
        for position, key in enumerate(row.get("chunks", [row["sha256"]])):
            offset = position * metadata["chunk_size"]
            length = min(metadata["chunk_size"], row["size"] - offset)
            if key in lengths and lengths[key] != length:
                raise ValueError("Inconsistent blob size")
            lengths[key] = length
            routes.setdefault(key, []).append((index, offset))
    with tempfile.TemporaryDirectory(prefix=".lpg-restore-", dir=root) as temporary:
        temporary = Path(temporary)
        remaining = set(routes)
        if pending:
            for pack in metadata["packs"]:
                with tarfile.open(cache / pack["name"], "r|gz") as archive:
                    for member in archive:
                        key = member.name.removeprefix("blobs/")
                        if not member.isfile() or member.name != "blobs/" + key or not HEX.fullmatch(key):
                            raise ValueError("Unsafe archive member: " + ascii(member.name))
                        if key not in remaining:
                            continue
                        if member.size != lengths[key]:
                            raise ValueError("Blob size mismatch: " + key)
                        source = archive.extractfile(member)
                        blob = temporary / "blob"
                        value = hashlib.sha256()
                        with blob.open("wb") as output:
                            for block in iter(lambda: source.read(BUFFER), b""):
                                value.update(block)
                                output.write(block)
                        if value.hexdigest() != key:
                            raise ValueError("Blob SHA-256 mismatch: " + key)
                        for index, offset in routes[key]:
                            output_path = temporary / str(index)
                            with output_path.open("r+b" if output_path.exists() else "w+b") as output:
                                output.seek(offset)
                                with blob.open("rb") as source_file:
                                    shutil.copyfileobj(source_file, output, BUFFER)
                        blob.unlink()
                        remaining.remove(key)
                print("Read " + pack["name"], flush=True)
                if prefix and not remaining:
                    break
        if remaining:
            raise ValueError("Missing %d required blobs" % len(remaining))
        for index, row in enumerate(pending):
            staged = temporary / str(index)
            check_file(staged, row)
            staged.chmod(row["mode"])
        for index, row in enumerate(pending):
            destination = safe_destination(root, row["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive publication; each destination has its own staged inode.
            os.link(temporary / str(index), destination)
            (temporary / str(index)).unlink()
        for row in links:
            check_link(root, row)
            destination = safe_destination(root, row["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(row["target"])
    print("Restored %d files and %d links; skipped %d matching paths" %
          (len(pending), len(links), skipped))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "restore", "verify"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--path-prefix", help="Restore or verify one path or directory")
    args = parser.parse_args()
    root = args.root.resolve()
    cache = (args.cache_dir or root / ".artifacts-cache").resolve()
    try:
        if args.command == "fetch":
            if args.path_prefix:
                parser.error("fetch downloads all packs; --path-prefix applies to restore/verify")
            fetch(root, cache)
        elif args.command == "restore":
            restore(root, cache, args.path_prefix)
        else:
            verify(root, args.path_prefix)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, subprocess.CalledProcessError) as error:
        print("ERROR: " + str(error).encode("ascii", "backslashreplace").decode(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
