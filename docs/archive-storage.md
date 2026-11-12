# Archive storage

The repository keeps the original experiment layout under `research/`. Code, documents, figures, and small result tables are checked into Git. Larger files and the full measurement history are in the [v1.0.0 release](https://github.com/AcE3742-SASA/LPG/releases/tag/v1.0.0).

## Contents

[`manifests/files.jsonl.gz`](../manifests/files.jsonl.gz) lists every retained research file with its path, size, SHA-256, permissions, and storage location. `storage: git` means the file is also available directly in the Git tree. The release contains all retained file contents, including those in Git.

The snapshot includes raw measurements, derived tables, trained models, original ZIPs, failed attempts, source copies embedded in results, manifests, and technical evaluation records. Files with identical contents share a stored blob. Restoration writes a separate file at each recorded path, so editing one restored file leaves its duplicates unchanged.



The v1.0.0 snapshot contains 1,253,340 research files totaling 87,470,921,715 bytes, plus three symlinks. Git carries 3,770 of those files. The 36 release packs total 12,453,551,374 bytes.

Three trace-directory symlinks use relative targets in this export. Their original absolute targets are recorded as `original_target`. Filenames are normalized to Unicode NFC for consistent checkouts; `original_path` records any change in Unicode representation. File contents keep their original bytes.

Installed environments, bytecode caches, report drafts, and internal authoring records are outside this public snapshot. Pinned requirements and recorded environment versions remain with the research.

## Download and restore

Run these commands from the repository root with Python 3.10+ and an authenticated GitHub CLI:

```bash
python3 scripts/artifacts.py fetch
python3 scripts/artifacts.py restore
python3 scripts/artifacts.py verify
```

Downloads go to `.artifacts-cache/`. `--cache-dir /path/to/packs` selects another location. `--path-prefix research/analysis/final_test_2026-11-11` limits restoration or verification to that directory. Fetch downloads the complete set of packs. The script checks pack checksums, available disk space, existing files, and reconstructed file hashes before publishing restored files. Matching existing files are reused; a different existing file stops restoration.

The complete tree contains over a million measurement files. Allow at least 110 GB of free space for the restored files, filesystem allocation, and downloaded packs. A prefix restore needs less space; its preflight reports the requirement.

Release payloads are ignored by Git. Existing tracked research files stay tracked. When adding a new source file to the archive, use `git add -f research/path/to/file` and update the snapshot manifests and release together.

## Format

[`manifests/artifacts.json`](../manifests/artifacts.json) records the release tag, manifest checksum, pack checksums, and coverage totals. Each `research-NNNN.tar.gz` holds regular members named `blobs/<sha256>`. Files larger than 256 MiB are split into ordered chunks; the file manifest records both the chunks and the complete file checksum. Each pack holds up to 1 GiB of uncompressed content. The helper assembles files from these blobs rather than extracting archive-supplied paths.

The pack format uses standard gzip and tar. The size choices fit GitHub's [release asset limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases). The manifest preserves path identity separately from content deduplication.

## Tool checks

```bash
python3 -m unittest discover -s tests -v
```

The tests cover chunked and empty files, independent duplicate restoration, prefix selection, existing files, corrupt or missing data, path traversal, symlinks, permissions, and disk-space checks. These checks exercise the archive tools. The saved research verification records describe the original measurements and model evaluation.
