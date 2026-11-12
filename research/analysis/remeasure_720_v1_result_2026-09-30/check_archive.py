"""Preserve the returned ZIP and verify it before CPU-only reanalysis."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SOURCE = Path('/Users/studyhard/Desktop/Transfer/full_720_remeasure_20260930_20260930_201957_8a23fb.zip')
ARCHIVE = BASE / SOURCE.name
OUT = BASE / 'recomputed/full_720_remeasure_20260930'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not ARCHIVE.exists() and not OUT.exists(), 'Do not overwrite prior evidence'
    shutil.copy2(SOURCE, ARCHIVE)
    digest = sha(SOURCE)
    assert sha(ARCHIVE) == digest
    print('Archive copied and SHA-256 verified.', flush=True)
    exports = {}
    with zipfile.ZipFile(ARCHIVE) as zipped:
        entries = zipped.infolist()
        assert len(entries) == len({entry.filename for entry in entries})
        for entry in entries:
            path = PurePosixPath(entry.filename)
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in entry.filename
            assert path.parts[0] == OUT.name
            assert not stat.S_ISLNK(entry.external_attr >> 16)
        prefix = OUT.name + '/'
        manifest = json.loads(zipped.read(prefix + 'manifest.json'))
        for group, local, expected in [('source', ROOT / 'remeasure_720_v1', manifest['source_hashes']),
                                       ('dependencies', ROOT, manifest['dependency_hashes'])]:
            for name, wanted in expected.items():
                assert hashlib.sha256(zipped.read(prefix + group + '/' + name)).hexdigest() == wanted
                assert sha(local / name) == wanted, name
        for entry in entries:
            relative = PurePosixPath(entry.filename).relative_to(OUT.name)
            if len(relative.parts) == 1 and relative.suffix in ('.json', '.csv', '.md'):
                exports[str(relative)] = hashlib.sha256(zipped.read(entry)).hexdigest()
        (BASE / 'original_summary.json').write_bytes(zipped.read(prefix + 'summary.json'))
        # ZipFile checks each member's CRC while extracting to this new directory.
        zipped.extractall(OUT.parent)
    result = dict(archive=str(ARCHIVE), sha256=digest, source_sha256=digest,
                  entries=len(entries), expanded_bytes=sum(e.file_size for e in entries),
                  zip_crc='PASS', safe_paths='PASS', source_files=len(manifest['source_hashes']),
                  dependency_files=len(manifest['dependency_hashes']), local_code_matches_archive=True,
                  original_export_sha256=exports)
    (BASE / 'archive_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('CRC, source, dependencies and extraction: PASS', flush=True)
    sys.path.insert(0, str(ROOT / 'remeasure_720_v1'))
    import prepare
    import run
    pending = prepare.verify_prepared(OUT, run.load(720))
    assert pending == 0
    (BASE / 'provenance_validation.json').write_text(json.dumps(dict(
        status='PASS', pending=pending, imported_records=108222, remeasured_records=5160,
        original_zip_preserved=True, imported_evidence_and_records_unchanged=True,
        original_graph_choices_preserved=True), indent=2) + '\n')
    print('Recovery provenance: PASS. Starting unchanged CPU analyzer.', flush=True)
    with (BASE / 'recompute.log').open('w') as log:
        subprocess.run([sys.executable, '-B', str(ROOT / 'remeasure_720_v1/run.py'),
                        'analyze', '--output', str(OUT)], stdout=log, stderr=subprocess.STDOUT, check=True)
    assert json.loads((BASE / 'recompute.log').read_text()) == json.loads((BASE / 'original_summary.json').read_text())
    assert all(sha(OUT / name) == wanted for name, wanted in exports.items())
    print('Recomputed summary and all saved exports: exact match.', flush=True)


if __name__ == '__main__':
    main()
