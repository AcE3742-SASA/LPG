#!/usr/bin/env python3
"""Import the reviewed clean prefix; leave measurement and admission to the frozen runner."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / 'abstract_sweep_725_v1'
sys.path.insert(0, str(RUNNER))
import run
import analyze_sweep

ARCHIVE_NAME = 'abstract_200_720_20260921_20260921_213639_262a4a.zip'
ARCHIVE_SHA = 'fec775a7d658ac4f2e65df76cbef573b94781dd0e1e7483e34d9c2ec110593fb'
ORIGINAL_DIGEST = '3551134f7df12d8d3215cc9b5993669f38ae9d8ed46dd2347b5abc4af7e3a95e'
SESSION = '20260921_193647_9ac17209'
KEEP = 1680
ORIGINAL_COUNT = 1800
OUTPUT = RUNNER / 'runs/abstract_200_720_recovery_20260921'
CHECK = RUNNER / 'runs/check_720_repeat_20260921_190318'
CHECK_MANIFEST_SHA = '03ebb174eaefba6e7292838e7b0bbf894ff211e97d1e71b35f622b980b59fd23'
CHECK_DIGEST = '3982763b20f26335d987209f7b7d33d5ffefc297a9a027687c4dc9e6d85ec434'
REVIEW_NOTE = (
    '2026-09-21 reviewed: 720 MHz check 48/48 PASS in two workers; correctness, '
    'timing, sampled clocks and reset accepted. User approved recovery: retain '
    '1680 measurements from 14 clean segments, redo all 120 from the flagged '
    'segment, then collect 1800 missing measurements. Original failed run is '
    'preserved in recovery/source_run.zip. Exact clock criteria unchanged; '
    'cumulative counter issue unresolved and retained.'
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def records_digest(entries):
    digest = hashlib.sha256()
    for seq, row in entries:
        value = json.dumps(dict(seq=seq, **row), ensure_ascii=False, sort_keys=True, allow_nan=False)
        digest.update((value + '\n').encode())
    return digest.hexdigest()


def extract_reviewed(archive, folder):
    require(run.sha(archive) == ARCHIVE_SHA, 'Source ZIP differs from the reviewed archive.')
    with zipfile.ZipFile(archive) as zipped:
        names = set()
        for entry in zipped.infolist():
            path = PurePosixPath(entry.filename)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in entry.filename
                    and path.parts and path.parts[0] == 'abstract_200_720_20260921'
                    and not stat.S_ISLNK(entry.external_attr >> 16)
                    and path.as_posix() not in names, 'Unsafe or duplicate archive path.')
            names.add(path.as_posix())
        require(zipped.testzip() is None, 'Source ZIP CRC failed.')
        zipped.extractall(folder)
    return folder / 'abstract_200_720_20260921'


def preserved_files(source):
    files = [source / 'manifest.json']
    for name in ('source', 'dependencies', 'launches'):
        files.extend(p for p in (source / name).rglob('*') if p.is_file())
    session = source / 'sessions' / SESSION
    files.extend((session / name) for name in ('session.json', 'environment.json'))
    for index in range(14):
        files.extend(p for p in (session / f'segment_{index:05d}').iterdir() if p.is_file())
    return [p.relative_to(source) for p in files]


def verify_import(out, source, original, loaded):
    manifest = run.manifest_for(out, 'full', loaded)
    entries = list(run.history(out))
    require(entries[:KEEP] == original[:KEEP], 'Imported records changed or import is incomplete.')
    require(all(run.sha(out / name) == run.sha(source / name) for name in preserved_files(source)),
            'Imported evidence changed.')
    require(run.sha(out / 'recovery/source_run.zip') == ARCHIVE_SHA, 'Preserved original ZIP changed.')
    _, done = run.choices_and_done(out, manifest, loaded)
    run.require_clean_resume(out, loaded)
    require('FAIL' not in done.values(), 'Saved FAIL requires review before resuming.')
    return len(manifest['tasks']) - len(done)


def prepare(archive, out, check):
    archive, out, check = archive.resolve(), out.resolve(), check.resolve()
    require(Path(run.__file__).resolve().parent == RUNNER, 'Wrong measurement module loaded.')
    loaded = run.load(720)
    require(run.sha(check / 'manifest.json') == CHECK_MANIFEST_SHA, 'Wrong admission check.')
    admission = run.check_admission(check, REVIEW_NOTE, loaded)
    require(admission['results_digest'] == CHECK_DIGEST, 'Admission check records changed.')
    with tempfile.TemporaryDirectory(prefix='gemm720-recovery-') as temporary:
        source = extract_reviewed(archive, Path(temporary))
        manifest = run.manifest_for(source, 'full', loaded)
        original = list(run.history(source))
        require(len(original) == ORIGINAL_COUNT and records_digest(original) == ORIGINAL_DIGEST,
                'Original records differ from reviewed results.')
        require([row['task'] for _, row in original] == manifest['tasks'][:ORIGINAL_COUNT],
                'Original task order differs.')
        require(all(row['segment'] == f'sessions/{SESSION}/segment_{(seq - 1) // 120:05d}'
                    for seq, row in original), 'Original segment boundaries differ.')
        provenance = dict(
            kind='reviewed_720_recovery_v1', source_archive_sha256=ARCHIVE_SHA,
            original_results_digest=ORIGINAL_DIGEST, imported_records=KEEP,
            imported_results_digest=records_digest(original[:KEEP]),
            excluded_segment=f'sessions/{SESSION}/segment_00014', redo_records=120,
            originally_missing_records=1800, initial_pending_records=1920,
            selected_clock_mhz=720, memory_mhz=6801, power_limit_w=70,
            check_manifest_sha256=CHECK_MANIFEST_SHA, check_results_digest=CHECK_DIGEST,
            review_note=REVIEW_NOTE,
            history_note='Original session ERROR/new_records=1800 and failed launch are preserved '
                         'unchanged as history. This derived collection imports only its clean '
                         '1680-record prefix. All 1800 original attempts remain in source_run.zip.')
        fresh = not out.exists()
        if fresh:
            out.mkdir(parents=True)
        with loaded['suite'].lock(out / '.launcher.lock'), loaded['suite'].lock(out / '.run.lock'):
            receipt = out / 'recovery_provenance.json'
            if not fresh:
                require(receipt.is_file(), 'Existing output lacks a completed recovery receipt; preserve and inspect it.')
                require(run.read(receipt) == provenance, 'Existing output belongs to a different recovery.')
                pending = verify_import(out, source, original, loaded)
            else:
                # Publish the receipt last: an interrupted import can never start GPU collection.
                for name in preserved_files(source):
                    (out / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / name, out / name)
                (out / 'recovery').mkdir()
                shutil.copy2(archive, out / 'recovery/source_run.zip')
                for path in HERE.iterdir():
                    if path.is_file() and path.suffix in ('.py', '.sh', '.md'):
                        shutil.copy2(path, out / 'recovery' / path.name)
                store = loaded['suite'].Store(out)
                try:
                    for _, row in original[:KEEP]:
                        store.append(row)
                finally:
                    store.close()
                pending = verify_import(out, source, original, loaded)
                summary = analyze_sweep.analyze(out, loaded)
                require(pending == 1920 and summary['counts'] == {'PASS': KEEP}
                        and summary['recorded'] == summary['attempts'] == KEEP
                        and summary['missing'] == 1920 and summary['gate'] == 'INCOMPLETE'
                        and not summary['clock_flags'] and not summary['between_round_flags']
                        and summary['all_restorations_accepted'], 'Imported quality check failed.')
                loaded['suite'].atomic_json(out / 'recovery/initial_summary.json', summary)
                loaded['suite'].atomic_json(receipt, provenance)
    print(f'RECOVERY READY: imported={KEEP}, pending={pending}, output={out}', flush=True)
    return pending


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=RUNNER / 'runs' / ARCHIVE_NAME)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', type=Path, default=CHECK)
    args = parser.parse_args()
    try:
        prepare(args.archive, args.output, args.check)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        raise SystemExit(f'RECOVERY STOP: {error}')
