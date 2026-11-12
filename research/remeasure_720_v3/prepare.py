#!/usr/bin/env python3
"""Import only reviewed PASS records into a separate v2-compatible collection."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

import run

HERE = Path(__file__).resolve().parent
ARCHIVE_NAME = 'full_720_remeasure_20260930_20260930_201957_8a23fb.zip'
ARCHIVE_SHA = '9bc86d586427662732062bff50da8830341163710933a6f892d0d0469e4e3b3a'
GRAPH_DIGEST = '66d072c65eb95e031a9bb43ac5b93c33e6f8364336fda5503fe3782f153024f6'
KEEP, REDO, TOTAL = 111582, 1800, 113382
ARCHIVE = HERE.parent / 'remeasure_720_v1/runs' / ARCHIVE_NAME
OUTPUT = HERE / 'runs/full_720_remeasure2_20260930'
REVIEW_NOTE = (
    '2026-09-30 user approved remeasurement of the remaining 1800 REVIEW tasks from the '
    'verified first remeasurement. Retain 111582 PASS records, original task IDs, '
    'seeds, splits and every Graph batch choice. Preserve the original ZIP and '
    'flag-and-continue policy; exact 720/6801 MHz and 70 W, correctness and fatal '
    'safety gates unchanged. Cumulative counter interpretation remains unresolved.'
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_prepared(out, loaded):
    receipt = run.read(out / 'recovery_provenance.json')
    require((receipt['source_archive_sha256'], receipt['imported_records'], receipt['redo_records'])
            == (ARCHIVE_SHA, KEEP, REDO), 'Output belongs to a different recovery')
    require(run.sha(out / 'recovery/source_run.zip') == ARCHIVE_SHA, 'Preserved ZIP changed')
    evidence = out / 'recovery/imported_evidence_sha256.json'
    require(run.sha(evidence) == receipt['evidence_index_sha256'], 'Evidence index changed')
    for name, wanted in run.read(evidence).items():
        path = (out / name).resolve()
        require(out in path.parents and run.sha(path) == wanted, 'Imported evidence changed: ' + name)
    manifest = run.manifest_for(out, 'full', loaded)
    choices, done = run.choices_and_done(out, manifest, loaded)
    require(loaded['plan'].digest(choices) == GRAPH_DIGEST, 'Original Graph choices changed')
    ids = set(run.read(out / 'recovery/remeasure_ids.json'))
    require(len(ids) == REDO, 'Remeasurement ID list changed')
    # SQLite reuses removed trailing row IDs; the cutoff is the last imported row.
    imported = hashlib.sha256()
    imported_count = 0
    for seq, row in run.history(out):
        if seq <= receipt['imported_last_seq']:
            require(row['task']['id'] not in ids, 'Excluded task appeared in imported records')
            value = json.dumps(dict(seq=seq, **row), ensure_ascii=False, sort_keys=True, allow_nan=False)
            imported.update((value + '\n').encode())
            imported_count += 1
        else:
            require(row['task']['id'] in ids, 'Unexpected task collected during remeasurement')
    require(imported_count == KEEP and imported.hexdigest() == receipt['imported_records_digest'],
            'Imported records changed')
    pending = {t['id'] for t in manifest['tasks']} - done.keys()
    require(pending <= ids, 'Unexpected pending task')
    return len(pending)


def prepare(archive, out):
    archive, out = archive.resolve(), out.resolve()
    require(out not in archive.parents, 'The source archive must be outside the output')
    loaded = run.load(720)
    out.parent.mkdir(parents=True, exist_ok=True)
    suite = loaded['suite']
    with suite.lock(out.parent / '.remeasure_prepare.lock'):
        if out.exists():
            require((out / 'recovery_provenance.json').is_file(),
                    'Existing output is incomplete or unrelated; preserve it and inspect')
            with suite.lock(out / '.launcher.lock'), suite.lock(out / '.run.lock'):
                pending = verify_prepared(out, loaded)
            print(f'REMEASUREMENT READY: imported={KEEP}, pending={pending}, output={out}', flush=True)
            return pending
        require(archive.is_file() and run.sha(archive) == ARCHIVE_SHA,
                'Missing or different reviewed source ZIP: ' + str(archive))
        require(shutil.disk_usage(out.parent).free >= 20 * 1024**3, 'Need at least 20 GiB free disk')
        print('CPU preparation: preserve original ZIP and import verified PASS records.', flush=True)
        with tempfile.TemporaryDirectory(prefix='.remeasure_', dir=out.parent) as temporary:
            stage = Path(temporary) / out.name
            stage.mkdir()
            manifest = run.manifest_for(stage, 'full', loaded)
            recovery = stage / 'recovery'
            recovery.mkdir()
            shutil.copy2(archive, recovery / 'source_run.zip')
            require(run.sha(recovery / 'source_run.zip') == ARCHIVE_SHA, 'ZIP copy mismatch')
            with zipfile.ZipFile(recovery / 'source_run.zip') as zipped:
                names = set()
                for entry in zipped.infolist():
                    p = PurePosixPath(entry.filename)
                    require(not p.is_absolute() and '..' not in p.parts and p.parts
                            and p.parts[0] == 'full_720_remeasure_20260930' and '\\' not in entry.filename
                            and not stat.S_ISLNK(entry.external_attr >> 16)
                            and p.as_posix() not in names, 'Unsafe or duplicate ZIP path')
                    names.add(p.as_posix())
                def read_member(name):
                    return json.loads(zipped.read('full_720_remeasure_20260930/' + name))
                old = read_member('manifest.json')
                summary = read_member('summary.json')
                require(all(old[k] == manifest[k] for k in ('tasks', 'protocol', 'shape_splits',
                        'modes', 'rounds', 'environment_signature')), 'Measurement plan changed')
                require(summary['collection_complete'] and summary['recorded'] == TOTAL
                        and summary['counts'] == {'PASS': KEEP, 'REVIEW': REDO}
                        and summary['all_restorations_accepted'], 'Source is not the reviewed completed run')
                review = read_member('remeasure_tasks.json')
                ids = {r['task']['id'] for r in review}
                require(len(ids) == len(review) == REDO, 'Wrong remeasurement list')
                suite.atomic_json(recovery / 'remeasure_ids.json', sorted(ids))
                for name in ('manifest.json', 'summary.json', 'remeasure_tasks.json'):
                    (recovery / ('source_' + name)).write_bytes(zipped.read('full_720_remeasure_20260930/' + name))
                source_digest, imported_digest = hashlib.sha256(), hashlib.sha256()
                kept_segments, excluded_segments, seen = set(), set(), set()
                graph = run.pinned_choices(loaded)
                count = last_seq = imported_last_seq = 0
                store = suite.Store(stage)
                try:
                    with store.con, zipped.open('full_720_remeasure_20260930/raw_attempts.jsonl') as raw:
                        for line in raw:
                            source_digest.update(line)
                            row = json.loads(line)
                            seq = row.pop('seq')
                            task = row['task']
                            require(task['id'] not in seen and seq > last_seq, 'Duplicate or unordered raw records')
                            seen.add(task['id']); last_seq = seq
                            if task['mode'] == 'graph':
                                key, nodes = loaded['plan'].graph_key(task), row['timing']['graph_nodes']
                                require(graph.setdefault(key, nodes) == nodes, 'Original Graph choice conflict')
                            if task['id'] in ids:
                                excluded_segments.add(row['segment'])
                                continue
                            require(row['status'] == 'PASS' and row['correctness']['passed']
                                    and row['correctness']['finite_all'], 'Non-PASS import refused')
                            imported_digest.update(line)
                            kept_segments.add(row['segment'])
                            store.con.execute('INSERT INTO attempts(seq,task_id,record) VALUES (?,?,?)',
                                (seq, task['id'], json.dumps(row, ensure_ascii=False, allow_nan=False)))
                            count += 1
                            imported_last_seq = seq
                finally:
                    store.close()
                require(source_digest.hexdigest() == summary['results_digest'] and len(seen) == TOTAL
                        and count == KEEP and len(excluded_segments) == 15
                        and not kept_segments & excluded_segments, 'Raw import or whole-segment exclusion mismatch')
                require(loaded['plan'].digest(graph) == GRAPH_DIGEST, 'Original Graph choices differ')
                evidence = {}
                for name in sorted(names):
                    p = PurePosixPath(name)
                    relative = PurePosixPath(*p.parts[1:])
                    selected = len(p.parts) >= 3 and p.parts[1] == 'launches'
                    if len(p.parts) >= 4 and p.parts[1] == 'sessions':
                        selected = len(p.parts) == 4 or str(PurePosixPath(*p.parts[1:4])) in kept_segments
                    if not selected or name.endswith('/'):
                        continue
                    target = stage / str(relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(name) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    if target.name != 'quality.json':
                        evidence[str(relative)] = run.sha(target)
                for segment in kept_segments:
                    quality = run.read(stage / segment / 'quality.json')
                    require(not quality['flags'] and all(q['status'] == 'PASS'
                            for q in quality['measurements'].values()), 'Imported segment is not PASS')
                suite.atomic_json(recovery / 'imported_evidence_sha256.json', evidence)
                suite.atomic_json(stage / 'recovery_provenance.json', dict(
                    kind='reviewed_full_720_remeasurement_round2_v3', source_archive_sha256=ARCHIVE_SHA,
                    source_results_digest=summary['results_digest'], source_last_seq=last_seq,
                    imported_last_seq=imported_last_seq,
                    imported_records=KEEP, redo_records=REDO, excluded_segments=sorted(excluded_segments),
                    imported_records_digest=imported_digest.hexdigest(), original_graph_digest=GRAPH_DIGEST,
                    evidence_index_sha256=run.sha(recovery / 'imported_evidence_sha256.json'),
                    review_note=REVIEW_NOTE, timestamp=suite.now()))
            require(verify_prepared(stage, loaded) == REDO, 'Unexpected pending tasks after import')
            suite.atomic_json(stage / 'progress.json', dict(state='READY_REMEASURE', recorded=KEEP,
                expected=TOTAL, flagged=0, updated_utc=suite.now()))
            stage.rename(out)
    print(f'REMEASUREMENT READY: imported={KEEP}, pending={REDO}, output={out}', flush=True)
    return REDO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.archive, args.output)
