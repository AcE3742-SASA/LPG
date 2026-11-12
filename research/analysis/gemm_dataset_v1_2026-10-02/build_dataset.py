"""Build a derived GEMM view without reading locked test timing fields."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'analysis/remeasure_720_v4_result_2026-10-01/recomputed/full_720_remeasure3_20261001'
VERSION = 'gemm_720_adopted_v1_2026-10-02'
ARCHIVE_SHA = '319c8480f15fd5f310bd0c211b326104ed341276b76e214bec688b673ecdc5c4'
ROLES = {'legacy_train': 'train_candidate_before_C_reservation',
         'legacy_validation': 'validation', 'development': 'development',
         'legacy_test_explored': 'explored_diagnostic',
         'fresh_holdout': 'locked_final_test',
         'boundary_diagnostic': 'boundary_diagnostic', 'calibration': 'calibration'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def save_csv(name, rows, fields):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def condition(task):
    return (task['phase'], task['split'], task.get('group', ''),
            tuple(task['shape']), task['dtype'], task['mode'])


def is_complete(tasks, held):
    return len(tasks) == 3 and {t['round'] for t in tasks} == {0, 1, 2} and not any(t['id'] in held for t in tasks)


def self_check():
    tasks = [dict(id=str(i), round=i) for i in range(3)]
    assert is_complete(tasks, set())
    assert not is_complete(tasks, {'1'})
    assert not is_complete(tasks[:2], set())
    assert not is_complete([tasks[0], tasks[0], tasks[2]], set())
    print('PASS: complete-round and quarantine checks')


def build():
    watched = ['manifest.json', 'remeasure_tasks.json', 'metrics.csv',
               'summary.json', 'recovery_provenance.json', 'results.sqlite3']
    before = {name: sha(SOURCE / name) for name in watched}
    archive_validation = read(SOURCE.parents[1] / 'archive_validation.json')
    for name, value in before.items():
        if name in archive_validation['original_export_sha256']:
            assert value == archive_validation['original_export_sha256'][name]
    archive = ROOT / 'full_720_remeasure3_20261001_20261001_163605_88321a.zip'
    assert sha(archive) == ARCHIVE_SHA
    manifest = read(SOURCE / 'manifest.json')
    tasks = manifest['tasks']
    expected = {t['id']: t for t in tasks}
    assert len(expected) == len(tasks) == 113382
    reviews = {r['task']['id']: r for r in read(SOURCE / 'remeasure_tasks.json')}
    assert len(reviews) == 54 and set(reviews) <= expected.keys()
    assert not any(expected[tid]['split'] == 'fresh_holdout' for tid in reviews)
    for tid, row in reviews.items():
        assert row['task'] == expected[tid]

    # SQL projects metadata only. Locked latency, block and performance columns never enter Python.
    query = """SELECT seq, task_id, json_extract(record,'$.task'),
        json_extract(record,'$.session'), json_extract(record,'$.segment'),
        json_extract(record,'$.input_seed'), json_extract(record,'$.timing.graph_nodes')
        FROM attempts ORDER BY seq"""
    metadata = {}
    with sqlite3.connect((SOURCE / 'results.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        for seq, tid, task_json, session, segment, seed, nodes in db.execute(query):
            assert tid not in metadata and json.loads(task_json) == expected[tid]
            metadata[tid] = dict(seq=seq, session=session, segment=segment,
                                 input_seed=seed, graph_nodes=nodes)
    assert metadata.keys() == expected.keys()

    quality = {}
    quality_files = sorted({m['segment'] + '/quality.json' for m in metadata.values()})
    for relative in quality_files:
        saved = read(SOURCE / relative)
        assert not (quality.keys() & saved['measurements'].keys())
        quality.update(saved['measurements'])
    assert quality.keys() == expected.keys()
    assert Counter(q['status'] for q in quality.values()) == {'PASS': 113331, 'REVIEW': 51}
    assert {tid for tid, q in quality.items() if q['status'] != 'PASS'} <= reviews.keys()

    unlocked = {}
    with (SOURCE / 'metrics.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            tid = row['id']
            assert tid not in unlocked and expected[tid]['split'] != 'fresh_holdout'
            assert row['status'] == quality[tid]['status']
            assert row['session'] == metadata[tid]['session']
            value = float(row['median_ms'])
            assert math.isfinite(value) and value > 0
            unlocked[tid] = value
    assert unlocked.keys() == {tid for tid, t in expected.items() if t['split'] != 'fresh_holdout'}

    origins = {}
    for label, directory, sessions in [
        ('full_720', 'full_720_result_2026-09-30', ['20260922_165030_e89635b2', '20260922_175658_0cf7c114']),
        ('remeasure_v1', 'remeasure_720_v1_result_2026-09-30', ['20260930_144016_61da8183']),
        ('remeasure_v4', 'remeasure_720_v4_result_2026-10-01', ['20261001_143341_57d18ef1'])]:
        receipt = read(ROOT / 'analysis' / directory / 'archive_validation.json')
        for session in sessions:
            origins[session] = dict(origin_run=label, origin_archive=receipt['archive'],
                                    origin_archive_sha256=receipt['sha256'])
    assert {m['session'] for m in metadata.values()} == origins.keys()

    grouped = defaultdict(list)
    index, excluded = [], []
    for task in tasks:
        tid = task['id']
        grouped[condition(task)].append(task)
        review = reviews.get(tid)
        row = {**task, 'shape': json.dumps(task['shape']),
               'group': task.get('group', ''), **metadata[tid],
               **origins[metadata[tid]['session']], 'quality_status': quality[tid]['status'],
               'disposition': 'quarantine' if review else 'accepted',
               'between_round_review': bool(review and review['between_round_review']),
               'role': ROLES[task['split']], 'source_archive_sha256': ARCHIVE_SHA,
               'raw_record_locator': 'results.sqlite3:attempts:' + str(metadata[tid]['seq']),
               'quality_locator': metadata[tid]['segment'] + '/quality.json',
               'flags': json.dumps(quality[tid]['flags'])}
        index.append(row)
        if review:
            excluded.append(row)
    fields = list(index[0])
    save_csv('measurement_index.csv', index, fields)
    save_csv('quarantine_54.csv', excluded, fields)
    accepted = [r for r in index if r['disposition'] == 'accepted']
    save_csv('accepted_unlocked_measurements.csv',
             [{**r, 'median_ms': unlocked[r['id']]} for r in accepted if r['split'] != 'fresh_holdout'], fields + ['median_ms'])
    save_csv('accepted_locked_index.csv', [r for r in accepted if r['split'] == 'fresh_holdout'], fields)

    model_rows, locked_conditions, partial = [], [], []
    for key, members in sorted(grouped.items()):
        task = members[0]
        row = dict(phase=task['phase'], split=task['split'], group=task.get('group', ''),
                   shape=json.dumps(task['shape']), dtype=task['dtype'], mode=task['mode'],
                   task_ids=json.dumps([t['id'] for t in sorted(members, key=lambda t: t['round'])]),
                   accepted_rounds=sum(t['id'] not in reviews for t in members),
                   role=ROLES[task['split']])
        if not is_complete(members, reviews):
            partial.append(row)
        elif task['phase'] == 'main':
            if task['split'] == 'fresh_holdout':
                locked_conditions.append(row)
            else:
                model_rows.append({**row, 'median_of_round_medians_ms': statistics.median(unlocked[t['id']] for t in members)})
    condition_fields = ['phase', 'split', 'group', 'shape', 'dtype', 'mode', 'task_ids', 'accepted_rounds', 'role']
    save_csv('main_complete_unlocked_conditions.csv', model_rows, condition_fields + ['median_of_round_medians_ms'])
    save_csv('main_complete_locked_index.csv', locked_conditions, condition_fields)
    save_csv('incomplete_conditions.csv', partial, condition_fields)

    counts = []
    for phase, split in sorted({(t['phase'], t['split']) for t in tasks}):
        selected = [t for t in tasks if (t['phase'], t['split']) == (phase, split)]
        good = [t for t in selected if t['id'] not in reviews]
        complete = [v for k, v in grouped.items() if k[:2] == (phase, split) and is_complete(v, reviews)]
        counts.append(dict(phase=phase, split=split, total_measurements=len(selected),
                           accepted_measurements=len(good), quarantined_measurements=len(selected)-len(good),
                           total_unique_MNK=len({tuple(t['shape']) for t in selected}),
                           accepted_unique_MNK=len({tuple(t['shape']) for t in good}),
                           complete_conditions=len(complete),
                           complete_fp16_shapes=len({tuple(v[0]['shape']) for v in complete if v[0]['dtype']=='fp16'}),
                           complete_fp32_shapes=len({tuple(v[0]['shape']) for v in complete if v[0]['dtype']=='fp32'}),
                           role=ROLES[split]))
    save_csv('split_phase_counts.csv', counts, list(counts[0]))
    scope = []
    for shape in sorted({tuple(expected[tid]['shape']) for tid in reviews}):
        for phase in ['main', 'controls']:
            for dtype in ['fp16', 'fp32']:
                selected = [t for t in tasks if tuple(t['shape']) == shape and t['phase'] == phase and t['dtype'] == dtype]
                if selected:
                    scope.append(dict(shape=json.dumps(shape), phase=phase, dtype=dtype,
                                      split=selected[0]['split'], total=len(selected),
                                      accepted=sum(t['id'] not in reviews for t in selected),
                                      quarantine=sum(t['id'] in reviews for t in selected),
                                      accepted_rounds_by_mode=json.dumps({mode: sorted(t['round'] for t in selected if t['mode']==mode and t['id'] not in reviews) for mode in manifest['modes']})))
    save_csv('shape_scope.csv', scope, list(scope[0]))
    assert len(accepted) == 113328 and len(excluded) == 54
    assert len(locked_conditions) == 6000
    assert len(model_rows) + len(locked_conditions) == 35994
    assert sum(r['accepted_rounds'] for r in partial if r['phase']=='main') == 18
    after = {name: sha(SOURCE / name) for name in watched}
    assert after == before
    summary = dict(version=VERSION, source_archive_sha256=ARCHIVE_SHA, measurements=len(tasks),
                   accepted_measurements=len(accepted), quarantine_measurements=len(excluded),
                   quality_statuses=dict(Counter(q['status'] for q in quality.values())),
                   main_complete_conditions=len(model_rows)+len(locked_conditions),
                   main_unlocked_complete_conditions=len(model_rows), main_locked_complete_conditions=len(locked_conditions),
                   main_accepted_partial_round_measurements=18, split_phase_counts=counts,
                   origin_counts=dict(Counter(origins[m['session']]['origin_run'] for m in metadata.values())),
                   source_unchanged=True, locked_timing_fields_read=False, training_started=False,
                   builder_sha256=sha(Path(__file__)),
                   C_reservation='pending_layer_operation_overlap_audit',
                   quality_files=len(quality_files), input_sha256=before,
                   quality_evidence_sha256={name: sha(SOURCE/name) for name in quality_files})
    save_json('summary.json', summary)
    outputs = sorted(p for p in OUT.iterdir() if p.suffix in ('.csv', '.json') and p.name != 'derived_hashes.json')
    save_json('derived_hashes.json', {p.name: sha(p) for p in outputs})
    print('PASS: derived dataset, source preservation and locked-test boundary')
    print('Accepted measurements: 113328; quarantine: 54; main complete conditions: 35994')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if not args.self_check:
        build()
