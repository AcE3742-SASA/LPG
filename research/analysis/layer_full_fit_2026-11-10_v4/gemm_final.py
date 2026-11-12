"""Freeze development-selected GEMM models, then separately evaluate sealed data."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics

import joblib
import numpy as np

import components

learners = components.learners
gemm = learners.old
ROOT = components.ROOT
RUN = ROOT / 'analysis/learner_comparison_2026-10-28/runs/v1'
DATASET = ROOT / 'analysis/gemm_dataset_v1_2026-10-02'
RAW = ROOT / 'analysis/remeasure_720_v4_result_2026-10-01/recomputed/full_720_remeasure3_20261001'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(path, digest):
    if sha(path) != digest:
        raise ValueError('Frozen file checksum mismatch: ' + str(path))


def choose_validation(scores):
    selected = []
    for rq in ('RQ1', 'RQ4'):
        for dtype in gemm.DTYPES:
            for mode in gemm.MODES if rq == 'RQ1' else ('prealloc', 'graph'):
                candidates = [r for r in scores if (r['rq'], r['dtype'], r['mode']) == (rq, dtype, mode)
                              and int(r['failures']) == 0
                              and math.isfinite(float(r['mean_mape_pct']))
                              and math.isfinite(float(r['mean_p90_pct']))]
                if not candidates:
                    raise ValueError('No finite development candidate')
                best = min(candidates, key=lambda r: (float(r['mean_mape_pct']), float(r['mean_p90_pct']),
                                                       int(r['n']), r['learner'], r['method']))
                selected.append({**best, 'n': int(best['n'])})
    return selected


def locked_metadata(condition_path, measurement_path, expected_shapes=1000):
    """Read time-free indexes and enforce exact PASS identities and three rounds."""
    conditions = rows(condition_path)
    measurements = rows(measurement_path)
    by_id, grouped, shapes = {}, {}, set()
    for row in measurements:
        if (row['phase'], row['split'], row['quality_status'], row['disposition'], row['role']) != (
                'main', 'fresh_holdout', 'PASS', 'accepted', 'locked_final_test'):
            raise ValueError('Invalid sealed measurement quality or role')
        if row['between_round_review'] != 'False' or json.loads(row['flags']):
            raise ValueError('Sealed measurement carries quality flags')
        shape = tuple(json.loads(row['shape']))
        if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError('Invalid sealed shape')
        if row['dtype'] not in gemm.DTYPES or row['mode'] not in gemm.MODES:
            raise ValueError('Invalid sealed dtype or mode')
        if row['id'] in by_id or int(row['round']) not in (0, 1, 2):
            raise ValueError('Duplicate sealed ID or invalid round')
        by_id[row['id']] = row
    used = set()
    for row in conditions:
        if (row['phase'], row['split'], row['role'], int(row['accepted_rounds'])) != (
                'main', 'fresh_holdout', 'locked_final_test', 3):
            raise ValueError('Invalid sealed condition')
        shape = tuple(json.loads(row['shape']))
        key = (shape, row['dtype'], row['mode'])
        tids = json.loads(row['task_ids'])
        if key in grouped or len(set(tids)) != 3 or used.intersection(tids):
            raise ValueError('Duplicate sealed condition or task')
        members = [by_id[tid] for tid in tids]
        if {int(m['round']) for m in members} != {0, 1, 2}:
            raise ValueError('Incomplete sealed rounds')
        if any((tuple(json.loads(m['shape'])), m['dtype'], m['mode']) != key for m in members):
            raise ValueError('Sealed condition identity mismatch')
        grouped[key] = tids
        shapes.add(shape)
        used.update(tids)
    if (len(shapes), len(grouped), len(by_id)) != (expected_shapes, expected_shapes * 6, expected_shapes * 18):
        raise ValueError('Sealed index counts changed')
    if used != set(by_id) or set(grouped) != {(s, d, m) for s in shapes for d in gemm.DTYPES for m in gemm.MODES}:
        raise ValueError('Incomplete sealed condition grid')
    return sorted(shapes), grouped, by_id


def predict(envelope, shapes):
    x, lower = gemm.design(shapes, envelope['dtype'])
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        if envelope['rq'] == 'RQ1':
            result = np.exp(envelope['model'].predict(x))
            return result * lower if envelope['method'] == 'normalized' else result
        source = envelope['source']
        prior = np.exp(source['model'].predict(x)) * (lower if source['target'] == 'normalized' else 1)
        method = envelope['method']
        if method == 'zero':
            return prior
        if method == 'constant_ratio':
            return prior * np.exp(envelope['model'])
        xx = np.column_stack((x, np.log(prior))) if method in ('residual', 'prior_direct') else x
        result = np.exp(envelope['model'].predict(xx))
        return result * prior if method == 'residual' else result


def prepare(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite prepared GEMM models')
    trusted = read(RUN / 'artifact_hashes_final.json')
    source_files = ['aggregate_metrics.csv', 'accuracy_first_selection_validation_only.csv',
                    'frozen_protocol.json', 'source_selection_before_target_access.json',
                    'validation_predictions.csv']
    for name in source_files:
        checked(RUN / name, trusted[name])
    development = read(RUN / 'frozen_protocol.json')
    for name, digest in development['sources'].items():
        checked(ROOT / name, digest)
    derived = read(DATASET / 'derived_hashes.json')
    index_names = ('main_complete_locked_index.csv', 'accepted_locked_index.csv', 'summary.json')
    for name in index_names:
        checked(DATASET / name, derived[name])
    shapes, _, _ = locked_metadata(DATASET / index_names[0], DATASET / index_names[1])
    selected = choose_validation(rows(RUN / 'aggregate_metrics.csv'))
    reference = rows(RUN / 'accuracy_first_selection_validation_only.csv')
    keys = ('rq', 'learner', 'dtype', 'mode', 'method', 'n')
    if [[str(r[k]) for k in keys] for r in selected] != [[str(r[k]) for k in keys] for r in reference]:
        raise ValueError('Development selection differs from preserved accuracy selection')
    # RQ1 chosen weights are preserved at the two endpoints; smaller choices need a separate replay.
    if any(r['rq'] == 'RQ1' and r['n'] not in (2652, 3345) for r in selected):
        raise ValueError('Selected RQ1 weights are not a preserved endpoint')
    output.mkdir(parents=True)
    (output / 'models').mkdir()
    gemm.save_json(output / 'selection_before_replay.json', dict(
        created_utc=gemm.stamp(), selected=selected, sealed_times_read=False,
        rule='Preserved full-grid validation MAPE minimum, then p90, n, learner, method.'))
    wanted = {(r['rq'], r['learner'], r['dtype'], r['mode'], r['method'], r['n']) for r in selected}
    wanted |= {('RQ1', r['learner'], r['dtype'], r['mode'],
                'normalized' if r['method'] == 'direct' else 'direct', r['n'])
               for r in selected if r['rq'] == 'RQ1'}
    validation = defaultdict(dict)
    with (RUN / 'validation_predictions.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['rq'], row['learner'], row['dtype'], row['mode'], row['method'], int(row['n']))
            if key in wanted:
                vkey = key + (int(row['seed']),)
                if row['shape'] in validation[vkey]:
                    raise ValueError('Duplicate stored validation prediction')
                validation[vkey][row['shape']] = float(row['predicted_ms'])
    source_targets = {(r['learner'], r['dtype']): r['target']
                      for r in read(RUN / 'source_selection_before_target_access.json')['selection']}
    inputs = {str((RUN / n).relative_to(ROOT)): trusted[n] for n in source_files}
    inputs.update({str((DATASET / n).relative_to(ROOT)): derived[n] for n in index_names})
    inputs.update(development['sources'])
    access, entries, replay = [], [], []
    for chosen in selected:
        for seed in gemm.SEEDS:
            source = None
            if chosen['rq'] == 'RQ4':
                source_target = source_targets[(chosen['learner'], chosen['dtype'])]
                name = f"models/rq1_{chosen['learner']}_{chosen['dtype']}_allocate_{source_target}_2652_{seed}.joblib"
                checked(RUN / name, trusted[name])
                source = joblib.load(RUN / name)
                inputs[str((RUN / name).relative_to(ROOT))] = trusted[name]
            variants = [('selected', chosen['method']), ('baseline',
                        ('normalized' if chosen['method'] == 'direct' else 'direct')
                        if chosen['rq'] == 'RQ1' else 'zero')]
            for variant, method in variants:
                envelope = dict(rq=chosen['rq'], learner=chosen['learner'], dtype=chosen['dtype'],
                                mode=chosen['mode'], method=method, n=chosen['n'], seed=seed,
                                features=gemm.FEATURES, source=source)
                if chosen['rq'] == 'RQ1':
                    name = f"models/rq1_{chosen['learner']}_{chosen['dtype']}_{chosen['mode']}_{method}_{chosen['n']}_{seed}.joblib"
                    checked(RUN / name, trusted[name])
                    saved = joblib.load(RUN / name)
                    expected = dict(learner=chosen['learner'], dtype=chosen['dtype'], mode=chosen['mode'],
                                    target=method, n=chosen['n'], seed=seed, features=gemm.FEATURES)
                    if any(saved.get(k) != v for k, v in expected.items()):
                        raise ValueError('Saved RQ1 envelope mismatch')
                    envelope['model'] = saved['model']
                    inputs[str((RUN / name).relative_to(ROOT))] = trusted[name]
                    diagnostics = dict(fit_seconds=0., saved_weight_reused=True)
                elif method == 'zero':
                    envelope['model'] = None
                    envelope['n'] = 0
                    diagnostics = dict(fit_seconds=0.)
                else:
                    train = list(map(tuple, development['c_orders'][str(seed)][:chosen['n']]))
                    labels = gemm.load_labels('legacy_train', train, [chosen['mode']], access)
                    y = gemm.target(labels, train, chosen['dtype'], chosen['mode'])
                    source_envelope = dict(rq='RQ1', dtype=chosen['dtype'], method=source['target'], model=source['model'])
                    prior = predict(source_envelope, train)
                    if not (np.isfinite(prior) & (prior > 0)).all():
                        raise ValueError('Invalid fixed GEMM source prediction')
                    if method == 'constant_ratio':
                        envelope['model'], seconds = gemm.fit_model(method, np.empty((len(y), 0)), y,
                                                                    np.ones(len(y)), seed, prior)
                        diagnostics = dict(fit_seconds=seconds)
                    else:
                        x, _ = gemm.design(train, chosen['dtype'])
                        xx = np.column_stack((x, np.log(prior))) if method != 'direct' else x
                        response = np.log(y) - (np.log(prior) if method == 'residual' else 0)
                        envelope['model'], diagnostics = learners.fit(chosen['learner'], xx, response, seed)
                if not (chosen['rq'] == 'RQ4' and variant == 'baseline'):
                    vg = list(map(tuple, development['validation_shapes'] if chosen['rq'] == 'RQ1'
                                  else development['rq4_validation_shapes']))
                    key = (chosen['rq'], chosen['learner'], chosen['dtype'], chosen['mode'], method, chosen['n'], seed)
                    expected_pred = np.asarray([validation[key][gemm.shape_id(s)] for s in vg])
                    if not np.array_equal(predict(envelope, vg), expected_pred):
                        raise ValueError('Development model replay changed validation predictions')
                name = f"models/{chosen['rq']}_{chosen['dtype']}_{chosen['mode']}_{variant}_{seed}.joblib"
                joblib.dump(envelope, output / name)
                entries.append(dict(path=name, sha256=sha(output / name), variant=variant,
                                    **{k: envelope[k] for k in ('rq', 'learner', 'dtype', 'mode', 'method', 'n', 'seed')}))
                replay.append(dict(path=name, **diagnostics))
    gemm.save_json(output / 'development_replay.json', dict(models=replay, label_access=access,
                                                           sealed_times_read=False))
    code_paths = [Path(__file__).resolve(), Path(components.__file__).resolve(),
                  Path(learners.__file__).resolve(), Path(gemm.__file__).resolve()]
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in code_paths})
    frozen = dict(version='gemm_final_v1', created_utc=gemm.stamp(), inputs=inputs,
                  selected=selected, models=entries, seeds=gemm.SEEDS, shapes=shapes,
                  expected_shapes=1000, condition_index=str((DATASET / index_names[0]).relative_to(ROOT)),
                  measurement_index=str((DATASET / index_names[1]).relative_to(ROOT)),
                  raw_database=str((RAW / 'results.sqlite3').relative_to(ROOT)),
                  raw_database_sha256=read(DATASET / 'summary.json')['input_sha256']['results.sqlite3'],
                  baselines='RQ1 same learner/n with opposite target; RQ4 fixed allocate source without adaptation.',
                  paired_bootstrap='1000 shape resamples per seed; seeds are not independent shape samples.',
                  sealed_times_read=False)
    gemm.save_json(output / 'frozen_selection.json', frozen)
    (output / 'frozen_selection.sha256').write_text(sha(output / 'frozen_selection.json') + '\n')
    print('PASS: GEMM development selection and 60 model envelopes frozen; sealed times closed')


def verify_prepared(prepared):
    prepared = Path(prepared)
    checked(prepared / 'frozen_selection.json', (prepared / 'frozen_selection.sha256').read_text().strip())
    protocol = read(prepared / 'frozen_selection.json')
    if protocol.get('version') != 'gemm_final_v1' or protocol.get('sealed_times_read') is not False:
        raise ValueError('Invalid pre-test GEMM freeze')
    for name, digest in protocol['inputs'].items():
        checked(ROOT / name, digest)
    for entry in protocol['models']:
        path = (prepared / entry['path']).resolve()
        if prepared.resolve() not in path.parents:
            raise ValueError('Model path escapes prepared directory')
        checked(path, entry['sha256'])
    return protocol


def load_sealed(protocol, allow_sealed_test=False):
    if not allow_sealed_test:
        raise ValueError('Sealed GEMM evaluation requires --allow-sealed-test')
    shapes, grouped, indexed = locked_metadata(ROOT / protocol['condition_index'],
                                               ROOT / protocol['measurement_index'], protocol['expected_shapes'])
    if shapes != list(map(tuple, protocol['shapes'])):
        raise ValueError('Frozen sealed shape allowlist changed')
    database = ROOT / protocol['raw_database']
    checked(database, protocol['raw_database_sha256'])
    measured = {}
    query = """SELECT task_id, json_extract(record,'$.task'),
               json_extract(record,'$.timing.median_ms') FROM attempts WHERE seq=? AND task_id=?"""
    with sqlite3.connect(database.as_uri() + '?mode=ro&immutable=1', uri=True) as connection:
        for tid, index in indexed.items():
            result = connection.execute(query, (int(index['seq']), tid)).fetchall()
            if len(result) != 1:
                raise ValueError('Missing or duplicate sealed raw measurement')
            raw_id, task_json, value = result[0]
            task = json.loads(task_json)
            expected = dict(phase='main', split='fresh_holdout', id=tid, round=int(index['round']),
                            shape=json.loads(index['shape']), dtype=index['dtype'], mode=index['mode'])
            if raw_id != tid or any(task.get(k) != v for k, v in expected.items()):
                raise ValueError('Sealed raw task identity mismatch')
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid sealed raw time')
            measured[tid] = float(value)
    return {key: statistics.median(measured[tid] for tid in tids) for key, tids in grouped.items()}


def evaluate(prepared, output, allow_sealed_test=False):
    if not allow_sealed_test:
        raise ValueError('Sealed GEMM evaluation requires --allow-sealed-test')
    prepared, output = Path(prepared), Path(output)
    protocol = verify_prepared(prepared)
    if output.exists():
        raise ValueError('Refusing to overwrite final GEMM results')
    marker = prepared / 'sealed_access_started.json'
    with marker.open('x') as stream:
        json.dump(dict(started_utc=gemm.stamp(), output=str(output.resolve()),
                       protocol_sha256=sha(prepared / 'frozen_selection.json')), stream, indent=2)
    output.mkdir(parents=True)
    labels = load_sealed(protocol, allow_sealed_test=True)
    shapes = list(map(tuple, protocol['shapes']))
    scores, predictions, bootstrap, subgroups, cached = [], [], [], [], {}
    for entry in protocol['models']:
        envelope = joblib.load(prepared / entry['path'])
        identity = {k: entry[k] for k in ('rq', 'learner', 'dtype', 'mode', 'method', 'n', 'seed', 'variant')}
        predicted = predict(envelope, shapes)
        actual = gemm.target(labels, shapes, entry['dtype'], entry['mode'])
        scores.append(dict(**identity, **gemm.metrics(actual, predicted)))
        for name, mask in gemm.subgroups(shapes).items():
            if mask.any():
                subgroups.append(dict(**identity, subgroup=name, **gemm.metrics(actual[mask], predicted[mask])))
        for shape, y, p in zip(shapes, actual, predicted):
            predictions.append(dict(**identity, shape=gemm.shape_id(shape), actual_ms=y, predicted_ms=p))
        key = (entry['rq'], entry['dtype'], entry['mode'], entry['seed'])
        cached.setdefault(key, {})[entry['variant']] = predicted
    for (rq, dtype, mode, seed), pair in cached.items():
        actual = gemm.target(labels, shapes, dtype, mode)
        bootstrap.append(dict(rq=rq, dtype=dtype, mode=mode, seed=seed,
                              **gemm.bootstrap_delta(actual, pair['selected'], pair['baseline'], seed)))
    gemm.save_csv(output / 'metrics.csv', scores)
    gemm.save_csv(output / 'predictions.csv', predictions)
    gemm.save_csv(output / 'paired_bootstrap.csv', bootstrap)
    gemm.save_csv(output / 'subgroup_metrics.csv', subgroups)
    shutil.copy2(prepared / 'frozen_selection.json', output / 'frozen_selection.json')
    gemm.save_json(output / 'completed.json', dict(completed_utc=gemm.stamp(), shapes=len(shapes),
                                                  conditions=len(labels), models=len(scores),
                                                  raw_measurements=len(labels) * 3,
                                                  post_test_selection=False))
    print('PASS: fixed GEMM final evaluation complete; no post-test model selection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('evaluate')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-sealed-test', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    else:
        evaluate(args.prepared, args.output, args.allow_sealed_test)


if __name__ == '__main__':
    main()
