"""Recheck frozen development results and replay saved models without fitting."""
from collections import defaultdict
import csv
from itertools import product
import json
from pathlib import Path
import sys

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = ROOT / 'analysis/layer_full_fit_2026-11-10_v5'
RUN = ANALYSIS / 'prepared_layer_full'
CAMPAIGN = ROOT / 'analysis/layer_full_result_2026-11-10/raw'
sys.path.insert(0, str(ANALYSIS))
import components
import data
import pipeline

KEY = ('target', 'layer', 'dtype', 'learner', 'method', 'k', 'seed')


def key(row):
    return tuple(int(row[k]) if k in ('k', 'seed') else row[k] for k in KEY)


def metric_values(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    assert (np.isfinite(actual) & (actual > 0)).all()
    valid = np.isfinite(predicted) & (predicted > 0)
    failures = int((~valid).sum())
    if failures:
        return dict(mape_pct=float('inf'), p90_pct=float('inf'), failures=failures)
    error = np.abs(predicted - actual) / actual * 100
    return dict(mape_pct=float(error.mean()), p90_pct=float(np.percentile(error, 90)), failures=0)


def best(candidates):
    candidates = [r for r in candidates if not r['failures'] and np.isfinite(r['mean_mape_pct'])]
    return min(candidates, key=lambda r: (r['mean_mape_pct'], r['mean_p90_pct'],
               r['actual_train_labels'], r['learner'], r['method'], r['k']))


def selfcheck():
    assert metric_values([1, 2], [1.1, 1.8])['failures'] == 0
    np.testing.assert_allclose(metric_values([1, 2], [1.1, 1.8])['mape_pct'], 10)
    assert metric_values([1, 2], [1, float('nan')])['failures'] == 1
    base = dict(mean_mape_pct=10, mean_p90_pct=20, actual_train_labels=5,
                learner='ridge', method='direct', k=5, failures=0)
    smaller = dict(base, actual_train_labels=4)
    invalid = dict(base, mean_mape_pct=0, failures=1)
    assert best([base, smaller, invalid]) == smaller


def main():
    selfcheck()
    protocol = pipeline.verify_protocol(RUN)
    freeze = data.check_freeze(RUN / 'frozen_protocol.json', RUN / 'selection_frozen.json')
    assert protocol['sealed_test_status'] == freeze['sealed_test_status'] == 'NOT_OPENED'
    assert not (RUN / 'sealed_evaluation_started.json').exists()
    assert freeze['selection_rule'] == protocol['selection']
    dataset = data.load_campaign(CAMPAIGN, split='dev', diagnostic=True,
                                 protocol_path=RUN / 'frozen_protocol.json')
    audit = data.read(RUN / 'input_audit.json')
    assert audit['provenance'] == dataset['provenance']
    assert not dataset['provenance']['test_opened']
    assert all(not p.startswith('test_') for p in dataset['provenance']['phases'])
    assert all(r['split'] in ('train', 'validation') for r in dataset['rows'] + dataset['excluded'])
    assert all(r['status'] == 'PASS' for r in dataset['rows'])
    groups = {g['group_id']: g for g in protocol['groups']}
    by_split = {split: {g for g, row in groups.items() if row['split'] == split}
                for split in ('train', 'validation', 'test_locked')}
    assert not (by_split['train'] & by_split['validation'] or by_split['train'] & by_split['test_locked']
                or by_split['validation'] & by_split['test_locked'])
    labels = {(r['case_id'], r['dtype'], r['implementation']): r['time_ms'] for r in dataset['rows']}
    cases_by_group = {(c['layer'], c['group_id']): c for cc in protocol['cases'].values() for c in cc}
    common, common_train_ids, coverage = {}, {}, []
    for layer, dtype in product(protocol['layers'], ('fp16', 'fp32')):
        required = {'standard', 'decomposed'} | (set(components.component_names(layer)) - {'gemm'})
        planned = sorted([c for (ll, _), c in cases_by_group.items()
                          if ll == layer and c['split'] in ('train', 'validation')], key=lambda c: c['group_id'])
        kept = [c for c in planned if all((c['id'], dtype, impl) in labels for impl in required)]
        common[layer, dtype] = kept
        common_train_ids[layer, dtype] = {c['group_id'] for c in kept if c['split'] == 'train'}
        coverage.append(dict(layer=layer, dtype=dtype, planned_groups=len(planned), retained_groups=len(kept),
                             excluded_group_ids=[c['group_id'] for c in planned if c not in kept]))
    assert coverage == data.read(RUN / 'quality_coverage.json')

    metrics = data.read(RUN / 'metrics.json')
    indexed = {key(r): r for r in metrics}
    expected = {tuple(values) for values in product(protocol['targets'], protocol['layers'],
                ('fp16', 'fp32'), protocol['learners'], protocol['methods'], [0], protocol['seeds'])}
    expected = {k[:5] + (budget, k[-1]) for k in expected
                for budget in ([0] if k[4] == 'zero' else protocol['budgets'])}
    assert len(indexed) == len(metrics) and set(indexed) == expected
    predictions = defaultdict(dict)
    with (RUN / 'validation_predictions.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            identity, group = key(row), row['group_id']
            assert identity in indexed and group not in predictions[identity]
            assert group in by_split['validation']
            case = cases_by_group[row['layer'], group]
            actual = float(row['actual_ms'])
            assert actual == labels[case['id'], row['dtype'], row['target']]
            predictions[identity][group] = (actual, float(row['predicted_ms']))
    assert set(predictions) == set(indexed)
    aggregates = defaultdict(list)
    for identity, row in indexed.items():
        target, layer, dtype, learner, method, budget, seed = identity
        kept = common[layer, dtype]
        valid_ids = [c['group_id'] for c in kept if c['split'] == 'validation']
        assert list(predictions[identity]) == valid_ids
        values = np.asarray(list(predictions[identity].values()))
        calculated = metric_values(values[:, 0], values[:, 1])
        for name, value in calculated.items():
            np.testing.assert_allclose(row[name], value, rtol=1e-12, atol=1e-12)
        assert row['planned'] == len(valid_ids) and row['valid'] == len(valid_ids) - calculated['failures']
        family = 'conv' if layer.startswith('conv') else 'core'
        planned_ids = [] if not budget else protocol['subsets']['families'][family][str(seed)]['by_k'][str(budget)]
        train_ids = [i for i in planned_ids if i in common_train_ids[layer, dtype]]
        assert set(train_ids) <= by_split['train'] and not set(train_ids) & set(valid_ids)
        assert row['actual_train_groups'] == len(train_ids)
        assert row['actual_train_labels'] == len(train_ids) * (len(components.component_names(layer)) if method == 'component_sum' else 1)
        aggregates[identity[:-1]].append(row)
    calculated_aggregates = []
    for identity, rows in aggregates.items():
        assert len(rows) == len(protocol['seeds']) and {r['seed'] for r in rows} == set(protocol['seeds'])
        result = dict(zip(KEY[:-1], identity), failures=sum(r['failures'] for r in rows))
        for source, dest in [('mape_pct', 'mean_mape_pct'), ('p90_pct', 'mean_p90_pct'),
                             ('actual_train_groups', 'actual_train_groups'), ('actual_train_labels', 'actual_train_labels')]:
            result[dest] = float(np.mean([r[source] for r in rows]))
        calculated_aggregates.append(result)
    stored_aggregates = data.read(RUN / 'aggregate_metrics.json')
    assert calculated_aggregates == stored_aggregates
    selected = data.read(RUN / 'selected_configurations.json')
    assert len(selected) == len(protocol['targets']) * len(protocol['layers']) * 2 * (len(protocol['methods']) + 1)
    assert {(r['target'], r['layer'], r['dtype'], r['selection_scope']) for r in selected} == set(
        product(protocol['targets'], protocol['layers'], ('fp16', 'fp32'), protocol['methods'] + ['overall']))
    replayed = {}
    for selection in selected:
        candidates = [r for r in calculated_aggregates if all(r[k] == selection[k] for k in ('target', 'layer', 'dtype'))
                      and (selection['selection_scope'] == 'overall' or r['method'] == selection['selection_scope'])]
        winner = best(candidates)
        assert all(selection[k] == value for k, value in winner.items())
        valid = [c for c in common[selection['layer'], selection['dtype']] if c['split'] == 'validation']
        assert selection['validation_group_ids'] == [c['group_id'] for c in valid]
        assert set(selection['model_files_by_seed']) == set(map(str, protocol['seeds']))
        for seed, name in selection['model_files_by_seed'].items():
            assert name in freeze['artifact_hashes'] and data.sha(RUN / name) == freeze['artifact_hashes'][name]
            identity = tuple(selection[k] for k in KEY[:-1]) + (int(seed),)
            training = selection['training_group_ids_by_seed'][seed]
            family = 'conv' if selection['layer'].startswith('conv') else 'core'
            planned_ids = [] if not selection['k'] else protocol['subsets']['families'][family][seed]['by_k'][str(selection['k'])]
            retained_train = common_train_ids[selection['layer'], selection['dtype']]
            assert training == [i for i in planned_ids if i in retained_train]
            if name in replayed:
                continue
            envelope = joblib.load(RUN / name)
            assert key(envelope) == identity and envelope['training_group_ids'] == training
            if selection['method'] == 'component_sum':
                assert envelope['model']['train_group_ids'] == training
            actual = pipeline.predict(envelope, valid, selection['dtype'], protocol)
            expected = np.asarray([predictions[identity][c['group_id']][1] for c in valid])
            np.testing.assert_array_equal(actual, expected)
            replayed[name] = dict(predictions=len(actual), exact=True, sha256=freeze['artifact_hashes'][name])
    assert set(replayed) == {name for name in freeze['artifact_hashes'] if name.startswith('models/')}
    overlap = pipeline.source_overlap()
    assert overlap == protocol['source_overlap']
    data.check_freeze(RUN / 'frozen_protocol.json', RUN / 'selection_frozen.json')
    result = dict(status='PASS', completed_utc=pipeline.gemm.stamp(), frozen_artifacts=len(freeze['artifact_hashes']),
                  metric_rows=len(metrics), aggregate_rows=len(aggregates), selections=len(selected),
                  validation_predictions=sum(map(len, predictions.values())), replayed_models=len(replayed),
                  replayed_predictions=sum(r['predictions'] for r in replayed.values()), models=replayed,
                  common_cohorts=coverage, source_overlap=overlap, sealed_test_status='NOT_OPENED',
                  training_performed=False, frozen_protocol_sha256=data.sha(RUN / 'frozen_protocol.json'),
                  selection_frozen_sha256=data.sha(RUN / 'selection_frozen.json'), script_sha256=data.sha(Path(__file__)))
    (HERE / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('models', 'common_cohorts')}, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--selfcheck']:
        selfcheck()
        print('PASS: independent metric and selection self-check')
    else:
        assert not sys.argv[1:]
        main()
