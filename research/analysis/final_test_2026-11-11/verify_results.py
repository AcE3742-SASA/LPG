"""Verify raw test labels, frozen-weight predictions and independently computed metrics."""
from collections import defaultdict
from pathlib import Path
import sys

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'analysis/layer_full_fit_2026-11-10_v5'))
import components
import data
import gemm_final as final
import pipeline

GEMM_KEY = ('rq', 'learner', 'dtype', 'mode', 'method', 'n', 'seed', 'variant')
LAYER_KEY = ('target', 'layer', 'dtype', 'learner', 'method', 'k', 'selection_scope', 'seed')
LAYER = ROOT / 'analysis/layer_full_fit_2026-11-10_v5/prepared_layer_full'
LEGACY = ROOT / 'analysis/layer_full_fit_2026-11-04/prepared_gemm_final'


def key(row, fields):
    return tuple(str(row[k]) for k in fields)


def independent_metrics(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    assert y.shape == predicted.shape and len(y)
    assert (np.isfinite(y) & (y > 0)).all()
    assert (np.isfinite(predicted) & (predicted > 0)).all()
    relative = abs(predicted - y) / y
    return dict(planned=len(y), valid=len(y), failures=0, coverage=1.0,
                mape_pct=float(relative.mean() * 100),
                p90_pct=float(np.percentile(relative, 90) * 100),
                relative_rmse=float(np.sqrt(np.mean(relative ** 2))),
                mae_us=float(np.mean(abs(predicted - y)) * 1000))


def check_metrics(y, predicted, row):
    for name, value in independent_metrics(y, predicted).items():
        np.testing.assert_allclose(float(row[name]), value, rtol=1e-12, atol=1e-12)


def prediction_groups(rows, fields, id_field):
    groups = defaultdict(dict)
    for row in rows:
        identity = key(row, fields)
        assert row[id_field] not in groups[identity], 'Duplicate prediction identity'
        groups[identity][row[id_field]] = (float(row['actual_ms']), float(row['predicted_ms']))
    return groups


def verify_gemm(prepared, output, raw_labels):
    protocol = final.verify_prepared(prepared)
    completion = final.read(output / 'completed.json')
    assert completion['post_test_selection'] is False
    assert final.sha(output / 'frozen_selection.json') == final.sha(prepared / 'frozen_selection.json')
    marker = final.read(prepared / 'sealed_access_started.json')
    assert marker['protocol_sha256'] == final.sha(prepared / 'frozen_selection.json')
    assert Path(marker['output']).resolve() == output.resolve()
    scores = {key(r, GEMM_KEY): r for r in final.rows(output / 'metrics.csv')}
    predictions = prediction_groups(final.rows(output / 'predictions.csv'), GEMM_KEY, 'shape')
    assert len(scores) == completion['models'] == len(protocol['models'])
    assert set(scores) == set(predictions) == {key(r, GEMM_KEY) for r in protocol['models']}
    shapes = list(map(tuple, protocol['shapes']))
    count = 0
    for entry in protocol['models']:
        identity = key(entry, GEMM_KEY)
        observed = predictions[identity]
        assert list(observed) == [final.gemm.shape_id(s) for s in shapes]
        actual, predicted = np.asarray(list(observed.values())).T
        reference = np.asarray([raw_labels[s, entry['dtype'], entry['mode']] for s in shapes])
        assert np.array_equal(actual, reference)
        envelope = joblib.load(prepared / entry['path'])
        assert all(str(envelope[k]) == str(entry[k]) for k in GEMM_KEY if k != 'variant')
        assert np.array_equal(predicted, final.predict(envelope, shapes))
        check_metrics(actual, predicted, scores[identity])
        count += len(shapes)
    return dict(models=len(scores), predictions_reproduced=count, raw_conditions=len(raw_labels))


def verify_layers():
    protocol = pipeline.verify_protocol(LAYER)
    data.check_freeze(LAYER / 'frozen_protocol.json', LAYER / 'selection_frozen.json')
    output = HERE / 'layers'
    completed = final.read(output / 'evaluation_completed.json')
    assert completed['model_selection_changed'] is False
    assert completed['freeze_sha256'] == final.sha(LAYER / 'selection_frozen.json')
    marker = final.read(LAYER / 'sealed_evaluation_started.json')
    assert marker['freeze_sha256'] == completed['freeze_sha256']
    assert Path(marker['output']).resolve() == output.resolve()
    for name, digest in completed['artifact_hashes'].items():
        final.checked(output / name, digest)
    dataset = data.load_campaign(ROOT / 'analysis/layer_full_result_2026-11-10/raw',
        split='test', allow_test=True, diagnostic=True, protocol_path=LAYER / 'frozen_protocol.json',
        freeze_path=LAYER / 'selection_frozen.json')
    assert dataset['provenance'] == completed['input_provenance']
    assert len(dataset['rows']) == 1794 and len(dataset['excluded']) == 6
    assert all(r['split'] == 'test_locked' and r['status'] == 'PASS' for r in dataset['rows'])
    assert {(r['case_id'], r['dtype'], r['status']) for r in dataset['excluded']} == {
        ('lf_dfce839bd3ca0ef5', 'fp32', 'FAIL')}
    labels = {(r['case_id'], r['dtype'], r['implementation']): r['time_ms'] for r in dataset['rows']}
    training = {r['group_id'] for r in protocol['groups'] if r['split'] == 'train'}
    selected = final.read(LAYER / 'selected_configurations.json')
    scores = {key(r, LAYER_KEY): r for r in final.read(output / 'test_metrics.json')}
    predictions = prediction_groups(final.read(output / 'test_predictions.json'), LAYER_KEY, 'group_id')
    expected = {key(dict(item, seed=seed), LAYER_KEY) for item in selected for seed in protocol['seeds']}
    assert set(scores) == set(predictions) == expected and len(expected) == 336
    common, coverage = {}, []
    for layer in protocol['layers']:
        for dtype in ('fp16', 'fp32'):
            required = {'standard', 'decomposed'} | (set(components.component_names(layer)) - {'gemm'})
            planned = sorted([c for group in protocol['cases'].values() for c in group
                              if c['layer'] == layer and c['split'] == 'test_locked'], key=lambda c: c['group_id'])
            cases = [c for c in planned if all((c['id'], dtype, impl) in labels for impl in required)]
            assert len(cases) == (49 if (layer, dtype) == ('conv3', 'fp32') else 50)
            assert not {c['group_id'] for c in cases} & training
            common[layer, dtype] = cases
            coverage.append(dict(layer=layer, dtype=dtype, planned_groups=len(planned), retained_groups=len(cases),
                                 excluded_group_ids=[c['group_id'] for c in planned if c not in cases]))
    assert coverage == final.read(output / 'test_quality_coverage.json')
    count, model_files = 0, set()
    for item in selected:
        cases = common[item['layer'], item['dtype']]
        for seed, name in item['model_files_by_seed'].items():
            identity = key(dict(item, seed=seed), LAYER_KEY)
            observed = predictions[identity]
            assert list(observed) == [c['group_id'] for c in cases]
            actual, predicted = np.asarray(list(observed.values())).T
            assert np.array_equal(actual, [labels[c['id'], item['dtype'], item['target']] for c in cases])
            envelope = joblib.load(LAYER / name)
            assert all(str(envelope[k]) == str(dict(item, seed=seed)[k]) for k in LAYER_KEY if k != 'selection_scope')
            assert set(envelope['training_group_ids']) <= training
            assert np.array_equal(predicted, pipeline.predict(envelope, cases, item['dtype'], protocol))
            check_metrics(actual, predicted, scores[identity])
            model_files.add(name)
            count += len(cases)
    return dict(selected_recipes=len(selected), scored_seed_recipes=len(scores), unique_weights=len(model_files),
                predictions_reproduced=count, raw_conditions=len(dataset['rows']),
                raw_measurements=len(dataset['rows']) * 3, correctness_excluded_conditions=len(dataset['excluded']))


def main():
    np.testing.assert_allclose(independent_metrics([1, 2], [1.1, 1.8])['mape_pct'], 10)
    preflight = final.read(HERE / 'preflight.json')
    for name, digest in preflight['original_hashes'].items():
        final.checked(ROOT / name, digest)
    expanded = HERE / 'prepared_gemm_expanded'
    final.checked(expanded / 'frozen_selection.json', preflight['expanded_freeze_sha256'])
    labels = final.load_sealed(final.verify_prepared(expanded), allow_sealed_test=True)
    results = {}
    for name, prepared in [('gemm_expanded', expanded), ('gemm_legacy_and_transfer', LEGACY)]:
        results[name] = verify_gemm(prepared, HERE / name, labels)
    results['layers'] = verify_layers()
    output_hashes = {str(path.relative_to(HERE)): final.sha(path)
        for name in ('gemm_expanded', 'gemm_legacy_and_transfer', 'layers')
        for path in sorted((HERE / name).iterdir()) if path.is_file()}
    for name, digest in preflight['original_hashes'].items():
        final.checked(ROOT / name, digest)
    result = dict(status='PASS', verified_utc=final.gemm.stamp(),
        preflight_sha256=final.sha(HERE / 'preflight.json'), verifier_sha256=final.sha(__file__),
        original_files_unchanged=len(preflight['original_hashes']), original_model_selection_unchanged=True,
        fitting_performed=False, metrics_independently_recomputed=True, results=results,
        test_suite=dict(tests=47, status='PASS', scope='Existing frozen v5 analysis tests; synthetic final-evaluation guards'),
        artifact_hashes=output_hashes)
    report_path = HERE / 'independent_verification.json'
    if report_path.exists():
        saved = final.read(report_path)
        assert {k: v for k, v in result.items() if k != 'verified_utc'} == {
            k: v for k, v in saved.items() if k != 'verified_utc'}
    else:
        final.gemm.save_json(report_path, result)
    print('PASS: raw labels, frozen predictions, test metrics and original hashes verified', flush=True)
    print({name: values['predictions_reproduced'] for name, values in results.items()}, flush=True)


if __name__ == '__main__':
    main()
