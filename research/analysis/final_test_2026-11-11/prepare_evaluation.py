"""Verify frozen inputs and adapt saved expanded GEMM weights for the existing evaluator."""
from collections import defaultdict
import csv
from pathlib import Path
import sys
from unittest.mock import patch

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = ROOT / 'analysis/layer_full_fit_2026-11-10_v5'
sys.path.insert(0, str(ANALYSIS))
import data
import gemm_final as final
import pipeline

SOURCE = ROOT / 'analysis/gemm_expansion_result_2026-11-06/fit_v1'
LEGACY = ROOT / 'analysis/layer_full_fit_2026-11-04/prepared_gemm_final'
LAYER = ANALYSIS / 'prepared_layer_full'
PREPARED = HERE / 'prepared_gemm_expanded'
IDENTITY = ('learner', 'target', 'dtype', 'mode', 'n', 'seed')


def identity(row):
    return tuple(int(row[k]) if k in ('n', 'seed') else row[k] for k in IDENTITY)


def adapt(saved, expected, protocol_hash):
    if (identity(saved) != identity(expected) or saved['features'] != final.gemm.FEATURES
            or saved['protocol_sha256'] != protocol_hash or saved['locked_test_opened'] is not False):
        raise ValueError('Expanded model envelope mismatch')
    return dict(rq='RQ1', learner=saved['learner'], dtype=saved['dtype'], mode=saved['mode'],
                method=saved['target'], n=saved['n'], seed=saved['seed'],
                features=saved['features'], model=saved['model'], source=None)


def selfcheck():
    saved = dict(learner='rf', target='normalized', dtype='fp32', mode='allocate', n=11498,
                 seed=20261028, features=final.gemm.FEATURES, protocol_sha256='test',
                 locked_test_opened=False, model=object())
    assert adapt(saved, saved, 'test')['model'] is saved['model']
    for changed in (dict(saved, n=1), dict(saved, dtype='fp16'),
                    dict(saved, features=[]), dict(saved, locked_test_opened=True)):
        try:
            adapt(changed, saved, 'test')
        except ValueError:
            pass
        else:
            raise AssertionError('Changed envelope accepted')


def main():
    selfcheck()
    if PREPARED.exists() or (HERE / 'preflight.json').exists():
        raise ValueError('Refusing to overwrite evaluation preparation')
    for path in (LEGACY / 'sealed_access_started.json', LAYER / 'sealed_evaluation_started.json'):
        if path.exists():
            raise ValueError('Existing test-access record: ' + str(path))
    legacy = final.verify_prepared(LEGACY)
    layer_protocol = pipeline.verify_protocol(LAYER)
    layer_freeze = data.check_freeze(LAYER / 'frozen_protocol.json', LAYER / 'selection_frozen.json')
    index_path = SOURCE.parent / 'artifact_hashes_final.json'
    trusted = {Path(p).resolve(): h for p, h in final.read(index_path)['sha256'].items()}
    original = {}

    def pin(path, digest=None):
        path = Path(path).resolve()
        value = final.sha(path)
        if digest is not None and value != digest:
            raise ValueError('Original artifact checksum mismatch: ' + str(path))
        original[str(path.relative_to(ROOT))] = value

    names = ['frozen_protocol.json', 'selection_frozen.json', 'accuracy_first_all_sizes.csv',
             'accuracy_first_model_refs.csv', 'selected_full_endpoints.csv',
             'validation_predictions.csv', 'independent_verification.json']
    for name in names:
        pin(SOURCE / name, trusted[(SOURCE / name).resolve()])
    development = final.read(SOURCE / 'frozen_protocol.json')
    freeze = final.read(SOURCE / 'selection_frozen.json')
    assert freeze['protocol_sha256'] == final.sha(SOURCE / 'frozen_protocol.json')
    assert freeze['accuracy_first_sha256'] == final.sha(SOURCE / 'accuracy_first_all_sizes.csv')
    assert freeze['selected_sha256'] == final.sha(SOURCE / 'selected_full_endpoints.csv')
    assert freeze['locked_test_opened'] is False
    assert development['runtime']['sklearn'] == final.gemm.sklearn.__version__
    for name, digest in {**legacy['inputs'], **development['inputs'],
                         **layer_protocol['inputs']}.items():
        pin(ROOT / name, digest)
    for entry in legacy['models']:
        pin(LEGACY / entry['path'], entry['sha256'])
    for name, digest in layer_freeze['artifact_hashes'].items():
        pin(LAYER / name, digest)
    for entry in layer_protocol['source_models'].values():
        pin(ROOT / entry['path'], entry['sha256'])
    for path in (LEGACY / 'frozen_selection.json', LEGACY / 'frozen_selection.sha256',
                 LAYER / 'selection_frozen.json', LAYER / 'frozen_protocol.json', index_path,
                 ROOT / 'analysis/layer_full_result_2026-11-10/raw/campaign.json',
                 Path(__file__), Path(final.__file__)):
        pin(path)
    shapes, _, _ = final.locked_metadata(ROOT / legacy['condition_index'], ROOT / legacy['measurement_index'])
    assert shapes == list(map(tuple, legacy['shapes']))
    dev_shapes = {tuple(s) for s in development['old_train_orders'][str(development['seeds'][0])]
                  + development['new_train_orders'][str(development['seeds'][0])]
                  + development['legacy_validation_shapes'] + development['new_validation_shapes']}
    assert len(dev_shapes) == 13109 and not set(shapes) & dev_shapes
    near = [(s, axis, delta) for s in shapes for axis in range(3) for delta in range(-8, 9)
            if delta and tuple(v + (delta if i == axis else 0) for i, v in enumerate(s)) in dev_shapes]
    chosen = final.rows(SOURCE / 'accuracy_first_all_sizes.csv')
    refs = final.rows(SOURCE / 'accuracy_first_model_refs.csv')
    assert len(chosen) == 6 and len(refs) == 18
    assert {(r['dtype'], r['mode']) for r in chosen} == {
        (d, m) for d in final.gemm.DTYPES for m in final.gemm.MODES}
    assert all(int(r['n']) == 11498 and int(r['failures']) == 0 for r in chosen)
    envelopes = []
    for choice in chosen:
        for seed in development['seeds']:
            match = [r for r in refs if identity(r) == identity(dict(choice, seed=seed))]
            assert len(match) == 1
            for variant, target in [('selected', choice['target']),
                                    ('baseline', 'direct' if choice['target'] == 'normalized' else 'normalized')]:
                spec = dict(choice, target=target, seed=seed)
                source = SOURCE / 'models' / f"{choice['learner']}_{choice['dtype']}_{choice['mode']}_{target}_{seed}.joblib"
                pin(source, trusted[source.resolve()])
                if variant == 'selected':
                    assert source.resolve() == (ROOT / match[0]['model_ref']).resolve()
                    assert final.sha(source) == match[0]['sha256']
                envelope = adapt(joblib.load(source), spec, freeze['protocol_sha256'])
                envelopes.append((spec, variant, envelope, source))
    wanted = {identity(spec) for spec, _, _, _ in envelopes}
    expected = defaultdict(dict)
    with (SOURCE / 'validation_predictions.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = identity(row)
            if key in wanted:
                assert row['shape'] not in expected[key]
                expected[key][row['shape']] = float(row['predicted_ms'])
    replayed = 0
    for spec, _, envelope, _ in envelopes:
        reference = expected[identity(spec)]
        assert len(reference) == 1611
        validation = [tuple(map(int, s.split('x'))) for s in reference]
        assert np.array_equal(final.predict(envelope, validation), np.asarray(list(reference.values())))
        replayed += len(reference)
    PREPARED.mkdir()
    (PREPARED / 'models').mkdir()
    entries = []
    for spec, variant, envelope, source in envelopes:
        name = f"models/RQ1_{spec['dtype']}_{spec['mode']}_{variant}_{spec['seed']}.joblib"
        joblib.dump(envelope, PREPARED / name)
        entries.append(dict(path=name, sha256=final.sha(PREPARED / name), variant=variant,
                            original_model=str(source.relative_to(ROOT)),
                            **{k: envelope[k] for k in ('rq', 'learner', 'dtype', 'mode', 'method', 'n', 'seed')}))
    protocol = dict(legacy, created_utc=final.gemm.stamp(), inputs=original, models=entries,
                    selected=[dict(r, rq='RQ1', method=r['target'], n=int(r['n'])) for r in chosen],
                    source='Expanded 11498-shape development selection, saved weights only.',
                    baselines='Same learner and training n, opposite direct/normalized target.',
                    development_predictions_reproduced=replayed,
                    near_development_shapes=[dict(test_shape=s, axis=axis, delta=delta)
                                             for s, axis, delta in near],
                    near_shape_reporting='Primary fixed 1000 shapes; additionally report all non-near shapes. Defined before timing access.')
    final.gemm.save_json(PREPARED / 'frozen_selection.json', protocol)
    (PREPARED / 'frozen_selection.sha256').write_text(final.sha(PREPARED / 'frozen_selection.json') + '\n')
    final.verify_prepared(PREPARED)
    final.gemm.save_json(HERE / 'preflight.json', dict(status='PASS', created_utc=final.gemm.stamp(),
        user_authorization='Evaluate frozen GEMM and layer models on final test; user request 2026-11-11 Asia/Seoul.',
        scope=['expanded GEMM RQ1', 'previous frozen GEMM RQ1 comparison and RQ4 adaptation',
               'layer v5 all preselected method and overall recipes'],
        original_hashes=original, development_predictions_reproduced=replayed,
        expanded_models=len(entries), legacy_models=len(legacy['models']),
        layer_frozen_artifacts=len(layer_freeze['artifact_hashes']),
        reserved_gemm_shapes=len(shapes), exact_development_overlap=0,
        one_axis_near_overlap_pairs=len(near), near_development_shapes=protocol['near_development_shapes'],
        near_shape_reporting=protocol['near_shape_reporting'],
        sealed_labels_opened=False, fitting_performed=False,
        expanded_freeze_sha256=final.sha(PREPARED / 'frozen_selection.json')))
    print(f'PASS: {replayed} expanded validation predictions replayed; all frozen inputs verified; test closed', flush=True)


if __name__ == '__main__':
    # Preparation must never access the reserved timing database or fit any model.
    with patch.object(final.sqlite3, 'connect', side_effect=AssertionError('Preparation opened test DB')), \
         patch.object(final.learners, 'fit', side_effect=AssertionError('Preparation fitted a model')):
        main()
