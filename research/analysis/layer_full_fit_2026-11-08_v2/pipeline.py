#!/usr/bin/env python3
"""Four-layer analysis with the selected expanded GEMM prior shared by all learners."""
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'layer_full_20261108_v2'
BASE = HERE
EXPANSION = ROOT / 'analysis/gemm_expansion_result_2026-11-06'
SOURCE = EXPANSION / 'fit_v1'
sys.path.insert(0, str(ROOT / 'analysis/layer_transfer_fit_2026-10-28'))
import compare_layers as previous
sys.path.insert(0, str(BASE))

SOURCE_FILES = ['frozen_protocol.json', 'selection_frozen.json',
                'accuracy_first_all_sizes.csv', 'accuracy_first_model_refs.csv',
                'execution_completed.json', 'independent_verification.json']

gemm = previous.gemm
LAYERS = ['linear', 'ffn', 'conv1', 'conv3']
TARGETS = ['standard', 'decomposed']
METHODS = ['zero', 'constant_ratio', 'direct', 'prior_direct', 'residual', 'component_sum']
SEEDS = [20261104, 20261105, 20261106]
BUDGETS = [5, 10, 20, 30, 50, 100, 200]
FEATURES = gemm.FEATURES + ['log_total_gemm_flops', 'log1p_gelu_elements',
    'log1p_batch', 'log1p_height', 'log1p_width', 'log1p_cin', 'log1p_cout', 'log1p_kernel']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return gemm.sha(Path(path))


def save(path, value):
    gemm.save_json(Path(path), value)


def logical_gemms(case):
    if case['layer'].startswith('conv'):
        return [(case['batch'] * case['height'] * case['width'], case['cout'],
                 case['cin'] * case['kernel'] ** 2)]
    m, d = case['rows'], case['input_width']
    if case['layer'] == 'linear':
        return [(m, case['output_width'], d)]
    f = case['hidden_width']
    return [(m, f, d), (m, d, f)]


def features(cases, dtype):
    shapes = [logical_gemms(c) for c in cases]
    x, _ = gemm.design([s[0] for s in shapes], dtype)
    total = [sum(2.0 * np.prod(s) for s in ss) for ss in shapes]
    gelu = [c['rows'] * c['hidden_width'] if c['layer'] == 'ffn' else 0 for c in cases]
    geometry = [[c.get(k, 0) for k in ('batch', 'height', 'width', 'cin', 'cout', 'kernel')] for c in cases]
    return np.column_stack((x, np.log(total), np.log1p(gelu), np.log1p(geometry)))


def source_prior(cases, dtype, envelope):
    shapes = [logical_gemms(c) for c in cases]
    prediction = np.zeros(len(cases))
    for index in range(max(map(len, shapes))):
        indices = [i for i, s in enumerate(shapes) if len(s) > index]
        x, lower = gemm.design([shapes[i][index] for i in indices], dtype)
        values, _ = gemm.predict(envelope['learner'] + '_' + envelope['target'],
                                 envelope['model'], x, lower)
        prediction[indices] += values
    if not (np.isfinite(prediction) & (prediction > 0)).all():
        raise ValueError('Invalid GEMM source prediction')
    return prediction


def frozen_sources():
    known = {Path(name).resolve(): digest for name, digest in
             read(EXPANSION / 'artifact_hashes_final.json')['sha256'].items()}
    for name in SOURCE_FILES:
        path = SOURCE / name
        if sha(path) != known.get(path.resolve()):
            raise ValueError('Expanded GEMM selection checksum mismatch: ' + name)
    frozen = read(SOURCE / 'selection_frozen.json')
    protocol = read(SOURCE / 'frozen_protocol.json')
    verification = read(SOURCE / 'independent_verification.json')
    if (frozen['status'] != 'DEVELOPMENT_SELECTION_FROZEN'
            or frozen['locked_test_opened'] is not False
            or frozen['protocol_sha256'] != sha(SOURCE / 'frozen_protocol.json')
            or frozen['accuracy_first_sha256'] != sha(SOURCE / 'accuracy_first_all_sizes.csv')
            or verification['status'] != 'PASS' or verification['locked_test_opened'] is not False
            or protocol['features'] != gemm.FEATURES
            or protocol['seeds'] != previous.layer.SEEDS
            or protocol['runtime']['sklearn'] != gemm.sklearn.__version__):
        raise ValueError('Invalid expanded GEMM selection or environment')
    with (SOURCE / 'accuracy_first_all_sizes.csv').open(newline='') as stream:
        selected = [r for r in csv.DictReader(stream) if r['mode'] == 'allocate']
    with (SOURCE / 'accuracy_first_model_refs.csv').open(newline='') as stream:
        refs = [r for r in csv.DictReader(stream) if r['mode'] == 'allocate']
    if len(selected) != 2 or len(refs) != 6:
        raise ValueError('Expected two selected allocate configurations and six weights')
    mapping = {}
    for item in selected:
        if (item['learner'] != {'fp16': 'mlp32', 'fp32': 'rf'}.get(item['dtype'])
                or item['target'] != 'normalized' or int(item['n']) != 11498
                or int(item['added']) != 8153 or int(item['failures']) != 0):
            raise ValueError('Unexpected selected expanded GEMM configuration')
        for seed, source_seed in zip(SEEDS, previous.layer.SEEDS):
            matches = [r for r in refs if int(r['seed']) == source_seed and
                       all(r[k] == item[k] for k in ('learner', 'dtype', 'mode', 'target', 'n', 'added'))]
            if len(matches) != 1:
                raise ValueError('Missing or duplicate selected GEMM weight')
            ref = matches[0]
            path = (ROOT / ref['model_ref']).resolve()
            if (path.parent != SOURCE / 'models' or path.is_symlink()
                    or sha(path) != ref['sha256'] or sha(path) != known.get(path)):
                raise ValueError('Expanded GEMM source checksum or path mismatch')
            spec = dict(path=str(path.relative_to(ROOT)), sha256=ref['sha256'],
                learner=item['learner'], dtype=item['dtype'], target=item['target'],
                mode='allocate', source_seed=source_seed, n=11498, features=gemm.FEATURES,
                protocol_sha256=frozen['protocol_sha256'])
            for learner in previous.learners.LEARNERS:
                mapping[f"{learner}/{item['dtype']}/{seed}"] = dict(spec)
    if len(mapping) != 24:
        raise ValueError('Incomplete shared GEMM source mapping')
    return mapping


def source_overlap():
    protocol = read(SOURCE / 'frozen_protocol.json')
    shapes = {tuple(s) for key in ('old_train_orders', 'new_train_orders')
              for rows in protocol[key].values() for s in rows}
    shapes.update(map(tuple, protocol['legacy_validation_shapes']))
    shapes.update(map(tuple, protocol['new_validation_shapes']))
    protected = set(map(tuple, read(PACKAGE / 'protected_shapes.json')))
    if not shapes <= protected:
        raise ValueError('Expanded GEMM source shapes missing from layer protection')
    spec = importlib.util.spec_from_file_location('shared_layer_manifest', PACKAGE / 'build_manifest.py')
    manifest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest)
    cases = [c for rows in read(PACKAGE / 'cases_by_phase.json').values() for c in rows]
    if any(manifest.overlaps(logical_gemms(c), shapes) for c in cases):
        raise ValueError('Layer shape overlaps expanded GEMM train or validation')
    return dict(source_train_validation_shapes=len(shapes), layer_cases=len(cases),
                logical_gemm_entries=sum(len(logical_gemms(c)) for c in cases),
                exact_or_one_axis_within_8_overlaps=0, measurement_labels_read=False)


def prepare(output):
    import components
    paths = list(HERE.glob('*.py')) + list(BASE.glob('*.py')) + [PACKAGE / n for n in (
        'groups.json', 'cases_by_phase.json', 'training_subsets.json', 'protocol.json',
        'release_hashes.json', 'protected_shapes.json', 'build_manifest.py')]
    paths += [SOURCE / name for name in SOURCE_FILES] + [EXPANSION / 'artifact_hashes_final.json']
    paths += [ROOT / 'analysis' / directory / name for directory, name in (
        ('layer_transfer_fit_2026-10-28', 'compare_layers.py'),
        ('layer_transfer_fit_2026-10-28', 'fit_layers.py'),
        ('learner_comparison_2026-10-28', 'compare_learners.py'),
        ('integrated_rq14_2026-10-28', 'run_rq14.py'),
        ('integrated_rq14_2026-10-28', 'requirements.txt'))]
    sources = frozen_sources()
    overlap = source_overlap()
    for dtype in ('fp16', 'fp32'):
        for seed in SEEDS:
            load_source({'source_models': sources}, 'rf', dtype, seed)
    subsets = read(PACKAGE / 'training_subsets.json')
    groups = read(PACKAGE / 'groups.json')
    by_id = {g['group_id']: g for g in groups}
    for family in ('core', 'conv'):
        for seed in SEEDS:
            seen = set()
            for k in BUDGETS:
                ids = subsets['families'][family][str(seed)]['by_k'][str(k)]
                if len(set(ids)) != k or not seen <= set(ids) or any(
                        by_id[i]['split'] != 'train' or by_id[i]['family'] != family for i in ids):
                    raise ValueError('Invalid fixed training subsets')
                seen = set(ids)
    protocol = dict(status='LAYER_FULL_PROTOCOL_FROZEN', version=2, created_utc=gemm.stamp(),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in paths}, source_models=sources,
        package_release_hashes=read(PACKAGE / 'release_hashes.json'),
        learners=previous.learners.LEARNERS, layers=LAYERS, targets=TARGETS, methods=METHODS,
        seeds=SEEDS, budgets=BUDGETS, features=FEATURES, component_features=components.FEATURES,
        subsets=subsets, groups=groups, cases=read(PACKAGE / 'cases_by_phase.json'),
        rf=gemm.RF, mlp=previous.learners.MLP, ridge=dict(alpha=1),
        quality_policy='Common PASS intersection across standard, decomposed and components for each layer/dtype; retain planned IDs and exclusions; no replacement or k relabeling.',
        selection='Mean validation MAPE across three seeds, then mean p90, actual training labels, learner, method, k. Invalid predictions disqualify. Save best per method and overall.',
        source_policy='Shared selected expanded allocate prior: FP16 MLP32 normalized and FP32 RF normalized, n=11498. Same six source weights for every layer learner; source seeds paired by ordinal to layer seeds. Layer transfer only; RQ4 retains its independent A/C split.',
        source_overlap=overlap,
        component_policy='Train separate log-time models from static features and component training labels; sum predicted components. Score the sum against both forward targets. Component validation labels used only for diagnostics.',
        access_policy='Development standard, decomposed and component labels are declared inputs. Test times require frozen artifacts and a separate explicit evaluation command.',
        runtime=dict(python=sys.version, numpy=np.__version__, sklearn=gemm.sklearn.__version__, joblib=joblib.__version__),
        sealed_test_status='NOT_OPENED')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'frozen_protocol.json', protocol)
    print('PASS: protocol prepared; no measurement labels read', flush=True)


def verify_protocol(output):
    p = read(output / 'frozen_protocol.json')
    if p['status'] != 'LAYER_FULL_PROTOCOL_FROZEN' or p['version'] != 2:
        raise ValueError('Invalid protocol status')
    for name, digest in p['inputs'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT / name) != digest:
            raise ValueError('Frozen code or metadata changed: ' + name)
    for item in p['source_models'].values():
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen GEMM model changed')
    if p['source_models'] != frozen_sources():
        raise ValueError('Frozen shared GEMM source mapping changed')
    if p['runtime']['sklearn'] != gemm.sklearn.__version__:
        raise ValueError('Use the frozen scikit-learn environment')
    return p


def load_source(p, learner, dtype, seed):
    spec = p['source_models'][f'{learner}/{dtype}/{seed}']
    if (spec['dtype'] != dtype or spec['mode'] != 'allocate' or spec['n'] != 11498
            or spec['features'] != gemm.FEATURES):
        raise ValueError('Invalid expanded GEMM source specification')
    path = ROOT / spec['path']
    if (Path(spec['path']).is_absolute() or '..' in Path(spec['path']).parts
            or path.is_symlink() or path.resolve().parent != SOURCE / 'models'
            or sha(path) != spec['sha256']):
        raise ValueError('Frozen GEMM model checksum or path mismatch')
    model = joblib.load(path)
    for key, expected in [('learner', spec['learner']), ('dtype', dtype), ('target', spec['target']),
                          ('mode', 'allocate'), ('seed', spec['source_seed']), ('n', spec['n']),
                          ('features', gemm.FEATURES), ('protocol_sha256', spec['protocol_sha256']),
                          ('locked_test_opened', False), ('validation_only', True)]:
        if model.get(key) != expected:
            raise ValueError('GEMM source envelope mismatch: ' + key)
    return model


def cohort(dataset, p, layer, dtype, test=False):
    import components
    labels = {(r['case_id'], r['dtype'], r['implementation']): r['time_ms'] for r in dataset['rows']}
    allowed = {'test_locked'} if test else {'train', 'validation'}
    planned = sorted([c for cc in p['cases'].values() for c in cc
                      if c['layer'] == layer and c['split'] in allowed], key=lambda c: c['group_id'])
    required = set(TARGETS) | set(components.component_names(layer))
    required.discard('gemm')
    retained = [c for c in planned if all((c['id'], dtype, impl) in labels for impl in required)]
    return retained, labels, dict(layer=layer, dtype=dtype, planned_groups=len(planned),
        retained_groups=len(retained), excluded_group_ids=[c['group_id'] for c in planned if c not in retained])


def predict(envelope, cases, dtype, p):
    if envelope['method'] == 'component_sum':
        import components
        return components.predict_sum(envelope['model'], cases)[0]
    if envelope['method'] == 'direct':
        return gemm.predict('direct', envelope['model'], features(cases, dtype), np.ones(len(cases)))[0]
    prior = source_prior(cases, dtype, load_source(p, envelope['learner'], dtype, envelope['seed']))
    return gemm.predict(envelope['method'], envelope['model'], features(cases, dtype),
                        np.ones(len(cases)), prior)[0]


def aggregate(scores):
    groups = defaultdict(list)
    for row in scores:
        groups[tuple(row[k] for k in ('target', 'layer', 'dtype', 'learner', 'method', 'k'))].append(row)
    result = []
    for key, rows in groups.items():
        item = dict(zip(('target', 'layer', 'dtype', 'learner', 'method', 'k'), key))
        item.update(mean_mape_pct=float(np.mean([r['mape_pct'] for r in rows])),
                    mean_p90_pct=float(np.mean([r['p90_pct'] for r in rows])),
                    failures=sum(r['failures'] for r in rows),
                    actual_train_groups=float(np.mean([r['actual_train_groups'] for r in rows])),
                    actual_train_labels=float(np.mean([r['actual_train_labels'] for r in rows])))
        result.append(item)
    return result


def choose(aggregate_rows):
    selected = []
    for target in TARGETS:
        for layer in LAYERS:
            for dtype in ('fp16', 'fp32'):
                candidates = [r for r in aggregate_rows if (r['target'], r['layer'], r['dtype']) == (target, layer, dtype)]
                for scope in METHODS + ['overall']:
                    valid = [r for r in candidates if (scope == 'overall' or r['method'] == scope)
                             and r['failures'] == 0 and np.isfinite(r['mean_mape_pct'])]
                    if not valid:
                        raise ValueError(f'No valid candidate for {target}/{layer}/{dtype}/{scope}')
                    best = min(valid, key=lambda r: (r['mean_mape_pct'], r['mean_p90_pct'],
                        r['actual_train_labels'], r['learner'], r['method'], r['k']))
                    selected.append(dict(best, selection_scope=scope))
    return selected


def label_rows(selection, cases_by_group):
    import components
    result = []
    for row in selection:
        for seed, ids in row['training_group_ids_by_seed'].items():
            base = dict(layer=row['layer'], dtype=row['dtype'], seed=int(seed))
            implementations = components.component_names(row['layer']) if row['method'] == 'component_sum' else [row['target']]
            for impl in implementations:
                result.append(dict(base, implementation='decomposed' if impl == 'gemm' else impl,
                    train_group_ids=ids, validation_group_ids=[]))
            result.append(dict(base, implementation=row['target'], train_group_ids=[],
                               validation_group_ids=row['validation_group_ids']))
    return result


def refit(envelope, training, labels, p):
    import components
    learner, dtype, seed, method = (envelope[k] for k in ('learner', 'dtype', 'seed', 'method'))
    if method == 'component_sum':
        return components.fit_sum(training, dtype, learner, labels, seed)
    if method == 'zero':
        return None, dict(fit_seconds=0.0, convergence_warnings=0)
    y = np.asarray([labels[c['id'], dtype, envelope['target']] for c in training])
    prior = source_prior(training, dtype, load_source(p, learner, dtype, seed))
    return previous.train(learner, method, features(training, dtype), y, prior, seed)


def implementation_diagnostics(dataset, p, test=False):
    import components
    result = []
    for layer in LAYERS:
        for dtype in ('fp16', 'fp32'):
            cases, labels, _ = cohort(dataset, p, layer, dtype, test=test)
            for case in cases:
                standard = labels[case['id'], dtype, 'standard']
                decomposed = labels[case['id'], dtype, 'decomposed']
                total = sum(labels[case['id'], dtype, 'decomposed' if name == 'gemm' else name]
                            for name in components.component_names(layer))
                result.append(dict(layer=layer, dtype=dtype, group_id=case['group_id'], split=case['split'],
                    standard_ms=standard, decomposed_ms=decomposed, measured_component_sum_ms=total,
                    decomposed_over_standard=decomposed / standard,
                    component_sum_over_decomposed=total / decomposed,
                    gemm_axis_below_64=any(min(s) < 64 for s in logical_gemms(case))))
    return result


def train(campaign, output):
    import components
    import data
    import costs
    p = verify_protocol(output)
    marker = output / 'execution_started.json'
    if marker.exists():
        raise ValueError('Execution already started; prepare a new output directory')
    save(marker, dict(time=gemm.stamp(), campaign=str(campaign.resolve())))
    dataset = data.load_campaign(campaign, diagnostic=True, protocol_path=output / 'frozen_protocol.json')
    save(output / 'input_audit.json', {k: v for k, v in dataset.items() if k != 'rows'})
    start = time.perf_counter()
    scores, predictions, diagnostics, coverage, component_scores = [], [], [], [], []
    models = {}
    retained_cases = {}
    for layer in LAYERS:
        for dtype in ('fp16', 'fp32'):
            cases, labels, quality = cohort(dataset, p, layer, dtype)
            coverage.append(quality)
            train_cases = {c['group_id']: c for c in cases if c['split'] == 'train'}
            valid = [c for c in cases if c['split'] == 'validation']
            if not valid:
                raise ValueError('Empty validation cohort')
            retained_cases[layer, dtype] = (train_cases, valid)
            xv = features(valid, dtype)
            family = 'conv' if layer.startswith('conv') else 'core'
            for learner in p['learners']:
                for seed in p['seeds']:
                    source = load_source(p, learner, dtype, seed)
                    before_prior = time.perf_counter()
                    pv = source_prior(valid, dtype, source)
                    diagnostics.append(dict(layer=layer, dtype=dtype, learner=learner, seed=seed,
                        k=0, target='shared', method='source_prior_validation', fit_seconds=0.0,
                        prediction_seconds=time.perf_counter() - before_prior))
                    for k in p['budgets']:
                        ids = p['subsets']['families'][family][str(seed)]['by_k'][str(k)]
                        training = [train_cases[i] for i in ids if i in train_cases]
                        if len(training) < 2:
                            raise ValueError('Too few retained training groups for fixed budget')
                        x = features(training, dtype)
                        before_prior = time.perf_counter()
                        pa = source_prior(training, dtype, source)
                        diagnostics.append(dict(layer=layer, dtype=dtype, learner=learner, seed=seed,
                            k=k, target='shared', method='source_prior_train', fit_seconds=0.0,
                            prediction_seconds=time.perf_counter() - before_prior))
                        comp_model, comp_detail = components.fit_sum(training, dtype, learner, labels, seed)
                        before_component_predict = time.perf_counter()
                        comp_pred, individual, _ = components.predict_sum(comp_model, valid)
                        comp_detail['prediction_seconds'] = time.perf_counter() - before_component_predict
                        for name, values in individual.items():
                            impl = 'decomposed' if name == 'gemm' else name
                            yy = np.asarray([labels[c['id'], dtype, impl] for c in valid])
                            component_scores.append(dict(layer=layer, dtype=dtype, learner=learner, seed=seed,
                                k=k, component=name, **gemm.metrics(yy, values)))
                        component_details = comp_detail if isinstance(comp_detail, list) else [comp_detail]
                        diagnostics.extend(dict(d, layer=layer, dtype=dtype, learner=learner, seed=seed,
                                                k=k, target='shared', method='component_sum') for d in component_details)
                        for target in TARGETS:
                            y = np.asarray([labels[c['id'], dtype, target] for c in training])
                            yv = np.asarray([labels[c['id'], dtype, target] for c in valid])
                            for method in METHODS:
                                if method == 'zero' and k != p['budgets'][0]:
                                    continue
                                info = dict(target=target, layer=layer, dtype=dtype, learner=learner,
                                            seed=seed, k=0 if method == 'zero' else k, method=method)
                                if method == 'component_sum':
                                    model, pred = comp_model, comp_pred
                                else:
                                    model, detail = previous.train(learner, method, x, y, pa, seed)
                                    pred, prediction_seconds = gemm.predict(method, model, xv, np.ones(len(valid)), pv)
                                    diagnostics.append(dict(info, **detail, prediction_seconds=prediction_seconds))
                                scores.append(dict(info, **gemm.metrics(yv, pred),
                                    actual_train_groups=0 if method == 'zero' else len(training),
                                    actual_train_labels=0 if method == 'zero' else len(training) *
                                        (len(components.component_names(layer)) if method == 'component_sum' else 1)))
                                predictions.extend(dict(info, group_id=c['group_id'], actual_ms=float(a), predicted_ms=float(b))
                                    for c, a, b in zip(valid, yv, pred))
                                identity = tuple(info[n] for n in ('target', 'layer', 'dtype', 'learner', 'method', 'k', 'seed'))
                                # Store recipes; refit selected seeds to keep forests out of the search-wide RAM set.
                                models[identity] = dict(info,
                                    training_group_ids=[] if method == 'zero' else [c['group_id'] for c in training])
            print(f'PASS: development models {layer} {dtype}', flush=True)
    aggregated = aggregate(scores)
    selected = choose(aggregated)
    (output / 'models').mkdir()
    hashes = {}
    all_labels = {(r['case_id'], r['dtype'], r['implementation']): r['time_ms'] for r in dataset['rows']}
    for item in selected:
        item['training_group_ids_by_seed'] = {}
        item['model_files_by_seed'] = {}
        _, valid = retained_cases[item['layer'], item['dtype']]
        item['validation_group_ids'] = [c['group_id'] for c in valid]
        for seed in p['seeds']:
            identity = tuple(item[n] for n in ('target', 'layer', 'dtype', 'learner', 'method', 'k')) + (seed,)
            envelope = models[identity]
            name = 'models/' + '_'.join(map(str, identity)) + '.joblib'
            if name not in hashes:
                train_cases, _ = retained_cases[item['layer'], item['dtype']]
                training = [train_cases[i] for i in envelope['training_group_ids']]
                envelope['model'], detail = refit(envelope, training, all_labels, p)
                diagnostics.append(dict(detail, phase='selected_refit', target=item['target'],
                    layer=item['layer'], dtype=item['dtype'], learner=item['learner'], method=item['method'],
                    seed=seed, k=item['k']))
                joblib.dump(envelope, output / name)
                reloaded = predict(joblib.load(output / name), valid, item['dtype'], p)
                expected = [r['predicted_ms'] for r in predictions if
                            tuple(r[n] for n in ('target', 'layer', 'dtype', 'learner', 'method', 'k', 'seed')) == identity]
                if not np.array_equal(reloaded, np.asarray(expected)):
                    raise ValueError('Model round-trip prediction mismatch')
                hashes[name] = sha(output / name)
                envelope.pop('model')
            item['model_files_by_seed'][str(seed)] = name
            item['training_group_ids_by_seed'][str(seed)] = envelope['training_group_ids']
    save(output / 'selected_configurations.json', selected)
    hashes['selected_configurations.json'] = sha(output / 'selected_configurations.json')
    for name, rows in [('metrics', scores), ('aggregate_metrics', aggregated), ('validation_predictions', predictions),
                       ('fit_diagnostics', diagnostics), ('quality_coverage', coverage), ('component_metrics', component_scores),
                       ('implementation_diagnostics', implementation_diagnostics(dataset, p))]:
        save(output / (name + '.json'), rows)
        if rows and name != 'fit_diagnostics':
            gemm.save_csv(output / (name + '.csv'), rows)
    fit_timing = dict(wall_seconds=time.perf_counter() - start,
                      total_model_fit_seconds=sum(d.get('fit_seconds', 0) for d in diagnostics),
                      prediction_seconds=sum(d.get('prediction_seconds', 0) for d in diagnostics))
    save(output / 'fit_timing.json', fit_timing)
    protocol_access = [dict(layer=r['layer'], dtype=r['dtype'], seed='shared', implementation=r['implementation'],
        train_group_ids=[r['group_id']] if r['split'] == 'train' else [],
        validation_group_ids=[r['group_id']] if r['split'] == 'validation' else [])
        for r in dataset['rows'] + dataset['excluded']]
    recipe = label_rows([r for r in selected if r['selection_scope'] == 'overall'], {})
    save(output / 'selected_label_budget.json', recipe)
    save(output / 'protocol_access_budget.json', protocol_access)
    save(output / 'cost_report.json', costs.build_report(campaign, recipe, fit_timing=fit_timing, protocol_access=protocol_access))
    for path in sorted(output.iterdir()):
        if path.suffix in ('.json', '.csv') and path.name not in ('frozen_protocol.json', 'execution_started.json'):
            hashes[path.name] = sha(path)
    verify_protocol(output)
    save(output / 'selection_frozen.json', dict(status='LAYER_FULL_SELECTION_FROZEN', timestamp=gemm.stamp(),
        protocol_sha256=sha(output / 'frozen_protocol.json'), artifact_hashes=hashes,
        sealed_test_status='NOT_OPENED', selection_rule=p['selection'], **fit_timing))
    print('PASS: development selection and model hashes frozen', flush=True)


def evaluate(campaign, output, destination, allow_sealed_test=False):
    if not allow_sealed_test:
        raise ValueError('Sealed evaluation requires --allow-sealed-test')
    import data
    p = verify_protocol(output)
    freeze = read(output / 'selection_frozen.json')
    if freeze['protocol_sha256'] != sha(output / 'frozen_protocol.json'):
        raise ValueError('Frozen protocol mismatch')
    for name, digest in freeze['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(output / name) != digest:
            raise ValueError('Frozen artifact mismatch')
    marker = output / 'sealed_evaluation_started.json'
    if marker.exists():
        raise ValueError('Sealed evaluation already started; preserve its audit trail')
    destination.mkdir(parents=True, exist_ok=False)
    save(marker, dict(timestamp=gemm.stamp(), output=str(destination.resolve()),
        freeze_sha256=sha(output / 'selection_frozen.json')))
    dataset = data.load_campaign(campaign, split='test', allow_test=True, diagnostic=True,
        protocol_path=output / 'frozen_protocol.json', freeze_path=output / 'selection_frozen.json')
    scores, predictions, coverage, slice_scores, paired, component_scores = [], [], [], [], [], []
    selected = read(output / 'selected_configurations.json')
    group_meta = {g['group_id']: g for g in p['groups']}
    for layer in LAYERS:
        for dtype in ('fp16', 'fp32'):
            cases, labels, quality = cohort(dataset, p, layer, dtype, test=True)
            coverage.append(quality)
            if not cases:
                raise ValueError('No retained test groups')
            for target in TARGETS:
                y = np.asarray([labels[c['id'], dtype, target] for c in cases])
                candidate_predictions = {}
                for item in [s for s in selected if (s['layer'], s['dtype'], s['target']) == (layer, dtype, target)]:
                    for seed, name in item['model_files_by_seed'].items():
                        envelope = joblib.load(output / name)
                        pred = predict(envelope, cases, dtype, p)
                        info = {k: item[k] for k in ('target', 'layer', 'dtype', 'learner', 'method', 'k', 'selection_scope')}
                        info['seed'] = int(seed)
                        scores.append(dict(info, **gemm.metrics(y, pred)))
                        if item['selection_scope'] == 'component_sum':
                            import components
                            _, component_predictions, _ = components.predict_sum(envelope['model'], cases)
                            for component, values in component_predictions.items():
                                impl = 'decomposed' if component == 'gemm' else component
                                actual = np.asarray([labels[c['id'], dtype, impl] for c in cases])
                                component_scores.append(dict(info, component=component, **gemm.metrics(actual, values)))
                        candidate_predictions[item['selection_scope'], seed] = pred
                        predictions.extend(dict(info, group_id=c['group_id'], actual_ms=float(a), predicted_ms=float(b))
                            for c, a, b in zip(cases, y, pred))
                        slices = {'all': np.ones(len(cases), dtype=bool),
                            'gemm_axis_below_64': np.asarray([any(min(s) < 64 for s in logical_gemms(c)) for c in cases])}
                        for key in ('alignment', 'work_class', 'aspect'):
                            for value in sorted({group_meta[c['group_id']][key] for c in cases}):
                                slices[key + '/' + value] = np.asarray([group_meta[c['group_id']][key] == value for c in cases])
                        for name, mask in slices.items():
                            if mask.any():
                                slice_scores.append(dict(info, slice=name, **gemm.metrics(y[mask], pred[mask])))
                for seed in map(str, p['seeds']):
                    for first in ['overall', 'residual', 'component_sum']:
                        paired.append(dict(target=target, layer=layer, dtype=dtype, seed=int(seed),
                            first=first, second='direct', **gemm.bootstrap_delta(y,
                            candidate_predictions[first, seed], candidate_predictions['direct', seed])))
    for name, rows in [('test_metrics', scores), ('test_predictions', predictions), ('test_quality_coverage', coverage),
                       ('test_slices', slice_scores), ('paired_bootstrap', paired), ('test_component_metrics', component_scores),
                       ('test_implementation_diagnostics', implementation_diagnostics(dataset, p, test=True))]:
        save(destination / (name + '.json'), rows)
        gemm.save_csv(destination / (name + '.csv'), rows)
    save(destination / 'evaluation_completed.json', dict(timestamp=gemm.stamp(),
        freeze_sha256=sha(output / 'selection_frozen.json'), model_selection_changed=False,
        input_provenance=dataset.get('provenance', {}), rows=len(scores),
        artifact_hashes={path.name: sha(path) for path in sorted(destination.iterdir())
                         if path.is_file() and path.suffix in ('.json', '.csv')}))
    print('PASS: fixed selections evaluated; no refitting or reselection', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--evaluation-output', type=Path)
    parser.add_argument('--allow-sealed-test', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
    elif args.action == 'train':
        if args.campaign is None:
            parser.error('--campaign is required')
        train(args.campaign, args.output)
    else:
        if args.campaign is None or args.evaluation_output is None:
            parser.error('--campaign and --evaluation-output are required')
        evaluate(args.campaign, args.output, args.evaluation_output, args.allow_sealed_test)


if __name__ == '__main__':
    main()
