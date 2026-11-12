"""Generate static cases and split audits without reading any timing labels."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def components(case):
    if case['layer'].startswith('conv'):
        return [(case['batch'] * case['height'] * case['width'], case['cout'],
                 case['cin'] * case['kernel'] ** 2)]
    m, d = case['rows'], case['input_width']
    return [(m, case['output_width'], d)] if case['layer'] == 'linear' else [
        (m, case['hidden_width'], d), (m, d, case['hidden_width'])]


def neighbours(shape):
    for axis in range(3):
        for delta in range(-8, 9):
            yield tuple(v + (delta if axis == j else 0) for j, v in enumerate(shape))


def overlaps(shapes, reserved):
    return any(n in reserved for s in shapes for n in neighbours(s))


def protected_shapes():
    meta = read('metadata/gemm_shape_metadata.json')
    reserved = {tuple(r['shape']) for r in meta['shape_splits']
                if r['split'] not in ('legacy_train', 'development')}
    reserved.update(map(tuple, meta['nonmain_shapes']))
    reserved.update(map(tuple, read('metadata/old_forbidden_shapes.json')))
    for c in read('metadata/old_development_cases.json'):
        reserved.update(components(c))
    for g in read('metadata/old_layer_spec_groups.json'):
        if g['split'] not in ('validation', 'test_locked'):
            continue
        m, d, f, ci, co = (g[k] for k in ('M', 'D', 'F', 'Cin', 'Cout'))
        reserved.update(((m, co, d), (m, co, ci), (m, f, d), (m, d, f), (m, co, 9 * ci)))
    return reserved


def make_core_cases(group):
    common = dict(rows=group['M'], input_width=group['D'], split=group['split'],
                  group_id=group['group_id'], selection_family=group['group_id'],
                  selection_reason=group['alignment'] + '_' + group['work_class'])
    result = [dict(common, layer='linear', output_width=group['F']),
              dict(common, layer='ffn', hidden_width=group['F'])]
    for c in result:
        c['id'] = 'lt_' + digest(c)[:16]
    return result


def training_subsets(groups):
    train = [g for g in groups if g['split'] == 'train']
    cycle = [('aligned16', 'small'), ('unaligned', 'large'),
             ('aligned16', 'large'), ('unaligned', 'small')]
    seeds = {}
    for seed in (20261028, 20261029, 20261030):
        buckets = {key: sorted((g for g in train if (g['alignment'], g['work_class']) == key),
                               key=lambda g: digest([seed, 'rq3_training', g['group_id']]))
                   for key in cycle}
        ordered = []
        while any(buckets.values()):
            for key in cycle:
                if buckets[key]:
                    ordered.append(buckets[key].pop(0))
        seeds[str(seed)] = dict(
            ordered_train_group_ids=[g['group_id'] for g in ordered],
            by_k={str(k): [g['group_id'] for g in ordered[:k]] for k in (5, 10, 20, 30)},
            static_balance_by_k={str(k): dict(
                alignment=dict(Counter(g['alignment'] for g in ordered[:k])),
                work_class=dict(Counter(g['work_class'] for g in ordered[:k]))) for k in (5, 10, 20, 30)})
    return dict(unit='M,D,F group; same IDs for both layers, dtypes, implementations and methods',
                timing_values_used=False, seeds=seeds,
                policy='seed-hash within static alignment/work buckets; deterministic round-robin bucket cycle',
                frozen_before_gpu_collection=True)


def build():
    reserved = protected_shapes()
    roles = {tuple(r['shape']): r['internal_role'] for r in read('metadata/gemm_internal_roles.json')}
    pool = []
    for m, d, ratio in itertools.product(
            (64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096),
            (64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024), (2, 4)):
        for delta in (0, 1):
            mm, dd, ff = m + delta, d + delta, d * ratio + delta
            if max(mm, dd, ff) > 4096:
                continue
            shapes = {(mm, ff, dd), (mm, dd, ff)}
            if overlaps(shapes, reserved):
                continue
            g = dict(M=mm, D=dd, F=ff, anchor=[m, d, ratio],
                     alignment='aligned16' if delta == 0 else 'unaligned',
                     work_class='small' if 2 * mm * dd * ff <= 100_000_000 else 'large')
            pool.append(g)
    groups, used, anchors, rejected = [], set(), set(), Counter()
    for split, count in [('train', 30), ('validation', 10), ('test_locked', 20)]:
        for alignment in ('aligned16', 'unaligned'):
            for i in range(count // 2):
                work = 'small' if i < count // 4 else 'large'
                candidates = sorted((g for g in pool if g['alignment'] == alignment and g['work_class'] == work),
                                    key=lambda g: digest([20261028, split, g]))
                for candidate in candidates:
                    shapes = {(candidate['M'], candidate['F'], candidate['D']),
                              (candidate['M'], candidate['D'], candidate['F'])}
                    if tuple(candidate['anchor']) in anchors or overlaps(shapes, used):
                        rejected['core_cross_group_overlap_or_anchor'] += 1
                        continue
                    g = dict(candidate, split=split)
                    g['group_id'] = 'ltg_' + digest(g)[:16]
                    g['component_existing_roles'] = [roles.get(s, 'unseen_MNK') for s in sorted(shapes)]
                    g['nested_train_position'] = sum(x['split'] == 'train' for x in groups) + 1 if split == 'train' else None
                    groups.append(g)
                    used.update(shapes)
                    anchors.add(tuple(candidate['anchor']))
                    break
                else:
                    raise RuntimeError('Static candidate pool exhausted before balanced split completion')
    core = [c for g in groups for c in make_core_cases(g)]
    confirm, conv_used = [], set()
    conv_pool = list(itertools.product((1, 2, 4), (10, 14, 18, 22, 28, 40),
                                      (32, 48, 64, 96, 128, 160, 192, 256), (64, 96, 128, 192, 256, 384)))
    conv_pool.sort(key=lambda x: digest([20261028, 'conv_confirm', x]))
    for b, h, ci, co in conv_pool:
        shapes = {(b * h * h, co, ci), (b * h * h, co, 9 * ci)}
        if max(max(s) for s in shapes) > 4096 or overlaps(shapes, reserved | used | conv_used):
            continue
        base_id = 'ltc_' + digest([b, h, ci, co])[:16]
        for k in (1, 3):
            c = dict(layer=f'conv{k}', batch=b, height=h, width=h, cin=ci, cout=co, kernel=k,
                     split='confirmation_reserved', group_id=base_id,
                     selection_family=base_id, selection_reason='new_static_confirmation')
            c['id'] = 'lt_' + digest(c)[:16]
            confirm.append(c)
        conv_used.update(shapes)
        if len(confirm) == 20:
            break
    assert len(confirm) == 20
    old = read('metadata/old_development_cases.json')
    diagnostic = []
    for kind in ('conv1', 'conv3'):
        candidates = [c for c in old if c['layer'] == kind and c['selection_reason'] == 'anchor']
        candidates.sort(key=lambda c: digest([20261028, 'diagnostic', c['id']]))
        diagnostic.extend(candidates[:4])
    cases = {'core_dev': [c for c in core if c['split'] != 'test_locked'],
             'core_locked': [c for c in core if c['split'] == 'test_locked'],
             'conv_confirm': confirm, 'conv_diagnostic': diagnostic}
    audit = dict(status='STATIC_ONLY_NO_TIMING_LABELS_OR_GPU', seed=20261028,
                 protected_shapes=len(reserved), adjacency='identical or one axis +/-8; same base anchor never split',
                 core_groups=len(groups), group_splits=dict(Counter(g['split'] for g in groups)),
                 alignment_by_split={s: dict(Counter(g['alignment'] for g in groups if g['split'] == s))
                                     for s in ('train', 'validation', 'test_locked')},
                 work_by_split={s: dict(Counter(g['work_class'] for g in groups if g['split'] == s))
                                for s in ('train', 'validation', 'test_locked')},
                 core_component_existing_roles=dict(Counter(r for g in groups for r in g['component_existing_roles'])),
                 cross_group_neighbour_overlaps=0, protected_neighbour_overlaps=0,
                 expected_measurements={'core_dev': 960, 'core_locked': 480, 'conv_confirm': 240, 'conv_diagnostic': 264},
                 expected_time_blocks={'core_dev': 6720, 'core_locked': 3360, 'conv_confirm': 1680, 'conv_diagnostic': 1848},
                 rejected=dict(rejected), no_protected_ids_reassigned=True,
                 diagnostic_scope='existing development only; 96 whole-path + 168 fixed-intermediate component records',
                 locked_test_times_read=False, collection_executed=False)
    return groups, cases, sorted(reserved), audit


if __name__ == '__main__':
    groups, cases, protected, audit = build()
    for name, value in [('groups.json', groups), ('cases_by_phase.json', cases),
                        ('protected_shapes.json', protected), ('selection_audit.json', audit),
                        ('rq3_training_subsets.json', training_subsets(groups))]:
        (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print('MANIFEST PASS: 60 groups; core 960 development and 480 locked records; optional Conv 240/264')
