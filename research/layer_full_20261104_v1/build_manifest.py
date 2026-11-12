"""Freeze balanced layer cases using static metadata only."""
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
SEED = 20261104
SPLITS = [('train', 200), ('validation', 50), ('test_locked', 50)]
STRATA = list(itertools.product(('aligned16', 'unaligned'),
                               ('small', 'medium', 'large'), ('compact', 'elongated')))
K_VALUES = (5, 10, 20, 30, 50, 100, 200)


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
    reserved = {tuple(r['shape']) for r in meta['shape_splits']}
    reserved.update(map(tuple, meta['nonmain_shapes']))
    reserved.update(map(tuple, read('metadata/old_forbidden_shapes.json')))
    cases = read('metadata/old_development_cases.json')
    cases += [c for rows in read('metadata/prior_transfer_cases.json').values() for c in rows]
    for case in cases:
        reserved.update(components(case))
    for g in read('metadata/old_layer_spec_groups.json'):
        m, d, f, ci, co = (g[k] for k in ('M', 'D', 'F', 'Cin', 'Cout'))
        reserved.update(((m, co, d), (m, co, ci), (m, f, d), (m, d, f), (m, co, 9 * ci)))
    reserved.update(tuple(r['shape']) for r in read('metadata/gemm_expansion_shapes.json'))
    return reserved


def make_cases(group):
    common = dict(split=group['split'], group_id=group['group_id'],
                  selection_family=group['group_id'], selection_reason=group['stratum'])
    if group['family'] == 'core':
        common.update(rows=group['M'], input_width=group['D'])
        cases = [dict(common, layer='linear', output_width=group['F']),
                 dict(common, layer='ffn', hidden_width=group['F'])]
    else:
        common.update({key: group[key] for key in ('batch', 'height', 'width', 'cin', 'cout')})
        cases = [dict(common, layer=f'conv{k}', kernel=k) for k in (1, 3)]
    for case in cases:
        case['id'] = 'lf_' + digest(case)[:16]
    return cases


def stratum(shape, aligned):
    flops = 2 * math.prod(shape)
    work = 'small' if flops <= 100_000_000 else 'medium' if flops <= 1_000_000_000 else 'large'
    aspect = 'elongated' if max(shape) / min(shape) >= 8 else 'compact'
    return ('aligned16' if aligned else 'unaligned', work, aspect)


def sample_family(family, reserved, used):
    rng = random.Random(SEED + (family == 'conv'))
    buckets = {key: [] for key in STRATA}
    anchors = set()
    attempts = 0
    while any(len(rows) < 25 for rows in buckets.values()):
        attempts += 1
        if attempts > 1_000_000:
            raise RuntimeError('Static candidate pool exhausted: ' + str({str(k): len(v) for k, v in buckets.items()}))
        aligned = bool(rng.randrange(2))
        delta = 0 if aligned else rng.choice((-1, 1))
        if family == 'core':
            m = 16 * round(math.exp(rng.uniform(math.log(64), math.log(4096))) / 16)
            d = 16 * round(math.exp(rng.uniform(math.log(64), math.log(2048))) / 16)
            ratio = rng.choice((2, 4))
            anchor = (m, d, ratio)
            m, d, f = m + delta, d + delta, d * ratio + delta
            if not 64 <= min(m, d, f) or max(m, d, f) > 4096:
                continue
            g = dict(family=family, M=m, D=d, F=f)
            shapes = {(m, f, d), (m, d, f)}
            representative = (m, f, d)
        else:
            b = rng.choice((1, 2, 4, 8, 16))
            h = rng.choice((8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64))
            w = h if rng.random() < .5 else rng.choice((8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64))
            ci = 16 * round(math.exp(rng.uniform(math.log(16), math.log(448))) / 16)
            co = 16 * round(math.exp(rng.uniform(math.log(16), math.log(4096))) / 16)
            anchor = (b, h, w, ci, co)
            ci, co = ci + delta, co + delta
            m = b * h * w
            if min(ci, co) < 16 or max(m, co, ci * 9) > 4096:
                continue
            g = dict(family=family, batch=b, height=h, width=w, cin=ci, cout=co)
            shapes = {(m, co, ci), (m, co, ci * 9)}
            representative = (m, co, ci * 9)
        bucket = stratum(representative, aligned)
        if len(buckets[bucket]) == 25 or anchor in anchors:
            continue
        if overlaps(shapes, reserved) or overlaps(shapes, used):
            continue
        g.update(anchor=list(anchor), alignment=bucket[0], work_class=bucket[1],
                 aspect=bucket[2], stratum='_'.join(bucket))
        buckets[bucket].append(g)
        anchors.add(anchor)
        used.update(shapes)
    for bucket, rows in buckets.items():
        rows.sort(key=lambda g: digest([SEED, 'split', bucket, g]))
    ordered = [buckets[bucket][i] for i in range(25) for bucket in STRATA]
    groups, start = [], 0
    for split, count in SPLITS:
        for candidate in ordered[start:start + count]:
            g = dict(candidate, split=split)
            g['group_id'] = 'lfg_' + digest(g)[:16]
            groups.append(g)
        start += count
    return groups, attempts


def training_subsets(groups):
    families = {}
    for family in ('core', 'conv'):
        train = [g for g in groups if g['family'] == family and g['split'] == 'train']
        seeds = {}
        for seed in (SEED, SEED + 1, SEED + 2):
            buckets = {key: sorted((g for g in train if g['stratum'] == '_'.join(key)),
                                  key=lambda g: digest([seed, g['group_id']])) for key in STRATA}
            order = []
            while any(buckets.values()):
                for key in STRATA:
                    if buckets[key]:
                        order.append(buckets[key].pop(0)['group_id'])
            seeds[str(seed)] = {'by_k': {str(k): order[:k] for k in K_VALUES}}
        families[family] = seeds
    return dict(timing_values_used=False, unit='paired base group', families=families)


def build():
    reserved, used = protected_shapes(), set()
    core, core_attempts = sample_family('core', reserved, used)
    conv, conv_attempts = sample_family('conv', reserved, used)
    groups, phases = core + conv, {}
    for split, count in SPLITS:
        families = [[g for g in rows if g['split'] == split] for rows in (core, conv)]
        prefix = 'test' if split == 'test_locked' else split
        for start in range(0, count, 10):
            batch = families[0][start:start + 10] + families[1][start:start + 10]
            phases[f'{prefix}_{start // 10 + 1:02d}'] = [c for g in batch for c in make_cases(g)]
    audit = dict(seed=SEED, groups=600, layer_cases=1200, batch_count=30,
                 per_layer_split_counts={kind: dict(SPLITS) for kind in ('linear', 'ffn', 'conv1', 'conv3')},
                 family_strata={family: dict(Counter(g['stratum'] for g in groups if g['family'] == family))
                                for family in ('core', 'conv')},
                 attempts=dict(core=core_attempts, conv=conv_attempts),
                 static_flop_boundaries=[100_000_000, 1_000_000_000],
                 protected_shape_count=len(reserved), existing_splits_changed=False,
                 protection='All earlier GEMM and layer shapes, including the 10000-shape expansion; exact and one-axis +/-8; no cross-group component overlap',
                 expected_measurements={phase: 1080 for phase in phases},
                 expected_forward_measurements=14400, expected_component_measurements=18000,
                 expected_total_measurements=32400, expected_time_blocks=226800,
                 linear_component_policy='decomposed forward is the standalone GEMM; reuse the same record',
                 timing_values_used=False, gpu_collection_executed=False)
    return groups, phases, sorted(reserved), audit


if __name__ == '__main__':
    targets = ('groups.json', 'cases_by_phase.json', 'protected_shapes.json',
               'selection_audit.json', 'training_subsets.json')
    if any((HERE / name).exists() for name in targets):
        raise SystemExit('Frozen manifest exists; preserve it and create a new release')
    groups, phases, protected, audit = build()
    values = (groups, phases, protected, audit, training_subsets(groups))
    for name, value in zip(targets, values):
        (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print('MANIFEST PASS: 600 base groups, 1200 layer cases, 32400 measurements, 30 batches')
