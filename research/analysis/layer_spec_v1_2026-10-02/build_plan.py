"""Metadata-only layer candidate plan and overlap audit; no GPU imports."""
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'analysis/remeasure_720_v4_result_2026-10-01/recomputed/full_720_remeasure3_20261001'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def calls(g):
    b, h, d, f, ci, co = (g[k] for k in ('batch', 'side', 'D', 'F', 'Cin', 'Cout'))
    m = b*h*h
    gemms = {'linear': [(m, co, d)], 'conv1x1': [(m, co, ci)],
             'gelu_ffn': [(m, f, d), (m, d, f)], 'conv3x3_explicit': [(m, co, 9*ci)]}
    aux = {'linear': [], 'conv1x1': [('pack_nchw_input', b, ci, h, h), ('pack_nchw_output', b, co, h, h)],
           'gelu_ffn': [('gelu_none', m, f)],
           'conv3x3_explicit': [('unfold3x3', b, ci, h, h, 1, 1, 1),
                                ('pack_unfold_columns', b, 9*ci, h*h),
                                ('pack_nchw_output', b, co, h, h)]}
    return gemms, aux


def neighbours(shape):
    yield shape
    for axis in range(3):
        for delta in range(-8, 9):
            if delta:
                value = list(shape)
                value[axis] += delta
                yield tuple(value)


def build():
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    splits = {tuple(r['shape']): r['split'] for r in manifest['shape_splits']}
    forbidden = {s for s, split in splits.items() if split not in ('legacy_train', 'development')}
    forbidden.update(tuple(t['shape']) for t in manifest['tasks'] if t['phase'] != 'main')
    # Shared component keys connect groups. Select disjoint groups before allocating splits.
    candidates = list(itertools.product((1, 2, 4), (8, 12, 16, 24, 32),
                      (64, 72, 128, 136, 256, 264, 512, 520, 1024), (2, 4),
                      (64, 72, 128, 136, 256), (64, 128, 256, 512, 1024)))
    random.Random(20261002).shuffle(candidates)
    selected, used_gemm, used_aux = [], set(), set()
    rejected = Counter()
    for b, h, d, ratio, ci, co in candidates:
        g = dict(batch=b, side=h, D=d, F=d*ratio, Cin=ci, Cout=co)
        gemms, aux = calls(g)
        gs = {s for parts in gemms.values() for s in parts}
        hs = {s for parts in aux.values() for s in parts}
        if max(g['F'], b*h*h, 9*ci, co) > 4096:
            rejected['dimension_over_4096'] += 1
            continue
        expanded = {n for s in gs for n in neighbours(s)}
        if expanded & forbidden:
            rejected['existing_nontraining_or_diagnostic_overlap'] += 1
            continue
        if expanded & used_gemm or hs & used_aux:
            rejected['cross_group_component_overlap'] += 1
            continue
        g.update(group_id='lg_' + digest(g)[:12], M=b*h*h, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1)
        selected.append(g)
        used_gemm.update(gs)
        used_aux.update(hs)
        if len(selected) == 50:
            break
    assert len(selected) == 50
    selected.sort(key=lambda g: digest(['layer_split_v1', g['group_id']]))
    for i, g in enumerate(selected):
        g['split'] = 'train' if i < 30 else 'validation' if i < 40 else 'test_locked'
        g['nested_train_position'] = i+1 if i < 30 else None
    assert Counter(g['split'] for g in selected) == {'train': 30, 'validation': 10, 'test_locked': 10}

    operations, overlap = [], []
    owners_g, owners_h = {}, {}
    for g in selected:
        gemms, aux = calls(g)
        for layer in gemms:
            for j, shape in enumerate(gemms[layer]):
                owners_g.setdefault(shape, set()).add(g['group_id'])
                overlap.append(dict(group_id=g['group_id'], layer_split=g['split'], layer=layer,
                                    component=j, M=shape[0], N=shape[1], K=shape[2],
                                    existing_shape_split=splits.get(shape, 'unseen_MNK'),
                                    timing_reuse='unconfirmed_operand_and_execution_contract'))
                for dtype, mode in itertools.product(('fp16', 'fp32'), ('ordinary', 'graph')):
                    operations.append(dict(group_id=g['group_id'], split=g['split'], layer=layer,
                                           kind='gemm', component=j, logical_key=list(shape), dtype=dtype, mode=mode,
                                           operand_source='actual_layer_intermediate_and_weight',
                                           rounds=[0, 1, 2], graph_policy='paid_probe_2ms_max64_provisional'))
            for j, key in enumerate(aux[layer]):
                owners_h.setdefault(key, set()).add(g['group_id'])
                for dtype, mode in itertools.product(('fp16', 'fp32'), ('ordinary', 'graph')):
                    operations.append(dict(group_id=g['group_id'], split=g['split'], layer=layer,
                                           kind='aux', component=j, logical_key=list(key), dtype=dtype, mode=mode,
                                           operand_source='actual_layer_intermediate', rounds=[0, 1, 2],
                                           graph_policy='paid_probe_2ms_max64_provisional'))
    assert all(len(v)==1 for v in owners_g.values()) and all(len(v)==1 for v in owners_h.values())
    assert not (set(owners_g) & forbidden)
    assert all(owners_g[s] == owners_g[n]
               for s in owners_g for n in neighbours(s) if n in owners_g)
    train_component_shapes = {tuple(r['logical_key']) for r in operations if r['kind']=='gemm' and r['split']=='train'}
    existing_train = sorted(s for s, split in splits.items() if split=='legacy_train')
    forced_c = train_component_shapes & set(existing_train)
    ordered = sorted(set(existing_train)-forced_c-forbidden, key=lambda s: digest(['C_reservation_v1', s]))
    reserve_size = max(len(forced_c), (len(existing_train)+4)//5)
    reserved_c = forced_c | set(ordered[:reserve_size-len(forced_c)])
    assert reserved_c <= set(existing_train) and not (reserved_c & forbidden)
    frontier = list(reserved_c)
    train_shapes = set(existing_train)
    while frontier:
        shape = frontier.pop()
        additions = set(neighbours(shape)) & train_shapes - reserved_c
        reserved_c.update(additions)
        frontier.extend(additions)
    assert all((s in reserved_c) == (n in reserved_c)
               for s in train_shapes for n in neighbours(s) if n in train_shapes)
    reserve_rows = [dict(shape=list(s), original_split='legacy_train',
                         internal_role='C_calibration' if s in reserved_c else 'A_train_candidate',
                         forced_by_layer_train=s in forced_c) for s in existing_train]
    original_neighbours = [dict(shape_a=list(s), split_a=splits[s], shape_b=list(n), split_b=splits[n])
                           for s in sorted(splits) for n in neighbours(s)
                           if n in splits and s < n and splits[s] != splits[n]]
    conditions = Counter(r['kind'] for r in operations)
    role_conditions = Counter()
    with (ROOT/'analysis/gemm_dataset_v1_2026-10-02/main_complete_unlocked_conditions.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split']=='legacy_train':
                role_conditions['C_calibration' if tuple(json.loads(row['shape'])) in reserved_c else 'A_train_candidate'] += 1
    static_gemm = {tuple(r['logical_key']) + (r['dtype'], r['mode']) for r in operations if r['kind']=='gemm'}
    static_aux = {tuple(r['logical_key']) + (r['dtype'], r['mode']) for r in operations if r['kind']=='aux'}
    summary = dict(status='LOCAL_REVIEWABLE_CANDIDATE_NO_GPU', groups=50,
                   group_splits=dict(Counter(g['split'] for g in selected)),
                   layer_measurements=50*4*2*2*3,
                   component_gemm_measurements_before_reuse=conditions['gemm']*3,
                   aux_measurements_before_reuse=conditions['aux']*3,
                   total_measurements_before_reuse=50*4*2*2*3+sum(conditions.values())*3,
                   component_gemm_static_dedup_lower_bound=len(static_gemm)*3,
                   aux_static_dedup_lower_bound=len(static_aux)*3,
                   static_dedup_is_not_confirmed_timing_reuse=True,
                   gemm_unique_MNK=len(owners_g), aux_unique_logical_keys=len(owners_h),
                   cross_group_gemm_overlaps=0, cross_group_aux_overlaps=0,
                   cross_group_one_axis_8_neighbour_overlaps=0,
                   existing_split_matches=dict(Counter(r['existing_shape_split'] for r in overlap)),
                   C_reserved_existing_shapes=len(reserved_c), C_forced_shapes=len(forced_c),
                   C_initial_reservation_size=reserve_size, A_C_one_axis_8_neighbour_overlaps=0,
                   original_split_one_axis_8_neighbour_pairs=len(original_neighbours),
                   A_candidate_existing_shapes=len(existing_train)-len(reserved_c),
                   complete_existing_conditions_by_internal_role=dict(role_conditions),
                   layer_split_existing_shape_matches={split: dict(Counter(r['existing_shape_split'] for r in overlap if r['layer_split']==split)) for split in ('train','validation','test_locked')},
                   builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   rejected_candidate_counts=dict(rejected), graph_policy_user_response='pending',
                   pending=['GPU correctness tolerances', 'capture memory and API validation',
                            'actual operand hash reuse', 'seed/model budget and target error'])
    for name, value in [('groups.json', selected), ('operations.json', operations),
                        ('gemm_internal_roles.json', reserve_rows), ('summary.json', summary)]:
        (OUT/name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    with (OUT/'original_split_neighbours.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=['shape_a','split_a','shape_b','split_b'])
        writer.writeheader();writer.writerows(original_neighbours)
    with (OUT/'existing_gemm_overlap.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(overlap[0]));writer.writeheader();writer.writerows(overlap)
    files=[OUT/name for name in ('groups.json','operations.json','gemm_internal_roles.json','summary.json','existing_gemm_overlap.csv','original_split_neighbours.csv')]
    (OUT/'derived_hashes.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2)+'\n')
    print('PASS: 50 groups, 30/10/10 split, zero cross-group operation overlap, C-before-A reservation')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    build()
