#!/usr/bin/env python3
"""Independently enumerate label IDs, check target minima, and hash inputs."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'analysis/layer_full_fit_2026-11-10_v5/prepared_layer_full'


def read(path):
    return json.loads(path.read_text())


def csv_rows(name):
    with (HERE / 'tables' / (name + '.csv')).open() as stream:
        return list(csv.DictReader(stream))


def main():
    protocol = read(RUN / 'frozen_protocol.json')
    aggregate = read(RUN / 'aggregate_metrics.json')
    coverage = {(r['layer'], r['dtype']): set(r['excluded_group_ids']) for r in read(RUN / 'quality_coverage.json')}
    implementations = {'linear': ['decomposed'], 'ffn': ['gemm1', 'gelu', 'gemm2'],
                       'conv1': ['input_pack', 'mm', 'output_pack'],
                       'conv3': ['unfold', 'input_pack', 'mm', 'output_pack']}
    all_rows = csv_rows('layer_all_configurations')
    assert len(all_rows) == len(aggregate) == 2304
    identity = lambda r: (r['target'], r['layer'], r['dtype'], r['learner'], r['method'], int(r['k']))
    lookup = {identity(r): r for r in aggregate}
    label_sets = {}
    for r in all_rows:
        original = lookup[identity(r)]
        assert math.isclose(float(r['mean_mape_pct']), original['mean_mape_pct'])
        assert math.isclose(float(r['mean_p90_pct']), original['mean_p90_pct'])
        labels = set()
        per_seed = []
        for seed in protocol['seeds']:
            ids = [] if r['method'] == 'zero' else protocol['subsets']['families'][
                'conv' if r['layer'].startswith('conv') else 'core'][str(seed)]['by_k'][r['k']]
            ids = set(ids) - coverage[r['layer'], r['dtype']]
            impls = implementations[r['layer']] if r['method'] == 'component_sum' else [r['target']]
            per_seed.append(len(ids) * len(impls))
            labels.update(('train', r['layer'], r['dtype'], g, i) for g in ids for i in impls)
        for cases in protocol['cases'].values():
            for c in cases:
                if c['split'] == 'validation' and c['layer'] == r['layer'] and c['group_id'] not in coverage[r['layer'], r['dtype']]:
                    labels.add(('validation', r['layer'], r['dtype'], c['group_id'], r['target']))
        assert len(labels) == int(r['total_labels_union'])
        assert len(labels) * 3 == int(r['raw_records_union'])
        assert math.isclose(sum(per_seed) / 3, original['actual_train_labels'])
        label_sets[identity(r)] = labels
    equivalents = csv_rows('layer_equivalent_accuracy')
    attained = {}
    prior = {'zero', 'constant_ratio', 'prior_direct', 'residual'}
    for r in equivalents:
        allowed = prior if r['method_family'] == 'best_prior' else (prior | {'direct'}
                  if r['method_family'] == 'prior_with_direct_fallback' else {r['method_family']})
        candidates = [x for x in aggregate if (x['target'], x['layer'], x['dtype']) ==
                      (r['target'], r['layer'], r['dtype']) and x['method'] in allowed
                      and x['failures'] == 0 and x['mean_mape_pct'] <= float(r['target_mape_pct']) + 1e-12
                      and (not r['target_p90_pct'] or x['mean_p90_pct'] <= float(r['target_p90_pct']) + 1e-12)]
        if not candidates:
            assert r['status'] == 'NOT_ATTAINED_ON_OBSERVED_GRID'
            continue
        best = sorted(candidates, key=lambda x: (x['actual_train_labels'], x['mean_mape_pct'],
                x['mean_p90_pct'], x['learner'], x['method'], x['k']))[0]
        assert r['status'] == 'ATTAINED' and identity(r) == identity(best)
        attained[r['goal'], r['target'], r['layer'], r['dtype'], r['method_family']] = label_sets[identity(best)]
    portfolios = csv_rows('layer_portfolios')
    for r in portfolios:
        targets = {'standard', 'decomposed'} if r['target'] == 'combined' else {r['target']}
        sets = [v for k, v in attained.items() if k[0] == r['goal'] and k[1] in targets and k[4] == r['method_family']]
        assert len(sets) == int(r['attained_conditions'])
        if len(sets) == int(r['total_conditions']):
            assert len(set().union(*sets)) == int(r['total_labels_union'])
    be = csv_rows('break_even_conditional')
    for r in be:
        g, d, reuse = map(float, (r['source_cost'], r['direct_per_task'], r['reuse_per_task']))
        if d <= reuse:
            assert r['status'] == 'NO_BREAK_EVEN'
        else:
            q = int(r['first_saving_task_count'])
            assert g + q * reuse < q * d
            assert q == 1 or g + (q - 1) * reuse >= (q - 1) * d
    old = read(ROOT / 'analysis/learner_comparison_2026-10-28/runs/v1/frozen_protocol.json')
    exp = read(ROOT / 'analysis/gemm_expansion_result_2026-11-06/fit_v1/frozen_protocol.json')
    assert len(exp['legacy_validation_shapes']) + len(exp['new_validation_shapes']) == 1611
    for r in csv_rows('gemm_learning_curves'):
        n = int(float(r['n']))
        train = set()
        for seed in old['seeds']:
            if r['cohort'] == 'legacy_716':
                train.update(map(tuple, (old['a_orders'][str(seed)] + old['c_orders'][str(seed)])[:n]))
            else:
                train.update(map(tuple, exp['old_train_orders'][str(seed)] + exp['new_train_orders'][str(seed)][:n-3345]))
        assert len(train) == int(r['train_shapes_union'])
        assert len(train) + int(r['validation_shapes']) == int(r['recipe_labels_union'])
    for name, digest in read(HERE / 'input_manifest.json').items():
        with (ROOT / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
    missing = []
    for link in re.findall(r'\]\(([^)]+)\)', (HERE / 'README.md').read_text()):
        if '://' not in link and link != 'independent_verification.json' and not (HERE / link.split('#')[0]).exists():
            missing.append(link)
    assert not missing, missing
    result = dict(status='PASS', enumerated_layer_configurations=len(all_rows),
        minimum_budget_queries=len(equivalents), portfolio_rows=len(portfolios),
        break_even_rows=len(be), inputs_unchanged=len(read(HERE / 'input_manifest.json')),
        source_prior_train_shapes_per_seed=11498, source_validation_shapes=1611,
        test_metrics_read=False, refitting=False, report_links='PASS')
    (HERE / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: independent ID unions, minima, break-even boundaries and source hashes')


if __name__ == '__main__':
    main()
