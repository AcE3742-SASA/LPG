#!/usr/bin/env python3
"""Read frozen development results and recover preparation-cost evidence."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
from collections import defaultdict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIT = ROOT / 'analysis/layer_full_fit_2026-11-10_v5'
RUN = FIT / 'prepared_layer_full'
EXP = ROOT / 'analysis/gemm_expansion_result_2026-11-06'
LEGACY = ROOT / 'analysis/learner_comparison_2026-10-28/runs/v1'
OLD_RAW = ROOT / 'analysis/remeasure_720_v4_result_2026-10-01/recomputed/full_720_remeasure3_20261001'
sys.path.insert(0, str(FIT))
import costs

METHODS = ('direct', 'prior_direct', 'residual', 'component_sum', 'zero', 'constant_ratio')
FAMILIES = METHODS + ('best_prior', 'prior_with_direct_fallback')
COMPONENTS = {'linear': ['decomposed'], 'ffn': ['gemm1', 'gelu', 'gemm2'],
              'conv1': ['input_pack', 'mm', 'output_pack'],
              'conv3': ['unfold', 'input_pack', 'mm', 'output_pack']}
INPUTS = {}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def watch(path):
    path = Path(path)
    key = str(path.relative_to(ROOT))
    if key not in INPUTS:
        INPUTS[key] = sha(path)
    return path


def read(path):
    return json.loads(watch(path).read_text())


def csv_read(path):
    with watch(path).open() as stream:
        return list(csv.DictReader(stream))


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def table(name, rows):
    if not rows:
        raise ValueError('Empty output table: ' + name)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / 'tables' / (name + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def meets(row, mape, p90=None):
    return (row['failures'] == 0 and math.isfinite(row['mean_mape_pct'])
            and math.isfinite(row['mean_p90_pct']) and row['mean_mape_pct'] <= mape + 1e-12
            and (p90 is None or row['mean_p90_pct'] <= p90 + 1e-12))


def cheapest(rows, mape, p90=None):
    valid = [r for r in rows if meets(r, mape, p90)]
    return min(valid, key=lambda r: (r['actual_train_labels'], r['mean_mape_pct'],
               r['mean_p90_pct'], r['learner'], r['method'], r['k'])) if valid else None


def breakeven(source, direct, reuse):
    if not all(math.isfinite(x) and x >= 0 for x in (source, direct, reuse)):
        raise ValueError('Invalid nonnegative cost')
    saving = direct - reuse
    if saving <= 0:
        return {'status': 'NO_BREAK_EVEN', 'saving_per_task': saving,
                'first_nonloss_task_count': None, 'first_saving_task_count': None}
    ratio = source / saving
    return {'status': 'CONDITIONAL', 'saving_per_task': saving,
            'first_nonloss_task_count': max(1, math.ceil(ratio)),
            'first_saving_task_count': max(1, math.floor(ratio) + 1)}


def recipe(row, protocol, coverage):
    family = 'conv' if row['layer'].startswith('conv') else 'core'
    excluded = set(coverage[row['layer'], row['dtype']]['excluded_group_ids'])
    cases = [c for cc in protocol['cases'].values() for c in cc
             if c['layer'] == row['layer'] and c['group_id'] not in excluded]
    valid = sorted(c['group_id'] for c in cases if c['split'] == 'validation')
    result = []
    for seed in protocol['seeds']:
        ids = [] if row['method'] == 'zero' else [g for g in
            protocol['subsets']['families'][family][str(seed)]['by_k'][str(row['k'])]
            if g not in excluded]
        impls = COMPONENTS[row['layer']] if row['method'] == 'component_sum' else [row['target']]
        base = dict(layer=row['layer'], dtype=row['dtype'], seed=seed)
        result += [dict(base, implementation=impl, train_group_ids=ids, validation_group_ids=[])
                   for impl in impls]
        result.append(dict(base, implementation=row['target'], train_group_ids=[],
                           validation_group_ids=valid))
    return result


def budget_fields(rows):
    b = costs.label_budget(rows)
    u = b['all_seed_union']
    per = list(b['per_seed'].values())
    groups = {g for r in rows for g in r['train_group_ids']}
    cases = {(r['layer'], g) for r in rows for g in r['train_group_ids']}
    return dict(train_family_groups_union=len(groups), train_layer_shapes_union=len(cases),
                train_labels_union=u['train_round_records'] // 3,
                validation_labels=u['validation_round_records'] // 3,
                total_labels_union=u['aggregated_labels'], raw_records_union=u['round_records'],
                train_labels_per_seed_min=min(x['train_round_records'] // 3 for x in per),
                train_labels_per_seed_max=max(x['train_round_records'] // 3 for x in per))


def layer_analysis():
    p = read(RUN / 'frozen_protocol.json')
    agg = read(RUN / 'aggregate_metrics.json')
    selected = read(RUN / 'selected_configurations.json')
    coverage = {(r['layer'], r['dtype']): r for r in read(RUN / 'quality_coverage.json')}
    metrics = read(RUN / 'metrics.json')
    diagnostics = read(RUN / 'fit_diagnostics.json')
    by_key = defaultdict(list)
    key = lambda r: tuple(r[k] for k in ('target', 'layer', 'dtype', 'learner', 'method', 'k'))
    for r in metrics:
        by_key[key(r)].append(r)
    details = {}
    for r in agg:
        b = budget_fields(recipe(r, p, coverage))
        seeds = by_key[key(r)]
        assert len(seeds) == 3
        assert math.isclose(statistics.mean(x['mape_pct'] for x in seeds), r['mean_mape_pct'])
        assert math.isclose(statistics.mean(x['actual_train_labels'] for x in seeds), r['actual_train_labels'])
        fits = [d for d in diagnostics if d.get('phase') != 'selected_refit'
                and d['layer'] == r['layer'] and d['dtype'] == r['dtype']
                and d['learner'] == r['learner'] and d['method'] == r['method']
                and d['k'] == r['k'] and d['target'] == ('shared' if r['method'] == 'component_sum' else r['target'])]
        prior = [d for d in diagnostics if d.get('phase') != 'selected_refit'
                 and d['layer'] == r['layer'] and d['dtype'] == r['dtype']
                 and d['learner'] == r['learner'] and
                 (d['method'] == 'source_prior_validation' or
                  (d['method'] == 'source_prior_train' and d['k'] == r['k']))]
        details[key(r)] = dict(r, **b, seed_mape_min=min(x['mape_pct'] for x in seeds),
            seed_mape_max=max(x['mape_pct'] for x in seeds),
            seed_p90_max=max(x['p90_pct'] for x in seeds),
            cpu_candidate_fit_seconds=sum(x.get('fit_seconds', 0) for x in fits),
            cpu_prior_prediction_seconds=sum(x.get('prediction_seconds', 0) for x in prior)
            if r['method'] in ('zero', 'constant_ratio', 'residual', 'prior_direct') else 0)
    table('layer_all_configurations', list(details.values()))
    equal, chosen, curve = [], {}, []
    conditions = [(t, l, d) for t in ('standard', 'decomposed')
                  for l in COMPONENTS for d in ('fp16', 'fp32')]
    for target, layer, dtype in conditions:
        rows = [r for r in agg if (r['target'], r['layer'], r['dtype']) == (target, layer, dtype)]
        direct = next(r for r in selected if (r['target'], r['layer'], r['dtype'], r['selection_scope'])
                      == (target, layer, dtype, 'direct'))
        targets = [('direct_mape', direct['mean_mape_pct'], None),
                   ('direct_mape_and_p90', direct['mean_mape_pct'], direct['mean_p90_pct']),
                   ('mape_10', 10, None), ('mape_15', 15, None), ('mape_20', 20, None),
                   ('legacy_15_30', 15, 30)]
        for method in FAMILIES:
            allowed = ('zero', 'constant_ratio', 'prior_direct', 'residual')
            if method == 'prior_with_direct_fallback':
                allowed += ('direct',)
            pool = [r for r in rows if r['method'] == method or (method in FAMILIES[-2:]
                    and r['method'] in allowed)]
            for goal, mape, p90 in targets:
                best = cheapest(pool, mape, p90)
                base = dict(goal=goal, target=target, layer=layer, dtype=dtype, method_family=method,
                            target_mape_pct=mape, target_p90_pct=p90)
                if best:
                    enriched = details[key(best)]
                    row = dict(base, status='ATTAINED', **enriched,
                        all_three_seeds_attain=enriched['seed_mape_max'] <= mape + 1e-12
                        and (p90 is None or enriched['seed_p90_max'] <= p90 + 1e-12))
                    chosen[goal, target, layer, dtype, method] = best
                else:
                    row = dict(base, status='NOT_ATTAINED_ON_OBSERVED_GRID')
                equal.append(row)
            if method in METHODS:
                for k in sorted({r['k'] for r in pool}):
                    valid = [r for r in pool if r['k'] == k and r['failures'] == 0]
                    best = min(valid, key=lambda r: (r['mean_mape_pct'], r['mean_p90_pct'], r['learner']))
                    curve.append(details[key(best)])
    table('layer_equivalent_accuracy', equal)
    table('layer_label_learning_curves', curve)
    portfolios = []
    for goal in ('direct_mape', 'direct_mape_and_p90', 'mape_10', 'mape_15', 'mape_20', 'legacy_15_30'):
        for target in ('standard', 'decomposed', 'combined'):
            target_names = ('standard', 'decomposed') if target == 'combined' else (target,)
            for method in FAMILIES:
                selected_rows = [chosen[goal, t, l, d, method] for t in target_names for l in COMPONENTS for d in ('fp16', 'fp32')
                                 if (goal, t, l, d, method) in chosen]
                total = 8 * len(target_names)
                row = dict(goal=goal, target=target, method_family=method, attained_conditions=len(selected_rows),
                           total_conditions=total)
                if len(selected_rows) == total:
                    recipes = [x for r in selected_rows for x in recipe(r, p, coverage)]
                    row.update(budget_fields(recipes), cpu_candidate_fit_seconds=sum(
                        details[key(r)]['cpu_candidate_fit_seconds'] for r in selected_rows),
                        cpu_prior_prediction_seconds=sum(details[key(r)]['cpu_prior_prediction_seconds'] for r in selected_rows))
                    if target == 'combined':
                        row['cpu_prior_prediction_seconds'] = None
                portfolios.append(row)
    table('layer_portfolios', portfolios)
    # Shared component fits appear once in the original diagnostics.
    cpu = defaultdict(lambda: dict(calls=0, fit_seconds=0., prediction_seconds=0.))
    for d in diagnostics:
        c = cpu[d.get('phase', 'search'), d['method']]
        c['calls'] += 1
        c['fit_seconds'] += d.get('fit_seconds', 0.)
        c['prediction_seconds'] += d.get('prediction_seconds', 0.)
    table('layer_cpu_by_method', [dict(phase=k[0], method=k[1], **v) for k, v in cpu.items()])
    cr = read(RUN / 'cost_report.json')
    assert costs.label_budget(read(RUN / 'selected_label_budget.json')) == cr['selected_recipe_label_budget']
    assert costs.label_budget(read(RUN / 'protocol_access_budget.json')) == cr['all_protocol_access_label_budget']
    assert math.isclose(sum(v['fit_seconds'] for v in cpu.values()), cr['fitting']['model_fit_seconds'])
    assert cr['selected_recipe_label_budget']['all_seed_union']['aggregated_labels'] == 5994
    return equal, portfolios, cr, curve


def observed_layer_costs(expected):
    campaign = ROOT / 'analysis/layer_full_result_2026-11-10/raw'
    for path in sorted(campaign.glob('*')):
        if path.is_file():
            watch(path)
    observed = costs.collection_costs(campaign)
    assert observed == expected['collection']
    table('layer_collection_attempts', observed['attempt_details'])
    rows = []
    for split in ('train', 'validation', 'test'):
        attempts = [r for r in observed['attempt_details'] if r['split'] == split]
        failed = [r for r in attempts if r['collection'] != 'COMPLETE']
        rows.append(dict(split=split, **observed['by_split'][split],
                         failed_attempts=len(failed),
                         failed_seconds=sum(r['controlled_collection_wall_seconds'] for r in failed),
                         failed_records=sum(r['recorded'] for r in failed)))
    table('layer_collection_splits', rows)
    return observed


def gemm_costs():
    sessions = []
    for stage, root in [('legacy', OLD_RAW), ('expansion', EXP / 'raw/expansion_10000')]:
        for path in sorted((root / 'sessions').glob('*/session.json')):
            r = read(path)
            sessions.append(dict(stage=stage, session=path.parent.name, source=str(path.relative_to(ROOT)), **r))
    assert len({r['session'] for r in sessions}) == len(sessions) == 7
    table('gemm_worker_sessions', sessions)
    # Project accounting fields only; latency labels and test errors never enter this calculation.
    parts = []
    for stage, root in [('legacy_final_retained', OLD_RAW), ('expansion', EXP / 'raw/expansion_10000')]:
        db = watch(root / 'results.sqlite3')
        for p in root.glob('results.sqlite3-*'):
            watch(p)
        with sqlite3.connect(db.as_uri() + '?mode=ro&immutable=1', uri=True) as connection:
            query = """SELECT json_extract(record,'$.task.split'), json_extract(record,'$.task.phase'),
                json_extract(record,'$.task.dtype'), json_extract(record,'$.task.mode'),
                json_extract(record,'$.session'), count(*),
                sum(json_extract(record,'$.task_wall_seconds')),
                count(json_extract(record,'$.task_wall_seconds')) FROM attempts GROUP BY 1,2,3,4,5"""
            for split, phase, dtype, mode, session, n, seconds, known in connection.execute(query):
                assert n == known and seconds >= 0
                parts.append(dict(stage=stage, split=split, phase=phase, dtype=dtype, mode=mode,
                                  session=session, records=n, task_wall_seconds=seconds, source=str(db.relative_to(ROOT))))
    table('gemm_task_costs_by_split_mode', parts)
    verification = read(EXP / 'verification.json')
    expansion_sum = sum(r['task_wall_seconds'] for r in parts if r['stage'] == 'expansion')
    assert math.isclose(expansion_sum, verification['task_wall_sum_seconds'], rel_tol=1e-11)
    assert sum(r['records'] for r in parts if r['stage'] == 'legacy_final_retained') == 113382
    assert sum(r['records'] for r in parts if r['stage'] == 'expansion') == 180000
    return sessions, parts


def gemm_curves_and_cpu():
    fields = ('mean_mape_pct', 'mean_p90_pct', 'failures', 'n')
    expanded = csv_read(EXP / 'fit_v1/aggregate_metrics.csv')
    old = csv_read(LEGACY / 'aggregate_metrics.csv')
    old_protocol = read(LEGACY / 'frozen_protocol.json')
    exp_protocol = read(EXP / 'fit_v1/frozen_protocol.json')
    def union_budget(cohort, n, validation):
        union = set()
        for seed in old_protocol['seeds']:
            if cohort == 'legacy_716':
                order = old_protocol['a_orders'][str(seed)] + old_protocol['c_orders'][str(seed)]
                training = order[:int(n)]
            else:
                training = exp_protocol['old_train_orders'][str(seed)] + exp_protocol['new_train_orders'][str(seed)][:int(n)-3345]
            assert len(training) == int(n)
            union.update(map(tuple, training))
        return dict(train_shapes_union=len(union), recipe_labels_union=len(union) + validation,
                    recipe_records_union=3 * (len(union) + validation))
    for rows in (expanded, old):
        for r in rows:
            for k in fields:
                r[k] = float(r[k])
    curves, targets = [], []
    for cohort, rows, v in [('legacy_716', [r for r in old if r['rq'] == 'RQ1'], 716),
                             ('pooled_1611', [r for r in expanded if r['validation'] == 'pooled'], 1611)]:
        for dtype in ('fp16', 'fp32'):
            for mode in ('allocate', 'prealloc', 'graph'):
                group = [r for r in rows if r['dtype'] == dtype and r['mode'] == mode]
                for n in sorted({r['n'] for r in group}):
                    best = min([r for r in group if r['n'] == n],
                               key=lambda r: (r['mean_mape_pct'], r['mean_p90_pct'], r['learner']))
                    curves.append(dict(cohort=cohort, validation_shapes=v, **best, **union_budget(cohort, n, v),
                        recipe_labels_per_seed=n + v, recipe_records_per_seed=3 * (n + v)))
                for goal, mape, p90 in [('mape_10', 10, None), ('mape_15', 15, None),
                                       ('mape_20', 20, None), ('legacy_15_30', 15, 30)]:
                    valid = [r for r in group if meets(r, mape, p90)]
                    base = dict(cohort=cohort, dtype=dtype, mode=mode, goal=goal,
                                validation_shapes=v, target_mape_pct=mape, target_p90_pct=p90)
                    if valid:
                        best = min(valid, key=lambda r: (r['n'], r['mean_mape_pct'], r['mean_p90_pct'], r['learner']))
                        targets.append(dict(base, status='ATTAINED', **best, **union_budget(cohort, best['n'], v),
                            recipe_labels_per_seed=best['n'] + v, recipe_records_per_seed=3 * (best['n'] + v)))
                    else:
                        targets.append(dict(base, status='NOT_ATTAINED_ON_OBSERVED_GRID'))
    assert len(curves) == 6 * (len(old_protocol['rq1_sizes']) + len(exp_protocol['added_sizes']))
    table('gemm_learning_curves', curves)
    table('gemm_target_budgets', targets)
    expfits = csv_read(EXP / 'fit_v1/fit_diagnostics.csv')
    oldfits = csv_read(LEGACY / 'fit_diagnostics.csv')
    refs = csv_read(EXP / 'fit_v1/accuracy_first_model_refs.csv')
    source_refs = [r for r in refs if r['mode'] == 'allocate']
    assert len(source_refs) == 6
    match = lambda r, ref: all(r[k] == ref[k] for k in ('learner', 'target', 'dtype', 'mode', 'n', 'seed'))
    source_fits = [r for r in expfits if any(match(r, ref) for ref in source_refs)]
    assert len(source_fits) == 6
    table('gemm_selected_prior_fit_calls', source_fits)
    cpu = []
    for stage, folder, fits in [('legacy_search', LEGACY, oldfits), ('expansion_search', EXP / 'fit_v1', expfits)]:
        complete = read(folder / 'execution_completed.json')
        total = sum(float(r['fit_seconds']) for r in fits)
        assert math.isclose(total, complete['total_fit_seconds'], rel_tol=1e-10)
        cpu.append(dict(stage=stage, wall_seconds=complete['wall_seconds'], fit_seconds=total,
                        scope='All recorded search fits; not six-weight recipe'))
    source_fit_seconds = sum(float(r['fit_seconds']) for r in source_fits)
    cpu.append(dict(stage='selected_six_prior_fit_recipe', wall_seconds=None,
                    fit_seconds=source_fit_seconds, scope='Subset of expansion_search; do not add twice'))
    table('gemm_cpu_costs', cpu)
    return curves, cpu, source_fit_seconds


def break_even_tables(equal, portfolios, collection, sessions, parts, source_cpu):
    source_labels = (11498 + 1611) * 2
    full_source_labels = source_labels * 3
    session_seconds = sum(r['elapsed_seconds'] for r in sessions)
    session_records = sum(r['new_records'] for r in sessions)
    dev = [r for r in collection['attempt_details'] if r['split'] != 'test']
    dev_seconds = sum(r['controlled_collection_wall_seconds'] for r in dev)
    completed_records = sum(r['recorded'] for r in dev if r['collection'] == 'COMPLETE')
    layer_seconds_per_label = dev_seconds / (completed_records / 3)
    old_recipe_seconds = sum(r['task_wall_seconds'] for r in parts if r['mode'] == 'allocate'
        and r['phase'] == 'main' and r['split'] in ('legacy_train', 'legacy_validation', 'expansion_train', 'expansion_validation'))
    source_scenarios = [('reused_prior_sunk', 0, 0),
                        ('allocate_label_recipe', source_labels, old_recipe_seconds),
                        ('six_condition_label_recipe', full_source_labels, None),
                        ('historical_collection_all_sessions', session_records / 3, session_seconds)]
    rows, sensitivity = [], []
    for target in ('standard', 'decomposed', 'combined'):
        for goal in ('direct_mape', 'direct_mape_and_p90', 'mape_10', 'mape_15', 'mape_20', 'legacy_15_30'):
            direct = next(r for r in portfolios if (r['target'], r['goal'], r['method_family']) == (target, goal, 'direct'))
            if direct['attained_conditions'] != direct['total_conditions']:
                continue
            for method in ('prior_direct', 'residual', 'best_prior', 'prior_with_direct_fallback'):
                reuse = next(r for r in portfolios if (r['target'], r['goal'], r['method_family']) == (target, goal, method))
                if reuse['attained_conditions'] != reuse['total_conditions']:
                    continue
                for scenario, source, seconds in source_scenarios:
                    rows.append(dict(target=target, goal=goal, method=method, accounting='aggregate_labels_union',
                        source_scenario=scenario, source_cost=source,
                        direct_per_task=direct['total_labels_union'], reuse_per_task=reuse['total_labels_union'],
                        **breakeven(source, direct['total_labels_union'], reuse['total_labels_union'])))
                if target == 'combined':
                    continue
                d = direct['cpu_candidate_fit_seconds']
                r = reuse['cpu_candidate_fit_seconds'] + reuse['cpu_prior_prediction_seconds']
                rows.append(dict(target=target, goal=goal, method=method, accounting='cpu_logged_fit_plus_prior_seconds',
                    source_scenario='six_prior_fit_recipe', source_cost=source_cpu,
                    direct_per_task=d, reuse_per_task=r, **breakeven(source_cpu, d, r)))
                for multiplier in (0.5, 1., 2.):
                    seconds_per_label = layer_seconds_per_label * multiplier
                    d = direct['total_labels_union'] * seconds_per_label
                    r = reuse['total_labels_union'] * seconds_per_label
                    sensitivity.append(dict(target=target, goal=goal, method=method,
                        scenario='historical_source_worker_vs_proportional_layer_controlled',
                        marginal_multiplier=multiplier, source_seconds=session_seconds,
                        allocated_layer_seconds_per_label=seconds_per_label,
                        direct_allocated_seconds=d, reuse_allocated_seconds=r,
                        **breakeven(session_seconds, d, r)))
    table('break_even_conditional', rows)
    table('gpu_time_allocation_sensitivity', sensitivity)
    individual = []
    for r in equal:
        if r['goal'] != 'direct_mape' or r['status'] != 'ATTAINED' or r['method_family'] not in ('prior_direct', 'residual', 'best_prior'):
            continue
        d = next(x for x in equal if x['goal'] == r['goal'] and x['target'] == r['target']
                 and x['layer'] == r['layer'] and x['dtype'] == r['dtype'] and x['method_family'] == 'direct')
        individual.append(dict(target=r['target'], layer=r['layer'], dtype=r['dtype'], method=r['method_family'],
            source_labels_one_dtype=source_labels // 2, direct_labels=d['total_labels_union'],
            reuse_labels=r['total_labels_union'], **breakeven(source_labels // 2, d['total_labels_union'], r['total_labels_union'])))
    table('break_even_single_condition', individual)
    return dict(selected_allocate_source_train_labels=11498 * 2,
                selected_allocate_source_validation_labels=1611 * 2,
                selected_allocate_source_total_labels=source_labels,
                selected_allocate_source_raw_records=source_labels * 3,
                full_six_condition_source_labels=full_source_labels,
                gemm_recorded_worker_seconds=session_seconds, gemm_all_session_records=session_records,
                gemm_allocate_train_validation_retained_task_seconds=old_recipe_seconds,
                layer_development_controlled_seconds=dev_seconds,
                layer_development_completed_records=completed_records,
                layer_failure_inclusive_allocated_seconds_per_label=layer_seconds_per_label,
                observed_like_for_like_time_break_even='UNKNOWN: narrow direct/reuse collections not executed; timing scopes differ',
                boundary='Historical GEMM worker total includes fresh_holdout, diagnostics, replaced records and preheat. '
                'Allocate task sum includes rejected/partial conditions and omits shared worker overhead. '
                'Proportional layer unit allocates failed attempts and shared setup to completed labels. '
                'Labels are heterogeneous, so time scenarios are assumptions rather than measured savings.')


def figures(curves, gemm_curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(direct='#303030', prior_direct='#1f77b4', residual='#d55e00',
                  component_sum='#009e73', zero='#999999', constant_ratio='#9467bd')
    for target in ('standard', 'decomposed'):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), layout='constrained')
        for i, dtype in enumerate(('fp16', 'fp32')):
            for j, layer in enumerate(COMPONENTS):
                ax = axes[i, j]
                for method in METHODS:
                    rs = sorted([r for r in curves if (r['target'], r['layer'], r['dtype'], r['method'])
                                 == (target, layer, dtype, method)], key=lambda r: r['total_labels_union'])
                    ax.plot([r['total_labels_union'] for r in rs], [r['mean_mape_pct'] for r in rs],
                            marker='o', markersize=3, linewidth=1.3, color=colors[method], label=method,
                            linestyle='--' if method in ('zero', 'constant_ratio') else '-')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_title(layer.upper() + ' ' + dtype.upper())
                ax.grid(alpha=.2)
                ax.set_xlabel('Train + validation labels')
                ax.set_ylabel('Validation MAPE (%)')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=6, frameon=False)
        fig.suptitle(target.capitalize() + ': distinct labels across 3 seeds; best learner at each k')
        for ext in ('png', 'pdf'):
            fig.savefig(HERE / 'figures' / f'layer_{target}.{ext}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), layout='constrained')
    for i, dtype in enumerate(('fp16', 'fp32')):
        for j, mode in enumerate(('allocate', 'prealloc', 'graph')):
            ax = axes[i, j]
            for cohort, color in [('legacy_716', '#555555'), ('pooled_1611', '#d55e00')]:
                rs = sorted([r for r in gemm_curves if (r['dtype'], r['mode'], r['cohort']) == (dtype, mode, cohort)], key=lambda r:r['n'])
                ax.plot([r['n'] for r in rs], [r['mean_mape_pct'] for r in rs], marker='o', markersize=3, label=cohort, color=color)
            ax.axhline(15, color='#aaa', linestyle=':', linewidth=1)
            ax.set_xscale('log'); ax.grid(alpha=.2)
            ax.set_title(dtype.upper() + ' / ' + mode)
            ax.set_xlabel('Training shapes per seed'); ax.set_ylabel('Validation MAPE (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('GEMM learning curves: compare points only within the same validation cohort')
    for ext in ('png', 'pdf'):
        fig.savefig(HERE / 'figures' / f'gemm_curves.{ext}', dpi=180)
    plt.close(fig)


def main():
    read(HERE / 'policy.json')
    watch(FIT / 'costs.py')
    final = ROOT / 'analysis/final_test_2026-11-11'
    prior_hashes = read(final / 'preflight.json')['original_hashes']
    for name, digest in prior_hashes.items():
        assert sha(watch(ROOT / name)) == digest, 'Frozen input mismatch: ' + name
    final_manifest = read(final / 'artifact_manifest.json')
    for name, digest in final_manifest['artifacts'].items():
        assert sha(watch(final / name)) == digest, 'Final artifact mismatch: ' + name
    freeze = watch(RUN / 'selection_frozen.json')
    assert sha(freeze) == 'beed9b4314c3f258f0fcd6f86d19b15d271fac2d1abe705875aa50a94e99cbd4'
    print('PASS: frozen model and final evaluation hashes', flush=True)
    equal, portfolios, cr, curves = layer_analysis()
    collection = observed_layer_costs(cr)
    print('PASS: layer budgets and all 33 collection ledgers', flush=True)
    sessions, parts = gemm_costs()
    print('PASS: GEMM session and task accounting', flush=True)
    gemm_curves, cpu, source_cpu = gemm_curves_and_cpu()
    summary = break_even_tables(equal, portfolios, collection, sessions, parts, source_cpu)
    summary.update(layer_cpu=cr['fitting'], gemm_cpu=cpu,
                   selected_layer_labels=cr['selected_recipe_label_budget']['all_seed_union'],
                   full_development_labels=cr['all_protocol_access_label_budget']['all_seed_union'])
    save('summary.json', summary)
    figures(curves, gemm_curves)
    for name, digest in INPUTS.items():
        assert sha(ROOT / name) == digest, 'Input changed: ' + name
    save('input_manifest.json', INPUTS)
    save('verification.json', dict(status='PASS', inputs_unchanged=len(INPUTS),
        original_final_preflight_files=len(prior_hashes), final_artifacts_checked=len(final_manifest['artifacts']),
        layer_configurations=len(read(RUN / 'aggregate_metrics.json')),
        learning_curve_points=len(curves), equivalent_accuracy_rows=len(equal),
        layer_attempts=33, gemm_sessions=len(sessions), refitting=False,
        model_selection_changed=False, test_metrics_used_for_efficiency=False))
    print('PASS: analysis, tables, figures and source preservation', flush=True)


if __name__ == '__main__':
    main()
