"""Summarize frozen development outputs without fitting or changing selections."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'layer_full_fit_2026-11-10_v5'
sys.path.insert(0, str(PACKAGE))
import data
from report import csv_write

KEY = ('target', 'layer', 'dtype', 'learner', 'method', 'k')
GROUP = KEY[:3]
PRIOR = ('zero', 'constant_ratio', 'prior_direct', 'residual')


def key(row, fields=KEY):
    return tuple(row[field] for field in fields)


def rank(row):
    return tuple(row[n] for n in ('mean_mape_pct', 'mean_p90_pct', 'actual_train_labels',
                                 'learner', 'method', 'k'))


def matched_pairs(aggregate):
    indexed = {key(row): row for row in aggregate}
    assert len(indexed) == len(aggregate)
    pairs, skipped = [], Counter()
    for row in aggregate:
        if row['method'] not in PRIOR:
            continue
        direct = indexed.get(key(dict(row, method='direct')))
        if direct is None:
            skipped[row['method'], 'no_same_k_direct'] += 1
            continue
        if row['failures'] or direct['failures']:
            skipped[row['method'], 'invalid_predictions'] += 1
            continue
        assert row['actual_train_groups'] == direct['actual_train_groups']
        assert row['actual_train_labels'] == direct['actual_train_labels']
        delta = row['mean_mape_pct'] - direct['mean_mape_pct']
        pairs.append(dict(**{n: row[n] for n in KEY},
            actual_train_groups=row['actual_train_groups'], actual_train_labels=row['actual_train_labels'],
            prior_mape_pct=row['mean_mape_pct'], direct_mape_pct=direct['mean_mape_pct'],
            prior_minus_direct_pp=delta,
            result='tie' if math.isclose(delta, 0.0, abs_tol=1e-12) else 'win' if delta < 0 else 'loss'))
    wins = []
    for target in sorted({r['target'] for r in aggregate}):
        for method in PRIOR:
            rows = [r for r in pairs if r['target'] == target and r['method'] == method]
            counts = Counter(r['result'] for r in rows)
            wins.append(dict(target=target, method=method, matched_configurations=len(rows),
                wins=counts['win'], ties=counts['tie'], losses=counts['loss'],
                mean_prior_minus_direct_pp=statistics.mean(r['prior_minus_direct_pp'] for r in rows) if rows else None))
    return pairs, wins, [dict(method=m, reason=r, count=n) for (m, r), n in sorted(skipped.items())]


def summarize(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError('Summary output must be outside the frozen run')
    freeze = data.check_freeze(run / 'frozen_protocol.json', run / 'selection_frozen.json')
    assert freeze['sealed_test_status'] == 'NOT_OPENED'
    names = ('aggregate_metrics', 'metrics', 'selected_configurations', 'validation_predictions',
             'fit_diagnostics', 'cost_report', 'implementation_diagnostics')
    assert all(name + '.json' in freeze['artifact_hashes'] for name in names)
    tables = {name: data.read(run / (name + '.json')) for name in names}
    aggregate, metrics, selected = (tables[n] for n in names[:3])
    protocol = data.read(run / 'frozen_protocol.json')
    seeds = set(protocol['seeds'])
    assert len(seeds) == 3
    indexed = {key(r): r for r in aggregate}
    assert len(indexed) == len(aggregate)
    by_seed = defaultdict(list)
    for row in metrics:
        by_seed[key(row)].append(row)
    assert set(by_seed) == set(indexed)
    for identity, rows in by_seed.items():
        assert len(rows) == 3 and {r['seed'] for r in rows} == seeds
        for metric in ('mape_pct', 'p90_pct'):
            assert math.isclose(statistics.mean(r[metric] for r in rows), indexed[identity]['mean_' + metric], rel_tol=1e-12)
        assert sum(r['failures'] for r in rows) == indexed[identity]['failures']
    prediction_ids, prediction_failures = defaultdict(set), Counter()
    for row in tables.pop('validation_predictions'):
        identity = key(row) + (row['seed'],)
        assert row['group_id'] not in prediction_ids[identity]
        prediction_ids[identity].add(row['group_id'])
        assert math.isfinite(row['actual_ms']) and row['actual_ms'] > 0
        if not math.isfinite(row['predicted_ms']) or row['predicted_ms'] <= 0:
            prediction_failures[identity] += 1
    for row in metrics:
        identity = key(row) + (row['seed'],)
        assert len(prediction_ids[identity]) == row['planned']
        assert prediction_failures[identity] == row['failures']
    assert len(prediction_ids) == len(metrics)
    chosen = {(key(r, GROUP), r['selection_scope']): r for r in selected}
    assert len(chosen) == len(selected) == 16 * 7
    groups = sorted({key(r, GROUP) for r in aggregate})
    assert len(groups) == 16
    for row in selected:
        assert all(row[n] == indexed[key(row)][n] for n in indexed[key(row)])
        assert set(map(int, row['training_group_ids_by_seed'])) == seeds
        assert set(map(int, row['model_files_by_seed'])) == seeds
        assert all(set(row['validation_group_ids']) == prediction_ids[key(row) + (seed,)] for seed in seeds)
    diagnostics = tables['fit_diagnostics']
    summaries = []
    for group in groups:
        views = [(scope, chosen[group, scope]) for scope in ('overall', 'direct', 'component_sum')]
        views.append(('best_prior_family', min((chosen[group, m] for m in PRIOR), key=rank)))
        for view, row in views:
            seed_rows = by_seed[key(row)]
            related = [d for d in diagnostics if all(d.get(n) == row[n] for n in ('layer','dtype','learner','method','k'))
                       and d.get('target') in (row['target'], 'shared')]
            summaries.append(dict(view=view, **{n: row[n] for n in KEY},
                mean_mape_pct=row['mean_mape_pct'], mean_p90_pct=row['mean_p90_pct'],
                mean_actual_train_groups=row['actual_train_groups'], mean_actual_train_labels=row['actual_train_labels'],
                validation_groups=len(row['validation_group_ids']), failures=row['failures'],
                seed_mape_min_pct=min(r['mape_pct'] for r in seed_rows),
                seed_mape_max_pct=max(r['mape_pct'] for r in seed_rows),
                seed_metrics=[{n: r[n] for n in ('seed','mape_pct','p90_pct','actual_train_groups','actual_train_labels')}
                              for r in sorted(seed_rows, key=lambda r: r['seed'])],
                search_convergence_warnings=sum(d.get('convergence_warnings', 0) for d in related if d.get('phase') != 'selected_refit'),
                selected_refit_convergence_warnings=sum(d.get('convergence_warnings', 0) for d in related if d.get('phase') == 'selected_refit')))
    pairs, wins, skipped = matched_pairs(aggregate)
    implementation_groups = defaultdict(list)
    for row in tables['implementation_diagnostics']:
        if row['split'] == 'validation':
            implementation_groups[row['layer'], row['dtype']].append(row)
    assert len(implementation_groups) == 8
    implementation_summary = []
    for (layer, dtype), rows in sorted(implementation_groups.items()):
        ids = [r['group_id'] for r in rows]
        assert len(ids) == len(set(ids))
        assert set(ids) == set(chosen[('standard', layer, dtype), 'overall']['validation_group_ids'])
        item = dict(split='validation', layer=layer, dtype=dtype, groups=len(rows))
        for ratio in ('decomposed_over_standard', 'component_sum_over_decomposed'):
            values = [r[ratio] for r in rows]
            assert all(math.isfinite(v) and v > 0 for v in values)
            item.update({ratio + '_median': statistics.median(values),
                         ratio + '_min': min(values), ratio + '_max': max(values)})
        implementation_summary.append(item)
    warning_rows = [{n: d.get(n) for n in (*KEY, 'seed', 'phase', 'epochs', 'convergence_warnings', 'warning_messages', 'components')}
                    for d in diagnostics if d.get('convergence_warnings', 0)]
    failures = [r for r in metrics if r['failures']]
    result = dict(source_run=str(run), selection_freeze_sha256=data.sha(run / 'selection_frozen.json'),
        original_selection_unchanged=True, sealed_test_status='NOT_OPENED',
        interpretation=dict(best_prior_family='Display-only minimum among four already frozen method selections; no new model selection.',
            matched_pairs='Same target, layer, dtype, learner and planned k; average MAPE over the same three seeds. Negative delta favors prior.',
            p90='Mean of seed-specific p90 values; seed ranges are descriptive, not confidence intervals.',
            label_budget='Planned k, retained groups and consumed labels are distinct. Recipe budgets are retrospective.'),
        counts=dict(aggregate_configurations=len(aggregate), seed_configurations=len(metrics),
            validation_prediction_rows=sum(map(len, prediction_ids.values())),
            failed_aggregate_configurations=sum(bool(r['failures']) for r in aggregate),
            failed_seed_configurations=len(failures), failed_prediction_rows=sum(r['failures'] for r in metrics),
            convergence_warning_diagnostic_entries=len(warning_rows),
            convergence_warning_count=sum(d['convergence_warnings'] for d in warning_rows)),
        selected_comparisons=summaries, matched_prior_direct_wins=wins, skipped_matched_pairs=skipped,
        validation_implementation_ratios=implementation_summary,
        convergence_warnings=warning_rows, failed_configurations=failures, costs=tables['cost_report'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'validation_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for filename, rows in (('selected_comparisons.csv', summaries), ('matched_prior_direct.csv', pairs),
                           ('matched_prior_direct_wins.csv', wins), ('convergence_warnings.csv', warning_rows),
                           ('failed_configurations.csv', failures),
                           ('validation_implementation_ratios.csv', implementation_summary)):
        csv_write(output / filename, rows)
    manifest = dict(source_selection_sha256=result['selection_freeze_sha256'],
                    output_hashes={p.name: data.sha(p) for p in sorted(output.iterdir())})
    (output / 'summary_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert data.sha(run / 'selection_frozen.json') == result['selection_freeze_sha256']
    print('PASS: frozen validation summary saved to ' + str(output))


def selfcheck():
    direct = dict(target='standard', layer='linear', dtype='fp16', learner='rf', method='direct',
                  k=5, mean_mape_pct=10.0, mean_p90_pct=20.0, actual_train_groups=5,
                  actual_train_labels=5, failures=0)
    pairs, wins, skipped = matched_pairs([direct, dict(direct, method='residual', mean_mape_pct=8.0),
                                         dict(direct, method='zero', k=0, actual_train_groups=0, actual_train_labels=0)])
    assert len(pairs) == 1 and pairs[0]['prior_minus_direct_pp'] == -2 and pairs[0]['result'] == 'win'
    assert next(r for r in wins if r['method'] == 'residual')['wins'] == 1
    assert skipped == [dict(method='zero', reason='no_same_k_direct', count=1)]
    try:
        matched_pairs([direct, direct])
    except AssertionError:
        pass
    else:
        raise AssertionError('Duplicate configuration accepted')
    print('PASS: summary self-check')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PACKAGE / 'prepared_layer_full')
    parser.add_argument('--output', type=Path, default=HERE / 'summary')
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    selfcheck() if args.selfcheck else summarize(args.run, args.output)
