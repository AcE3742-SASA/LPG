#!/usr/bin/env python3
"""Render frozen layer results and cost tables without fitting or opening test data."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import data

LAYERS = ('linear', 'ffn', 'conv1', 'conv3')
METHODS = ('zero', 'constant_ratio', 'direct', 'prior_direct', 'residual', 'component_sum')


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)


def summarize_test(rows):
    groups = defaultdict(list)
    keys = ('target', 'layer', 'dtype', 'selection_scope', 'learner', 'method', 'k')
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, scores in sorted(groups.items()):
        result.append(dict(zip(keys, key), seeds=len(scores),
            mean_mape_pct=statistics.mean(row['mape_pct'] for row in scores),
            mean_p90_pct=statistics.mean(row['p90_pct'] for row in scores),
            failures=sum(row['failures'] for row in scores),
            minimum_test_groups=min(row['planned'] for row in scores),
            maximum_test_groups=max(row['planned'] for row in scores)))
    return result


def curves(rows, selected, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = []
    for target in sorted({row['target'] for row in rows}):
        for dtype in sorted({row['dtype'] for row in rows if row['target'] == target}):
            fig, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
            for column, layer in enumerate(LAYERS):
                candidates = [row for row in rows if
                              (row['target'], row['dtype'], row['layer']) == (target, dtype, layer)]
                for method in METHODS:
                    points = []
                    for k in sorted({row['k'] for row in candidates if row['method'] == method}):
                        valid = [row for row in candidates if row['method'] == method and row['k'] == k
                                 and row['failures'] == 0 and math.isfinite(row['mean_mape_pct'])
                                 and math.isfinite(row['mean_p90_pct'])]
                        if valid:
                            points.append(min(valid, key=lambda row:
                                (row['mean_mape_pct'], row['mean_p90_pct'], row['learner'])))
                    for index, metric in enumerate(('mean_mape_pct', 'mean_p90_pct')):
                        if points:
                            axes[index, column].plot([row['k'] for row in points],
                                [row[metric] for row in points], marker='o', markersize=3, label=method)
                chosen = [row for row in selected if row['selection_scope'] == 'overall' and
                          (row['target'], row['dtype'], row['layer']) == (target, dtype, layer)]
                for index, metric in enumerate(('mean_mape_pct', 'mean_p90_pct')):
                    axis = axes[index, column]
                    for row in chosen:
                        axis.scatter(row['k'], row[metric], color='black', marker='*', s=90, zorder=10)
                    axis.set_xscale('symlog', linthresh=5)
                    ticks = sorted({row['k'] for row in candidates})
                    axis.set_xticks(ticks, labels=[str(k) for k in ticks])
                    axis.tick_params(axis='x', labelsize=8)
                    axis.set_title(layer)
                    axis.set_ylabel('Mean MAPE (%)' if index == 0 else 'Mean p90 error (%)')
                    axis.set_xlabel('Planned training groups')
                    axis.grid(alpha=.2)
            handles, labels = [], []
            for axis in axes.flat:
                hh, ll = axis.get_legend_handles_labels()
                for handle, label in zip(hh, ll):
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)
            fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
            fig.suptitle(f'{target} / {dtype}: validation learning curves\n'
                         'Best validation learner per method and budget; star = overall selection', fontsize=12)
            fig.tight_layout(rect=(0, .09, 1, .91))
            name = f'validation_{target}_{dtype}'
            for extension in ('png', 'pdf'):
                fig.savefig(destination / f'{name}.{extension}', dpi=160)
            plt.close(fig)
            figures.append(name)
    return figures


def build(run, destination, evaluation=None):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if destination == run or run in destination.parents:
        raise ValueError('Keep reports outside the frozen run directory')
    freeze = data.check_freeze(run / 'frozen_protocol.json', run / 'selection_frozen.json')
    required = ('aggregate_metrics.json', 'selected_configurations.json', 'cost_report.json')
    if not set(required) <= set(freeze['artifact_hashes']):
        raise ValueError('Report inputs must be included in the frozen artifact hashes')
    aggregate = read(run / 'aggregate_metrics.json')
    selected = read(run / 'selected_configurations.json')
    costs = read(run / 'cost_report.json')
    if not aggregate or not selected:
        raise ValueError('Frozen result tables are empty')
    test_summary, completion = [], None
    if evaluation is not None:
        evaluation = Path(evaluation).resolve()
        completion = read(evaluation / 'evaluation_completed.json')
        if completion['freeze_sha256'] != data.sha(run / 'selection_frozen.json') or completion['model_selection_changed'] is not False:
            raise ValueError('Evaluation belongs to a different frozen selection')
        hashes = completion.get('artifact_hashes', {})
        if 'test_metrics.json' not in hashes:
            raise ValueError('Evaluation metrics must have completion artifact hashes')
        for name, digest in hashes.items():
            data.safe_name(name)
            path = evaluation / name
            if path.is_symlink() or evaluation not in path.resolve().parents or data.sha(path) != digest:
                raise ValueError('Evaluation output checksum mismatch')
        test_summary = summarize_test(read(evaluation / 'test_metrics.json'))
    destination.mkdir(parents=True, exist_ok=False)
    csv_write(destination / 'learning_curves.csv', aggregate)
    summary_fields = ('target', 'layer', 'dtype', 'selection_scope', 'learner', 'method', 'k',
                      'mean_mape_pct', 'mean_p90_pct', 'failures', 'actual_train_groups')
    csv_write(destination / 'selected_summary.csv', [{key: row[key] for key in summary_fields} for row in selected])
    collection = costs['collection']
    csv_write(destination / 'collection_costs.csv',
              [dict(split=split, **row) for split, row in collection['by_split'].items()] +
              [dict(split='all_attempts', **collection['all_attempts'])])
    label_rows = []
    for name in ('selected_recipe_label_budget', 'all_protocol_access_label_budget'):
        budget = costs[name]
        label_rows.extend(dict(budget=name, seed=seed, **row) for seed, row in budget['per_seed'].items())
        label_rows.append(dict(budget=name, seed='all_seed_union', **budget['all_seed_union']))
    csv_write(destination / 'label_budgets.csv', label_rows)
    csv_write(destination / 'fitting_costs.csv', [costs['fitting']])
    if test_summary:
        csv_write(destination / 'test_summary.csv', test_summary)
    figures = curves(aggregate, selected, destination)
    failed = sum(row['failures'] > 0 for row in aggregate)
    observed = collection['all_attempts']['observed_controlled_wall_seconds']
    complete = collection['all_attempts']['verified_complete_attempt_sum_seconds']
    lines = ['# Layer prediction results', '',
        f'Frozen run: `{run}`.', '',
        'Development results use the predeclared validation cohort. Each curve point is the lowest validation MAPE learner within one method and budget; p90 uses that same learner. Black stars show the overall frozen selection.', '',
        f'Aggregate configurations: {len(aggregate)}. Configurations with prediction failures: {failed}. Failed configurations remain in the CSV and are excluded from selection and curves.', '',
        '## Result tables', '',
        '- [All validation learning curves](learning_curves.csv)',
        '- [Selected recipes](selected_summary.csv)',
        '- [Collection costs](collection_costs.csv)',
        '- [Distinct label budgets](label_budgets.csv)',
        '- [CPU fit costs](fitting_costs.csv)', '',
        '## Cost interpretation', '',
        f'Observed controlled-collection wall across recorded attempts: {observed:.3f} seconds. '
        f'Verified sum for all listed attempts: {complete if complete is not None else "UNKNOWN"}.', '',
        'Retry attempts and shared setup are included in collection ledgers. Wrapper setup, ZIP packaging and campaign idle gaps remain outside the recorded sum. Test collection is listed separately. Selected-recipe label budgets are retrospective and distinct from the full development search access. A smaller label budget does not remove the measured bundled collection cost.', '',
        f'Common GEMM source cost: {"UNKNOWN" if costs.get("common_source_cost") is None else json.dumps(costs["common_source_cost"], sort_keys=True)}. '
        f'Break-even status: {costs["break_even"]["status"]}. CPU fit time and GPU collection wall are kept separate.', '',
        '## Validation figures', '']
    for name in figures:
        lines += [f'![{name}]({name}.png)', '', f'[PDF]({name}.pdf)', '']
    lines += ['## Final evaluation', '']
    if test_summary:
        lines += ['[Fixed-selection test summary](test_summary.csv). Mean p90 is the mean of seed-specific p90 values. '
                  'The evaluation uses the previously selected method and budget; the report performs no selection.', '',
                  f'Evaluation source: `{evaluation}`.', '']
    else:
        lines += ['Final evaluation is pending. This report reads frozen development outputs only.', '']
    (destination / 'report.md').write_text('\n'.join(lines))
    (destination / 'report_manifest.json').write_text(json.dumps(dict(
        source_run=str(run), selection_freeze_sha256=data.sha(run / 'selection_frozen.json'),
        evaluation_source=str(evaluation) if evaluation else None,
        output_hashes={path.name: data.sha(path) for path in sorted(destination.iterdir())}), indent=2) + '\n')
    print('PASS: frozen result tables, cost tables and figures saved')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path)
    args = parser.parse_args()
    build(args.run, args.output, args.evaluation)


if __name__ == '__main__':
    main()
