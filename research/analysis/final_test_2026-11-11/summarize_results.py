"""Summarize preselected final-test models without selecting or fitting models."""
from collections import defaultdict
from pathlib import Path
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'analysis/layer_full_fit_2026-11-10_v5'))
import gemm_final as final
from verify_results import independent_metrics

OUT = HERE / 'summary'
GEMM = ('rq', 'dtype', 'mode', 'variant', 'learner', 'method', 'n')
LAYER = ('target', 'layer', 'dtype', 'selection_scope', 'learner', 'method', 'k')


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[k]) for k in fields)].append(row)
    result = []
    for key, members in groups.items():
        assert len(members) == 3 and len({int(r['seed']) for r in members}) == 3
        assert len({int(r['planned']) for r in members}) == 1
        info = dict(zip(fields, key), test_groups=int(members[0]['planned']), seeds=3,
                    failures=sum(int(r['failures']) for r in members))
        for metric in ('mape_pct', 'p90_pct', 'relative_rmse', 'mae_us'):
            values = [float(r[metric]) for r in members]
            info['mean_' + metric] = statistics.mean(values)
            info['min_seed_' + metric] = min(values)
            info['max_seed_' + metric] = max(values)
        result.append(info)
    return result


def bootstrap(first, second, id_field):
    def group_errors(rows):
        errors = defaultdict(list)
        for r in rows:
            actual, predicted = float(r['actual_ms']), float(r['predicted_ms'])
            errors[r[id_field]].append((int(r['seed']), actual, abs(predicted - actual) / actual * 100))
        assert all(len(v) == 3 and len({r[0] for r in v}) == 3 for v in errors.values())
        return {k: sorted(v) for k, v in errors.items()}
    a, b = group_errors(first), group_errors(second)
    assert set(a) == set(b)
    assert all([r[1] for r in a[k]] == [r[1] for r in b[k]] for k in a)
    delta = np.array([np.mean([r[2] for r in a[k]]) - np.mean([r[2] for r in b[k]]) for k in sorted(a)])
    rng = np.random.default_rng(20261111)
    samples = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
    return dict(groups=len(delta), delta_mape_pp=float(delta.mean()),
                ci95_low_pp=float(np.quantile(samples, .025)), ci95_high_pp=float(np.quantile(samples, .975)),
                seed_unit='Mean of three seed errors per shape; resample shapes, 2000 draws.')


def main():
    synthetic = [dict(shape=str(group), seed=seed, actual_ms=1, predicted_ms=1.2)
                 for group in range(2) for seed in range(3)]
    check = bootstrap(synthetic, [dict(r, predicted_ms=1.1) for r in synthetic], 'shape')
    np.testing.assert_allclose([check[k] for k in ('delta_mape_pp', 'ci95_low_pp', 'ci95_high_pp')], 10)
    verification = final.read(HERE / 'independent_verification.json')
    assert verification['status'] == 'PASS'
    for name, digest in verification['artifact_hashes'].items():
        final.checked(HERE / name, digest)
    OUT.mkdir(exist_ok=True)
    expanded = summarize(final.rows(HERE / 'gemm_expanded/metrics.csv'), GEMM)
    legacy = summarize(final.rows(HERE / 'gemm_legacy_and_transfer/metrics.csv'), GEMM)
    layers = summarize(final.read(HERE / 'layers/test_metrics.json'), LAYER)
    dev_gemm = final.rows(ROOT / 'analysis/gemm_expansion_result_2026-11-06/fit_v1/accuracy_first_all_sizes.csv')
    dev_layer = final.read(ROOT / 'analysis/layer_full_fit_2026-11-10_v5/prepared_layer_full/selected_configurations.json')
    for row in expanded:
        if row['variant'] == 'selected':
            source = next(r for r in dev_gemm if (r['dtype'], r['mode']) == (row['dtype'], row['mode']))
            row['validation_mape_pct'] = float(source['mean_mape_pct'])
            row['validation_p90_pct'] = float(source['mean_p90_pct'])
    for row in layers:
        source = next(r for r in dev_layer if all(str(r[k]) == row[k] for k in LAYER))
        row['validation_mape_pct'] = source['mean_mape_pct']
        row['validation_p90_pct'] = source['mean_p90_pct']
        row['actual_train_groups'] = source['actual_train_groups']
    for name, rows in [('gemm_expanded', expanded), ('gemm_legacy_and_transfer', legacy), ('layers', layers)]:
        final.gemm.save_csv(OUT / (name + '.csv'), rows)
    ep = final.rows(HERE / 'gemm_expanded/predictions.csv')
    lp = final.rows(HERE / 'gemm_legacy_and_transfer/predictions.csv')
    yp = final.read(HERE / 'layers/test_predictions.json')
    pairs = []
    for dtype in final.gemm.DTYPES:
        for mode in final.gemm.MODES:
            a = [r for r in ep if (r['dtype'], r['mode'], r['variant']) == (dtype, mode, 'selected')]
            b = [r for r in lp if (r['rq'], r['dtype'], r['mode'], r['variant']) == ('RQ1', dtype, mode, 'selected')]
            pairs.append(dict(comparison='expanded_selected_minus_legacy_selected', dtype=dtype, mode=mode, target='', layer='',
                              **bootstrap(a, b, 'shape')))
        for mode in ('prealloc', 'graph'):
            group = [r for r in lp if (r['rq'], r['dtype'], r['mode']) == ('RQ4', dtype, mode)]
            pairs.append(dict(comparison='rq4_selected_minus_unadapted_source', dtype=dtype, mode=mode, target='', layer='',
                **bootstrap([r for r in group if r['variant'] == 'selected'],
                            [r for r in group if r['variant'] == 'baseline'], 'shape')))
    for target in ('standard', 'decomposed'):
        for layer in ('linear', 'ffn', 'conv1', 'conv3'):
            for dtype in ('fp16', 'fp32'):
                group = [r for r in yp if (r['target'], r['layer'], r['dtype']) == (target, layer, dtype)]
                pairs.append(dict(comparison='layer_overall_minus_direct', target=target, layer=layer, dtype=dtype, mode='',
                    **bootstrap([r for r in group if r['selection_scope'] == 'overall'],
                                [r for r in group if r['selection_scope'] == 'direct'], 'group_id')))
    final.gemm.save_csv(OUT / 'paired_group_bootstrap.csv', pairs)
    near = {final.gemm.shape_id(r['test_shape']) for r in final.read(HERE / 'preflight.json')['near_development_shapes']}
    assert near == {'216x304x80'}
    groups = defaultdict(list)
    for row in ep:
        if row['variant'] == 'selected' and row['shape'] not in near:
            groups[tuple(row[k] for k in GEMM + ('seed',))].append(row)
    sensitivity = []
    for key, rows in groups.items():
        assert len(rows) == 999
        sensitivity.append(dict(zip(GEMM + ('seed',), key),
            **independent_metrics([float(r['actual_ms']) for r in rows], [float(r['predicted_ms']) for r in rows])))
    final.gemm.save_csv(OUT / 'gemm_non_near_sensitivity.csv', summarize(sensitivity, GEMM))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained')
    selected_gemm = [r for r in expanded if r['variant'] == 'selected']
    selected_layer = [r for r in layers if r['target'] == 'standard' and r['selection_scope'] == 'overall']
    panels = [(selected_gemm, [r['dtype'].upper() + '\n' + r['mode'] for r in selected_gemm],
               'GEMM: 1,000 reserved test shapes'),
              (selected_layer, [dict(linear='Linear', ffn='FFN', conv1='Conv1x1', conv3='Conv3x3')[r['layer']]
                                + '\n' + r['dtype'].upper() for r in selected_layer],
               'Standard layers: 50 test shapes per condition; Conv3x3 FP32: 49')]
    for ax, (rows, labels, title) in zip(axes, panels):
        x = np.arange(len(rows))
        for offset, field, label, color in [(-.19, 'validation_mape_pct', 'Validation', '#a7b3c2'),
                                           (.19, 'mean_mape_pct', 'Final test', '#2369a1')]:
            bars = ax.bar(x + offset, [r[field] for r in rows], .36, label=label, color=color)
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 20)
        ax.set_ylabel('MAPE (%)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc='upper right')
    fig.suptitle('Frozen model selections: validation and final test', fontsize=14)
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / ('final_mape.' + extension), dpi=180)
    plt.close(fig)
    final.gemm.save_json(OUT / 'manifest.json', dict(created_utc=final.gemm.stamp(),
        script_sha256=final.sha(__file__),
        source_verification_sha256=final.sha(HERE / 'independent_verification.json'),
        selection_changed=False, artifacts={p.name: final.sha(p) for p in OUT.iterdir() if p.name != 'manifest.json'}))
    print('PASS: fixed-selection summaries, paired group intervals and comparison figure saved', flush=True)


if __name__ == '__main__':
    main()
