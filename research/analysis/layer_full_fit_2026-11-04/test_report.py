"""Synthetic report smoke test; every result stays in a temporary directory."""
import json
from pathlib import Path
import tempfile
import unittest

import costs
import data
import report


def write(path, value):
    path.write_text(json.dumps(value))


def run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / 'fit'
        source.mkdir()
        (source / 'models').mkdir()
        write(source / 'models/synthetic.json', {'synthetic': True})
        write(source / 'frozen_protocol.json', {'synthetic': True})
        rows, selected = [], []
        for target in ('standard', 'decomposed'):
            for dtype in ('fp16', 'fp32'):
                for layer in report.LAYERS:
                    for method in ('direct', 'component_sum'):
                        for k in (5, 10):
                            row = dict(target=target, dtype=dtype, layer=layer, method=method, k=k,
                                learner='ridge', mean_mape_pct=10 / k, mean_p90_pct=20 / k,
                                failures=0, actual_train_groups=k)
                            rows.append(row)
                        selected.append(dict(row, selection_scope=method))
                    selected.append(dict(row, selection_scope='overall'))
        write(source / 'aggregate_metrics.json', rows)
        write(source / 'selected_configurations.json', selected)
        total = dict(attempts=1, observed_controlled_wall_seconds=50,
                     verified_complete_attempt_sum_seconds=50, missing_ledgers=[], unverified_ledgers=[])
        ledger = dict(collection=dict(by_split={name: total for name in ('train', 'validation', 'test')},
                                      all_attempts=total),
            selected_recipe_label_budget=costs.label_budget([]),
            all_protocol_access_label_budget=costs.label_budget([]), fitting=costs.fit_costs(None),
            common_source_cost=None, break_even=costs.break_even(None, None, None))
        write(source / 'cost_report.json', ledger)
        freeze = dict(status='LAYER_FULL_SELECTION_FROZEN', protocol_sha256=data.sha(source / 'frozen_protocol.json'),
            artifact_hashes={name: data.sha(source / name) for name in
                            ('aggregate_metrics.json', 'selected_configurations.json', 'cost_report.json', 'models/synthetic.json')})
        write(source / 'selection_frozen.json', freeze)
        output = report.build(source, root / 'report')
        assert len(list(output.glob('*.png'))) == len(list(output.glob('*.pdf'))) == 4
        assert all(path.stat().st_size > 1000 for path in output.glob('*.pdf'))
        text = (output / 'report.md').read_text()
        assert 'Final evaluation is pending' in text and 'UNKNOWN' in text
        assert (output / 'collection_costs.csv').is_file()
        assert report.summarize_test([dict(target='standard', layer='linear', dtype='fp16',
            selection_scope='overall', learner='ridge', method='direct', k=10,
            mape_pct=2, p90_pct=4, failures=0, planned=50)])[0]['mean_mape_pct'] == 2
        evaluation = root / 'evaluation'
        evaluation.mkdir()
        write(evaluation / 'evaluation_completed.json', dict(
            freeze_sha256=data.sha(source / 'selection_frozen.json'), model_selection_changed=False))
        try:
            report.build(source, root / 'unchecked_evaluation', evaluation)
        except ValueError as error:
            assert 'completion artifact hashes' in str(error)
        else:
            raise AssertionError('Unchecked evaluation metrics accepted')
        write(source / 'aggregate_metrics.json', [])
        try:
            report.build(source, root / 'changed')
        except ValueError as error:
            assert 'Frozen artifact mismatch' in str(error)
        else:
            raise AssertionError('Changed frozen metrics accepted')
    print('PASS: synthetic plots, tables and frozen-artifact checks')


class ReportTests(unittest.TestCase):
    def test_synthetic_report(self):
        run()


if __name__ == '__main__':
    run()
