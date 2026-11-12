"""Synthetic end-to-end development/freeze/evaluation without measured labels."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pipeline as p
import components
import data


def fixture():
    cases = p.read(p.PACKAGE / 'cases_by_phase.json')
    subsets = p.read(p.PACKAGE / 'training_subsets.json')
    groups = p.read(p.PACKAGE / 'groups.json')
    wanted = set()
    for family in ('core', 'conv'):
        wanted.update(subsets['families'][family][str(p.SEEDS[0])]['by_k']['5'])
        for split in ('validation', 'test_locked'):
            wanted.update(g['group_id'] for g in groups if g['family'] == family and g['split'] == split)
    chosen = [c for cc in cases.values() for c in cc if c['group_id'] in wanted]
    # Keep enough distinct validation/test groups to exercise bootstrap and geometry slices.
    for split in ('validation', 'test_locked'):
        for family in ('core', 'conv'):
            keep = {g['group_id'] for g in groups if g['family'] == family and g['split'] == split}
            keep = set(sorted(keep)[:3])
            chosen = [c for c in chosen if c['split'] != split or
                      (c['layer'].startswith('conv') != (family == 'conv')) or c['group_id'] in keep]
    protocol = dict(status='LAYER_FULL_PROTOCOL_FROZEN', learners=['rf', 'ridge'], seeds=[p.SEEDS[0]],
        budgets=[5], subsets=subsets, groups=groups, cases={'synthetic': chosen},
        source_models=p.frozen_sources(), selection='synthetic fixed selection')
    rows = []
    for c in chosen:
        for dtype in ('fp16', 'fp32'):
            first = p.logical_gemms(c)[0]
            base = .02 + np.prod(first) / 1e10
            names = components.component_names(c['layer'])
            for impl in data.implementations(c):
                value = base if impl in ('standard', 'decomposed') else base / len(names)
                rows.append(dict(case_id=c['id'], group_id=c['group_id'], layer=c['layer'], dtype=dtype,
                    split=c['split'], case=c, implementation=impl, time_ms=float(value),
                    round_times_ms=[float(value)] * 3, status='PASS', reasons=[], record_costs=[]))
    dev = dict(rows=[r for r in rows if r['split'] != 'test_locked'], excluded=[], provenance={})
    test = dict(rows=[r for r in rows if r['split'] == 'test_locked'], excluded=[], provenance={})
    return protocol, dev, test


class PipelineTests(unittest.TestCase):
    def test_features_and_common_quality(self):
        protocol, dev, _ = fixture()
        cases, labels, coverage = p.cohort(dev, protocol, 'conv3', 'fp32')
        self.assertEqual(p.features(cases, 'fp32').shape, (len(cases), len(p.FEATURES)))
        self.assertTrue(np.isfinite(p.features(cases, 'fp32')).all())
        first = cases[0]
        dev['rows'] = [r for r in dev['rows'] if not (r['case_id'] == first['id'] and
                      r['dtype'] == 'fp32' and r['implementation'] == 'unfold')]
        retained, _, coverage = p.cohort(dev, protocol, 'conv3', 'fp32')
        self.assertNotIn(first, retained)
        self.assertIn(first['group_id'], coverage['excluded_group_ids'])

    def test_synthetic_train_freeze_and_final(self):
        import costs
        protocol, dev, test = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'run'
            output.mkdir()
            p.save(output / 'frozen_protocol.json', protocol)
            with patch.object(p, 'verify_protocol', return_value=protocol), \
                 patch.object(data, 'load_campaign', return_value=dev) as loader, \
                 patch.object(costs, 'build_report', return_value={'synthetic': True}):
                p.train(root / 'campaign', output)
                self.assertNotIn('allow_test', loader.call_args.kwargs)
                freeze = p.read(output / 'selection_frozen.json')
                self.assertEqual(freeze['sealed_test_status'], 'NOT_OPENED')
                data.check_freeze(output / 'frozen_protocol.json', output / 'selection_frozen.json')
                self.assertEqual(len(p.read(output / 'selected_configurations.json')), 112)
                with self.assertRaises(ValueError):
                    p.train(root / 'campaign', output)
                with self.assertRaises(ValueError):
                    p.evaluate(root / 'campaign', output, root / 'denied')
                self.assertFalse((root / 'denied').exists())
                loader.return_value = test
                p.evaluate(root / 'campaign', output, root / 'final', allow_sealed_test=True)
                self.assertTrue(loader.call_args.kwargs['allow_test'])
                self.assertEqual(p.read(root / 'final/evaluation_completed.json')['rows'], 112)
                self.assertTrue(p.read(root / 'final/paired_bootstrap.json'))
                with self.assertRaises(ValueError):
                    p.evaluate(root / 'campaign', output, root / 'repeat', allow_sealed_test=True)
                model = next(iter(freeze['artifact_hashes']))
                (output / model).write_text('tampered')
                with self.assertRaises(ValueError):
                    data.check_freeze(output / 'frozen_protocol.json', output / 'selection_frozen.json')


if __name__ == '__main__':
    unittest.main()
