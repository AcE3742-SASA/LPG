"""Expanded GEMM source identity, prediction replay and frozen-input guards."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pipeline as p


class ExpandedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = dict(source_models=p.frozen_sources())

    def test_selected_allocate_sources_are_shared_across_target_learners(self):
        mapping = self.protocol['source_models']
        self.assertEqual(len(mapping), 24)
        self.assertEqual(len({s['path'] for s in mapping.values()}), 6)
        for dtype, source_learner in [('fp16', 'mlp32'), ('fp32', 'rf')]:
            for target_seed, source_seed in zip(p.SEEDS, [20261028, 20261029, 20261030]):
                specs = [mapping[f'{learner}/{dtype}/{target_seed}']
                         for learner in p.previous.learners.LEARNERS]
                self.assertTrue(all(s == specs[0] for s in specs))
                self.assertEqual((specs[0]['learner'], specs[0]['mode'], specs[0]['target'],
                                  specs[0]['source_seed'], specs[0]['n']),
                                 (source_learner, 'allocate', 'normalized', source_seed, 11498))
                self.assertEqual(specs[0]['features'], p.gemm.FEATURES)

    def test_all_selected_development_predictions_replay_exactly(self):
        wanted = {(s['learner'], s['dtype'], str(s['source_seed']))
                  for s in self.protocol['source_models'].values()}
        stored = {}
        with (p.SOURCE / 'validation_predictions.csv').open(newline='') as handle:
            for row in csv.DictReader(handle):
                key = row['learner'], row['dtype'], row['seed']
                if key not in wanted or (row['mode'], row['target'], row['n']) != (
                        'allocate', 'normalized', '11498'):
                    continue
                m, n, k = map(int, row['shape'].split('x'))
                case = dict(layer='linear', rows=m, input_width=k, output_width=n)
                stored.setdefault(key + (row['validation'],), []).append(
                    (case, float(row['predicted_ms'])))
        self.assertEqual(len(stored), 12)
        count = 0
        for dtype in ('fp16', 'fp32'):
            for target_seed in p.SEEDS:
                source = p.load_source(self.protocol, 'ridge', dtype, target_seed)
                for split, expected_count in [('legacy', 716), ('expansion', 895)]:
                    rows = stored[source['learner'], dtype, str(source['seed']), split]
                    self.assertEqual(len(rows), expected_count)
                    predicted = p.source_prior([r[0] for r in rows], dtype, source)
                    np.testing.assert_array_equal(predicted, [r[1] for r in rows])
                    count += len(rows)
        self.assertEqual(count, 9666)

    def test_prior_sums_ffn_gemms_and_preserves_convolution_geometry(self):
        cases = [
            dict(layer='linear', rows=96, input_width=127, output_width=256),
            dict(layer='ffn', rows=96, input_width=127, hidden_width=511),
            dict(layer='conv1', batch=2, height=8, width=16, cin=33, cout=65, kernel=1),
            dict(layer='conv3', batch=2, height=8, width=16, cin=33, cout=65, kernel=3),
        ]
        expected_shapes = [[(96, 256, 127)], [(96, 511, 127), (96, 127, 511)],
                           [(256, 65, 33)], [(256, 65, 297)]]
        for dtype in ('fp16', 'fp32'):
            source = p.load_source(self.protocol, 'mlp64', dtype, p.SEEDS[0])
            for case, shapes in zip(cases, expected_shapes):
                self.assertEqual(p.logical_gemms(case), shapes)
                expected = 0.0
                for shape in shapes:
                    features, lower = p.gemm.design([shape], dtype)
                    expected += float(np.exp(source['model'].predict(features))[0] * lower[0])
                np.testing.assert_allclose(p.source_prior([case], dtype, source), [expected],
                                           rtol=1e-14, atol=0)

    def test_source_hash_is_checked_before_deserialization(self):
        protocol = copy.deepcopy(self.protocol)
        protocol['source_models'][f'ridge/fp16/{p.SEEDS[0]}']['sha256'] = '0' * 64
        with patch.object(p.joblib, 'load') as deserialize:
            with self.assertRaises(ValueError):
                p.load_source(protocol, 'ridge', 'fp16', p.SEEDS[0])
        deserialize.assert_not_called()

    def test_source_envelope_identity_and_feature_mismatches_are_rejected(self):
        original = p.load_source(self.protocol, 'ridge', 'fp16', p.SEEDS[0])
        changes = dict(learner='rf', dtype='fp32', mode='graph', target='direct',
                       seed=20261029, n=2652, features=list(reversed(p.gemm.FEATURES)),
                       protocol_sha256='0' * 64, locked_test_opened=True, validation_only=False)
        for key, value in changes.items():
            with self.subTest(field=key):
                altered = dict(original, **{key: value})
                with patch.object(p.joblib, 'load', return_value=altered):
                    with self.assertRaises(ValueError):
                        p.load_source(self.protocol, 'ridge', 'fp16', p.SEEDS[0])

    def test_prepare_freezes_source_inputs_and_detects_input_hash_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'prepared'
            p.prepare(output)
            protocol = p.verify_protocol(output)
            self.assertEqual(protocol['version'], 2)
            self.assertEqual(protocol['sealed_test_status'], 'NOT_OPENED')
            selected_file = str((p.SOURCE / 'accuracy_first_model_refs.csv').relative_to(p.ROOT))
            self.assertIn(selected_file, protocol['inputs'])
            with self.assertRaises(FileExistsError):
                p.prepare(output)
            protocol['inputs'][selected_file] = '0' * 64
            p.save(output / 'frozen_protocol.json', protocol)
            with self.assertRaises(ValueError):
                p.verify_protocol(output)

    def test_source_training_and_selection_shapes_do_not_overlap_layer_shapes(self):
        source = p.read(p.SOURCE / 'frozen_protocol.json')
        protected = set(map(tuple, source['old_train_orders']['20261028'] +
                            source['new_train_orders']['20261028'] +
                            source['legacy_validation_shapes'] + source['new_validation_shapes']))
        self.assertEqual(len(protected), 13109)
        cases = p.read(p.PACKAGE / 'cases_by_phase.json')
        for batch in cases.values():
            for case in batch:
                for shape in p.logical_gemms(case):
                    self.assertNotIn(shape, protected)
                    for axis in range(3):
                        for delta in range(-8, 9):
                            neighbor = list(shape)
                            neighbor[axis] += delta
                            self.assertNotIn(tuple(neighbor), protected)


if __name__ == '__main__':
    unittest.main()
