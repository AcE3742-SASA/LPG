"""Synthetic feature, summed-model and failure-contract checks."""
import copy
import unittest

import numpy as np

import components as c


def cases(layer, count=6):
    result = []
    for i in range(count):
        case = dict(id=f'case_{i}', group_id=f'group_{i}', split='train', layer=layer)
        if layer in ('linear', 'ffn'):
            case.update(rows=64 + 16 * i, input_width=64)
            case['output_width' if layer == 'linear' else 'hidden_width'] = 128
        else:
            case.update(batch=2, height=8 + i, width=12, cin=16, cout=32,
                        kernel=int(layer[-1]))
        result.append(case)
    return result


class Constant:
    def __init__(self, response):
        self.response = response

    def predict(self, x):
        return np.full(len(x), self.response)


class ComponentTests(unittest.TestCase):
    def test_static_feature_geometry(self):
        for layer in c.COMPONENTS:
            for name in c.component_names(layer):
                x = c.component_features(cases(layer), 'fp16', name)
                self.assertEqual(x.shape, (6, len(c.FEATURES)))
                self.assertTrue(np.isfinite(x).all())
        ffn = cases('ffn', 1)
        first = c.component_features(ffn, 'fp16', 'gemm1')[0]
        second = c.component_features(ffn, 'fp16', 'gemm2')[0]
        np.testing.assert_allclose(np.exp(first[:3]), [64, 128, 64])
        np.testing.assert_allclose(np.exp(second[:3]), [64, 64, 128])
        conv = c.component_features(cases('conv3', 1), 'fp32', 'unfold')[0]
        np.testing.assert_allclose(np.exp(conv[-2:]), [2 * 8 * 12 * 16, 2 * 8 * 12 * 16 * 9])
        changed = copy.deepcopy(ffn)
        changed[0].update(time_ms=9999, actual_gelu_ms=0.001, measured_stride=[100, 1])
        np.testing.assert_array_equal(c.component_features(ffn, 'fp16', 'gelu'),
                                      c.component_features(changed, 'fp16', 'gelu'))
        with self.assertRaises(ValueError):
            c.component_features(ffn, 'fp64', 'gelu')
        with self.assertRaises(ValueError):
            c.component_features(ffn, 'fp16', 'unfold')

    def test_sum_and_training_boundary(self):
        for layer in c.COMPONENTS:
            sample = cases(layer)
            names = c.component_names(layer)
            labels = {(row['id'], 'fp16', 'decomposed' if name == 'gemm' else name): float(i + 1)
                      for row in sample for i, name in enumerate(names)}
            bundle, diagnostic = c.fit_sum(sample, 'fp16', 'rf', labels, 20261104)
            total, parts, valid = c.predict_sum(bundle, sample)
            self.assertTrue(valid.all())
            np.testing.assert_allclose(total, sum(range(1, len(names) + 1)))
            np.testing.assert_allclose(total, np.sum(list(parts.values()), axis=0))
            self.assertEqual(diagnostic['component_fits'], len(names))
            self.assertEqual(bundle['train_case_ids'], [row['id'] for row in sample])
            bad = copy.deepcopy(sample)
            bad[0]['split'] = 'validation'
            with self.assertRaises(ValueError):
                c.fit_sum(bad, 'fp16', 'rf', labels, 20261104)
            labels[next(iter(labels))] = 0
            with self.assertRaises(ValueError):
                c.fit_sum(sample, 'fp16', 'rf', labels, 20261104)

    def test_all_learners_and_invalid_predictions(self):
        sample = cases('linear')
        labels = {(row['id'], 'fp32', 'decomposed'): 2.0 for row in sample}
        for learner in c.learners.LEARNERS:
            bundle, _ = c.fit_sum(sample, 'fp32', learner, labels, 20261104)
            total, _, valid = c.predict_sum(bundle, sample)
            self.assertTrue(valid.all())
            np.testing.assert_allclose(total, 2.0, rtol=0.03)
        bundle = dict(version='component_sum_v1', features=c.FEATURES, layer='ffn',
                      dtype='fp16', models={name: Constant(0) for name in c.component_names('ffn')})
        for bad in (np.nan, np.inf, -np.inf, 1000, -1000):
            bundle['models']['gelu'] = Constant(bad)
            total, parts, valid = c.predict_sum(bundle, cases('ffn'))
            self.assertFalse(valid.any())
            self.assertTrue(np.isnan(total).all())
            self.assertTrue((~np.isfinite(parts['gelu']) | (parts['gelu'] <= 0)).all())
        with self.assertRaises(ValueError):
            c.predict_sum(bundle, sample)


if __name__ == '__main__':
    unittest.main()
