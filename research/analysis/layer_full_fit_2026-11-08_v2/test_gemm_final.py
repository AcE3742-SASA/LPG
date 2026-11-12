"""Sealed-evaluation boundary and exact-ID tests using synthetic records only."""
import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import gemm_final as f


def write_csv(path, values):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def fixture(root):
    conditions, measurements = [], []
    shape = [64, 80, 96]
    database = root / 'synthetic.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE attempts(seq INTEGER PRIMARY KEY, task_id TEXT, record TEXT)')
        for dtype in f.gemm.DTYPES:
            for mode in f.gemm.MODES:
                ids = []
                for rd in range(3):
                    tid = f'{dtype}_{mode}_{rd}'
                    task = dict(id=tid, phase='main', split='fresh_holdout', shape=shape,
                                dtype=dtype, mode=mode, round=rd)
                    seq = len(measurements) + 1
                    connection.execute('INSERT INTO attempts VALUES(?,?,?)',
                                       (seq, tid, json.dumps(dict(task=task, timing=dict(median_ms=1.0 + rd)))))
                    measurements.append(dict(**task, seq=seq, quality_status='PASS', disposition='accepted',
                                             role='locked_final_test', between_round_review='False', flags='[]'))
                    measurements[-1]['shape'] = json.dumps(shape)
                    ids.append(tid)
                conditions.append(dict(phase='main', split='fresh_holdout', shape=json.dumps(shape),
                                       dtype=dtype, mode=mode, task_ids=json.dumps(ids), accepted_rounds=3,
                                       role='locked_final_test'))
        connection.execute('INSERT INTO attempts VALUES(?,?,?)',
                           (1000, 'UNALLOWED', 'This record must never be decoded'))
    condition_path, measurement_path = root / 'conditions.csv', root / 'measurements.csv'
    write_csv(condition_path, conditions)
    write_csv(measurement_path, measurements)
    protocol = dict(condition_index=str(condition_path), measurement_index=str(measurement_path),
                    expected_shapes=1, shapes=[shape], raw_database=str(database),
                    raw_database_sha256=f.sha(database))
    return protocol, conditions, measurements


class Constant:
    def __init__(self, response):
        self.response = response

    def predict(self, x):
        return np.full(len(x), self.response)


class GemmFinalTests(unittest.TestCase):
    def test_selection_filters_failed_candidates_and_ties(self):
        scores = []
        for rq in ('RQ1', 'RQ4'):
            for dtype in f.gemm.DTYPES:
                for mode in f.gemm.MODES if rq == 'RQ1' else ('prealloc', 'graph'):
                    base = dict(rq=rq, dtype=dtype, mode=mode, learner='rf', method='direct',
                                failures=0, n=2652, mean_mape_pct=10, mean_p90_pct=20)
                    scores.extend([base, dict(base, n=100, mean_mape_pct=0, failures=1),
                                   dict(base, n=3345), dict(base, n=30, mean_mape_pct='nan')])
        chosen = f.choose_validation(scores)
        self.assertEqual(len(chosen), 10)
        self.assertTrue(all(row['n'] == 2652 for row in chosen))

    def test_denial_precedes_any_read(self):
        with patch.object(Path, 'open', side_effect=AssertionError('Unexpected file read')), \
                patch.object(sqlite3, 'connect', side_effect=AssertionError('Unexpected database access')):
            with self.assertRaisesRegex(ValueError, 'allow-sealed-test'):
                f.load_sealed({})
            with self.assertRaisesRegex(ValueError, 'allow-sealed-test'):
                f.evaluate('missing', 'missing')

    def test_exact_allowlist_three_round_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, _, _ = fixture(root)
            with patch.object(sqlite3, 'connect', side_effect=AssertionError('Metadata opened DB')):
                shape, grouped, records = f.locked_metadata(root / 'conditions.csv', root / 'measurements.csv', 1)
            self.assertEqual((len(shape), len(grouped), len(records)), (1, 6, 18))
            labels = f.load_sealed(protocol, allow_sealed_test=True)
            self.assertEqual(len(labels), 6)
            self.assertEqual(set(labels.values()), {2.0})
            protocol['raw_database_sha256'] = 'invalid'
            with patch.object(sqlite3, 'connect', side_effect=AssertionError('Hash failure opened DB')):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    f.load_sealed(protocol, allow_sealed_test=True)

    def test_review_and_duplicate_round_rejected_before_raw_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, _, measurements = fixture(root)
            original = measurements[0].copy()
            for key, value in [('quality_status', 'REVIEW'), ('round', 1), ('flags', '["clock_deviation"]')]:
                measurements[0] = dict(original, **{key: value})
                write_csv(root / 'measurements.csv', measurements)
                with patch.object(sqlite3, 'connect', side_effect=AssertionError('Bad metadata opened DB')):
                    with self.assertRaises(ValueError):
                        f.load_sealed(protocol, allow_sealed_test=True)

    def test_prediction_formulas_and_invalid_visibility(self):
        shapes = [(64, 80, 96), (128, 192, 256)]
        direct = dict(rq='RQ1', dtype='fp16', method='direct', model=Constant(np.log(2)))
        np.testing.assert_allclose(f.predict(direct, shapes), 2)
        normalized = dict(direct, method='normalized')
        np.testing.assert_allclose(f.predict(normalized, shapes), 2 * f.gemm.design(shapes, 'fp16')[1])
        source = dict(model=Constant(np.log(2)), target='direct')
        for method, model, expected in [('zero', None, 2), ('constant_ratio', np.log(3), 6),
                                        ('residual', Constant(np.log(3)), 6),
                                        ('direct', Constant(np.log(3)), 3),
                                        ('prior_direct', Constant(np.log(3)), 3)]:
            envelope = dict(rq='RQ4', dtype='fp16', source=source, method=method, model=model)
            np.testing.assert_allclose(f.predict(envelope, shapes), expected)
        result = f.predict(dict(direct, model=Constant(1000)), shapes)
        self.assertTrue(np.isinf(result).all())
        self.assertEqual(f.gemm.metrics(np.ones(2), result)['failures'], 2)

    def test_changed_freeze_and_one_time_guard_precede_test_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / 'frozen_selection.json'
            frozen.write_text('{"version":"gemm_final_v1","sealed_times_read":false,"inputs":{},"models":[]}')
            (root / 'frozen_selection.sha256').write_text(f.sha(frozen))
            self.assertEqual(f.verify_prepared(root)['version'], 'gemm_final_v1')
            frozen.write_text(frozen.read_text() + ' ')
            with patch.object(f, 'load_sealed', side_effect=AssertionError('Changed freeze accessed test')):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    f.evaluate(root, root / 'out', True)
            (root / 'frozen_selection.sha256').write_text(f.sha(frozen))
            (root / 'sealed_access_started.json').write_text('{}')
            with patch.object(f, 'load_sealed', side_effect=AssertionError('Repeated test read')):
                with self.assertRaises(FileExistsError):
                    f.evaluate(root, root / 'out', True)


if __name__ == '__main__':
    unittest.main()
