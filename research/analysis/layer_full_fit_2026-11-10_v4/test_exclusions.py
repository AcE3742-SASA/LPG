"""Keep the fixed correctness failure visible without inventing timing labels."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import data
from test_data import fixture, rehash, write


class CorrectnessExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.actual_cases = data.read(data.PACKAGE / 'cases_by_phase.json')
        self.package, self.campaign, self.frozen = fixture(self.root, exclusion=True)
        self.patcher = patch.object(data, 'PACKAGE', self.package)
        self.patcher.start()
        self.protocol = data.read(self.package / 'protocol.json')
        self.sources = data.read(self.package / 'release_hashes.json')
        self.exclusion = self.protocol['correctness_exclusions'][0]

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def load(self, **options):
        return data.load_campaign(self.campaign, protocol_path=self.frozen, **options)

    def test_skipped_fp32_has_six_explicit_failures_and_fp16_remains(self):
        result = self.load(diagnostic=True)
        self.assertEqual((len(result['rows']), len(result['excluded'])), (78, 6))
        target = self.exclusion['case_id']
        self.assertEqual({row['implementation'] for row in result['excluded']},
                         {'standard', 'decomposed', 'unfold', 'input_pack', 'mm', 'output_pack'})
        for row in result['excluded']:
            self.assertEqual((row['case_id'], row['dtype'], row['status']), (target, 'fp32', 'FAIL'))
            self.assertEqual(row['reasons'], ['correctness_exclusion'])
            self.assertIsNone(row['time_ms'])
            self.assertEqual(row['round_times_ms'], [])
            self.assertEqual(row['record_costs'], [])
        self.assertEqual(len([row for row in result['rows'] if row['case_id'] == target and row['dtype'] == 'fp16']), 6)
        standard = self.load()
        self.assertEqual((len(standard['rows']), len(standard['excluded'])), (17, 1))
        self.assertEqual(standard['excluded'][0]['implementation'], 'standard')
        ledger = next(item['ledger'] for item in result['batch_costs'] if item['phase'] == 'validation_01')
        self.assertEqual((ledger['recorded'], ledger['expected']), (126, 126))

    def test_wrong_case_dtype_phase_or_reason_is_not_an_exclusion(self):
        cases = data.read(self.package / 'cases_by_phase.json')['validation_01']
        for key, value in {'case_id': cases[0]['id'], 'dtype': 'fp16', 'phase': 'train_01',
                           'reason': 'any_correctness_failure'}.items():
            with self.subTest(field=key), data.Artifact(self.campaign, 'validation_01_attempt_001') as artifact:
                protocol = copy.deepcopy(self.protocol)
                protocol['correctness_exclusions'][0][key] = value
                with self.assertRaisesRegex(ValueError, 'exclusion identity or reason'):
                    data.correctness_exclusions(artifact, 'validation_01', cases, protocol, self.sources)

    def test_rehashed_evidence_must_still_show_original_criterion_failure(self):
        report = data.read(self.package / self.exclusion['evidence'])
        cases = data.read(self.package / 'cases_by_phase.json')['validation_01']
        changes = [lambda r: r.update(passed=True),
                   lambda r: r.update(atol=0.001),
                   lambda r: r['full_pair_adjudication']['implementations']['standard'].update(passed=True),
                   lambda r: r['full_pair_adjudication']['implementations']['standard'].update(max_tolerance_ratio=0.9),
                   lambda r: r['full_pair_adjudication']['implementations']['standard'].update(
                       observed=list(r['full_pair_adjudication']['reference'])),
                   lambda r: r['full_pair_adjudication'].update(checked_count=1)]
        for change in changes:
            altered = copy.deepcopy(report)
            change(altered)
            blob = json.dumps(altered).encode()
            digest = hashlib.sha256(blob).hexdigest()
            protocol = copy.deepcopy(self.protocol)
            protocol['correctness_exclusions'][0]['evidence_sha256'] = digest
            sources = dict(self.sources, **{self.exclusion['evidence']: digest})
            with self.assertRaisesRegex(ValueError, 'original-criterion failure|evidence is incomplete'):
                data.correctness_exclusions(SimpleNamespace(bytes=lambda name: blob), 'validation_01', cases, protocol, sources)

    def test_embedded_evidence_tamper_and_undeclared_missing_record_are_rejected(self):
        attempt = self.campaign / 'validation_01_attempt_001'
        evidence = attempt / 'source' / self.exclusion['evidence']
        original = evidence.read_bytes()
        evidence.write_bytes(original + b' ')
        rehash(attempt)
        with self.assertRaisesRegex(ValueError, 'evidence hash mismatch'):
            self.load(diagnostic=True)
        evidence.write_bytes(original)
        next((attempt / 'records').glob('*.json')).unlink()
        rehash(attempt)
        with self.assertRaisesRegex(ValueError, 'record inventory'):
            self.load(diagnostic=True)

    def test_manifest_and_summary_cannot_widen_or_hide_the_exclusion(self):
        attempt = self.campaign / 'validation_01_attempt_001'
        for filename, key, value in [('manifest.json', 'correctness_exclusions', []),
                                     ('summary.json', 'excluded', 0),
                                     ('summary.json', 'excluded_conditions', [])]:
            path = attempt / filename
            original = path.read_bytes()
            altered = data.read(path)
            altered[key] = value
            write(path, altered)
            rehash(attempt)
            with self.assertRaisesRegex(ValueError, 'correctness exclusions'):
                self.load(diagnostic=True)
            path.write_bytes(original)
            rehash(attempt)

    def test_common_validation_cohort_is_49_for_fp32_and_50_for_fp16(self):
        import pipeline
        cases = [case for phase, group in self.actual_cases.items() if phase.startswith('validation_')
                 for case in group if case['layer'] == 'conv3']
        self.assertEqual(len(cases), 50)
        rows = [dict(case_id=case['id'], dtype=dtype, implementation=implementation, time_ms=1.0)
                for case in cases for dtype in ('fp16', 'fp32')
                for implementation in data.implementations(case)
                if (case['id'], dtype) != (self.exclusion['case_id'], 'fp32')]
        for dtype, expected in [('fp32', 49), ('fp16', 50)]:
            retained, _, coverage = pipeline.cohort({'rows': rows}, {'cases': {'validation': cases}}, 'conv3', dtype)
            self.assertEqual(len(retained), expected)
            self.assertEqual(coverage['planned_groups'], 50)
            self.assertEqual(coverage['retained_groups'], expected)


if __name__ == '__main__':
    unittest.main()
