"""Synthetic campaign checks; no measured or sealed research labels are read."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import data


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def rehash(attempt):
    write(attempt/'result_hashes.json', {str(p.relative_to(attempt)): data.sha(p)
          for p in attempt.rglob('*') if p.is_file() and p.name != 'result_hashes.json'})


def fixture(root):
    package, campaign = root/'package', root/'campaign'
    protocol = data.read(data.PACKAGE/'protocol.json')
    original_cases = data.read(data.PACKAGE/'cases_by_phase.json')
    phases = ['train_01', 'validation_01', 'test_01']
    cases = {phase: [copy.deepcopy(next(c for c in original_cases[phase] if c['layer'] == layer))
                    for layer in ('linear', 'ffn', 'conv1', 'conv3')] for phase in phases}
    write(package/'protocol.json', protocol)
    write(package/'cases_by_phase.json', cases)
    sources = {name: data.sha(package/name) for name in ('protocol.json', 'cases_by_phase.json')}
    write(package/'release_hashes.json', sources)
    write(root/'protocol.json', dict(status='LAYER_FULL_PROTOCOL_FROZEN', package_release_hashes=sources))
    state = dict(source_hashes=sources, batch_order=phases, attempts={})
    for phase in phases:
        attempt = campaign/(phase + '_attempt_001')
        state['attempts'][phase] = [attempt.name]
        batch_cases = cases[phase]
        count = sum(len(data.implementations(c)) * 6 for c in batch_cases)
        config = dict(protocol, phase=phase, health_case_ids=[c['id'] for c in batch_cases])
        write(attempt/'manifest.json', dict(kind='layer_full_20261104_v1', phase=phase, protocol=config,
              cases=batch_cases, source_hashes=sources, expected_measurements=count,
              split=batch_cases[0]['split'], sealed_test=phase.startswith('test_')))
        flags = {}
        for name in sources:
            target = attempt/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((package/name).read_bytes())
        for case in batch_cases:
            for dtype in protocol['dtypes']:
                identity = f'inputs/{case["id"]}_{dtype}.json'
                write(attempt/identity, dict(input={'sha256': 'input'}, weights=[], prepared=[], seed=1))
                for rd in range(3):
                    correct = {edge: f'correctness/{case["id"]}_{dtype}_r{rd}_{edge}.json'
                               for edge in ('before', 'after')}
                    for name in correct.values():
                        write(attempt/name, {'passed': True})
                    component_identity = f'inputs/{case["id"]}_{dtype}_r{rd}_components.json'
                    write(attempt/component_identity, {'midpoint': {'sha256': 'component'}})
                    for implementation in data.implementations(case):
                        rid = data.key(case, dtype, rd, implementation)
                        directory = 'records' if implementation == 'standard' else 'restricted_explicit/records'
                        value = 1.0 + rd * .001
                        block = dict(elapsed_ms=value*100, repeats=100, operations_per_call=1,
                                     latency_ms=value, host_envelope_start_utc='2026-11-04T00:00:00+00:00',
                                     host_envelope_end_utc='2026-11-04T00:00:01+00:00')
                        timing = dict(median_ms=value, flags=[], blocks=[block]*7,
                                      relative_iqr=0.0, first_last_drift=0.0, min_ms=value, max_ms=value,
                                      setup=dict(settle_event_ms=value*3100, settle_blocks=[block]*31,
                                                 calibration=[block], prepare_wall_ms=3200))
                        write(attempt/directory/(rid+'.json'), dict(id=rid, case=case, dtype=dtype,
                              round=rd, implementation=implementation,
                              timing=timing,
                              correctness_before=correct['before'], correctness_after=correct['after'],
                              input_identity=identity, component_identity=component_identity,
                              preparation=dict(pair_creation_wall_ms=2, before_correctness_wall_ms=3)))
                        flags[rid] = []
        write(attempt/'summary.json', dict(scope=phase, collection='COMPLETE', gate='PASS', comparisons=[],
              missing=0, expected=count, recorded=count, counts={'PASS': count}, measurement_flags=flags))
        write(attempt/'status.json', dict(collection='COMPLETE', restoration='COMMANDS_ACCEPTED'))
        write(attempt/'worker/complete.json', {'complete': True})
        write(attempt/'cost_ledger.json', dict(phase=phase, source_hashes=sources,
              expected=count, recorded=count, collection_wall_seconds=50.0))
        rehash(attempt)
    write(campaign/'campaign.json', state)
    return package, campaign, root/'protocol.json'


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package, self.campaign, self.protocol = fixture(self.root)
        self.patcher = patch.object(data, 'PACKAGE', self.package)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def load(self, **options):
        return data.load_campaign(self.campaign, protocol_path=self.protocol, **options)

    def test_dev_ignores_test_and_standard_ignores_components(self):
        (self.campaign/'test_01_attempt_001/manifest.json').write_text('DO NOT OPEN TEST')
        for phase in ('train_01', 'validation_01'):
            for path in (self.campaign/(phase+'_attempt_001')/'restricted_explicit/records').glob('*.json'):
                path.write_text('DO NOT OPEN COMPONENT')
        result = self.load()
        self.assertEqual(len(result['rows']), 16)
        self.assertEqual(result['excluded'], [])
        self.assertFalse(result['provenance']['test_opened'])

    def test_component_labels_and_three_round_medians(self):
        result = self.load(diagnostic=True)
        self.assertEqual(len(result['rows']), 72)
        self.assertAlmostEqual(result['rows'][0]['time_ms'], 1.001)
        self.assertEqual({r['implementation'] for r in result['rows']},
                         {'standard', 'decomposed', 'gemm1', 'gemm2', 'gelu', 'unfold', 'input_pack', 'mm', 'output_pack'})

    def test_review_is_retained_as_explicit_exclusion(self):
        attempt = self.campaign/'train_01_attempt_001'
        path = next((attempt/'records').glob('*.json'))
        row = data.read(path)
        summary = data.read(attempt/'summary.json')
        summary['measurement_flags'][row['id']] = ['clock_deviation']
        summary['gate'] = 'REVIEW'
        write(attempt/'summary.json', summary)
        rehash(attempt)
        result = self.load()
        self.assertEqual(len(result['excluded']), 1)
        self.assertEqual(result['excluded'][0]['reasons'], ['clock_deviation'])
        self.assertEqual(len(result['rows']), 15)

    def test_missing_round_and_hash_change_fail(self):
        attempt = self.campaign/'train_01_attempt_001'
        path = next((attempt/'records').glob('*.json'))
        original = path.read_bytes()
        row = data.read(path)
        row['timing']['median_ms'] = 2
        write(path, row)
        with self.assertRaisesRegex(ValueError, 'Result hash mismatch'):
            self.load()
        path.write_bytes(original)
        path.unlink()
        rehash(attempt)
        with self.assertRaisesRegex(ValueError, 'record inventory'):
            self.load()

    def test_rehashed_raw_statistic_and_source_corruption_fail(self):
        attempt = self.campaign/'train_01_attempt_001'
        path = next((attempt/'records').glob('*.json'))
        original = path.read_bytes()
        row = data.read(path)
        row['timing']['median_ms'] *= 2
        write(path, row)
        rehash(attempt)
        with self.assertRaisesRegex(ValueError, 'Stored timing statistics mismatch'):
            self.load()
        path.write_bytes(original)
        source = attempt/'source/protocol.json'
        source.write_text('{}')
        rehash(attempt)
        with self.assertRaisesRegex(ValueError, 'Measurement source mismatch'):
            self.load()

    def test_sealed_guard_and_frozen_artifacts(self):
        with self.assertRaisesRegex(ValueError, 'explicit authorization'):
            self.load(split='test')
        with self.assertRaisesRegex(ValueError, 'frozen selection'):
            self.load(split='test', allow_test=True)
        run = self.root/'fit'
        write(run/'selected_configurations.json', [{'method': 'direct'}])
        write(run/'models/model.json', {'weights': [1]})
        write(run/'selection_frozen.json', dict(status='LAYER_FULL_SELECTION_FROZEN',
              protocol_sha256=data.sha(self.protocol), artifact_hashes={
                  n: data.sha(run/n) for n in ('selected_configurations.json', 'models/model.json')}))
        result = self.load(split='test', allow_test=True, freeze_path=run/'selection_frozen.json')
        self.assertEqual(len(result['rows']), 8)
        (run/'models/model.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Frozen artifact mismatch'):
            self.load(split='test', allow_test=True, freeze_path=run/'selection_frozen.json')

    def test_returned_zips_match_extracted_inputs(self):
        expected = self.load(diagnostic=True)
        archive_campaign = self.root/'zip_return'
        archive_campaign.mkdir()
        (archive_campaign/'campaign.json').write_bytes((self.campaign/'campaign.json').read_bytes())
        for phase in ('train_01', 'validation_01'):
            attempt = self.campaign/(phase+'_attempt_001')
            archive = archive_campaign/(attempt.name+'_fixture.zip')
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
                for path in attempt.rglob('*'):
                    if path.is_file():
                        output.write(path, str(Path(attempt.name)/path.relative_to(attempt)))
            archive.with_suffix('.zip.sha256').write_text(data.sha(archive)+'  '+archive.name+'\n')
        result = data.load_campaign(archive_campaign, protocol_path=self.protocol, diagnostic=True)
        self.assertEqual(result['rows'], expected['rows'])
        archive.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            data.load_campaign(archive_campaign, protocol_path=self.protocol)

    def test_path_escape_and_unfrozen_protocol_fail(self):
        with self.assertRaisesRegex(ValueError, 'Invalid attempt name'):
            data.Artifact(self.campaign, '../outside')
        write(self.protocol, {'status': 'DRAFT'})
        with self.assertRaisesRegex(ValueError, 'not frozen'):
            self.load()


if __name__ == '__main__':
    unittest.main()
