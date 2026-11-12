"""Check the v4-to-v5 provenance layer with synthetic public metadata only."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import costs
import data
from test_data import write, rehash
from test_mixed import pack


class V5InheritanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package, self.campaign = self.root/'package', self.root/'campaign'
        actual = data.PACKAGE
        research = actual.parent
        names = ('protocol.json', 'cases_by_phase.json', 'inheritance.json',
                 'inherited_v2_protocol.json', 'inherited_v2_release_hashes.json',
                 'inherited_v3_protocol.json', 'inherited_v3_release_hashes.json', 'inherited_v3_campaign.json',
                 'inherited_v4_protocol.json', 'inherited_v4_release_hashes.json', 'inherited_v4_campaign.json')
        self.package.mkdir(); self.campaign.mkdir()
        for name in names:
            shutil.copy2(actual/name, self.package/name)
        sources = {name: data.sha(self.package/name) for name in names}
        write(self.package/'release_hashes.json', sources)
        shutil.copy2(research/'analysis/layer_full_result_2026-11-09/raw/campaign.json', self.campaign/'inherited_campaign.json')
        shutil.copy2(actual/'inherited_v3_campaign.json', self.campaign/'previous_campaign.json')
        shutil.copy2(actual/'inherited_v4_campaign.json', self.campaign/'v4_campaign.json')
        old_state = data.read(self.campaign/'v4_campaign.json')
        old_sources = old_state['source_hashes']
        entries = {}
        for phase in old_state['batch_order'][20:28]:
            attempt = self.campaign/old_state['attempts'][phase][-1]
            count = 1062 if phase == 'validation_01' else 1080
            write(attempt/'summary.json', dict(collection='COMPLETE', gate='REVIEW' if count==1062 else 'PASS'))
            write(attempt/'cost_ledger.json', dict(phase=phase, source_hashes=old_sources,
                  recorded=count, expected=count, collection_wall_seconds=10.0, scope='synthetic public cost'))
            rehash(attempt)
            archive = pack(attempt)
            entries[phase] = dict(phase=phase, attempt=attempt.name, filename=archive.name,
                sha256=data.sha(archive), complete=True, source_kind='layer_full_20261110_v4', source_hashes=old_sources)
            shutil.rmtree(attempt)
        spec = data.read(actual/'inheritance.json')
        filenames = [spec['batches']['validation_01']['filename'], spec['prior_failures'][0]['filename'],
                     'test_04_attempt_001_c1bc4101.zip']
        folders = ['analysis/layer_full_result_2026-11-09/raw', 'analysis/layer_full_probe_result_2026-11-10/raw',
                   'analysis/layer_full_test04_result_2026-11-10/raw']
        for name, folder in zip(filenames, folders):
            destination = self.campaign/name
            shutil.copy2(research/folder/name, destination)
            destination.with_suffix('.zip.sha256').write_text(data.sha(destination)+'  '+name+'\n')
        failure = dict(phase='test_04', attempt='test_04_attempt_001', filename=filenames[-1],
            sha256=data.sha(self.campaign/filenames[-1]), complete=False, source_kind='layer_full_20261110_v4', source_hashes=old_sources)
        self.state = dict(old_state, source_hashes=sources, inherited_v4_batches=entries,
            inherited_v4_failures=[failure], inherited_v4_campaign_sha256=data.sha(self.campaign/'v4_campaign.json'))
        write(self.campaign/'campaign.json', self.state)
        self.patcher = patch.object(data, 'PACKAGE', self.package)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop(); self.temp.cleanup()

    def test_all_eight_v4_completions_and_three_failures_have_provenance(self):
        inheritance = data.validate_inheritance(self.campaign, self.state)
        self.assertEqual(len(inheritance['v4_batches']), 8)
        for phase, item in self.state['inherited_v4_batches'].items():
            self.assertEqual(data.inherited_attempt(inheritance, phase, item['attempt']), item)
        original = data.Artifact.read
        def public_only(artifact, name):
            self.assertIn(name, ('result_hashes.json', 'summary.json', 'cost_ledger.json'))
            return original(artifact, name)
        with patch.object(data.Artifact, 'read', public_only):
            report = costs.collection_costs(self.campaign)
        failures = [r for r in report['attempt_details'] if r['collection']=='INCOMPLETE']
        self.assertEqual([(r['attempt'], r['recorded']) for r in failures],
            [('validation_01_attempt_001',272), ('validation_01_attempt_002',0), ('test_04_attempt_001',30)])
        self.assertEqual(report['all_attempts']['attempts'], 31)
        self.assertEqual(sum(r['recorded'] for r in report['attempt_details'] if r['collection']=='COMPLETE'), 8622)
        self.assertIsNone(report['all_attempts']['verified_complete_attempt_sum_seconds'])

    def test_completed_validation_dispatches_to_its_original_v4_protocol(self):
        frozen = self.root/'frozen.json'
        write(frozen, dict(status='LAYER_FULL_PROTOCOL_FROZEN', package_release_hashes=self.state['source_hashes']))
        calls = []
        def load(artifact, phase, cases, protocol, sources, diagnostic):
            calls.append((phase, protocol['version'], sources))
            return [], [], {'public': True}
        with patch.object(data, '_load_batch', load):
            result = data.load_campaign(self.campaign, split='validation', protocol_path=frozen)
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(version=='layer_full_20261110_v4' for _,version,_ in calls))
        self.assertTrue(all(item['inherited'] for item in result['provenance']['inputs'].values()))
        self.assertTrue(all(sources==data.read(self.package/'inherited_v4_release_hashes.json') for _,_,sources in calls))

    def test_changed_v4_history_identity_or_failure_is_rejected(self):
        for alter in [lambda s: s.pop('inherited_v4_batches'),
                      lambda s: s['attempts']['validation_01'].pop(),
                      lambda s: s['attempts']['validation_01'].append('validation_01_attempt_004'),
                      lambda s: s['inherited_v4_batches']['test_01'].update(source_kind='layer_full_20261110_v5'),
                      lambda s: s['inherited_v4_failures'][0].update(sha256='0'*64)]:
            state = copy.deepcopy(self.state); alter(state)
            with self.assertRaisesRegex(ValueError, 'Inherited v4'):
                data.validate_inheritance(self.campaign, state)
        path = self.campaign/'v4_campaign.json'; path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'Inherited v4 campaign provenance'):
            data.validate_inheritance(self.campaign, self.state)

    def test_rechecksummed_v4_archive_and_shadow_directory_are_rejected(self):
        entry = self.state['inherited_v4_batches']['validation_01']
        path = self.campaign/entry['filename']; path.write_bytes(path.read_bytes()+b'changed')
        path.with_suffix('.zip.sha256').write_text(data.sha(path)+'  '+path.name+'\n')
        with self.assertRaisesRegex(ValueError, 'pinned ZIP'):
            costs.collection_costs(self.campaign)
        write(self.campaign/entry['attempt']/'result_hashes.json', {})
        with self.assertRaisesRegex(ValueError, 'pinned ZIP'):
            costs.collection_costs(self.campaign)


if __name__ == '__main__':
    unittest.main()
