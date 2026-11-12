"""Exercise pinned v2 ZIP reuse with two failed releases and v4 retries and metadata-only costs."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import costs
import data
from test_data import fixture, rehash, write


def pack(attempt):
    archive = attempt.with_name(attempt.name + '_fixture.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in attempt.rglob('*'):
            if path.is_file():
                output.write(path, str(Path(attempt.name) / path.relative_to(attempt)))
    archive.with_suffix('.zip.sha256').write_text(data.sha(archive) + '  ' + archive.name + '\n')
    return archive


def mixed_fixture(root):
    package, campaign, frozen = fixture(root)
    protocol = data.read(package / 'protocol.json')
    old_protocol = {k: v for k, v in protocol.items() if k != 'correctness_exclusions'}
    old_protocol['version'] = 'layer_full_20261108_v2'
    old_package = root / 'old_package'
    write(old_package / 'protocol.json', old_protocol)
    shutil.copy2(package / 'cases_by_phase.json', old_package / 'cases_by_phase.json')
    old_sources = {name: data.sha(old_package / name) for name in ('protocol.json', 'cases_by_phase.json')}
    state = data.read(campaign / 'campaign.json')
    old_state = dict(source_hashes=old_sources, batch_order=state['batch_order'], completed_batches=1,
                     attempts={p: list(state['attempts'][p]) for p in ('train_01', 'validation_01')})
    write(campaign / 'inherited_campaign.json', old_state)
    retry = campaign / 'validation_01_attempt_003'
    failure = campaign / 'validation_01_attempt_002'
    shutil.copytree(campaign / 'validation_01_attempt_001', retry)
    shutil.copytree(campaign / 'validation_01_attempt_001', failure)
    batches = {}
    for phase in ('train_01', 'validation_01'):
        attempt = campaign / (phase + '_attempt_001')
        manifest = data.read(attempt / 'manifest.json')
        manifest.update(kind=old_protocol['version'], source_hashes=old_sources,
                        protocol=dict(old_protocol, phase=phase, health_case_ids=manifest['protocol']['health_case_ids']))
        for field in ('planned_measurements', 'correctness_exclusions'):
            manifest.pop(field)
        write(attempt / 'manifest.json', manifest)
        for name in old_sources:
            shutil.copy2(old_package / name, attempt / 'source' / name)
        ledger = data.read(attempt / 'cost_ledger.json')
        ledger.update(source_hashes=old_sources, scope='controlled run')
        summary = data.read(attempt / 'summary.json')
        for field in ('planned', 'excluded', 'excluded_conditions'):
            summary.pop(field)
        if phase == 'validation_01':
            ledger.update(recorded=0, collection_wall_seconds=7.0)
            summary.update(collection='INCOMPLETE', gate='FAIL', recorded=0, missing=summary['expected'], counts={})
            write(attempt / 'failure.json', {'error': 'synthetic correctness failure'})
            shutil.rmtree(attempt / 'records')
            shutil.rmtree(attempt / 'restricted_explicit')
        write(attempt / 'summary.json', summary)
        write(attempt / 'cost_ledger.json', ledger)
        rehash(attempt)
        archive = pack(attempt)
        batches[phase] = dict(attempt=attempt.name, filename=archive.name, sha256=data.sha(archive),
                              complete=phase == 'train_01')
        shutil.rmtree(attempt)
    prior_package = root / 'prior_package'
    prior_protocol = dict(old_protocol, version='layer_full_20261110_v3')
    write(prior_package / 'protocol.json', prior_protocol)
    shutil.copy2(package / 'cases_by_phase.json', prior_package / 'cases_by_phase.json')
    prior_sources = {name: data.sha(prior_package / name) for name in ('protocol.json', 'cases_by_phase.json')}
    manifest = data.read(failure / 'manifest.json')
    manifest.update(kind=prior_protocol['version'], source_hashes=prior_sources,
                    protocol=dict(prior_protocol, phase='validation_01', health_case_ids=manifest['protocol']['health_case_ids']))
    for field in ('planned_measurements', 'correctness_exclusions'):
        manifest.pop(field)
    write(failure / 'manifest.json', manifest)
    for name in prior_sources:
        shutil.copy2(prior_package / name, failure / 'source' / name)
    ledger = data.read(failure / 'cost_ledger.json')
    ledger.update(source_hashes=prior_sources, recorded=0, collection_wall_seconds=11.0, scope='controlled run')
    write(failure / 'cost_ledger.json', ledger)
    summary = data.read(failure / 'summary.json')
    summary.update(collection='INCOMPLETE', gate='FAIL', recorded=0, missing=summary['expected'], counts={})
    write(failure / 'summary.json', summary)
    write(failure / 'failure.json', {'error': 'synthetic v3 correctness failure'})
    shutil.rmtree(failure / 'records')
    shutil.rmtree(failure / 'restricted_explicit')
    rehash(failure)
    archive = pack(failure)
    prior_failure = dict(phase='validation_01', attempt=failure.name, filename=archive.name,
                         sha256=data.sha(archive), source_kind=prior_protocol['version'], source_hashes=prior_sources)
    shutil.rmtree(failure)
    inheritance = dict(source_kind=old_protocol['version'], source_campaign='synthetic_previous_campaign',
                       campaign_sha256=data.sha(campaign / 'inherited_campaign.json'), source_hashes=old_sources,
                       batches=batches, prior_failures=[prior_failure])
    write(package / 'inheritance.json', inheritance)
    write(package / 'inherited_v2_protocol.json', old_protocol)
    write(package / 'inherited_v2_release_hashes.json', old_sources)
    write(package / 'inherited_v3_protocol.json', prior_protocol)
    write(package / 'inherited_v3_release_hashes.json', prior_sources)
    prior_state = copy.deepcopy(old_state)
    prior_state.update(source_hashes=prior_sources, inherited_campaign_sha256=inheritance['campaign_sha256'],
                       inherited_batches={'train_01': batches['train_01']})
    prior_state['attempts']['validation_01'].append(failure.name)
    write(package / 'inherited_v3_campaign.json', prior_state)
    shutil.copy2(package / 'inherited_v3_campaign.json', campaign / 'previous_campaign.json')
    names = ('protocol.json', 'cases_by_phase.json', 'inheritance.json',
             'inherited_v2_protocol.json', 'inherited_v2_release_hashes.json',
             'inherited_v3_protocol.json', 'inherited_v3_release_hashes.json', 'inherited_v3_campaign.json')
    sources = {name: data.sha(package / name) for name in names}
    write(package / 'release_hashes.json', sources)
    write(frozen, dict(status='LAYER_FULL_PROTOCOL_FROZEN', package_release_hashes=sources))
    state.update(source_hashes=sources, inherited_campaign_sha256=inheritance['campaign_sha256'],
                 inherited_batches={'train_01': batches['train_01']},
                 previous_campaign_sha256=data.sha(campaign / 'previous_campaign.json'))
    state['attempts']['validation_01'] += [failure.name, retry.name]
    write(campaign / 'campaign.json', state)
    for attempt in (retry, campaign / 'test_01_attempt_001'):
        manifest = data.read(attempt / 'manifest.json')
        manifest['source_hashes'] = sources
        write(attempt / 'manifest.json', manifest)
        ledger = data.read(attempt / 'cost_ledger.json')
        ledger.update(source_hashes=sources, scope='controlled run')
        write(attempt / 'cost_ledger.json', ledger)
        for name in names:
            shutil.copy2(package / name, attempt / 'source' / name)
        rehash(attempt)
    return package, campaign, frozen


class MixedCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package, self.campaign, self.frozen = mixed_fixture(self.root)
        self.patcher = patch.object(data, 'PACKAGE', self.package)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def load(self):
        return data.load_campaign(self.campaign, protocol_path=self.frozen, diagnostic=True)

    def test_pinned_train_and_v4_retry_keep_both_failed_costs(self):
        result = self.load()
        self.assertEqual((len(result['rows']), len(result['excluded'])), (72, 0))
        inputs = result['provenance']['inputs']
        self.assertTrue(inputs['train_01']['inherited'])
        self.assertEqual(inputs['train_01']['collection_version'], 'layer_full_20261108_v2')
        self.assertFalse(inputs['validation_01']['inherited'])
        self.assertEqual(inputs['validation_01']['attempt'], 'validation_01_attempt_003')
        original = data.Artifact.read
        def metadata_only(artifact, name):
            self.assertIn(name, ('result_hashes.json', 'summary.json', 'cost_ledger.json'))
            return original(artifact, name)
        with patch.object(data.Artifact, 'read', metadata_only):
            report = costs.collection_costs(self.campaign)
        self.assertEqual(report['all_attempts']['verified_complete_attempt_sum_seconds'], 168.0)
        self.assertEqual(report['by_split']['validation']['observed_controlled_wall_seconds'], 68.0)
        self.assertEqual(report['all_attempts']['attempts'], 5)
        with self.assertRaisesRegex(ValueError, 'explicit authorization'):
            data.load_campaign(self.campaign, split='test', protocol_path=self.frozen)

    def test_changed_metadata_or_dropped_history_is_rejected(self):
        path = self.campaign / 'campaign.json'
        original = data.read(path)
        for field in ('inherited_batches', 'inherited_campaign_sha256', 'previous_campaign_sha256'):
            altered = copy.deepcopy(original)
            del altered[field]
            write(path, altered)
            with self.assertRaisesRegex(ValueError, 'Inherited|Previous'):
                self.load()
        altered = copy.deepcopy(original)
        altered['attempts']['validation_01'] = ['validation_01_attempt_003']
        write(path, altered)
        with self.assertRaisesRegex(ValueError, 'history changed'):
            costs.collection_costs(self.campaign)

    def test_rechecksummed_zip_or_shadow_directory_cannot_replace_pinned_train(self):
        entry = data.read(self.package / 'inheritance.json')['batches']['train_01']
        archive = self.campaign / entry['filename']
        with zipfile.ZipFile(archive, 'a') as output:
            output.writestr(entry['attempt'] + '/extra.txt', 'changed')
        archive.with_suffix('.zip.sha256').write_text(data.sha(archive) + '  ' + archive.name + '\n')
        with self.assertRaisesRegex(ValueError, 'pinned ZIP'):
            self.load()
        shadow = self.campaign / entry['attempt']
        write(shadow / 'result_hashes.json', {})
        with self.assertRaisesRegex(ValueError, 'pinned ZIP'):
            costs.collection_costs(self.campaign)

    def test_failed_inherited_validation_and_wrong_current_source_are_rejected(self):
        path = self.campaign / 'campaign.json'
        state = data.read(path)
        state['attempts']['validation_01'].pop()
        write(path, state)
        with self.assertRaisesRegex(ValueError, 'completed v4 retry'):
            self.load()
        state['attempts']['validation_01'].append('validation_01_attempt_003')
        write(path, state)
        attempt = self.campaign / 'validation_01_attempt_003'
        manifest = data.read(attempt / 'manifest.json')
        manifest['kind'] = 'layer_full_20261108_v2'
        write(attempt / 'manifest.json', manifest)
        rehash(attempt)
        with self.assertRaisesRegex(ValueError, 'frozen collection'):
            self.load()

    def test_unpinned_old_campaign_cannot_bypass_cost_source_validation(self):
        original = data.read(self.campaign / 'inherited_campaign.json')
        write(self.campaign / 'campaign.json', original)
        (self.campaign / 'inherited_campaign.json').unlink()
        with self.assertRaisesRegex(ValueError, 'release mismatch'):
            costs.collection_costs(self.campaign)

    def test_previous_campaign_and_v3_failure_zip_are_pinned(self):
        previous = self.campaign / 'previous_campaign.json'
        original = previous.read_bytes()
        previous.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'Previous v3 campaign provenance'):
            self.load()
        previous.write_bytes(original)
        entry = data.read(self.package / 'inheritance.json')['prior_failures'][0]
        archive = self.campaign / entry['filename']
        with zipfile.ZipFile(archive, 'a') as output:
            output.writestr(entry['attempt'] + '/extra.txt', 'changed')
        archive.with_suffix('.zip.sha256').write_text(data.sha(archive) + '  ' + archive.name + '\n')
        with self.assertRaisesRegex(ValueError, 'pinned ZIP'):
            costs.collection_costs(self.campaign)


if __name__ == '__main__':
    unittest.main()
