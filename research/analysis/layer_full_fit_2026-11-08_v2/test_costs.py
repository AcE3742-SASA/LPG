"""Small assertions for shared labels, missing costs and metadata-only accounting."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import costs


def run():
    row = dict(layer='ffn', dtype='fp16', seed=1, implementation='standard',
               train_group_ids=['a', 'b'], validation_group_ids=['v'])
    rows = [row, dict(row), dict(row, seed=2, train_group_ids=['b', 'c'])]
    result = costs.label_budget(rows)
    assert result['per_seed']['1']['round_records'] == 9
    assert result['all_seed_union']['round_records'] == 12
    component = dict(row, implementation='gemm1', train_group_ids=['a'], validation_group_ids=[])
    assert costs.label_budget(rows + [component])['all_seed_union']['round_records'] == 15
    assert costs.label_budget([dict(row, train_group_ids=[])])['all_seed_union']['round_records'] == 3
    for bad in (dict(row, validation_group_ids=['a']), dict(row, implementation='unfold')):
        try:
            costs.label_budget([bad])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid label access accepted')
    def evidence(seconds):
        return dict(seconds=seconds, evidence='measured.json', scope='controlled_gpu_wall')
    assert costs.break_even(None, evidence(10), evidence(5))['status'] == 'UNKNOWN'
    result = costs.break_even(evidence(20), evidence(10), evidence(5))
    assert (result['first_nonloss_task_count'], result['first_saving_task_count']) == (4, 5)
    assert costs.break_even(evidence(20), evidence(5), evidence(10))['status'] == 'NO_BREAK_EVEN'
    assert costs.fit_costs([dict(fit_seconds=2, predict_seconds=1), dict(fit_seconds=3)])['prediction_seconds'] is None
    for invalid in (float('nan'), -1, True):
        try:
            costs.break_even(evidence(invalid), evidence(10), evidence(5))
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid measured cost accepted')
    accessed = []
    ledgers = {
        'train_01_attempt_001': 2,
        'train_01_attempt_002': 10,
        'validation_01_attempt_001': 4,
        'test_01_attempt_001': 6,
    }
    class Artifact:
        def __init__(self, campaign, name, require_hashes=False):
            self.name = name
            self.hashes = {'cost_ledger.json': 'hash'}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def exists(self, name):
            return name in ('cost_ledger.json', 'summary.json') and self.name in ledgers
        def read(self, name):
            accessed.append(name)
            if name == 'summary.json':
                return dict(collection='COMPLETE', gate='PASS')
            assert name == 'cost_ledger.json'
            return dict(phase=self.name.split('_attempt_')[0], source_hashes={'x': 'y'},
                        collection_wall_seconds=ledgers[self.name], recorded=12, expected=12, scope='controlled run')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        state = dict(source_hashes={'x': 'y'}, batch_order=['train_01', 'validation_01', 'test_01'],
                     attempts={'train_01': ['train_01_attempt_001', 'train_01_attempt_002'],
                               'validation_01': ['validation_01_attempt_001'], 'test_01': ['test_01_attempt_001']})
        (path / 'campaign.json').write_text(json.dumps(state))
        with patch.dict(sys.modules, {'data': SimpleNamespace(Artifact=Artifact)}):
            result = costs.collection_costs(path)
            assert result['all_attempts']['verified_complete_attempt_sum_seconds'] == 22
            assert result['by_split']['test']['observed_controlled_wall_seconds'] == 6
            assert set(accessed) == {'summary.json', 'cost_ledger.json'}
            state['attempts']['train_01'].append('train_01_attempt_003')
            (path / 'campaign.json').write_text(json.dumps(state))
            result = costs.collection_costs(path)
            assert result['all_attempts']['verified_complete_attempt_sum_seconds'] is None
            assert result['all_attempts']['observed_controlled_wall_seconds'] == 22
            state['attempts']['train_01'].append('../outside')
            (path / 'campaign.json').write_text(json.dumps(state))
            try:
                costs.collection_costs(path)
            except ValueError:
                pass
            else:
                raise AssertionError('Unsafe attempt path accepted')
    print('PASS: cost accounting, shared labels and metadata-only tests')


def integration():
    import data
    from test_data import fixture, rehash, write
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _, campaign, _ = fixture(root)
        for phase in ('train_01', 'validation_01', 'test_01'):
            attempt = campaign / (phase + '_attempt_001')
            ledger = data.read(attempt / 'cost_ledger.json')
            ledger['scope'] = 'controlled run'
            write(attempt / 'cost_ledger.json', ledger)
            rehash(attempt)
        original = data.Artifact.read
        def guarded(self, name):
            assert name in {'cost_ledger.json', 'summary.json', 'result_hashes.json'}
            return original(self, name)
        with patch.object(data.Artifact, 'read', guarded):
            extracted = costs.collection_costs(campaign)
        assert extracted['all_attempts']['verified_complete_attempt_sum_seconds'] == 150
        zipped = root / 'zipped'
        zipped.mkdir()
        (zipped / 'campaign.json').write_bytes((campaign / 'campaign.json').read_bytes())
        for attempt in campaign.iterdir():
            if not attempt.is_dir():
                continue
            archive = zipped / (attempt.name + '_fixture.zip')
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
                for path in attempt.rglob('*'):
                    if path.is_file():
                        package.write(path, str(Path(attempt.name) / path.relative_to(attempt)))
            archive.with_suffix('.zip.sha256').write_text(data.sha(archive) + '  ' + archive.name + '\n')
        with patch.object(data.Artifact, 'read', guarded):
            assert costs.collection_costs(zipped) == extracted
        archive.write_bytes(b'broken')
        try:
            costs.collection_costs(zipped)
        except ValueError as error:
            assert 'checksum mismatch' in str(error)
        else:
            raise AssertionError('Broken archive accepted')
    print('PASS: actual Artifact directory/ZIP cost integration')


class CostTests(unittest.TestCase):
    def test_accounting(self):
        run()

    def test_artifact_integration(self):
        integration()


if __name__ == '__main__':
    run()
    integration()
