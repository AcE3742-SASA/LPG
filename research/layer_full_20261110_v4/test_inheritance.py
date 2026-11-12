"""Exercise audited ZIP import and interrupted continuation without GPU work."""
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import collect_all
import compare
import inherit_campaign


def must_fail(call):
    try:
        call()
    except RuntimeError:
        return
    raise AssertionError('Invalid inheritance accepted')


def checks(source):
    source = Path(source)
    frozen = inherit_campaign.spec()
    inputs = ['campaign.json', 'inherited_campaign.json'] + [x['filename'] for x in
              list(frozen['batches'].values()) + frozen['prior_failures']]
    before = {p: compare.sha(source/p) for p in inputs}
    calls = []
    def execute(command):
        phase, out = command[2], Path(command[3])
        calls.append((phase, out.name))
        assert not phase.startswith('train_')
        assert ('--collect-sealed-test' in command) == phase.startswith('test_')
        out.mkdir()
        if len(calls) == 1:
            assert phase == 'validation_01' and out.name.endswith('_003')
            (out/'failure.json').write_text('{"interrupted": true}')
            return 130
        manifest = dict(phase=phase, source_hashes=compare.read(compare.HERE/'release_hashes.json'))
        config = dict(compare.read(compare.HERE/'protocol.json'), phase=phase)
        cases = compare.read(compare.HERE/'cases_by_phase.json')[phase]
        expected = compare.expected_records(cases, config)
        exclusions = compare.excluded_conditions(cases, config)
        summary = dict(collection='COMPLETE', gate='REVIEW' if exclusions else 'PASS',
                       missing=0, recorded=expected, expected=expected, planned=1080,
                       excluded=1080-expected, excluded_conditions=exclusions,
                       counts={'PASS': expected}, restoration='COMMANDS_ACCEPTED', worker_complete=True)
        for name, value in [('manifest.json', manifest), ('summary.json', summary)]:
            (out/name).write_text(json.dumps(value))
        archive = out.with_name(out.name+'_fixture.zip')
        with zipfile.ZipFile(archive, 'x') as package:
            for name in ('manifest.json','summary.json'):
                package.write(out/name,out.name+'/'+name)
        (out/'archive_receipt.json').write_text(json.dumps(dict(filename=archive.name,sha256=compare.sha(archive))))
        return 0
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
        runs = Path(directory)/'runs'
        campaign = inherit_campaign.prepare(source, runs)
        assert len(list(campaign.glob('*.zip'))) == 22
        assert inherit_campaign.prepare(source, runs) == campaign
        assert collect_all.collect(campaign, execute) == 130
        assert collect_all.collect(campaign, execute) == 2
        assert Counter(p for p,_ in calls) == dict(validation_01=2, validation_02=1, validation_03=1,
             validation_04=1, validation_05=1, test_01=1, test_02=1, test_03=1, test_04=1, test_05=1)
        assert len(calls) == 11
        assert collect_all.collect(campaign, execute) == 2 and len(calls) == 11
        state_path = campaign/'campaign.json'
        state = compare.read(state_path)
        assert state['attempts']['validation_01'] == ['validation_01_attempt_001',
             'validation_01_attempt_002','validation_01_attempt_003','validation_01_attempt_004']
        assert state['completed_batches'] == 30
        final = compare.read(campaign/'collection_summary.json')
        assert final['review_batches'] == ['train_03','train_09','validation_01']
        assert final['measurements'] == 32382 and final['excluded_measurements'] == 18
        assert final['planned_measurements'] == 32400
        original = state_path.read_bytes()
        state['inherited_batches']['train_01']['sha256'] = '0'*64
        state_path.write_text(json.dumps(state))
        must_fail(lambda: collect_all.collect(campaign, execute))
        state_path.write_bytes(original)
        previous = campaign/'previous_campaign.json'
        original_previous = previous.read_bytes()
        previous.write_bytes(b'{}')
        must_fail(lambda: collect_all.collect(campaign, execute))
        previous.write_bytes(original_previous)
        prior = campaign/frozen['prior_failures'][0]['filename']
        original_prior = prior.read_bytes()
        prior.write_bytes(b'corrupt')
        must_fail(lambda: collect_all.collect(campaign, execute))
        prior.write_bytes(original_prior)
        path = campaign/frozen['batches']['train_01']['filename']
        blob = path.read_bytes()
        path.write_bytes(b'corrupt')
        must_fail(lambda: collect_all.collect(campaign, execute))
        path.write_bytes(blob)
        (campaign/'train_01_attempt_001').mkdir()
        must_fail(lambda: collect_all.collect(campaign, execute))
    assert all(compare.sha(source/p)==value for p,value in before.items())
    result = dict(inherited_train_batches=20, preserved_archives=22, interrupted_validation_resumed=True,
                  gpu_calls=0, train_recollection_calls=0, new_batch_calls=11,
                  measured_records=32382, excluded_records=18, first_attempt='validation_01_attempt_003',
                  both_prior_failures_and_campaign_preserved=True,
                  altered_state_corrupt_archive_and_shadow_directory_rejected=True,
                  original_files_unchanged=True)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    checks(parser.parse_args().source)
