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
    before = {p: compare.sha(source/p) for p in ['campaign.json'] + [x['filename'] for x in frozen['batches'].values()]}
    calls = []
    def execute(command):
        phase, out = command[2], Path(command[3])
        calls.append((phase, out.name))
        assert not phase.startswith('train_')
        assert ('--collect-sealed-test' in command) == phase.startswith('test_')
        out.mkdir()
        if len(calls) == 1:
            assert phase == 'validation_01' and out.name.endswith('_002')
            (out/'failure.json').write_text('{"interrupted": true}')
            return 130
        manifest = dict(phase=phase, source_hashes=compare.read(compare.HERE/'release_hashes.json'))
        summary = dict(collection='COMPLETE', gate='PASS', missing=0, recorded=1080,
                       expected=1080, counts={'PASS': 1080}, restoration='COMMANDS_ACCEPTED', worker_complete=True)
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
        assert len(list(campaign.glob('*.zip'))) == 21
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
             'validation_01_attempt_002','validation_01_attempt_003']
        assert state['completed_batches'] == 30
        assert compare.read(campaign/'collection_summary.json')['review_batches'] == ['train_03','train_09']
        original = state_path.read_bytes()
        state['inherited_batches']['train_01']['sha256'] = '0'*64
        state_path.write_text(json.dumps(state))
        must_fail(lambda: collect_all.collect(campaign, execute))
        state_path.write_bytes(original)
        path = campaign/frozen['batches']['train_01']['filename']
        blob = path.read_bytes()
        path.write_bytes(b'corrupt')
        must_fail(lambda: collect_all.collect(campaign, execute))
        path.write_bytes(blob)
        (campaign/'train_01_attempt_001').mkdir()
        must_fail(lambda: collect_all.collect(campaign, execute))
    assert all(compare.sha(source/p)==value for p,value in before.items())
    result = dict(inherited_train_batches=20, preserved_archives=21, interrupted_validation_resumed=True,
                  gpu_calls=0, train_recollection_calls=0, new_batch_calls=11,
                  altered_state_corrupt_archive_and_shadow_directory_rejected=True,
                  original_files_unchanged=True)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    checks(parser.parse_args().source)
