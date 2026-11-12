"""Check v5 import/resume using 23 real ZIPs and 8 synthetic completed v4 ZIPs."""
from collections import Counter
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import collect_all
import compare
import inherit_campaign


def must_fail(call):
    try:
        call()
    except (RuntimeError, ValueError):
        return
    raise AssertionError('Invalid inheritance accepted')


def archive(path, attempt, blobs):
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for name, blob in blobs.items():
            package.writestr(attempt+'/'+name, blob)
        package.writestr(attempt+'/result_hashes.json', json.dumps(hashes))
    path.with_suffix('.zip.sha256').write_text(compare.sha(path)+'  '+path.name+'\n')


def build_fixture(directory):
    """Copy real prior evidence; missing eight completed v4 batches are opaque fixtures."""
    source = Path(directory)
    source.mkdir(parents=True, exist_ok=True)
    root = compare.HERE.parent
    old_raw = root/'analysis/layer_full_result_2026-11-09/raw'
    v3_raw = root/'analysis/layer_full_probe_result_2026-11-10/raw'
    v4_raw = root/'analysis/layer_full_test04_result_2026-11-10/raw'
    spec = inherit_campaign.spec()
    real = [(old_raw/item['filename']) for item in spec['batches'].values()]
    real += [(v3_raw/item['filename']) for item in spec['prior_failures']]
    real += [v4_raw/inherit_campaign.V4_FAILURE['filename']]
    assert len(real) == 23
    for path in real:
        shutil.copy2(path, source/path.name)
        (source/(path.name+'.sha256')).write_text(compare.sha(path)+'  '+path.name+'\n')
    shutil.copy2(old_raw/'campaign.json', source/'inherited_campaign.json')
    shutil.copy2(compare.HERE/'inherited_v3_campaign.json', source/'previous_campaign.json')
    shutil.copy2(compare.HERE/'inherited_v4_campaign.json', source/'campaign.json')
    previous = root/'layer_full_20261110_v4'
    sources = inherit_campaign.v4_sources()
    phases = compare.read(compare.HERE/'cases_by_phase.json')
    source_blobs = {'source/'+name:(previous/name).read_bytes() for name in sources}
    assert all(hashlib.sha256(source_blobs['source/'+name]).hexdigest() == digest
               for name,digest in sources.items())
    source_blobs['source/release_hashes.json'] = (previous/'release_hashes.json').read_bytes()
    for phase, attempt in inherit_campaign.v4_completed().items():
        cases = phases[phase]
        config = dict(compare.read(compare.HERE/'inherited_v4_protocol.json'), phase=phase)
        config['health_case_ids'] = [next(c['id'] for c in cases if c['layer']==layer)
                                    for layer in ('linear','ffn','conv1','conv3')]
        exclusions = [item for item in config['correctness_exclusions'] if item['phase']==phase]
        expected = 1062 if exclusions else 1080
        manifest = dict(kind='layer_full_20261110_v4', phase=phase, cases=cases, protocol=config,
                        source_hashes=sources, expected_measurements=expected,
                        planned_measurements=1080, correctness_exclusions=exclusions)
        summary = dict(collection='COMPLETE', gate='REVIEW' if exclusions else 'PASS',
                       missing=0, recorded=expected, expected=expected, planned=1080,
                       excluded=1080-expected, excluded_conditions=exclusions,
                       counts={'PASS':expected}, restoration='COMMANDS_ACCEPTED', worker_complete=True)
        blobs = dict(source_blobs, **{'manifest.json':json.dumps(manifest).encode(),
                                     'summary.json':json.dumps(summary).encode()})
        for case in cases:
            for dtype in config['dtypes']:
                if any(item['case_id']==case['id'] and item['dtype']==dtype for item in exclusions):
                    continue
                for rd in range(config['rounds']):
                    for implementation in compare.implementations(case, config):
                        name = compare.record_path(Path('.'), case, dtype, rd, implementation).as_posix()
                        blobs[name] = b'{}'
        archive(source/(attempt+'_synthetic.zip'), attempt, blobs)
    assert len(list(source.glob('*.zip'))) == 31
    return source


def checks(source):
    source = Path(source)
    before = {p.name:compare.sha(p) for p in source.iterdir() if p.is_file()}
    calls = []
    def execute(command):
        phase, out = command[2], Path(command[3])
        calls.append((phase, out.name))
        assert phase in ('test_04','test_05') and '--collect-sealed-test' in command
        out.mkdir()
        if len(calls) == 1:
            assert phase == 'test_04' and out.name.endswith('_002')
            (out/'failure.json').write_text('{"interrupted": true}')
            return 130
        config = dict(compare.read(compare.HERE/'protocol.json'), phase=phase)
        cases = compare.read(compare.HERE/'cases_by_phase.json')[phase]
        expected = compare.expected_records(cases, config)
        exclusions = compare.excluded_conditions(cases, config)
        manifest = dict(phase=phase, source_hashes=compare.read(compare.HERE/'release_hashes.json'))
        summary = dict(collection='COMPLETE', gate='REVIEW' if exclusions else 'PASS',
                       missing=0, recorded=expected, expected=expected, planned=1080,
                       excluded=1080-expected, excluded_conditions=exclusions,
                       counts={'PASS':expected}, restoration='COMMANDS_ACCEPTED', worker_complete=True)
        for name,value in [('manifest.json',manifest),('summary.json',summary)]:
            (out/name).write_text(json.dumps(value))
        path = out.with_name(out.name+'_fixture.zip')
        with zipfile.ZipFile(path,'x') as package:
            for name in ('manifest.json','summary.json'):
                package.write(out/name,out.name+'/'+name)
        (out/'archive_receipt.json').write_text(json.dumps(dict(filename=path.name,sha256=compare.sha(path))))
        return 2 if exclusions else 0
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
        runs = Path(directory)/'runs'
        campaign = inherit_campaign.prepare(source, runs)
        assert len(list(campaign.glob('*.zip'))) == 31
        assert inherit_campaign.prepare(source, runs) == campaign
        assert collect_all.collect(campaign, execute) == 130
        assert collect_all.collect(campaign, execute) == 2
        assert Counter(p for p,_ in calls) == dict(test_04=2,test_05=1) and len(calls)==3
        assert collect_all.collect(campaign, execute) == 2 and len(calls)==3
        state_path = campaign/'campaign.json'
        state = compare.read(state_path)
        assert state['attempts']['test_04'] == ['test_04_attempt_001','test_04_attempt_002','test_04_attempt_003']
        assert state['attempts']['validation_01'] == ['validation_01_attempt_001','validation_01_attempt_002','validation_01_attempt_003']
        assert state['completed_batches'] == 30
        final = compare.read(campaign/'collection_summary.json')
        assert final['review_batches'] == ['train_03','train_09','validation_01','test_04']
        assert (final['measurements'], final['excluded_measurements'], final['planned_measurements']) == (32364,36,32400)
        original = state_path.read_bytes()
        variants = []
        altered = copy.deepcopy(state);altered['inherited_batches']['train_01']['sha256']='0'*64;variants.append(altered)
        altered = copy.deepcopy(state);altered['inherited_v4_batches'].pop('test_03');variants.append(altered)
        altered = copy.deepcopy(state);altered['inherited_v4_campaign_sha256']='0'*64;variants.append(altered)
        altered = copy.deepcopy(state);altered['attempts']['test_04'].pop(0);variants.append(altered)
        for altered in variants:
            state_path.write_text(json.dumps(altered))
            must_fail(lambda:collect_all.collect(campaign, execute))
        state_path.write_bytes(original)
        for name in ('previous_campaign.json','v4_campaign.json'):
            path = campaign/name; blob = path.read_bytes();path.write_bytes(b'{}')
            must_fail(lambda:collect_all.collect(campaign, execute));path.write_bytes(blob)
        entry = state['inherited_v4_batches']['validation_02']
        path = campaign/entry['filename']; original_zip = path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(original_zip)) as package:
            prefix=entry['attempt']+'/'
            blobs = {name[len(prefix):]:package.read(name) for name in package.namelist()
                     if name!=prefix+'result_hashes.json'}
        altered = dict(blobs); manifest=json.loads(altered['manifest.json']);manifest['protocol']['tolerances']['fp32'][0]*=2
        altered['manifest.json']=json.dumps(manifest).encode()
        variants = [altered]
        altered = dict(blobs);summary=json.loads(altered['summary.json']);summary['excluded_conditions']=[{'case_id':'forged'}]
        altered['summary.json']=json.dumps(summary).encode();variants.append(altered)
        altered = dict(blobs);altered.pop(next(name for name in altered if name.startswith('records/')));variants.append(altered)
        for altered in variants:
            archive(path,entry['attempt'],altered)
            must_fail(lambda:inherit_campaign.check_v4_archive(campaign,dict(entry,sha256=compare.sha(path))))
        path.write_bytes(original_zip)
        path.with_suffix('.zip.sha256').write_text(entry['sha256']+'  '+path.name+'\n')
        backup = path.with_suffix('.saved');path.rename(backup);path.symlink_to(backup)
        must_fail(lambda:collect_all.collect(campaign,execute));path.unlink();backup.rename(path)
        shadow=campaign/entry['attempt'];shadow.mkdir()
        must_fail(lambda:collect_all.collect(campaign,execute));shadow.rmdir()
        prior=campaign/inherit_campaign.V4_FAILURE['filename'];blob=prior.read_bytes();prior.write_bytes(b'corrupt')
        must_fail(lambda:collect_all.collect(campaign,execute));prior.write_bytes(blob)
        assert len(calls)==3
    assert all(compare.sha(source/name)==digest for name,digest in before.items())
    result = dict(real_prior_archives=23, synthetic_completed_v4_archives=8,
                  preserved_archives=31, inherited_completed_batches=28, inherited_train_batches=20,
                  interrupted_test04_resumed=True, first_attempt='test_04_attempt_002',
                  gpu_calls=0, completed_batch_recollection_calls=0, new_batch_calls=3,
                  measured_records=32364, excluded_records=36, all_prior_failures_preserved=True,
                  altered_state_metadata_inventory_archive_symlink_shadow_rejected=True,
                  original_files_unchanged=True)
    print(json.dumps(result,sort_keys=True))
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path,nargs='?')
    args=parser.parse_args()
    if args.source:
        checks(args.source)
    else:
        with tempfile.TemporaryDirectory() as directory:
            checks(build_fixture(Path(directory)/'source'))
