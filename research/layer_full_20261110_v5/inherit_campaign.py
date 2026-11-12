"""Preserve 28 completed batches and all prior attempts in a separate v5 campaign."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import compare

V4_FAILURE = dict(phase='test_04', attempt='test_04_attempt_001',
    filename='test_04_attempt_001_c1bc4101.zip',
    sha256='047d2e27fb52c38978cc8ff620907370ad2a81b1e90e97713eb5a2c78e1cfc7f', complete=False)


def v4_sources():
    return compare.read(compare.HERE/'inherited_v4_release_hashes.json')


def v4_completed():
    prior = compare.read(compare.HERE/'inherited_v4_campaign.json')
    return {phase: prior['attempts'][phase][-1] for phase in prior['batch_order'][20:28]}


def check_v4_archive(campaign, entry, *, allow_directory=False):
    phase, attempt, filename = entry['phase'], entry['attempt'], entry['filename']
    if (Path(filename).name != filename or not filename.startswith(attempt+'_')
            or not filename.endswith('.zip') or entry['source_kind'] != 'layer_full_20261110_v4'
            or entry['source_hashes'] != v4_sources()):
        raise RuntimeError('Invalid inherited v4 archive identity')
    path = Path(campaign)/filename
    if (path.is_symlink() or not path.is_file() or compare.sha(path) != entry['sha256']
            or (not allow_directory and (Path(campaign)/attempt).exists())):
        raise RuntimeError('Inherited v4 archive missing, changed or shadowed: '+phase)
    with zipfile.ZipFile(path) as archive:
        prefix = attempt+'/'
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError('Duplicate archive members')
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if (name.is_absolute() or '..' in name.parts or '\\' in info.filename
                    or name.parts[0] != attempt or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise RuntimeError('Unsafe archive member')
        hashes = json.loads(archive.read(prefix+'result_hashes.json'))
        if set(names) != {prefix+n for n in hashes} | {prefix+'result_hashes.json'}:
            raise RuntimeError('Archive inventory differs from its hashes')
        for name, digest in hashes.items():
            # Hash opaque bytes; sealed timing values are not decoded here.
            if hashlib.sha256(archive.read(prefix+name)).hexdigest() != digest:
                raise RuntimeError('Inherited v4 result checksum mismatch: '+name)
        manifest = json.loads(archive.read(prefix+'manifest.json'))
        summary = json.loads(archive.read(prefix+'summary.json'))
        cases = compare.read(compare.HERE/'cases_by_phase.json')[phase]
        protocol = dict(compare.read(compare.HERE/'inherited_v4_protocol.json'), phase=phase)
        protocol['health_case_ids'] = [next(c['id'] for c in cases if c['layer']==kind)
                                      for kind in ('linear','ffn','conv1','conv3')]
        exclusions = [e for e in protocol['correctness_exclusions'] if e['phase']==phase]
        expected = 1062 if exclusions else 1080
        if (manifest['kind'] != entry['source_kind'] or manifest['phase'] != phase
                or manifest['source_hashes'] != v4_sources() or manifest['cases'] != cases
                or manifest['protocol'] != protocol or manifest['expected_measurements'] != expected
                or manifest['planned_measurements'] != 1080
                or manifest['correctness_exclusions'] != exclusions):
            raise RuntimeError('Inherited v4 manifest mismatch: '+phase)
        for name, digest in v4_sources().items():
            if hashes.get('source/'+name) != digest:
                raise RuntimeError('Inherited v4 source snapshot mismatch: '+name)
        if json.loads(archive.read(prefix+'source/release_hashes.json')) != v4_sources():
            raise RuntimeError('Inherited v4 embedded release mismatch')
        if entry['complete']:
            wanted = {('records/' if implementation=='standard' else 'restricted_explicit/records/')
                      +compare.key(case,dtype,rd,implementation)+'.json'
                      for case in cases for dtype in protocol['dtypes']
                      if not any((e['case_id'],e['dtype'])==(case['id'],dtype) for e in exclusions)
                      for rd in range(protocol['rounds']) for implementation in compare.implementations(case,protocol)}
            recorded = {name for name in hashes if name.startswith(('records/','restricted_explicit/records/'))}
            if recorded != wanted:
                raise RuntimeError('Inherited v4 measurement inventory mismatch: '+phase)
            if not (summary['collection']=='COMPLETE' and summary['gate'] in ('PASS','REVIEW')
                    and summary['expected']==summary['recorded']==expected and summary['missing']==0
                    and summary['planned']==1080 and summary['excluded']==1080-expected
                    and summary['excluded_conditions']==exclusions
                    and (not exclusions or summary['gate']=='REVIEW')
                    and summary['counts'].get('FAIL',0)==0
                    and sum(summary['counts'].values())==expected
                    and summary['restoration']=='COMMANDS_ACCEPTED' and summary['worker_complete']):
                raise RuntimeError('Inherited v4 batch is incomplete: '+phase)
        elif (any(entry.get(k)!=v for k,v in V4_FAILURE.items())
                or summary['collection']!='INCOMPLETE' or summary['recorded']!=30 or summary['gate']!='FAIL'):
            raise RuntimeError('Prior test_04 failure differs')
        return summary


def spec():
    return compare.read(compare.HERE / 'inheritance.json')


def inherited_batches():
    return {phase: item for phase, item in spec()['batches'].items() if item['complete']}


def check_archive(campaign, phase):
    frozen = spec()
    item = frozen['batches'][phase]
    path = Path(campaign) / item['filename']
    if path.is_symlink() or not path.is_file() or compare.sha(path) != item['sha256']:
        raise RuntimeError('Inherited archive missing or checksum mismatch: ' + phase)
    if (Path(campaign) / item['attempt']).exists():
        raise RuntimeError('Inherited ZIP shadowed by an attempt directory: ' + phase)
    with zipfile.ZipFile(path) as archive:
        prefix = item['attempt'] + '/'
        manifest = json.loads(archive.read(prefix + 'manifest.json'))
        summary = json.loads(archive.read(prefix + 'summary.json'))
        if (manifest['kind'] != frozen['source_kind'] or manifest['phase'] != phase
                or manifest['source_hashes'] != frozen['source_hashes']
                or manifest['cases'] != compare.read(compare.HERE / 'cases_by_phase.json')[phase]):
            raise RuntimeError('Inherited manifest mismatch: ' + phase)
        if item['complete'] and not (summary['collection'] == 'COMPLETE'
                and summary['gate'] in ('PASS', 'REVIEW') and summary['missing'] == 0
                and summary['recorded'] == summary['expected'] == 1080
                and summary['restoration'] == 'COMMANDS_ACCEPTED' and summary['worker_complete']):
            raise RuntimeError('Inherited batch is incomplete: ' + phase)
    return summary


def check_prior_failure(campaign, item):
    path = Path(campaign) / item['filename']
    if (path.is_symlink() or not path.is_file() or compare.sha(path) != item['sha256']
            or (Path(campaign) / item['attempt']).exists()):
        raise RuntimeError('Prior failure archive missing, changed or shadowed')
    with zipfile.ZipFile(path) as archive:
        prefix = item['attempt'] + '/'
        manifest = json.loads(archive.read(prefix + 'manifest.json'))
        summary = json.loads(archive.read(prefix + 'summary.json'))
        if (manifest['kind'] != item['source_kind'] or manifest['phase'] != item['phase']
                or manifest['source_hashes'] != item['source_hashes']
                or summary['collection'] != 'INCOMPLETE' or summary['recorded'] != 0):
            raise RuntimeError('Prior failure metadata mismatch')


def validate_state(campaign, state):
    frozen = spec()
    original = Path(campaign) / 'inherited_campaign.json'
    if (state.get('inherited_batches') != inherited_batches()
            or state.get('inherited_campaign_sha256') != frozen['campaign_sha256']
            or original.is_symlink() or not original.is_file()
            or compare.sha(original) != frozen['campaign_sha256']):
        raise RuntimeError('Inherited campaign provenance mismatch')
    old = compare.read(original)
    if old['source_hashes'] != frozen['source_hashes'] or old['batch_order'] != state['batch_order']:
        raise RuntimeError('Inherited campaign release or order mismatch')
    for phase, item in frozen['batches'].items():
        attempts = state['attempts'].get(phase, [])
        if (not attempts or attempts[0] != item['attempt']
                or (item['complete'] and attempts != [item['attempt']])):
            raise RuntimeError('Inherited attempt history changed: ' + phase)
        check_archive(campaign, phase)
    expected = compare.HERE / 'inherited_v3_campaign.json'
    previous = Path(campaign) / 'previous_campaign.json'
    if (previous.is_symlink() or not previous.is_file() or compare.sha(previous) != compare.sha(expected)
            or state.get('previous_campaign_sha256') != compare.sha(expected)):
        raise RuntimeError('Previous v3 campaign provenance mismatch')
    prior_state = compare.read(previous)
    if prior_state['source_hashes'] != compare.read(compare.HERE / 'inherited_v3_release_hashes.json'):
        raise RuntimeError('Previous v3 release mismatch')
    for phase, attempts in prior_state['attempts'].items():
        if state['attempts'].get(phase, [])[:len(attempts)] != attempts:
            raise RuntimeError('Previous attempt history changed: ' + phase)
    for item in frozen['prior_failures']:
        check_prior_failure(campaign, item)
    previous = Path(campaign)/'v4_campaign.json'
    expected = compare.HERE/'inherited_v4_campaign.json'
    if (previous.is_symlink() or not previous.is_file() or compare.sha(previous)!=compare.sha(expected)
            or state.get('inherited_v4_campaign_sha256')!=compare.sha(expected)):
        raise RuntimeError('Previous v4 campaign provenance mismatch')
    prior = compare.read(previous)
    if prior['source_hashes']!=v4_sources() or prior['completed_batches']!=28:
        raise RuntimeError('Previous v4 release or completion count mismatch')
    for phase, attempts in prior['attempts'].items():
        if state['attempts'].get(phase, [])[:len(attempts)] != attempts:
            raise RuntimeError('Previous v4 attempt history changed: '+phase)
    entries = state.get('inherited_v4_batches', {})
    if set(entries)!=set(v4_completed()):
        raise RuntimeError('Inherited v4 completed batch inventory mismatch')
    for phase, attempt in v4_completed().items():
        entry = entries[phase]
        if (entry.get('phase')!=phase or entry.get('attempt')!=attempt or entry.get('complete') is not True
                or state['attempts'][phase]!=prior['attempts'][phase]):
            raise RuntimeError('Inherited v4 completed attempt changed: '+phase)
        check_v4_archive(campaign, entry)
    failures = state.get('inherited_v4_failures', [])
    if len(failures)!=1 or any(failures[0].get(k)!=v for k,v in V4_FAILURE.items()):
        raise RuntimeError('Inherited v4 failed attempt history mismatch')
    check_v4_archive(campaign, failures[0])


def copy_checked(source, destination, expected):
    if source.is_symlink() or not source.is_file() or compare.sha(source) != expected:
        raise RuntimeError('Source checksum mismatch: ' + str(source))
    if destination.exists():
        if destination.is_symlink() or compare.sha(destination) != expected:
            raise RuntimeError('Existing inherited file differs: ' + str(destination))
        return
    temporary = destination.with_name(destination.name + '.copying')
    if temporary.is_symlink():
        raise RuntimeError('Symlink copy destination denied')
    shutil.copy2(source, temporary)
    if compare.sha(temporary) != expected:
        raise RuntimeError('Copied file checksum mismatch: ' + str(destination))
    temporary.replace(destination)


def prepare(source, runs):
    source, runs = Path(source).resolve(), Path(runs).resolve()
    frozen = spec()
    original = source / 'inherited_campaign.json'
    previous = source / 'previous_campaign.json'
    expected_previous = compare.HERE / 'inherited_v3_campaign.json'
    current = source/'campaign.json'
    expected_current = compare.HERE/'inherited_v4_campaign.json'
    if original.is_symlink() or not original.is_file() or compare.sha(original) != frozen['campaign_sha256']:
        raise RuntimeError('Use the v4 resumed campaign with its original inherited_campaign.json')
    if (previous.is_symlink() or not previous.is_file()
            or compare.sha(previous) != compare.sha(expected_previous)):
        raise RuntimeError('Previous v3 campaign differs')
    if (current.is_symlink() or not current.is_file() or compare.sha(current)!=compare.sha(expected_current)):
        raise RuntimeError('Expected the supplied v4 campaign stopped at test_04_attempt_001')
    old = compare.read(current)
    entries = {}
    for phase, attempt in v4_completed().items():
        archives = sorted(source.glob(attempt+'_*.zip'))
        if len(archives)!=1:
            raise RuntimeError('Expected one completed v4 archive: '+attempt)
        archive = archives[0]
        sidecar = archive.with_suffix('.zip.sha256')
        fields = sidecar.read_text().split() if sidecar.is_file() and not sidecar.is_symlink() else []
        digest = compare.sha(archive)
        if fields != [digest, archive.name]:
            raise RuntimeError('Completed v4 ZIP checksum sidecar mismatch: '+phase)
        entry = dict(phase=phase,attempt=attempt,filename=archive.name,sha256=digest,complete=True,
                     source_kind='layer_full_20261110_v4',source_hashes=v4_sources())
        check_v4_archive(source, entry, allow_directory=True)
        entries[phase] = entry
    failure = dict(V4_FAILURE,source_kind='layer_full_20261110_v4',source_hashes=v4_sources())
    check_v4_archive(source, failure, allow_directory=True)
    campaign = runs / ('resume_' + frozen['source_campaign'])
    if campaign.is_symlink():
        raise RuntimeError('Symlink campaign denied')
    campaign.mkdir(parents=True, exist_ok=True)
    for phase, item in frozen['batches'].items():
        target = campaign / item['filename']
        copy_checked(source / item['filename'], target, item['sha256'])
        sidecar = target.with_suffix('.zip.sha256')
        expected = item['sha256'] + '  ' + target.name + '\n'
        if sidecar.is_symlink() or (sidecar.exists() and sidecar.read_text() != expected):
            raise RuntimeError('Inherited checksum sidecar differs: ' + phase)
        sidecar.write_text(expected)
        check_archive(campaign, phase)
    for item in frozen['prior_failures']:
        target = campaign / item['filename']
        copy_checked(source / item['filename'], target, item['sha256'])
        sidecar = target.with_suffix('.zip.sha256')
        expected = item['sha256'] + '  ' + target.name + '\n'
        if sidecar.is_symlink() or (sidecar.exists() and sidecar.read_text() != expected):
            raise RuntimeError('Prior failure checksum sidecar differs')
        sidecar.write_text(expected)
        check_prior_failure(campaign, item)
    copy_checked(original, campaign / 'inherited_campaign.json', frozen['campaign_sha256'])
    copy_checked(previous, campaign / 'previous_campaign.json', compare.sha(expected_previous))
    copy_checked(current, campaign/'v4_campaign.json', compare.sha(expected_current))
    for item in list(entries.values())+[failure]:
        target = campaign/item['filename']
        copy_checked(source/item['filename'],target,item['sha256'])
        sidecar = target.with_suffix('.zip.sha256')
        expected = item['sha256']+'  '+target.name+'\n'
        if sidecar.is_symlink() or (sidecar.exists() and sidecar.read_text()!=expected):
            raise RuntimeError('Inherited v4 checksum sidecar differs')
        sidecar.write_text(expected)
    state_path = campaign / 'campaign.json'
    if state_path.is_symlink():
        raise RuntimeError('Symlink campaign state denied')
    if state_path.exists():
        state = compare.read(state_path)
        if state['source_hashes'] != compare.read(compare.HERE / 'release_hashes.json'):
            raise RuntimeError('Existing resumed campaign uses a different release')
    else:
        state = dict(old, source_hashes=compare.read(compare.HERE / 'release_hashes.json'),
                     inherited_v4_campaign_sha256=compare.sha(expected_current),
                     inherited_v4_batches=entries,inherited_v4_failures=[failure])
        temporary = state_path.with_suffix('.json.tmp')
        if temporary.is_symlink():
            raise RuntimeError('Symlink campaign state denied')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)
    validate_state(campaign, state)
    print('INHERIT PASS: 28 completed batches; all 31 prior attempt archives preserved.', flush=True)
    return campaign
