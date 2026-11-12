"""Preserve the audited v2 train and both failed validation attempts in v4."""
import json
from pathlib import Path
import shutil
import zipfile

import compare


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
    previous = source / 'campaign.json'
    expected_previous = compare.HERE / 'inherited_v3_campaign.json'
    if original.is_symlink() or not original.is_file() or compare.sha(original) != frozen['campaign_sha256']:
        raise RuntimeError('Use the v3 resumed campaign with its original inherited_campaign.json')
    if (previous.is_symlink() or not previous.is_file()
            or compare.sha(previous) != compare.sha(expected_previous)):
        raise RuntimeError('Expected v3 campaign after validation_01_attempt_002; campaign.json differs')
    old = compare.read(previous)
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
    state_path = campaign / 'campaign.json'
    if state_path.is_symlink():
        raise RuntimeError('Symlink campaign state denied')
    if state_path.exists():
        state = compare.read(state_path)
        if state['source_hashes'] != compare.read(compare.HERE / 'release_hashes.json'):
            raise RuntimeError('Existing resumed campaign uses a different release')
    else:
        state = dict(old, source_hashes=compare.read(compare.HERE / 'release_hashes.json'),
                     previous_campaign_sha256=compare.sha(expected_previous))
        temporary = state_path.with_suffix('.json.tmp')
        if temporary.is_symlink():
            raise RuntimeError('Symlink campaign state denied')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)
    validate_state(campaign, state)
    print('INHERIT PASS: 20 train batches; both failed validation attempts preserved.', flush=True)
    return campaign
