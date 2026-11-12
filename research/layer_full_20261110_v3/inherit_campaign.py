"""Preserve the audited v2 attempts and continue in a separate v3 campaign."""
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
    original = source / 'campaign.json'
    if original.is_symlink() or not original.is_file() or compare.sha(original) != frozen['campaign_sha256']:
        raise RuntimeError('Use the audited v2 campaign; campaign.json checksum differs')
    old = compare.read(original)
    if old['source_hashes'] != frozen['source_hashes'] or old['completed_batches'] != 20:
        raise RuntimeError('Expected the audited v2 campaign with 20 completed train batches')
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
    copy_checked(original, campaign / 'inherited_campaign.json', frozen['campaign_sha256'])
    state_path = campaign / 'campaign.json'
    if state_path.is_symlink():
        raise RuntimeError('Symlink campaign state denied')
    if state_path.exists():
        state = compare.read(state_path)
        if state['source_hashes'] != compare.read(compare.HERE / 'release_hashes.json'):
            raise RuntimeError('Existing resumed campaign uses a different release')
    else:
        state = dict(old, source_hashes=compare.read(compare.HERE / 'release_hashes.json'),
                     inherited_batches=inherited_batches(),
                     inherited_campaign_sha256=frozen['campaign_sha256'])
        temporary = state_path.with_suffix('.json.tmp')
        if temporary.is_symlink():
            raise RuntimeError('Symlink campaign state denied')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)
    validate_state(campaign, state)
    print('INHERIT PASS: 20 train batches; original failed validation attempt preserved.', flush=True)
    return campaign
