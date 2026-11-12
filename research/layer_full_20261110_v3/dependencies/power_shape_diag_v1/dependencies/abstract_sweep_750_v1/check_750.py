"""Read-only admission from the exact reviewed 750 MHz stage, in an isolated process."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'clock_recheck_v1'))
import lower
from analyze_lower import analyze_stage


def inspect(path):
    loaded = lower.load()
    expected = lower.read(HERE/'protocol.json')['reviewed_check']
    stage_name = expected['stage']
    hashes = {key: lower.digest(path/name) for key, name in (
        ('manifest_sha256', 'manifest.json'),
        ('results_sha256', stage_name+'/results.json'),
        ('telemetry_sha256', stage_name+'/telemetry.csv'))}
    if any(value != expected[key] for key, value in hashes.items()):
        raise RuntimeError('Not the reviewed 750 MHz evidence; preserve and inspect it')
    manifest = lower.read(path/'manifest.json')
    if (manifest['kind'] != 'clock_recheck_v1' or
        manifest['config'] != loaded[-1] or manifest['baseline'] != loaded[5] or
        manifest['cases'] != lower.cases(loaded[-1]) or
        manifest['source_hashes'] != lower.own_hashes() or
        manifest['dependency_hashes'] != lower.read(lower.HERE/'dependency_hashes.json')):
        raise RuntimeError('Reviewed recheck manifest/source differs')
    for folder, key in [('source','source_hashes'), ('dependencies','dependency_hashes')]:
        for name, wanted in manifest[key].items():
            if lower.digest(path/folder/name) != wanted:
                raise RuntimeError('Reviewed source snapshot changed: '+name)
    stage = next(s for s in loaded[-1]['stages'] if s['name'] == stage_name)
    if stage['sm_mhz'] != 750:
        raise RuntimeError('Expected the 750 MHz stage')
    result, rows, windows, triplets, _ = analyze_stage(path, stage, loaded)
    if not result['timing_and_sampled_clocks_pass'] or (len(rows),len(windows),len(triplets)) != (36,252,12):
        raise RuntimeError('750 MHz timing/clock admission failed')
    state = lower.read(path/'status.json')
    restore = lower.read(path/'restore.json')
    gpu = manifest['baseline']['environment_signature']['gpu_uuid']
    if state != dict(collection='COMPLETE', restoration='COMMANDS_ACCEPTED', full_sweep='HOLD'):
        raise RuntimeError('Reviewed recheck did not complete and restore')
    if not restore['commands_accepted'] or any(
        restore['commands'][name].get('exit_code') != 0 or
        restore['commands'][name]['command'][-1] != option or
        gpu not in restore['commands'][name]['command']
        for name, option in [('graphics','-rgc'), ('memory','-rmc')]):
        raise RuntimeError('Reviewed clock reset evidence differs')
    return dict(**hashes, stage=stage_name, zip_sha256=expected['zip_sha256'],
                check_gate='REVIEW', timing_and_sampled_clocks_pass=True,
                measurements=36, blocks=252, triplets=12,
                limitation='Three diagnostic shapes only; cumulative counter unresolved. '
                           'The reviewed return to 780 MHz had clock/power violations.')


if __name__ == '__main__':
    print(json.dumps(inspect(Path(sys.argv[1]).resolve())))
