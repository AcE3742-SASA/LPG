"""CPU-only checks for manifests, release gates and the real telemetry analyzer."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace

import build_manifest
import compare


def static_checks():
    groups, cases, protected, audit = build_manifest.build()
    for name, value in [('groups.json', groups), ('cases_by_phase.json', cases),
                        ('protected_shapes.json', protected), ('selection_audit.json', audit)]:
        assert json.loads(json.dumps(value)) == compare.read(compare.HERE / name), name
    assert Counter(g['split'] for g in groups) == {'train': 30, 'validation': 10, 'test_locked': 20}
    owners = {}
    for group in groups:
        assert max(group['M'], group['D'], group['F']) <= 4096
        for case in build_manifest.make_core_cases(group):
            for shape in compare.components(case):
                owners.setdefault(shape, set()).add(group['group_id'])
                assert not compare.forbidden_overlap(shape, set(map(tuple, protected)))
    assert all(len(v) == 1 for v in owners.values())
    for shape in owners:
        assert all(owners[shape] == owners[n] for n in build_manifest.neighbours(shape) if n in owners)
    assert not any(c['split'] == 'test_locked' for c in cases['core_dev'])
    assert len(cases['core_locked']) == 40
    subsets = build_manifest.training_subsets(groups)
    assert subsets == compare.read(compare.HERE/'rq3_training_subsets.json')
    by_id = {g['group_id']: g for g in groups}
    train_ids = {g['group_id'] for g in groups if g['split'] == 'train'}
    protected_ids = set(by_id)-train_ids
    for seed in ('20261028', '20261029', '20261030'):
        previous = set()
        for k in (5, 10, 20, 30):
            ids = subsets['seeds'][seed]['by_k'][str(k)]
            assert len(ids) == len(set(ids)) == k
            assert previous <= set(ids) <= train_ids and not set(ids) & protected_ids
            alignment = Counter(by_id[g]['alignment'] for g in ids)
            work = Counter(by_id[g]['work_class'] for g in ids)
            assert abs(alignment['aligned16']-alignment['unaligned']) <= 1
            assert abs(work['small']-work['large']) <= (2 if k == 30 else 1)
            previous = set(ids)
    for phase, expected in audit['expected_measurements'].items():
        config = dict(compare.read(compare.HERE / 'protocol.json'), phase=phase)
        assert compare.expected_records(cases[phase], config) == expected
    compare.PHASE, compare.LOCKED_RELEASE = 'core_locked', False
    try:
        compare.verify()
    except RuntimeError as error:
        assert 'Locked test' in str(error)
    else:
        raise AssertionError('Locked phase unexpectedly accessible')
    compare.PHASE = 'core_dev'
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out/'manifest.json').write_text(json.dumps({'phase': 'core_locked'}))
        (out/'records').mkdir()
        (out/'records/locked.json').write_bytes(b'INVALID RECORD: MUST NEVER BE PARSED OR HASHED')
        for phase in ('core_dev', 'core_locked'):
            result = subprocess.run([sys.executable, '-B', str(compare.HERE/'compare.py'), 'pack',
                                     '--output', str(out), '--phase', phase], capture_output=True, text=True)
            assert result.returncode != 0 and 'Locked result access' in result.stderr
        assert not (out/'.run.lock').exists()
        assert not list(out.parent.glob(out.name+'_*.zip'))
        assert not (out/'result_hashes.json').exists()
    return compare.verify()


def analyzer_checks(config, cases, telemetry):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    sensor = {'clocks.sm': 720, 'clocks.mem': 6801, 'clocks_event_reasons.sw_power_cap': False,
              'utc': '2026-10-28T00:00:00.500000+00:00', **dict.fromkeys(telemetry.COUNTERS, 0)}
    loaded = dict(suite=SimpleNamespace(atomic_json=write),
                  telemetry=SimpleNamespace(read=lambda *args: ([sensor], {}), window=telemetry.window))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        write(out / 'worker/telemetry_metadata.json', {'timezone_offset_seconds': 0})
        write(out / 'correct.json', {'passed': True})
        write(out / 'status.json', {'collection': 'COMPLETE', 'restoration': 'COMMANDS_ACCEPTED'})
        write(out / 'worker/complete.json', {'complete': True})
        for case in cases:
            for dtype in config['dtypes']:
                for rd in range(3):
                    for implementation in compare.implementations(case, config):
                        row_id = compare.key(case, dtype, rd, implementation)
                        write(compare.record_path(out, case, dtype, rd, implementation),
                              dict(id=row_id, case=case, dtype=dtype,
                                   timing=dict(median_ms=1 if implementation == 'standard' else 2, flags=[],
                                               blocks=[dict(host_envelope_start_utc='2026-10-28T00:00:00+00:00',
                                                            host_envelope_end_utc='2026-10-28T00:00:01+00:00')]),
                                   memory=dict(extra_peak_allocated_bytes=10), prepared_weight_bytes=20,
                                   correctness_before='correct.json', correctness_after='correct.json'))
        result = compare.analyze(out, config, cases, loaded)
        assert result['gate'] == 'PASS' and result['recorded'] == compare.expected_records(cases, config)
        assert not result['comparisons']
        assert all('standard_ms' not in row for row in compare.read(out / 'summary.json').get('comparisons', []))
        assert (out / 'restricted_explicit/comparison.csv').exists()
        health = compare.analyze(out, config, cases, loaded, health_only=True)
        assert health['gate'] == 'PASS' and not health['comparisons']
        sensor['clocks.sm'] = 705
        assert compare.analyze(out, config, cases, loaded)['gate'] == 'REVIEW'
        sensor['clocks.sm'] = 720
        write(out / 'correct.json', {'passed': False})
        assert compare.analyze(out, config, cases, loaded)['gate'] == 'FAIL'
        write(out / 'correct.json', {'passed': True})
        write(out / 'status.json', {'collection': 'COMPLETE', 'restoration': 'FAILED'})
        assert compare.analyze(out, config, cases, loaded)['gate'] == 'FAIL'
        if config['phase'] == 'conv_diagnostic':
            write(out / 'status.json', {'collection': 'COMPLETE', 'restoration': 'COMMANDS_ACCEPTED'})
            case = cases[0]
            path = compare.record_path(out, case, 'fp16', 2, 'mm')
            row = compare.read(path)
            row['timing']['median_ms'] = 3
            write(path, row)
            assert compare.analyze(out, config, cases, loaded)['gate'] == 'REVIEW'


if __name__ == '__main__':
    config, cases = static_checks()
    _, loaded = compare.load_frozen()
    analyzer_checks(config, cases, loaded['telemetry'])
    compare.PHASE = 'conv_diagnostic'
    config, cases = compare.verify()
    analyzer_checks(config, cases, loaded['telemetry'])
    print('SELF CHECK PASS: deterministic manifests, split protection, locked denial, 960/264 synthetic records, failure gates')
    print('NO GPU RUN: CUDA correctness, live environment and timing quality remain unverified')
