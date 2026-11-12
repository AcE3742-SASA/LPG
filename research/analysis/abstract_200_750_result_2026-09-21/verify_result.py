"""Offline verification using the frozen runner and analyzer; never executes GPU work."""
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'abstract_sweep_750_v1'))
import run
import analyze_sweep

RAW = HERE / 'raw/abstract_200_750'
ZIP = 'abstract_200_750_20260921_172036_6a2a50.zip'
ZIP_SHA = 'db1c9ee31d49f5bd12dc27e53cf4c5cfc5af5cc380e3959a6467e0df21f71659'
KST = dt.timezone(dt.timedelta(hours=9))


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def table(name, rows):
    assert rows
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def kst(utc):
    return dt.datetime.fromisoformat(utc).astimezone(KST).isoformat()


def verify_raw():
    actual = {str(p.relative_to(HERE / 'raw')): run.sha(p)
              for p in (HERE / 'raw').rglob('*') if p.is_file()}
    assert actual == run.read(HERE / 'raw_sha256.json')
    assert run.sha(HERE / ZIP) == ZIP_SHA


def main():
    verify_raw()
    loaded = run.load()
    manifest = run.manifest_for(RAW, 'full', loaded)
    for name, wanted in manifest['dependency_hashes'].items():
        assert run.sha(ROOT / name) == wanted
    derived = HERE / 'recomputed'
    if not derived.exists():
        shutil.copytree(RAW, derived)
    analyze_sweep.analyze(derived, loaded)
    assert run.read(derived / 'summary.json') == run.read(RAW / 'summary.json')
    names = ['raw_attempts.jsonl', 'metrics.csv', 'fresh_holdout_LOCKED.csv',
             'blocks.csv', 'blocks_fresh_holdout_LOCKED.csv', 'between_round.json',
             'review_queue.csv', 'review_fresh_holdout_LOCKED.csv', 'triplets.csv',
             'triplets_fresh_holdout_LOCKED.csv', 'shape_splits.csv', 'summary.json', 'report.md']
    names += [str(p.relative_to(RAW)) for p in (RAW / 'sessions').glob('*/segment_*/quality.json')]
    assert all(run.sha(derived / name) == run.sha(RAW / name) for name in names)
    entries = list(run.history(RAW))
    rows = [r for _, r in entries]
    assert len(rows) == 1236 == len({r['task']['id'] for r in rows})
    assert [r['task'] for r in rows] == manifest['tasks'][:1236]
    assert all(r['correctness']['passed'] and r['correctness']['finite_all'] for r in rows)
    assert sum(len(r['timing']['blocks']) for r in rows) == 8652
    for r in rows:
        blocks = r['timing']['blocks']
        for b in blocks:
            assert math.isclose(b['latency_ms'], b['elapsed_ms'] /
                                (b['repeats'] * b['operations_per_call']), rel_tol=1e-12)
        assert math.isclose(st.median(b['latency_ms'] for b in blocks),
                            r['timing']['median_ms'], rel_tol=1e-12)
    with (derived / 'metrics.csv').open() as f:
        metrics = {r['id']: r for r in csv.DictReader(f)}
    assert Counter(r['status'] for r in metrics.values()) == {'PASS': 1233, 'REVIEW': 3}
    assert Counter(r['task']['round'] for r in rows) == {0: 1200, 1: 36}
    assert len({tuple(r['task']['shape']) for r in rows[:1200]}) == 200
    assert len({tuple(r['task']['shape']) for r in rows[1200:]}) == 6
    sessions = {p.parent.name: run.read(p) for p in (RAW / 'sessions').glob('*/session.json')}
    first, second = sorted(sessions)
    assert sessions[first]['state'] == 'STOPPED' and sessions[first]['new_records'] == 36
    assert sessions[second]['state'] == 'ERROR' and sessions[second]['new_records'] == 1200
    assert all(r['session'] == first for r in rows[:36])
    assert all(r['session'] == second for r in rows[36:])
    resets = [run.read(p) for p in (RAW / 'launches').glob('*/restore.json')]
    assert len(resets) == 2 and all(s['commands_accepted'] for s in resets)
    assert all(s['commands'][name]['exit_code'] == 0 for s in resets for name in ('graphics', 'memory'))
    try:
        run.require_clean_resume(RAW, loaded)
    except RuntimeError as exc:
        resume_error = str(exc)
        assert resume_error.startswith('Saved clock/telemetry/data problem: resume refused.')
    else:
        raise AssertionError('Clock-violating run was accepted for resume')
    samples, segments = [], []
    for folder in sorted((RAW / 'sessions').glob('*/segment_*')):
        meta = run.read(folder / 'telemetry_metadata.json')
        found, cadence = loaded['telemetry'].read(folder / 'telemetry.csv', meta['timezone_offset_seconds'], 50)
        samples.extend(found)
        quality = run.read(folder / 'quality.json')
        segments.append(dict(segment=str(folder.relative_to(RAW)), cadence=cadence,
                             flags=quality['flags'], ranges=quality['ranges']))
    samples.sort(key=lambda s: s['utc'])
    times = [dt.datetime.fromisoformat(s['utc']).timestamp() for s in samples]
    active = 'clocks_event_reasons.sw_power_cap'
    bad = lambda s: s['clocks.sm'] != 750 or s[active] is True
    bad_samples = [s for s in samples if bad(s)]
    incident_blocks, reviewed = [], []
    for seq, r in entries:
        tm, task = r['timing'], r['task']
        for phase, blocks in [('settle', tm['setup']['settle_blocks']),
                              ('calibration', tm['setup']['calibration']), ('timed', tm['blocks'])]:
            for i, b in enumerate(blocks):
                start, end = (dt.datetime.fromisoformat(b[k]).timestamp() for k in
                              ('host_envelope_start_utc', 'host_envelope_end_utc'))
                inside = samples[bisect_left(times, start):bisect_right(times, end)]
                if any(bad(s) for s in inside):
                    incident_blocks.append(dict(seq=seq, task_id=task['id'], shape=task['shape'],
                        dtype=task['dtype'], mode=task['mode'], round=task['round'], phase=phase, block=i,
                        start_kst=kst(b['host_envelope_start_utc']), end_kst=kst(b['host_envelope_end_utc']),
                        samples=len(inside), low_clock_samples=sum(s['clocks.sm'] != 750 for s in inside),
                        power_cap_samples=sum(s[active] is True for s in inside),
                        core_min_mhz=min(s['clocks.sm'] for s in inside),
                        power_max_w=max(s['power.draw'] for s in inside),
                        temperature_max_c=max(s['temperature.gpu'] for s in inside), latency_us=b['latency_ms']*1000))
        if metrics[task['id']]['status'] != 'PASS':
            reviewed.append(dict(seq=seq, **task, median_us=tm['median_ms']*1000,
                relative_iqr=tm['relative_iqr'], first_last_drift=tm['first_last_drift'],
                stored_status=r['status'], analyzed_status=metrics[task['id']]['status'],
                flags=metrics[task['id']]['flags']))
    assert {tuple(b['shape']) for b in incident_blocks} == {(2329, 3998, 66)}
    assert {b['dtype'] for b in incident_blocks if b['phase'] == 'timed'} == {'fp32'}
    assert [(b['seq'], b['phase'], b['block']) for b in incident_blocks if b['dtype'] == 'fp16'] == [
        (1174, 'settle', i) for i in range(4)]
    table('incident_blocks.csv', incident_blocks)
    table('reviewed_measurements.csv', reviewed)
    table('anomalous_samples.csv', [dict(utc=s['utc'], kst=kst(s['utc']), core_mhz=s['clocks.sm'],
        memory_mhz=s['clocks.mem'], power_w=s['power.draw'], temperature_c=s['temperature.gpu'],
        sw_power_cap=s[active]) for s in bad_samples])
    old_problem = [metrics[r['task']['id']] for r in rows if r['task']['shape'] == [3128, 680, 488]]
    assert len(old_problem) == 6 and all(r['status'] == 'PASS' for r in old_problem)
    clean = [r for r in rows if metrics[r['task']['id']]['status'] == 'PASS']
    result = dict(evidence_verification='PASS', collection_decision='INCOMPLETE_CLOCK_VIOLATION_HOLD',
        zip_sha256=ZIP_SHA, raw_files=len(run.read(HERE / 'raw_sha256.json')),
        own_sources_verified=len(manifest['source_hashes']), dependencies_verified=len(manifest['dependency_hashes']),
        identical_recomputed_outputs=len(names), gpu_execution_on_mac=False,
        original_summary=run.read(RAW / 'summary.json'), recorded_shapes=200,
        round_record_counts=dict(Counter(r['task']['round'] for r in rows)), complete_three_round_conditions=0,
        correctness_scopes=dict(Counter(r['correctness']['scope'] for r in rows)), correctness_passed=len(rows),
        telemetry_samples=len(samples), core_counts=dict(Counter(s['clocks.sm'] for s in samples)),
        active_reason_counts={k:sum(s[k] is True for s in samples) for k in loaded['telemetry'].REASONS},
        ranges={k:[min(s[k] for s in samples),max(s[k] for s in samples)] for k in
                ('clocks.sm','clocks.mem','power.limit','power.draw','temperature.gpu')},
        anomaly_first_kst=kst(bad_samples[0]['utc']), anomaly_last_kst=kst(bad_samples[-1]['utc']),
        anomaly_sample_span_seconds=(dt.datetime.fromisoformat(bad_samples[-1]['utc'])-
                                     dt.datetime.fromisoformat(bad_samples[0]['utc'])).total_seconds(),
        anomaly_note='First-to-last sample span, not continuous throttling duration.',
        incident_blocks_by_phase=dict(Counter(b['phase'] for b in incident_blocks)),
        incident_timed_blocks_by_mode=dict(Counter(b['mode'] for b in incident_blocks if b['phase']=='timed')),
        fp16_setup_incident_note='Record 1174 allocate has 4 affected settling blocks; its 7 timed blocks pass. Do not interpret timed PASS as no setup clock event.',
        reviewed_measurements=reviewed, previous_problem_750_six_measurements='PASS (round 0 only)',
        pass_only_max_relative_iqr=max(r['timing']['relative_iqr'] for r in clean),
        pass_only_max_abs_drift=max(abs(r['timing']['first_last_drift']) for r in clean),
        sessions=sessions, segments=segments, restoration_commands_accepted=True, resume_guard=resume_error,
        exit_codes={str(p.relative_to(RAW)):int(p.read_text()) for p in RAW.rglob('*exit_code.txt')},
        interpretation='First-round coverage is complete; three-round collection and stability are not. Keep REVIEW rows and all raw evidence.')
    save('validation.json', result)
    verify_raw()
    print(json.dumps({k: result[k] for k in ('evidence_verification','raw_files','own_sources_verified',
        'dependencies_verified','identical_recomputed_outputs','core_counts','active_reason_counts',
        'incident_blocks_by_phase','incident_timed_blocks_by_mode','pass_only_max_relative_iqr',
        'pass_only_max_abs_drift','resume_guard')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
