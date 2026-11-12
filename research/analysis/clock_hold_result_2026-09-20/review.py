"""Recompute the uploaded hold check without changing any raw artifact."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
RAW = HERE / 'raw/clock_hold_20260920_160922_DN7Cqs'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def seconds(a, b):
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()


def describe(rows):
    fields = ('clocks.sm', 'clocks.mem', 'power.draw', 'power.limit', 'temperature.gpu')
    return {
        'samples': len(rows),
        'ranges': {key: [min(r[key] for r in rows), max(r[key] for r in rows)] for key in fields},
        'active_reason_counts': {key: sum(r[key] is True for r in rows) for key in telemetry.REASONS},
        'counter_first_to_last_us': {key: rows[-1][key] - rows[0][key] for key in telemetry.COUNTERS},
        'sample_span_s': seconds(rows[0]['utc'], rows[-1]['utc']),
    }


manifest = read(RAW / 'manifest.json')
verified_sources = {}
for snapshot, local, hashes in (
    ('source', 'clock_hold_v1', manifest['source_hashes']),
    ('audit_source', 'clock_audit_v1', manifest['dependency_hashes']),
    ('baseline_source', 'gpu_sweep_v6', manifest['baseline']['code_hashes']),
):
    for name, expected in hashes.items():
        assert digest(RAW / snapshot / name) == expected, (snapshot, name)
        assert digest(WORKSPACE / local / name) == expected, (local, name)
    verified_sources[snapshot] = len(hashes)

# Import only the verified local analysis helpers, never run GPU/clock commands.
sys.path[:0] = [str(WORKSPACE / 'clock_hold_v1'), str(WORKSPACE / 'clock_audit_v1')]
import hold
import audit
import analyze
import telemetry

baseline = read(WORKSPACE / 'clock_audit_v1/baseline_v6.json')
assert manifest['baseline'] == baseline
assert manifest['cases'] == audit.cases()
assert manifest['target'] == hold.TARGET
assert manifest['dependency_hashes'] == read(WORKSPACE / 'clock_hold_v1/dependency_hashes.json')
assert manifest['new_automatic_control'] is False
recomputed = HERE / 'recomputed'
recomputed.mkdir(exist_ok=True)
shutil.copytree(RAW / 'dense_before', recomputed / 'dense_before', dirs_exist_ok=True)
summary = hold.summarize(recomputed, audit, analyze, telemetry, baseline)
outputs = ('measurement_summary.json', 'analysis.json', 'clock_windows.json', 'report.md')
for name in outputs:
    assert (RAW / name).read_bytes() == (recomputed / name).read_bytes(), name
assert read(RAW / 'summary.json') == dict(summary, restoration='COMMANDS_ACCEPTED')
assert read(RAW / 'status.json') == dict(collection='COMPLETE', restoration='COMMANDS_ACCEPTED', full_sweep='HOLD')
for path, expected in ((RAW / 'exit_code.txt', '2'), (RAW / 'worker.exit_code.txt', '0'),
                       (RAW / 'tee_exit_code.txt', '0'), (next(RAW.glob('checked/runs/*/exit_code.txt')), '2')):
    assert path.read_text().strip() == expected, str(path)
for name, option in (('set_graphics', '-lgc'), ('set_memory', '-lmc'),
                     ('reset_graphics', '-rgc'), ('reset_memory', '-rmc')):
    command = read(RAW / (name + '.json'))
    assert command['exit_code'] == 0 and option in command['command']
    assert baseline['environment_signature']['gpu_uuid'] in command['command']
restore = read(RAW / 'restore.json')
assert restore['commands_accepted'] is True
for name in ('graphics', 'memory'):
    assert restore['commands'][name] == read(RAW / ('reset_' + name + '.json'))

env = read(RAW / 'dense_before/environment.json')
assert env['executable'] == read(RAW / 'worker.command.json')[0] == '/home/eu/miniconda3/envs/ai_env/bin/python'
assert env['environment']['CONDA_DEFAULT_ENV'] == 'ai_env'
environment_log = next(RAW.glob('checked/runs/*/environment.txt')).read_text()
assert 'ENV CHECK: PASS' in environment_log and 'python=' + env['executable'] in environment_log
assert 'runner_sha256=' + manifest['baseline']['code_hashes']['run_research_checked.sh'] in environment_log
metadata = read(RAW / 'dense_before/telemetry_metadata.json')
samples, cadence = telemetry.read(RAW / 'dense_before/telemetry.csv', metadata['timezone_offset_seconds'], 100)
windows = read(recomputed / 'clock_windows.json')
inside = [sample for block in windows for sample in block['window']['inside']]
gaps = []
for block in windows:
    w = block['window']
    if w['inside']:
        continue
    assert w['before'] is not None and w['after'] is not None
    gaps.append(dict(task_id=block['task_id'], block_index_zero_based=block['block'],
                     block_ms=1000 * seconds(w['start_utc'], w['end_utc']),
                     poll_gap_ms=1000 * seconds(w['before']['utc'], w['after']['utc']),
                     before_to_start_ms=1000 * seconds(w['before']['utc'], w['start_utc']),
                     end_to_after_ms=1000 * seconds(w['end_utc'], w['after']['utc']),
                     before_sm_mem=[w['before']['clocks.sm'], w['before']['clocks.mem']],
                     after_sm_mem=[w['after']['clocks.sm'], w['after']['clocks.mem']]))
analysis = read(recomputed / 'analysis.json')
rows = read(RAW / 'dense_before/results.json')
measurements = []
for item, source in zip(analysis['measurements'], rows):
    correct = item['correctness']
    assert correct['scope'] == 'sampled' and correct['points'] == 256
    task = item['task']
    measurements.append(dict(task_id=task['id'], shape=task['shape'], dtype=task['dtype'], mode=task['mode'],
                             round=task['round'], median_us=item['median_us'], iqr_pct=100*item['relative_iqr'],
                             drift_pct=100*item['first_last_drift'], inside_samples=item['inside_samples'],
                             block_us=[1000*b['latency_ms'] for b in source['result']['timing']['blocks']],
                             flags=item['flags']))
with (HERE / 'measurements.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(measurements[0]))
    writer.writeheader()
    writer.writerows(measurements)

def counters(text):
    return {key: int(re.search(r'^\s*' + re.escape(key) + r'\s*:\s*(\d+) us', text, re.M).group(1))
            for key in ('SW Power Capping', 'SW Thermal Slowdown', 'HW Thermal Slowdown')}

before = counters(read(RAW / 'gpu_before_settings.json')['stdout'])
after_text = read(RAW / 'gpu_after_restore.json')['stdout']
after = counters(after_text)
after_clocks = re.search(r'\n    Clocks\n(.*?)\n    Applications Clocks', after_text, re.S).group(1)
post_clocks = {key: int(re.search(key + r'\s*:\s*(\d+) MHz', after_clocks).group(1)) for key in ('Graphics', 'SM', 'Memory')}
metrics = dict(summary=read(RAW / 'summary.json'), verified_sources=verified_sources,
               recomputed_outputs_byte_identical=list(outputs), environment_executable=env['executable'],
               cadence=cadence, full_telemetry=describe(samples), timed_inside=describe(inside),
               gaps=gaps, max_iqr_pct=max(r['iqr_pct'] for r in measurements),
               max_abs_drift_pct=max(abs(r['drift_pct']) for r in measurements),
               rounds=analysis['rounds'], start_to_end_counter_delta_us={key:after[key]-before[key] for key in before},
               post_restore_observed_clocks=post_clocks,
               telemetry_process_exit_code=metadata['process_exit_code'],
               telemetry_capture_s=seconds(metadata['started_utc'], metadata['stopped_utc']),
               correctness_scope='256 sampled output elements per measurement; finite_all recorded true')
counter_steps = [b[telemetry.COUNTERS[0]] - a[telemetry.COUNTERS[0]] for a, b in zip(samples, samples[1:])]
assert min(counter_steps) >= 0
assert len(inside) == 52 and len(gaps) == 4
assert all(r['clocks.sm'] == 900 and r['clocks.mem'] == 6801 for r in samples)
metrics['power_counter_steps'] = dict(positive_updates=sum(x > 0 for x in counter_steps),
                                     min_positive_us=min(x for x in counter_steps if x > 0),
                                     max_us=max(counter_steps))
metrics['manual_review'] = dict(
    timing_stability='Within-run block and two-round checks pass; no independent-session reproducibility claim.',
    recorded_clocks='All 2067 recorded samples match 900/6801 MHz; not continuous observation.',
    unresolved=['Four timed blocks have no inside clock sample.',
                'Power-cap counter increases by 210.790582 s while all instantaneous power-cap samples are inactive.'],
    counter_definition='https://docs.nvidia.com/deploy/nvidia-smi/#clock-event-reasons-counters',
    conclusion='Stable timing under requested clocks observed; absence of power capping and causal isolation unproven.',
    full_sweep='HOLD')
save('review_metrics.json', metrics)
integrity = read(HERE / 'raw_integrity.json')
assert {str(p.relative_to(RAW)): digest(p) for p in RAW.rglob('*') if p.is_file()} == integrity['files']
save('verification.json', dict(status='PASS', raw_files=len(integrity['files']), zip_sha256=integrity['sha256'],
                               verified_sources=verified_sources, reproduced_files=list(outputs), raw_unchanged=True))
print(json.dumps(metrics, indent=2, ensure_ascii=False))
