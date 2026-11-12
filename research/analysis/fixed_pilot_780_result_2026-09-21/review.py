"""Recheck the uploaded fixed pilot offline, preserving every original byte."""
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
RAW = HERE / 'raw/fixed_pilot_780_20260921_004805_GtgB1c'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_save(name, rows):
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


integrity = read(HERE / 'raw_integrity.json')


def check_raw():
    actual = {str(p.relative_to(HERE / 'raw')): digest(p)
              for p in (HERE / 'raw').rglob('*') if p.is_file()}
    assert actual == integrity['file_sha256']


check_raw()
manifest = read(RAW / 'manifest.json')
for folder, local, hashes in [('source', WORKSPACE / 'fixed_pilot_780_v1', manifest['source_hashes']),
                              ('dependencies', WORKSPACE, manifest['dependency_hashes'])]:
    for name, wanted in hashes.items():
        assert digest(RAW / folder / name) == digest(local / name) == wanted, name

# Import verified local analysis helpers only. No GPU worker or setting command runs.
sys.path.insert(0, str(WORKSPACE / 'fixed_pilot_780_v1'))
import fixed
import analyze_pilot

loaded = fixed.load()
telemetry = loaded[4]
recomputed = HERE / 'recomputed'
shutil.copytree(RAW, recomputed, dirs_exist_ok=True)
summary = analyze_pilot.analyze(recomputed, loaded)
outputs = ['analysis.json', 'clock_windows.json', 'report.md', 'measurements.csv',
           'paired_metrics.csv', 'review_queue.csv']
assert all((RAW / name).read_bytes() == (recomputed / name).read_bytes() for name in outputs)
assert summary == read(RAW / 'summary.json')
assert read(RAW / 'status.json') == dict(collection='COMPLETE', restoration='COMMANDS_ACCEPTED', full_sweep='HOLD')
expected_exits = {'worker.exit_code.txt': '0', 'exit_code.txt': '2', 'tee_exit_code.txt': '0'}
for name, expected in expected_exits.items():
    assert (RAW / name).read_text().strip() == expected
assert next(RAW.glob('checked/runs/*/exit_code.txt')).read_text().strip() == '2'
assert all(r['exit_code'] == 0 for r in read(RAW / 'sudo_refresh.json'))
gpu = manifest['baseline']['environment_signature']['gpu_uuid']
for name, option, value in [('set_graphics', '-lgc', '780,780'), ('set_memory', '-lmc', '6801,6801')]:
    cmd = read(RAW / (name + '.json'))
    assert cmd['exit_code'] == 0 and cmd['command'][-2:] == [option, value] and gpu in cmd['command']
restore = read(RAW / 'restore.json')
assert restore['commands_accepted']
for key, option in [('graphics', '-rgc'), ('memory', '-rmc')]:
    cmd = read(RAW / ('reset_' + key + '.json'))
    assert cmd == restore['commands'][key]
    assert cmd['exit_code'] == 0 and cmd['command'][-1] == option and gpu in cmd['command']
env = read(RAW / 'pilot/environment.json')
assert env['executable'] == read(RAW / 'worker.command.json')[0] == '/home/eu/miniconda3/envs/ai_env/bin/python'
assert env['environment']['CONDA_DEFAULT_ENV'] == 'ai_env'

analysis = read(recomputed / 'analysis.json')
rows = analysis['measurements']
windows = read(recomputed / 'clock_windows.json')
meta = read(RAW / 'pilot/telemetry_metadata.json')
samples, cadence = telemetry.read(RAW / 'pilot/telemetry.csv', meta['timezone_offset_seconds'], 50)
by_id = defaultdict(list)
for w in windows:
    by_id[w['task_id']].extend(w['window']['inside'])
inside = [s for w in windows for s in w['window']['inside']]
case_groups, details, counter_examples = defaultdict(list), [], []
for row in rows:
    task = row['task']
    ss = by_id[task['id']]
    assert ss and all(s['clocks.mem'] == 6801 and s['power.limit'] == 70 for s in ss)
    counter = 'clocks_event_reasons_counters.sw_power_cap'
    reason = 'clocks_event_reasons.sw_power_cap'
    detail = dict(task_id=task['id'], shape=task['shape'], dtype=task['dtype'], mode=task['mode'],
                  round=task['round'], median_us=row['median_us'], iqr_pct=100*row['relative_iqr'],
                  drift_pct=100*row['first_last_drift'], sampled_sm_mhz=sorted({s['clocks.sm'] for s in ss}),
                  samples=len(ss), sm_mismatch_samples=sum(s['clocks.sm'] != 780 for s in ss),
                  power_cap_active_samples=sum(s[reason] for s in ss),
                  power_min_w=min(s['power.draw'] for s in ss), power_max_w=max(s['power.draw'] for s in ss),
                  temperature_min_c=min(s['temperature.gpu'] for s in ss), temperature_max_c=max(s['temperature.gpu'] for s in ss),
                  status=row['status'], flags=row['flags'])
    details.append(detail)
    case_groups[(tuple(task['shape']), task['dtype'])].append(detail)
    if not any(s[reason] for s in ss) and ss[-1][counter] > ss[0][counter]:
        counter_examples.append(dict(task=task, samples=len(ss),
            sample_span_s=(datetime.fromisoformat(ss[-1]['utc'])-datetime.fromisoformat(ss[0]['utc'])).total_seconds(),
            counter_delta_us=ss[-1][counter]-ss[0][counter], all_timed_samples_sm_780=all(s['clocks.sm']==780 for s in ss)))

cases = []
for (shape, dtype), members in case_groups.items():
    cases.append(dict(shape=shape, dtype=dtype, measurements=len(members),
        review_measurements=sum(x['status'] == 'REVIEW' for x in members),
        sampled_sm_mhz=sorted({s for x in members for s in x['sampled_sm_mhz']}),
        power_cap_active_samples=sum(x['power_cap_active_samples'] for x in members),
        samples=sum(x['samples'] for x in members),
        power_max_w=max(x['power_max_w'] for x in members),
        temperature_max_c=max(x['temperature_max_c'] for x in members)))

def observed(ss):
    return dict(samples=len(ss),
        ranges={key:[min(s[key] for s in ss),max(s[key] for s in ss)]
                for key in ['clocks.sm','clocks.mem','power.limit','power.draw','temperature.gpu']},
        sm_counts=dict(Counter(s['clocks.sm'] for s in ss)),
        active_counts={key:sum(s[key] is True for s in ss) for key in telemetry.REASONS})

before = read(RAW / 'gpu_before_settings.json')['stdout']
after = read(RAW / 'gpu_after_restore.json')['stdout']
times = [re.search(r'Timestamp\s*:\s*(.*)', t).group(1) for t in [before, after]]
duration = (datetime.strptime(times[1], '%a %b %d %H:%M:%S %Y') -
            datetime.strptime(times[0], '%a %b %d %H:%M:%S %Y')).total_seconds()
clocks_text = re.search(r'\n    Clocks\n(.*?)\n    Applications Clocks', after, re.S).group(1)
post_clocks = {key:int(re.search(key + r'\s*:\s*(\d+) MHz', clocks_text).group(1))
               for key in ['Graphics', 'SM', 'Memory']}
raw_results = read(RAW / 'pilot/results.json')
counter_windows = []
for entry in windows:
    window = entry['window']
    delta = window['counter_deltas_us']['clocks_event_reasons_counters.sw_power_cap']
    if delta and not any(s['clocks_event_reasons.sw_power_cap'] for s in window['inside']):
        counter_windows.append(dict(task_id=entry['task_id'], block=entry['block'],
            span_s=window['counter_interval_seconds'], delta_us=delta,
            sm_mhz=[s['clocks.sm'] for s in window['inside']]))
process_logs = list((RAW / 'pilot').glob('processes_*.json'))
clients = Counter()
for path in process_logs:
    data = read(path)
    assert data['exit_code'] == 0
    for line in data['stdout'].splitlines():
        pid, process, memory = line.split(',', 2)
        if int(pid) != data['diagnostic_pid']:
            clients[process.strip()] += 1
assert len(rows) == 192 and len(windows) == 1344 and len(cases) == 48
assert all(r['correctness']['passed'] and r['correctness']['finite_all'] for r in rows)
assert all(r['result']['timing']['flags'] == [] for r in raw_results)
assert all(r['relative_range'] <= .05 for r in analysis['rounds'])
assert len({s['utc'] for s in inside}) == len(inside)
metrics = dict(summary=summary, elapsed_snapshot_s=duration, snapshot_times=times,
    timed=observed(inside), worker=observed(samples), cadence=cadence,
    missing_inside_windows=sum(not w['window']['inside'] for w in windows),
    max_iqr_pct=100*max(r['relative_iqr'] for r in rows),
    max_abs_drift_pct=100*max(abs(r['first_last_drift']) for r in rows),
    round_groups=len(analysis['rounds']), max_round_range_pct=100*max(r['relative_range'] for r in analysis['rounds']),
    worst_round=max(analysis['rounds'],key=lambda r:r['relative_range']),
    review_cases=[r for r in cases if r['review_measurements']],
    correctness_scopes=dict(Counter(r['correctness']['scope'] for r in rows)),
    max_stored_correctness_tolerance_ratio=max(r['correctness']['max_tolerance_ratio'] for r in rows),
    counter_examples_no_active_timed_samples=counter_examples,
    counter_windows_no_active_inside_samples=counter_windows,
    sm_mismatch_windows=sum(any(s['clocks.sm'] != 780 for s in w['window']['inside']) for w in windows),
    power_cap_active_windows=sum(any(s['clocks_event_reasons.sw_power_cap'] for s in w['window']['inside']) for w in windows),
    process_snapshots=len(process_logs), other_logged_clients=dict(clients),
    counters=analysis['telemetry']['counters'], post_restore_clocks=post_clocks,
    limits=['Host envelopes around timed CUDA event blocks; samples are not continuous observation.',
            'Stored full or 256-element sampled correctness checks are not reexecuted on GPU.',
            'Same-process rounds do not establish independent-session reproducibility.',
            'No allocating torch.mm path, new Nsight capture, or matched automatic-core run.',
            '780 MHz is user-selected; full remains HOLD pending review.'])
csv_save('measurements_review.csv', details)
csv_save('case_clocks.csv', cases)
save('review_metrics.json', metrics)
check_raw()
save('verification.json', dict(status='PASS', zip_sha256=integrity['zip_sha256'],
    raw_files=integrity['file_count'], raw_unchanged=True,
    sources_verified=len(manifest['source_hashes']), dependencies_verified=len(manifest['dependency_hashes']),
    reproduced_outputs_byte_equal=outputs, summary_equal=True,
    expected_exits=expected_exits, checked_exit_code=2,
    task_order_inputs_graph_denominators_timing_statistics_correctness_records='verified by unchanged analyzer',
    environment_executable=env['executable'], clock_set_and_reset_commands_verified=True))
print(json.dumps({k:v for k,v in metrics.items() if k not in ['summary','counter_examples_no_active_timed_samples',
                                                          'counter_windows_no_active_inside_samples']}, indent=2))
print('Counter examples without active timed samples:', len(counter_examples))
