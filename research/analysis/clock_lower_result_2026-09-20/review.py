"""Offline verification of the lower-clock screening; raw bytes stay unchanged."""
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
RAW = HERE / 'raw/clock_lower_20260920_192959_la2ofx'


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
for folder, local, hashes in [('source', WORKSPACE / 'clock_lower_v1', manifest['source_hashes']),
                              ('dependencies', WORKSPACE, manifest['dependency_hashes'])]:
    for name, wanted in hashes.items():
        assert digest(RAW / folder / name) == digest(local / name) == wanted, name

# Only verified local offline analysis helpers are imported; no GPU worker runs.
sys.path.insert(0, str(WORKSPACE / 'clock_lower_v1'))
import lower
import analyze_lower

loaded = lower.load()
telemetry = loaded[4]
recomputed = HERE / 'recomputed'
shutil.copytree(RAW, recomputed, dirs_exist_ok=True)
summary = analyze_lower.analyze(recomputed, loaded)
outputs = ['analysis.json', 'clock_windows.json', 'report.md', 'measurements.csv',
           'paired_metrics.csv', 'review_queue.csv', 'repeat_750.csv']
assert all((RAW / name).read_bytes() == (recomputed / name).read_bytes() for name in outputs)
assert summary == read(RAW / 'summary.json')
assert read(RAW / 'status.json') == dict(collection='COMPLETE', restoration='COMMANDS_ACCEPTED', full_sweep='HOLD')
expected_exits = {'exit_code.txt': '2', 'tee_exit_code.txt': '0'}
for name, expected in expected_exits.items():
    assert (RAW / name).read_text().strip() == expected
assert next(RAW.glob('checked/runs/*/exit_code.txt')).read_text().strip() == '2'
assert all(r['exit_code'] == 0 for r in read(RAW / 'sudo_refresh.json'))
gpu = manifest['baseline']['environment_signature']['gpu_uuid']
restore = read(RAW / 'restore.json')
assert restore['commands_accepted']
for key, option in [('graphics', '-rgc'), ('memory', '-rmc')]:
    cmd = read(RAW / ('reset_' + key + '.json'))
    assert cmd == restore['commands'][key]
    assert cmd['exit_code'] == 0 and cmd['command'][-1] == option and gpu in cmd['command']

analysis = read(recomputed / 'analysis.json')
rows = analysis['measurements']
windows = read(recomputed / 'clock_windows.json')
stage_metrics, block_checks, clocks = [], [], defaultdict(dict)
all_samples, all_timestamps, result_hashes = [], [], []
for stage in analysis['stages']:
    config, info = stage['stage'], stage['telemetry']
    name, mhz = config['name'], config['sm_mhz']
    assert (RAW / (name + '.exit_code.txt')).read_text().strip() == '0'
    for suffix, option, value in [('core', '-lgc', f'{mhz},{mhz}'), ('memory', '-lmc', '6801,6801')]:
        cmd = read(RAW / (name + '_' + suffix + '.json'))
        assert cmd['exit_code'] == 0 and cmd['command'][-2:] == [option, value] and gpu in cmd['command']
    env = read(RAW / name / 'environment.json')
    command = read(RAW / (name + '.command.json'))
    assert env['executable'] == command[0] == '/home/eu/miniconda3/envs/ai_env/bin/python'
    assert env['environment']['CONDA_DEFAULT_ENV'] == 'ai_env' and command[-2:] == ['--stage', name]
    meta = read(RAW / name / 'telemetry_metadata.json')
    samples, cadence = telemetry.read(RAW / name / 'telemetry.csv', meta['timezone_offset_seconds'], 50)
    assert all(s['clocks.sm'] == mhz and s['clocks.mem'] == 6801 and s['power.limit'] == 70 for s in samples)
    assert all(not s[key] for s in samples for key in telemetry.REASONS)
    all_samples.extend(samples)
    results = read(RAW / name / 'results.json')
    result_hashes.append(digest(RAW / name / 'results.json'))
    for record in results:
        task, timing = record['task'], record['result']['timing']
        assert timing['flags'] == []
        for block in timing['blocks']:
            denominator = block['repeats'] * block['operations_per_call']
            assert math.isclose(block['latency_ms'], block['elapsed_ms'] / denominator, rel_tol=1e-12)
            assert math.isclose(block['host_latency_ms'], block['host_wall_ms'] / denominator, rel_tol=1e-12)
            all_timestamps.append(block['timestamp'])
        block_checks.append(dict(stage=name, task_id=task['id'], shape=task['shape'],
            dtype=task['dtype'], mode=task['mode'], round=task['round'],
            event_median_us=1000*st.median(b['latency_ms'] for b in timing['blocks']),
            host_median_us=1000*st.median(b['host_latency_ms'] for b in timing['blocks']),
            max_abs_host_event_difference_pct=100*max(abs(b['host_wall_ms']/b['elapsed_ms']-1) for b in timing['blocks'])))
    for r in stage['rounds']:
        clocks[(tuple(r['shape']),r['dtype'],r['mode'])][name] = st.median(r['round_us'])
    stage_metrics.append(dict(stage=name, sm_mhz=mhz, samples=len(samples), span_s=info['span_s'],
        max_iqr_pct=100*stage['max_relative_iqr'], max_abs_drift_pct=100*stage['max_abs_drift'],
        max_round_range_pct=100*stage['max_between_round_range'],
        memory_mhz=info['ranges']['clocks.mem'], power_limit_w=info['ranges']['power.limit'],
        power_draw_w=info['ranges']['power.draw'], temperature_c=info['ranges']['temperature.gpu'],
        power_cap_active_samples=info['counters'][telemetry.COUNTERS[0]]['active_samples'],
        power_cap_counter_delta_us=info['counters'][telemetry.COUNTERS[0]]['delta_us'],
        counter_anomaly_blocks=stage['counter_anomaly_blocks'], cadence=cadence))

assert len(set(all_timestamps)) == len(all_timestamps) == 1680
assert len(set(result_hashes)) == 4
assert len({s['utc'] for s in all_samples}) == len(all_samples)
assert len(rows) == 240 and len(windows) == 1680 and len(clocks) == 30
assert all(r['correctness']['passed'] and r['correctness']['finite_all'] for r in rows)
assert all(w['window']['inside'] for w in windows)
sensitivity, modes = [], []
for (shape, dtype, mode), times in clocks.items():
    sensitivity.append(dict(shape=shape, dtype=dtype, mode=mode, **times,
        change_780_vs_750before_pct=100*(times['s1_780']/times['s0_750_before']-1),
        change_810_vs_750before_pct=100*(times['s2_810']/times['s0_750_before']-1),
        four_stage_range_pct=100*(max(times.values())-min(times.values()))/st.median(times.values())))
    if mode == 'prealloc':
        for name in times:
            aa, pp, gg = [clocks[(shape,dtype,m)][name] for m in ['allocate','prealloc','graph']]
            modes.append(dict(stage=name, shape=shape, dtype=dtype, allocate_us=aa, prealloc_us=pp,
                graph_us=gg, allocate_vs_prealloc_pct=100*(aa/pp-1), graph_vs_prealloc_pct=100*(gg/pp-1)))

snapshots = [read(RAW / name)['stdout'] for name in ['gpu_before_settings.json','gpu_after_restore.json']]
times = [re.search(r'Timestamp\s*:\s*(.*)', s).group(1) for s in snapshots]
duration = (datetime.strptime(times[1], '%a %b %d %H:%M:%S %Y') -
            datetime.strptime(times[0], '%a %b %d %H:%M:%S %Y')).total_seconds()
clocks_text = re.search(r'\n    Clocks\n(.*?)\n    Applications Clocks', snapshots[1], re.S).group(1)
post_clocks = {key:int(re.search(key + r'\s*:\s*(\d+) MHz', clocks_text).group(1)) for key in ['Graphics','SM','Memory']}
metrics = dict(summary=summary, stages=stage_metrics, elapsed_snapshot_s=duration, snapshot_times=times,
    total_samples=len(all_samples), total_sample_spans_s=sum(s['span_s'] for s in stage_metrics),
    max_iqr_pct=max(s['max_iqr_pct'] for s in stage_metrics),
    max_abs_drift_pct=max(s['max_abs_drift_pct'] for s in stage_metrics),
    max_round_range_pct=max(s['max_round_range_pct'] for s in stage_metrics),
    max_abs_repeat750_change_pct=100*max(abs(r['after_over_before_minus_one']) for r in analysis['repeat_750']),
    worst_repeat750=max(analysis['repeat_750'],key=lambda r:abs(r['after_over_before_minus_one'])),
    max_abs_810_vs_750_change_pct=max(abs(r['change_810_vs_750before_pct']) for r in sensitivity),
    max_abs_allocate_vs_prealloc_at810_pct=max(abs(r['allocate_vs_prealloc_pct']) for r in modes if r['stage']=='s2_810'),
    max_abs_allocate_vs_prealloc_all_stages_pct=max(abs(r['allocate_vs_prealloc_pct']) for r in modes),
    max_abs_host_event_block_difference_pct=max(r['max_abs_host_event_difference_pct'] for r in block_checks),
    correctness_scopes=dict(Counter(r['correctness']['scope'] for r in rows)),
    post_restore_clocks=post_clocks,
    limits=['Sampled clocks do not establish continuous or effective kernel frequency.',
            'Stored correctness records were checked, not rerun on GPU.',
            'Ten shape/dtype combinations; not independent-session or all-shape evidence.',
            'No new profiler capture or automatic-core load comparison.',
            'Mode differences are not causal CPU wait or allocation cost.',
            'Counter interpretation unresolved; final clock unselected and full HOLD.'])
csv_save('stage_metrics.csv', stage_metrics)
csv_save('mode_comparison.csv', modes)
csv_save('clock_sensitivity.csv', sensitivity)
csv_save('host_time_checks.csv', block_checks)
save('review_metrics.json', metrics)
check_raw()
save('verification.json', dict(status='PASS', zip_sha256=integrity['zip_sha256'],
    raw_files=integrity['file_count'], raw_unchanged=True,
    sources_verified=len(manifest['source_hashes']), dependencies_verified=len(manifest['dependency_hashes']),
    reproduced_outputs_byte_equal=outputs, summary_equal=True, worker_exit_codes=[0]*4,
    expected_exits=expected_exits, checked_exit_code=2, unique_block_timestamps=len(set(all_timestamps)),
    unique_stage_result_hashes=len(set(result_hashes)),
    task_order_inputs_graph_denominators_statistics_correctness_records='verified by unchanged analyzer',
    environment_executable=env['executable'], clock_set_and_reset_commands_verified=True))
print(json.dumps({k:v for k,v in metrics.items() if k not in ['summary','stages','limits']}, indent=2))
