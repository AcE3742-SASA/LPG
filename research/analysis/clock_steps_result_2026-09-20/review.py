"""Verify the uploaded ladder and summarize raw timing/telemetry without GPU calls."""
from collections import Counter
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
RAW = HERE / 'raw/clock_steps_20260920_170208_OHnJ1h'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def seconds(a, b):
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()


integrity = read(HERE / 'raw_integrity.json')


def check_raw():
    actual = {str(p.relative_to(HERE / 'raw')): digest(p)
              for p in (HERE / 'raw').rglob('*') if p.is_file()}
    assert actual == integrity['files']


check_raw()
manifest = read(RAW / 'manifest.json')
for folder, local, hashes in [('source', WORKSPACE / 'clock_steps_v1', manifest['source_hashes']),
                              ('dependencies', WORKSPACE, manifest['dependency_hashes'])]:
    for name, wanted in hashes.items():
        assert digest(RAW / folder / name) == digest(local / name) == wanted, name

# Only verified local helpers are imported; run/worker/clock setters are never called.
sys.path.insert(0, str(WORKSPACE / 'clock_steps_v1'))
import steps
import analyze_steps

loaded = steps.load()
telemetry = loaded[3]
recomputed = HERE / 'recomputed'
shutil.copytree(RAW, recomputed, dirs_exist_ok=True)
summary = analyze_steps.analyze(recomputed, loaded)
outputs = ['analysis.json', 'clock_windows.json', 'report.md']
assert all((RAW / name).read_bytes() == (recomputed / name).read_bytes() for name in outputs)
assert summary == read(RAW / 'summary.json')
assert read(RAW / 'status.json') == dict(collection='COMPLETE', restoration='COMMANDS_ACCEPTED', full_sweep='HOLD')
assert (RAW / 'exit_code.txt').read_text().strip() == '2'
assert (RAW / 'tee_exit_code.txt').read_text().strip() == '0'
assert next(RAW.glob('checked/runs/*/exit_code.txt')).read_text().strip() == '2'
assert read(RAW / 'progress.json') == dict(stage_completed=7, total_stages=7)
assert all(r['exit_code'] == 0 for r in read(RAW / 'sudo_refresh.json'))

gpu = manifest['baseline']['environment_signature']['gpu_uuid']
restore = read(RAW / 'restore.json')
assert restore['commands_accepted']
for key, option in [('graphics', '-rgc'), ('memory', '-rmc')]:
    item = read(RAW / ('reset_' + key + '.json'))
    assert item == restore['commands'][key]
    assert item['exit_code'] == 0 and item['command'][-1] == option and gpu in item['command']

analysis = read(recomputed / 'analysis.json')
windows = read(recomputed / 'clock_windows.json')
metrics = []
measurements = []
case_clock_rows = []
missing = []
for stage in analysis['stages']:
    spec = stage['stage']
    name = spec['name']
    for suffix, option, value in [('core', '-lgc', f"{spec['sm_mhz']},{spec['sm_mhz']}"),
                                  ('memory', '-lmc', '6801,6801')]:
        command = read(RAW / (name + '_' + suffix + '.json'))
        assert command['exit_code'] == 0 and command['command'][-2:] == [option, value]
        assert gpu in command['command']
    env = read(RAW / name / 'environment.json')
    assert env['executable'] == read(RAW / (name + '.command.json'))[0] == '/home/eu/miniconda3/envs/ai_env/bin/python'
    assert env['environment']['CONDA_DEFAULT_ENV'] == 'ai_env'
    ws = [r for r in windows if r['stage'] == name]
    rows = [r for r in analysis['measurements'] if r['stage'] == name]
    inside = [s for w in ws for s in w['window']['inside']]
    assert len(ws) == 112 and len(rows) == 16
    for w in ws:
        v = w['window']
        if not v['inside']:
            missing.append(dict(stage=name, task_id=w['task_id'], block=w['block'],
                host_envelope_ms=1000*seconds(v['start_utc'], v['end_utc']),
                surrounding_poll_gap_ms=1000*seconds(v['before']['utc'], v['after']['utc']),
                before_sm_mem=[v['before']['clocks.sm'], v['before']['clocks.mem']],
                after_sm_mem=[v['after']['clocks.sm'], v['after']['clocks.mem']]))
    metrics.append(dict(stage=name, target_sm=spec['sm_mhz'], polling_ms=spec['interval_ms'],
        statuses=dict(Counter(r['status'] for r in rows)),
        max_iqr_pct=100*max(r['relative_iqr'] for r in rows),
        max_abs_drift_pct=100*max(abs(r['first_last_drift']) for r in rows),
        max_round_range_pct=100*max(r['relative_range'] for r in stage['rounds']),
        timed_inside_samples=len(inside), blocks_without_sample=sum(not w['window']['inside'] for w in ws),
        timed_ranges={key:[min(s[key] for s in inside), max(s[key] for s in inside)]
                      for key in ['clocks.sm','clocks.mem','power.draw','power.limit','temperature.gpu']},
        timed_active_counts={key:sum(s[key] is True for s in inside) for key in telemetry.REASONS},
        worker_telemetry=stage['counters']))
    for row in rows:
        t = row['task']
        measurements.append(dict(stage=name, task_id=t['id'], shape=t['shape'], dtype=t['dtype'],
            mode=t['mode'], round=t['round'], median_us=row['median_us'],
            iqr_pct=100*row['relative_iqr'], drift_pct=100*row['first_last_drift'],
            sampled_clocks=row['observed_values']['clocks.sm'], flags=row['flags'], status=row['status']))
    for index, case in enumerate(manifest['config']['fixed_cases']):
        ss = [s for w in ws if w['task_id'].startswith(f'c{index}_') for s in w['window']['inside']]
        case_clock_rows.append(dict(stage=name, case=index, shape=case['shape'], dtype=case['dtype'],
            samples=len(ss), sm_values=sorted({s['clocks.sm'] for s in ss}),
            sm_median=st.median(s['clocks.sm'] for s in ss),
            power_range=[min(s['power.draw'] for s in ss), max(s['power.draw'] for s in ss)],
            active_sw_power_cap=sum(s['clocks_event_reasons.sw_power_cap'] is True for s in ss)))

comparisons = {}
for label in sorted({r['comparison'] for r in analysis['comparisons']}):
    rows = [r for r in analysis['comparisons'] if r['comparison'] == label]
    comparisons[label] = dict(rows=rows,
        max_absolute_change_pct=100*max(abs(r['second_over_first_minus_one']) for r in rows),
        max_relative_range_pct=100*max(r['relative_range'] for r in rows))

before = read(RAW / 'gpu_before_settings.json')['stdout']
after = read(RAW / 'gpu_after_restore.json')['stdout']
times = [re.search(r'Timestamp\s*:\s*(.*)', t).group(1) for t in [before, after]]
duration = (datetime.strptime(times[1], '%a %b %d %H:%M:%S %Y') -
            datetime.strptime(times[0], '%a %b %d %H:%M:%S %Y')).total_seconds()
clocks_text = re.search(r'\n    Clocks\n(.*?)\n    Applications Clocks', after, re.S).group(1)
post_clocks = {key:int(re.search(key + r'\s*:\s*(\d+) MHz', clocks_text).group(1))
               for key in ['Graphics', 'SM', 'Memory']}
idle = [dict(name=r['phase']['name'], **r['telemetry']) for r in analysis['idle']]
assert len(missing) == 5 and all('100' in r['stage'] for r in missing)
assert summary['recorded'] == 112 and summary['timed_blocks'] == 784 and len(idle) == 16
assert all(r['timed_ranges']['clocks.mem'] == [6801,6801] for r in metrics)
assert all(r['timed_ranges']['power.limit'] == [70,70] for r in metrics)
assert all(r['correctness']['passed'] and r['correctness']['finite_all'] and
           r['correctness']['scope'] == 'sampled' and r['correctness']['points'] == 256
           for r in analysis['measurements'])
for name, rows in [('measurements.csv', measurements), ('case_clocks.csv', case_clock_rows)]:
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
save('review_metrics.json', dict(summary=summary, stages=metrics, case_clocks=case_clock_rows,
    idle=idle, comparisons=comparisons, missing_windows=missing,
    gpu_snapshots_local_time=times, elapsed_snapshot_s=duration, post_restore_clocks=post_clocks,
    interpretation=dict(candidate_for_broader_pilot_mhz=900, memory_mhz=6801, polling_ms=50,
        final_clock_selected=False, full_sweep='HOLD',
        counter_issue='Unresolved. Also grows in automatic-core/no-diagnostic-worker phases; not a GEMM slowdown fraction.',
        limits=['Four development conditions; no all-shape or independent-session certification.',
                'No new automatic-core load control, no profiler trace.',
                '50/100 ms comparison includes process/order/time effects; zero overhead is unproven.',
                'Intermediate 915-1035 MHz settings were not tested.',
                'Recorded instantaneous status is not continuous observation.'])))
check_raw()
save('verification.json', dict(status='PASS', raw_files=integrity['file_count'],
    archive_sha256=integrity['archive_sha256'], raw_unchanged=True,
    sources_verified=7, dependencies_verified=28, reproduced_outputs_byte_equal=outputs,
    summary_equal=True, worker_exit_codes=[0]*7, final_exit_code=2,
    environment_executable='/home/eu/miniconda3/envs/ai_env/bin/python',
    correctness='All 112 stored 256-element sampled comparisons passed; full numerical validation not claimed.',
    clock_set_and_reset_commands_verified=True))
print(json.dumps(dict(verification='PASS', stages=[{k:v for k,v in x.items() if k!='worker_telemetry'} for x in metrics],
    missing_windows=missing, duration_s=duration, post_restore_clocks=post_clocks),indent=2))
