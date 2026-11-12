"""Offline verification of the 780/750/780 recheck; raw bytes stay unchanged."""
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
RAW = HERE / 'raw/clock_recheck_20260921_143452_YZ9HN5'


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


integrity = read(HERE / 'raw_sha256.json')
ZIP_SHA = '274e47dcaaed7aabd6312c931e91130d29ca853bc757bc89b1e66f4dea7b74ab'


def check_raw():
    actual = {str(p.relative_to(RAW)): digest(p)
              for p in RAW.rglob('*') if p.is_file()}
    assert actual == integrity


check_raw()
manifest = read(RAW / 'manifest.json')
for folder, local, hashes in [('source', WORKSPACE / 'clock_recheck_v1', manifest['source_hashes']),
                              ('dependencies', WORKSPACE, manifest['dependency_hashes'])]:
    for name, wanted in hashes.items():
        assert digest(RAW / folder / name) == digest(local / name) == wanted, name

# Only verified local offline analysis helpers are imported; no GPU worker runs.
sys.path.insert(0, str(WORKSPACE / 'clock_recheck_v1'))
import lower
import analyze_lower

loaded = lower.load()
telemetry = loaded[4]
recomputed = HERE / 'recomputed'
shutil.copytree(RAW, recomputed, dirs_exist_ok=True)
summary = analyze_lower.analyze(recomputed, loaded)
outputs = ['analysis.json', 'clock_windows.json', 'report.md', 'measurements.csv',
           'paired_metrics.csv', 'review_queue.csv', 'repeat_780.csv']
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
    assert all(s['clocks.mem'] == 6801 and s['power.limit'] == 70 for s in samples)
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
    timed = [sample for w in windows if w['stage']==name for sample in w['window']['inside']]
    stage_rows = [r for r in rows if r['stage']==name]
    stage_metrics.append(dict(stage=name, sm_mhz=mhz, samples=len(samples), span_s=info['span_s'],
        measurements=len(stage_rows),review_measurements=sum(r['status']=='REVIEW' for r in stage_rows),
        timing_and_sampled_clocks_pass=stage['timing_and_sampled_clocks_pass'],
        worker_sm_mhz=sorted({s['clocks.sm'] for s in samples}),
        worker_mismatch_samples=sum(s['clocks.sm']!=mhz for s in samples),
        timed_samples=len(timed),timed_sm_mhz=sorted({s['clocks.sm'] for s in timed}),
        timed_mismatch_samples=sum(s['clocks.sm']!=mhz for s in timed),
        timed_power_cap_samples=sum(s[telemetry.REASONS[0]] is True for s in timed),
        thermal_active_samples={k:sum(s[k] is True for s in samples) for k in telemetry.REASONS[1:]},
        max_iqr_pct=100*stage['max_relative_iqr'], max_abs_drift_pct=100*stage['max_abs_drift'],
        max_round_range_pct=100*stage['max_between_round_range'],
        memory_mhz=info['ranges']['clocks.mem'], power_limit_w=info['ranges']['power.limit'],
        power_draw_w=info['ranges']['power.draw'], temperature_c=info['ranges']['temperature.gpu'],
        power_cap_active_samples=info['counters'][telemetry.COUNTERS[0]]['active_samples'],
        power_cap_counter_delta_us=info['counters'][telemetry.COUNTERS[0]]['delta_us'],
        counter_anomaly_blocks=stage['counter_anomaly_blocks'], cadence=cadence))

assert len(set(all_timestamps)) == len(all_timestamps) == 756
assert len(set(result_hashes)) == 3
assert len({s['utc'] for s in all_samples}) == len(all_samples)
assert len(rows) == 108 and len(windows) == 756 and len(clocks) == 18
assert all(r['correctness']['passed'] and r['correctness']['finite_all'] for r in rows)
assert all(w['window']['inside'] for w in windows)
assert digest(HERE / 'clock_recheck_20260921_143452_YZ9HN5.zip') == ZIP_SHA
assert (RAW / 'launcher_source.sh').read_bytes() == (WORKSPACE / 'clock_recheck_v1/launch.sh').read_bytes()
assert Counter(r['status'] for r in rows) == {'PASS':106,'REVIEW':2}
assert len({(r['stage'],r['task']['id']) for r in rows}) == 108
sensitivity=[]
for (shape,dtype,mode),times in clocks.items():
    sensitivity.append(dict(shape=shape,dtype=dtype,mode=mode,**times,
        change_750_vs_780before_pct=100*(times['s1_750']/times['s0_780_before']-1),
        change_750_vs_780after_pct=100*(times['s1_750']/times['s2_780_after']-1)))

# Reuse recorded sample windows, retaining active flags separately from frequency drops.
incident=[]
for w in windows:
    inside=w['window']['inside'];target=next(s['sm_mhz'] for s in loaded[-1]['stages'] if s['name']==w['stage'])
    caps=sum(s[telemetry.REASONS[0]] is True for s in inside)
    drops=sum(s['clocks.sm']!=target for s in inside)
    if caps or drops:
        task=next(r['task'] for r in rows if (r['stage'],r['task']['id'])==(w['stage'],w['task_id']))
        incident.append(dict(stage=w['stage'],task_id=w['task_id'],block=w['block'],shape=task['shape'],dtype=task['dtype'],mode=task['mode'],round=task['round'],power_cap_samples=caps,clock_mismatch_samples=drops,start_utc=w['window']['start_utc'],end_utc=w['window']['end_utc']))
bad=[s for s in all_samples if s[telemetry.REASONS[0]] is True or s['clocks.sm'] not in (750,780)]
assert bad
for s in stage_metrics:
    meta=read(RAW/s['stage']/'telemetry_metadata.json')
    sample,_=telemetry.read(RAW/s['stage']/'telemetry.csv',meta['timezone_offset_seconds'],50)
    s['worker_sm_counts']=dict(Counter(x['clocks.sm'] for x in sample))

# Cross-run comparison uses the original incident's same shape, dtype, seed and mode.
oldraw=WORKSPACE/'analysis/abstract_200_780_result_2026-09-21/raw/abstract_200_780/raw_attempts.jsonl'
prior=[json.loads(line) for line in oldraw.read_text().splitlines()]
prior={r['task']['mode']:r for r in prior if r['task']['shape']==[3128,680,488] and r['task']['dtype']=='fp32'}
cross_run=[]
for name in ('s0_780_before','s2_780_after'):
    for record in read(RAW/name/'results.json'):
        t,r=record['task'],record['result']
        if t['shape']!=[3128,680,488] or t['dtype']!='fp32':continue
        old=prior[t['mode']]
        assert r['input_seed']==old['input_seed']
        assert r['correctness']==old['correctness']
        if t['mode']=='graph':assert r['timing']['graph_nodes']==old['timing']['graph_nodes']==5
        cross_run.append(dict(stage=name,mode=t['mode'],round=t['round'],old_us=old['timing']['median_ms']*1000,new_us=r['timing']['median_ms']*1000,change_pct=100*(r['timing']['median_ms']/old['timing']['median_ms']-1)))
metrics=dict(summary=summary,stages=stage_metrics,total_samples=len(all_samples),
    clock_counts=dict(Counter(s['clocks.sm'] for s in all_samples)),power_cap_samples=sum(s[telemetry.REASONS[0]] is True for s in all_samples),
    incident_sample_first_kst=bad[0]['timestamp'],incident_sample_last_kst=bad[-1]['timestamp'],
    incident_sample_span_s=(datetime.fromisoformat(bad[-1]['utc'])-datetime.fromisoformat(bad[0]['utc'])).total_seconds(),
    incident_timed_blocks=len(incident),max_iqr_pct=max(s['max_iqr_pct'] for s in stage_metrics),
    max_abs_drift_pct=max(s['max_abs_drift_pct'] for s in stage_metrics),max_round_range_pct=max(s['max_round_range_pct'] for s in stage_metrics),
    max_abs_repeat780_change_pct=100*max(abs(r['after_over_before_minus_one']) for r in analysis['repeat_780']),
    matched750_latency_change_pct=dict(minimum=min(r['change_750_vs_780before_pct'] for r in sensitivity),median=st.median(r['change_750_vs_780before_pct'] for r in sensitivity),maximum=max(r['change_750_vs_780before_pct'] for r in sensitivity),condition_modes=len(sensitivity)),
    same_problem_cross_run_change_pct=dict(minimum=min(r['change_pct'] for r in cross_run),maximum=max(r['change_pct'] for r in cross_run)),
    limitations=['Three diagnostic shapes; 750 MHz full-sweep validity not established.',
        'Order, temperature and process state are not independently randomized.',
        'Sample counts and first-to-last span are not throttled duration or duty cycle.',
        'Original 200-shape workload history not replayed; previous raw result remains unchanged.',
        'Counter interpretation unresolved; REVIEW preserved separately from instantaneous reason/clock evidence.'])
csv_save('stage_metrics.csv',stage_metrics)
csv_save('clock_sensitivity.csv',sensitivity)
csv_save('incident_timed_blocks.csv',incident)
csv_save('incident_samples.csv',bad)
csv_save('host_time_checks.csv',block_checks)
csv_save('same_problem_cross_run.csv',cross_run)
save('review_metrics.json',metrics)
check_raw()
save('validation.json',dict(status='PASS',zip_sha256=ZIP_SHA,raw_files=len(integrity),raw_unchanged=True,
    sources_verified=len(manifest['source_hashes']),dependencies_verified=len(manifest['dependency_hashes']),
    reproduced_outputs_byte_equal=outputs,summary_equal=True,worker_exit_codes=[0]*3,expected_exits=expected_exits,
    unique_block_timestamps=len(set(all_timestamps)),unique_stage_result_hashes=len(set(result_hashes)),
    correctness_records_passed=len(rows),new_gpu_executions=0,measurement_count=108,blocks=756,triplets=36,
    task_order_inputs_graph_denominators_statistics='verified by unchanged analyzer and block arithmetic',
    clock_set_and_reset_commands_verified=True))
print(json.dumps({k:v for k,v in metrics.items() if k not in ('summary','stages','limitations')},indent=2))
