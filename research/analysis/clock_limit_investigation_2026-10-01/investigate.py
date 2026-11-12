"""Read-only comparison of clock evidence for six reviewed, non-holdout shapes."""
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime
import csv
import json
from pathlib import Path
import sqlite3
import statistics as stats
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'clock_audit_v1'))
import telemetry

RUNS = {
    'original': 'full_720_result_2026-09-30/recomputed/full_720',
    'v1': 'remeasure_720_v1_result_2026-09-30/recomputed/full_720_remeasure_20260930',
    'v3': 'remeasure_720_v3_result_2026-10-01/recomputed/full_720_remeasure2_20260930',
    'v4': 'remeasure_720_v4_result_2026-10-01/recomputed/full_720_remeasure3_20261001',
}
V4 = ROOT / 'analysis' / RUNS['v4']
SESSION = '20261001_143341_57d18ef1'
CAP = 'clocks_event_reasons.sw_power_cap'


def read(path):
    return json.loads(path.read_text())


def epoch(value):
    return datetime.fromisoformat(value).timestamp()


def summarize(samples):
    assert samples
    result = {'samples': len(samples)}
    for key in ('clocks.sm', 'power.draw', 'temperature.gpu', 'utilization.gpu'):
        values = [s[key] for s in samples]
        result[key] = dict(min=min(values), median=stats.median(values), max=max(values))
    result['cap_samples'] = sum(s[CAP] for s in samples)
    result['clock_mismatch_samples'] = sum(s['clocks.sm'] != 720 for s in samples)
    result['reason_masks'] = dict(Counter(hex(s['clocks_event_reasons.active']) for s in samples))
    return result


def main():
    manifest = read(V4 / 'manifest.json')
    shapes = {tuple(s) for s in manifest['config']['remeasurement_schedule']['isolated_fp32_shapes']}
    selected = {t['id']: t for t in manifest['tasks']
                if tuple(t['shape']) in shapes and t['split'] != 'fresh_holdout'}
    isolated_path = V4 / 'sessions' / SESSION / 'segment_00015'
    isolated_ids = set(read(isolated_path / 'collection_group.json')['task_ids'])
    assert len(isolated_ids) == 51 and isolated_ids <= selected.keys()
    all_results, all_rows = [], {}
    timeline, processes, segment_results = None, [], []
    for label, relative in RUNS.items():
        run = ROOT / 'analysis' / relative
        cache = {}

        def load_segment(relative_path):
            if relative_path not in cache:
                folder = run / relative_path
                meta = read(folder / 'telemetry_metadata.json')
                samples, _ = telemetry.read(folder / 'telemetry.csv', meta['timezone_offset_seconds'], 50)
                cache[relative_path] = (samples, [epoch(s['utc']) for s in samples], read(folder / 'quality.json'))
            return cache[relative_path]

        with sqlite3.connect((run / 'results.sqlite3').as_uri()+'?mode=ro&immutable=1', uri=True) as db:
            placeholders = ','.join('?' for _ in selected)
            rows = [json.loads(r[0]) for r in db.execute(
                f'SELECT record FROM attempts WHERE task_id IN ({placeholders}) ORDER BY seq', list(selected))]
        assert len(rows) == len(selected) and len({r['task']['id'] for r in rows}) == len(selected)
        all_rows[label] = rows
        for row in rows:
            task = row['task']
            assert task['split'] != 'fresh_holdout'
            samples, times, quality = load_segment(row['segment'])
            inside = []
            for block in row['timing']['blocks']:
                lo = bisect_left(times, epoch(block['host_envelope_start_utc']))
                hi = bisect_right(times, epoch(block['host_envelope_end_utc']))
                inside.extend(samples[lo:hi])
            q = quality['measurements'][task['id']]
            summary = summarize(inside)
            assert bool(summary['cap_samples']) == ('observed_active:'+CAP in q['flags'])
            assert bool(summary['clock_mismatch_samples']) == ('observed_mismatch:clocks.sm' in q['flags'])
            all_results.append(dict(run=label, task=task, original_session=row['session'],
                isolated_id=task['id'] in isolated_ids, timestamp=row['timestamp'],
                timing_samples=summary, quality_status=q['status'], flags=q['flags'],
                median_ms=row['timing']['median_ms'], graph_nodes=row['timing'].get('graph_nodes'),
                settle_event_ms=row['timing']['setup']['settle_event_ms']))
        if label == 'v4':
            timeline, _, _ = load_segment(str(isolated_path.relative_to(V4)))
            for folder in sorted((run/'sessions'/SESSION).glob('segment_*')):
                samples, _, _ = load_segment(str(folder.relative_to(run)))
                segment_results.append(dict(segment=folder.name, **summarize(samples)))
            for p in sorted(isolated_path.glob('processes*.json')):
                data = read(p)
                lines = [line for line in data['stdout'].splitlines() if line.strip()]
                assert data['exit_code'] == 0
                assert all(int(line.split(',')[0]) == data['diagnostic_pid'] for line in lines)
                processes.append(p.name)

    paired = []
    for label in RUNS:
        rows = [r for r in all_results if r['run'] == label and r['isolated_id']]
        samples = [r['timing_samples'] for r in rows]
        paired.append(dict(run=label, tasks=len(rows),
            direct_cap=sum(bool(s['cap_samples']) for s in samples),
            direct_clock=sum(bool(s['clock_mismatch_samples']) for s in samples),
            sm_min=min(s['clocks.sm']['min'] for s in samples),
            power_median_of_task_medians=stats.median(s['power.draw']['median'] for s in samples),
            temperature_median_of_task_medians=stats.median(s['temperature.gpu']['median'] for s in samples)))
    assert paired[-1]['direct_cap'] == paired[-1]['direct_clock'] == 51
    first = min((r for r in all_rows['v4'] if r['task']['id'] in isolated_ids), key=lambda r:r['timestamp'])
    first_cap = next(s for s in timeline if s[CAP])
    first_down = next(s for s in timeline if s['clocks.sm'] < 720)
    t0 = epoch(timeline[0]['utc'])
    quarters = [dict(start_s=i*50, end_s=(i+1)*50, **summarize(
        [s for s in timeline if i*50 <= epoch(s['utc'])-t0 < (i+1)*50])) for i in range(4)]
    result = dict(validation='PASS', selected_non_holdout_tasks=len(selected),
        comparisons_of_same_51_tasks=paired, new_segments=segment_results,
        isolated=dict(duration_s=epoch(timeline[-1]['utc'])-t0, **summarize(timeline),
            first_cap={k:first_cap[k] for k in ('utc','power.draw','temperature.gpu','clocks.sm')},
            first_clock_drop={k:first_down[k] for k in ('utc','power.draw','temperature.gpu','clocks.sm')},
            first_task_id=first['task']['id'], first_task_start_utc=first['timestamp'],
            first_task_timing_start_utc=first['timing']['blocks'][0]['host_envelope_start_utc'],
            first_cap_seconds_after_task_start=epoch(first_cap['utc'])-epoch(first['timestamp']),
            quarters=quarters, compute_process_snapshots_checked=len(processes)),
        comparison_details=all_results, holdout_latencies_disclosed=False,
        notes=['Sample counts are not duration fractions.',
               'Clock, power and reason fields can have different sensor update rates.',
               'Existing artifacts are opened read-only; no GPU calls or configuration changes.'])
    isolated_rows = [r for r in all_rows['v4'] if r['task']['id'] in isolated_ids]
    windows = []
    for row in isolated_rows:
        for phase, blocks in [('settling', row['timing']['setup']['settle_blocks']),
                              ('calibration', row['timing']['setup']['calibration']),
                              ('timing', row['timing']['blocks'])]:
            for block in blocks:
                windows.append((epoch(block['host_envelope_start_utc']),
                                epoch(block['host_envelope_end_utc']), phase, row['task']['id']))
    low_samples = []
    for sample in timeline:
        if sample['clocks.sm'] == 615:
            matches = [(phase, task_id) for lo, hi, phase, task_id in windows
                       if lo <= epoch(sample['utc']) <= hi]
            assert len(matches) == 1 and matches[0][0] == 'settling'
            low_samples.append(dict(utc=sample['utc'], matches=matches,
                **{k:sample[k] for k in ('temperature.gpu','power.draw')}))
    phase_check = dict(minimum_615_samples=low_samples,
        total_settling_event_seconds=sum(r['timing']['setup']['settle_event_ms'] for r in isolated_rows)/1000,
        total_timing_event_seconds=sum(b['elapsed_ms'] for r in isolated_rows for b in r['timing']['blocks'])/1000,
        graph_nodes=sorted({r['timing']['graph_nodes'] for r in isolated_rows if r['task']['mode']=='graph'}),
        timing_window_minimum_mhz=paired[-1]['sm_min'])
    assert len(low_samples) == 18 and phase_check['timing_window_minimum_mhz'] == 645
    (HERE/'phase_check.json').write_text(json.dumps(phase_check,indent=2)+'\n')
    (HERE/'diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    with (HERE/'isolated_timeline.csv').open('w') as f:
        keys = ['utc','clocks.sm','power.draw','temperature.gpu','utilization.gpu',CAP,'clocks_event_reasons.active']
        writer=csv.DictWriter(f, fieldnames=['elapsed_s']+keys)
        writer.writeheader()
        for s in timeline:
            writer.writerow(dict(elapsed_s=round(epoch(s['utc'])-t0,3), **{k:s[k] for k in keys}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
    xs = [epoch(s['utc'])-t0 for s in timeline]
    for ax, field, color, ylabel in zip(axes, ['clocks.sm','power.draw','temperature.gpu'],
            ['#2563eb','#dc2626','#d97706'], ['SM clock (MHz)','Reported power (W)','GPU temperature (C)']):
        ax.plot(xs,[s[field] for s in timeline],lw=1,color=color)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].axhline(720,color='#6b7280',ls='--',lw=1,label='Requested clock: 720 MHz')
    axes[0].legend(loc='lower left')
    axes[1].axhline(70,color='#6b7280',ls='--',lw=1,label='Power limit: 70 W')
    axes[1].legend(loc='lower right')
    for ax in axes:
        ax.axvline(epoch(first_cap['utc'])-t0,color='#7c3aed',ls=':',lw=1)
    axes[0].set_title('v4 isolated FP32 segment: 51 tasks, October 1, 2026',loc='left',fontweight='bold')
    axes[2].set_xlabel('Seconds from 16:29:52.216 KST | dotted line: first observed SW power cap')
    fig.savefig(HERE/'clock_power_temperature.png',dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparison_details','new_segments')},indent=2))


if __name__ == '__main__':
    main()
