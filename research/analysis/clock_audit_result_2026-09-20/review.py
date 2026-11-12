"""Reproduce the clock-audit review from immutable uploaded artifacts.

Run with the installed Python 3 and matplotlib. Existing clock_audit_v1/analyze.py
performs source, environment, timing, trace and preparation validation first.
This script summarizes its verified results without changing benchmark data.
"""
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'raw/clock_audit_20260920_145534_0K3jyQ'
sys.path.insert(0, str(HERE.parents[1] / 'clock_audit_v1'))
import telemetry


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def median_or_none(values):
    return st.median(values) if values else None


def main():
    integrity = read(HERE / 'raw_integrity.json')
    actual = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in ROOT.rglob('*') if p.is_file()}
    assert actual == integrity['files'], 'Raw files changed'
    for name in ['analysis.json', 'clock_windows.json', 'conditions.csv', 'report.md', 'summary.json']:
        assert (ROOT / name).read_bytes() == (HERE / 'recomputed' / name).read_bytes()
    analysis = read(ROOT / 'analysis.json')
    windows = read(ROOT / 'clock_windows.json')
    timed = [w for w in windows if w['phase'] == 'timed']
    assert len(timed) == 280
    record_rows = []
    for pass_name in analysis['telemetry']:
        for record in read(ROOT / pass_name / 'results.json'):
            task, result = record['task'], record['result']
            timing = result['timing']
            matched = [w for w in timed if w['pass_name'] == pass_name and w['task']['id'] == task['id']]
            assert len(matched) == 7
            samples = [sample for w in matched for sample in w['clock_window']['inside']]
            sample_span = ((dt.datetime.fromisoformat(samples[-1]['utc']) - dt.datetime.fromisoformat(samples[0]['utc'])).total_seconds()
                           if len(samples) >= 2 else None)
            reported_delta = ((samples[-1][telemetry.COUNTERS[0]]-samples[0][telemetry.COUNTERS[0]])/1e6
                              if len(samples) >= 2 else None)
            record_rows.append(dict(
                pass_name=pass_name, task_id=task['id'], shape=str(task['shape']),
                dtype=task['dtype'], mode=task['mode'], round=task['round'],
                median_us=timing['median_ms']*1000, iqr_pct=timing['relative_iqr']*100,
                first_last_pct=timing['first_last_drift']*100,
                block_us=[b['latency_ms']*1000 for b in timing['blocks']],
                flags=timing['flags'], inside_samples=len(samples),
                sm_mhz=median_or_none([s['clocks.sm'] for s in samples]),
                sm_min_mhz=min((s['clocks.sm'] for s in samples), default=None),
                sm_max_mhz=max((s['clocks.sm'] for s in samples), default=None),
                power_w=median_or_none([s['power.draw'] for s in samples]),
                temperature_c=median_or_none([s['temperature.gpu'] for s in samples]),
                power_cap_samples=sum(s[telemetry.REASONS[0]] for s in samples),
                inside_sample_span_s=sample_span,
                reported_power_counter_delta_s=reported_delta,
                reported_counter_exceeds_sample_span=(reported_delta > sample_span if sample_span else None),
                correctness_scope=result['correctness']['scope'],
                correctness_points=result['correctness']['points']))
    assert len(record_rows) == 40
    write_csv('measurements.csv', record_rows)
    trace_rows = []
    for label, block in analysis['trace_blocks'].items():
        gap = block['activity_gap_breakdown']
        assert gap['supported']
        assert sum(gap['phase_ns'].values()) == gap['total_gap_ns']
        assert block['kernel_count'] == block['expected_gemm_calls']
        trace_rows.append(dict(
            label=label, task_id=label.split('/')[1], gemm_calls=block['expected_gemm_calls'],
            event_us=block['event_ms']*1000/block['expected_gemm_calls'],
            kernel_us=block['kernel_union_us_per_gemm'],
            gap_us=gap['gap_us_per_gemm'], gap_pct=gap['gap_fraction_of_activity_span']*100,
            before_api_us=gap['phase_us_per_gemm']['before_api'],
            during_api_us=gap['phase_us_per_gemm']['during_api'],
            after_api_us=gap['phase_us_per_gemm']['after_api'],
            other_kernel_count=block['unassociated_kernel_count_in_span'],
            transfers_in_span=block['recorded_transfer_counts_in_kernel_span'],
            kernels=block['kernel_name_counts_per_gemm']))
    assert len(trace_rows) == 56
    write_csv('trace_blocks.csv', trace_rows)
    trace_summary = []
    for task_id in dict.fromkeys(r['task_id'] for r in trace_rows):
        rows = [r for r in trace_rows if r['task_id'] == task_id]
        trace_summary.append(dict(task_id=task_id, **{
            key: st.median(r[key] for r in rows)
            for key in ['event_us', 'kernel_us', 'gap_us', 'gap_pct', 'before_api_us', 'during_api_us']
        }))
    write_csv('trace_summary.csv', trace_summary)
    counter_rows = []
    for pass_name in analysis['telemetry']:
        meta = read(ROOT / pass_name / 'telemetry_metadata.json')
        samples, info = telemetry.read(ROOT / pass_name / 'telemetry.csv',
                                       meta['timezone_offset_seconds'], meta['requested_interval_ms'])
        span = (dt.datetime.fromisoformat(samples[-1]['utc']) - dt.datetime.fromisoformat(samples[0]['utc'])).total_seconds()
        counter_rows.append(dict(pass_name=pass_name, sample_span_s=span,
            sw_power_cap_s=(samples[-1][telemetry.COUNTERS[0]]-samples[0][telemetry.COUNTERS[0]])/1e6,
            sw_thermal_s=(samples[-1][telemetry.COUNTERS[1]]-samples[0][telemetry.COUNTERS[1]])/1e6,
            hw_thermal_s=(samples[-1][telemetry.COUNTERS[2]]-samples[0][telemetry.COUNTERS[2]])/1e6,
            thermal_active_samples=sum(any(s[k] for k in telemetry.REASONS[1:]) for s in samples),
            power_limits_w=sorted(set(s['power.limit'] for s in samples)),
            memory_clocks_mhz=sorted(set(s['clocks.mem'] for s in samples))))
    write_csv('process_counters.csv', counter_rows)
    coverage = {p:dict(Counter(str(w['clock_window']['samples_inside']) for w in timed if w['pass_name']==p))
                for p in analysis['telemetry']}
    warnings = [w for meta in analysis['trace_metadata'].values() for w in meta['warnings']]
    main_pids = {b['nvtx_global_tid'] & ~0xffffff for b in analysis['trace_blocks'].values()}
    dense_samples = [s for w in timed if w['pass_name'] in ['dense_before','nsys','dense_after']
                     for s in w['clock_window']['inside']]
    metrics = dict(
        raw_files_unchanged=len(actual), measurements=record_rows, trace_summary=trace_summary,
        counters=counter_rows, timed_block_sample_coverage=coverage,
        blocks_with_two_inside_samples=sum(w['clock_window']['samples_inside']>=2 for w in timed),
        dense_samples=len(dense_samples), dense_power_cap_samples=sum(s[telemetry.REASONS[0]] for s in dense_samples),
        dense_temperature_range=[min(s['temperature.gpu'] for s in dense_samples),max(s['temperature.gpu'] for s in dense_samples)],
        kernel_count=sum(b['gemm_calls'] for b in trace_rows),
        warning_count=len(warnings), main_process_warning_count=sum(w['globalPid'] in main_pids for w in warnings),
        warning_text_counts=dict(Counter(w['text'] for w in warnings)),
        measurement_counter_deltas_exceeding_sample_span=sum(r['reported_counter_exceeds_sample_span'] is True for r in record_rows),
        counter_observation_note='Readings update in steps; short-window counter delta is not a precise duration or fraction within that window.',
        interpretation='Power-limited clock variation strongly supported; not an isolated causal intervention.',
        full_sweep='HOLD')
    (HERE/'review_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n')
    plot(record_rows)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['measurements','trace_summary','counters']},indent=2))


def plot(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {'dense_before':'#2367ba', 'nsys':'#cf602a', 'dense_after':'#298866'}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7), constrained_layout=True)
    for ax, (case, mode) in zip(axes.flat, [('c0','graph'),('c1','graph'),('c0','prealloc'),('c1','prealloc')]):
        selected=[r for r in records if r['task_id'].startswith(case) and r['mode']==mode and r['pass_name'] in colors]
        for row in selected:
            x, y = row['sm_mhz'], row['median_us']
            ax.errorbar(x,y,xerr=[[x-row['sm_min_mhz']],[row['sm_max_mhz']-x]],
                        fmt='^' if row['pass_name']=='nsys' else 'o', color=colors[row['pass_name']], capsize=3,markersize=7)
        shape='1888 × 832 × 632 · FP16' if case=='c0' else '392 × 2776 × 80 · FP32'
        ax.set_title(shape+' · '+mode,fontsize=11)
        ax.set_xlabel('SM clock: median of inside samples (MHz)')
        ax.set_ylabel('GEMM time: median of 7 blocks (µs)')
        ax.set_xlim(980,1440)
        ax.set_ylim((54,69) if case=='c0' else (30,40))
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Clock audit: lower observed clocks accompany longer GEMM times',fontsize=14)
    legend=[Line2D([0],[0],color=c,marker='^' if p=='nsys' else 'o',linestyle='',label=p.replace('_',' ')) for p,c in colors.items()]
    fig.legend(handles=legend,loc='outside lower center',ncol=3,frameon=False,
               title='Each point = one measurement; horizontal bars = sampled clock range. No causal fit.')
    fig.savefig(HERE/'clock_latency.png',dpi=180)
    fig.savefig(HERE/'clock_latency.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
