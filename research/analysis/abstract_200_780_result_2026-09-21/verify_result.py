"""Reproduce the received partial run and diagnose sampled clock violations.

Run with the existing system Python, which provides matplotlib.
The ZIP and raw directory are read-only inputs. No GPU execution takes place.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
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
sys.path.insert(0, str(ROOT / 'abstract_sweep_780_v1'))
import run
import analyze_sweep

RAW = HERE / 'raw/abstract_200_780'
ZIP_SHA = 'f722419bea5b83b54894a4d0db8efdd9c680d63f6e2f1e25c14a7110c759ac20'
KST = dt.timezone(dt.timedelta(hours=9))


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def table(name, rows):
    if not rows:
        return
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def kst(utc):
    return dt.datetime.fromisoformat(utc).astimezone(KST).isoformat()


def verify_raw():
    wanted = run.read(HERE / 'raw_sha256.json')
    actual = {str(p.relative_to(HERE / 'raw')): run.sha(p)
              for p in (HERE / 'raw').rglob('*') if p.is_file()}
    assert actual == wanted, 'Preserved raw evidence changed'
    assert run.sha(HERE / 'abstract_200_780_20260921_083037_8b68d7.zip') == ZIP_SHA


def main():
    verify_raw()
    manifest = run.read(RAW / 'manifest.json')
    assert manifest['source_hashes'] == run.own_hashes()
    for folder, base, hashes in [
        ('source', ROOT / 'abstract_sweep_780_v1', manifest['source_hashes']),
        ('dependencies', ROOT, manifest['dependency_hashes']),
    ]:
        for name, wanted in hashes.items():
            assert run.sha(RAW / folder / name) == run.sha(base / name) == wanted
    loaded = run.load()
    assert run.make_plan('full', loaded)['tasks'] == manifest['tasks']
    derived = HERE / 'recomputed'
    if not derived.exists():
        shutil.copytree(RAW, derived)
    assert run.manifest_for(derived, 'full', loaded) == manifest
    result = analyze_sweep.analyze(derived, loaded)
    assert result == run.read(RAW / 'summary.json')
    names = ['raw_attempts.jsonl', 'metrics.csv', 'fresh_holdout_LOCKED.csv',
             'blocks.csv', 'blocks_fresh_holdout_LOCKED.csv', 'between_round.json',
             'review_queue.csv', 'review_fresh_holdout_LOCKED.csv', 'triplets.csv',
             'triplets_fresh_holdout_LOCKED.csv', 'shape_splits.csv', 'summary.json', 'report.md']
    names += [str(p.relative_to(RAW)) for p in (RAW / 'sessions').glob('*/segment_*/quality.json')]
    assert all(run.sha(derived / name) == run.sha(RAW / name) for name in names)

    entries = list(run.history(RAW))
    rows = [row for _, row in entries]
    assert len(rows) == 240 == len({r['task']['id'] for r in rows})
    assert [r['task'] for r in rows] == manifest['tasks'][:240]
    assert len(manifest['tasks']) == 3600
    assert all(r['correctness']['passed'] and r['correctness']['finite_all'] for r in rows)
    assert sum(len(r['timing']['blocks']) for r in rows) == 1680
    with (derived / 'metrics.csv').open() as f:
        metrics = {r['id']: r for r in csv.DictReader(f)}
    with (derived / 'triplets.csv').open() as f:
        triplets = list(csv.DictReader(f))
    assert len(triplets) == 80 and all(r['complete'] == 'True' for r in triplets)
    assert Counter(r['status'] for r in metrics.values()) == {'PASS': 237, 'REVIEW': 3}
    shapes = {tuple(r['task']['shape']) for r in rows}
    assert len(shapes) == 40 and {r['task']['round'] for r in rows} == {0}
    for r in rows:
        blocks = r['timing']['blocks']
        for b in blocks:
            factor = b['repeats'] * b['operations_per_call']
            assert math.isclose(b['latency_ms'], b['elapsed_ms'] / factor, rel_tol=1e-12)
        assert math.isclose(st.median(b['latency_ms'] for b in blocks),
                            r['timing']['median_ms'], rel_tol=1e-12)

    samples, segments = [], []
    for folder in sorted((RAW / 'sessions').glob('*/segment_*')):
        meta = run.read(folder / 'telemetry_metadata.json')
        found, cadence = loaded['telemetry'].read(folder / 'telemetry.csv',
                                                 meta['timezone_offset_seconds'], 50)
        samples.extend(found)
        quality = run.read(folder / 'quality.json')
        segments.append(dict(segment=str(folder.relative_to(RAW)), cadence=cadence,
                             flags=quality['flags'], ranges=quality['ranges']))
    samples.sort(key=lambda s: s['utc'])
    times = [dt.datetime.fromisoformat(s['utc']).timestamp() for s in samples]
    active_key = 'clocks_event_reasons.sw_power_cap'
    anomaly = lambda s: s['clocks.sm'] != 780 or s[active_key] is True
    bad_samples = [s for s in samples if anomaly(s)]
    assert bad_samples
    incident_blocks, reviewed = [], []
    for seq, r in entries:
        timing, task = r['timing'], r['task']
        phases = [('settle', timing['setup']['settle_blocks']),
                  ('calibration', timing['setup']['calibration']), ('timed', timing['blocks'])]
        mode_incidents = []
        for phase, blocks in phases:
            for i, b in enumerate(blocks):
                start, end = (dt.datetime.fromisoformat(b[k]).timestamp() for k in
                              ('host_envelope_start_utc', 'host_envelope_end_utc'))
                inside = samples[bisect_left(times, start):bisect_right(times, end)]
                bad = [s for s in inside if anomaly(s)]
                if bad:
                    record = dict(seq=seq, task_id=task['id'], shape=task['shape'],
                                  dtype=task['dtype'], mode=task['mode'], round=task['round'],
                                  phase=phase, block=i, start_kst=kst(b['host_envelope_start_utc']),
                                  end_kst=kst(b['host_envelope_end_utc']),
                                  samples=len(inside), low_clock_samples=sum(s['clocks.sm'] != 780 for s in inside),
                                  power_cap_samples=sum(s[active_key] is True for s in inside),
                                  core_min_mhz=min(s['clocks.sm'] for s in inside),
                                  power_max_w=max(s['power.draw'] for s in inside),
                                  temperature_max_c=max(s['temperature.gpu'] for s in inside),
                                  latency_us=b['latency_ms'] * 1000)
                    incident_blocks.append(record)
                    mode_incidents.append(record)
        if metrics[task['id']]['status'] != 'PASS':
            reviewed.append(dict(seq=seq, **task, median_us=timing['median_ms'] * 1000,
                                 stored_status=r['status'], analyzed_status=metrics[task['id']]['status'],
                                 flags=metrics[task['id']]['flags'],
                                 incident_timed_blocks=sum(b['phase'] == 'timed' for b in mode_incidents)))
    table('incident_blocks.csv', incident_blocks)
    table('reviewed_measurements.csv', reviewed)
    table('anomalous_samples.csv', [dict(utc=s['utc'], kst=kst(s['utc']),
        core_mhz=s['clocks.sm'], memory_mhz=s['clocks.mem'], power_w=s['power.draw'],
        power_limit_w=s['power.limit'], temperature_c=s['temperature.gpu'],
        sw_power_cap=s[active_key]) for s in bad_samples])
    missing = [t for t in manifest['tasks'] if t['id'] not in metrics]
    table('missing_tasks.csv', missing)
    selection = run.read(ROOT / 'abstract_sweep_780_v1/selection.json')
    planned_strata = Counter(s['stratum'] for s in selection['shapes'])
    recorded_strata = Counter(s['stratum'] for s in selection['shapes'] if tuple(s['shape']) in shapes)
    table('strata_coverage.csv', [dict(stratum=s, planned_shapes=n,
        recorded_shapes=recorded_strata[s], complete_three_round_shapes=0)
        for s, n in sorted(planned_strata.items())])
    sessions = {p.parent.name: run.read(p) for p in (RAW / 'sessions').glob('*/session.json')}
    resets = [run.read(p) for p in (RAW / 'launches').glob('*/restore.json')]
    assert len(resets) == 1 and resets[0]['commands_accepted']
    assert all(resets[0]['commands'][name]['exit_code'] == 0 for name in ('graphics', 'memory'))
    exit_codes = {str(p.relative_to(RAW)): int(p.read_text()) for p in RAW.rglob('*exit_code.txt')}
    choices, done = run.choices_and_done(RAW, manifest, loaded)
    assert len(done) == 240 and all(done[r['id']] == 'PASS' for r in reviewed)
    summary = dict(
        evidence_verification='PASS', collection_decision='INCOMPLETE_CLOCK_VIOLATION_HOLD',
        zip_sha256=ZIP_SHA, raw_files=len(run.read(HERE / 'raw_sha256.json')),
        own_sources_verified=len(manifest['source_hashes']), dependencies_verified=len(manifest['dependency_hashes']),
        identical_recomputed_outputs=len(names), gpu_execution_on_mac=False,
        original_summary=result, recorded_shapes=len(shapes), planned_shapes=200,
        recorded_rounds=[0], complete_three_round_conditions=0,
        between_round_variation=None,
        between_round_note='Original summary 0.0 comes from one value per condition; no between-round comparison is available.',
        correctness_scopes=dict(Counter(r['correctness']['scope'] for r in rows)),
        recorded_correctness_passed=len(rows), raw_status_counts=dict(Counter(r['status'] for r in rows)),
        telemetry_samples=len(samples), sampled_core_counts=dict(Counter(s['clocks.sm'] for s in samples)),
        active_reason_counts={key: sum(s[key] is True for s in samples) for key in loaded['telemetry'].REASONS},
        ranges={key: [min(s[key] for s in samples), max(s[key] for s in samples)] for key in
                ('clocks.sm', 'clocks.mem', 'power.limit', 'power.draw', 'temperature.gpu')},
        anomaly_first_kst=kst(bad_samples[0]['utc']), anomaly_last_kst=kst(bad_samples[-1]['utc']),
        anomaly_sample_span_seconds=(dt.datetime.fromisoformat(bad_samples[-1]['utc'])-
                                     dt.datetime.fromisoformat(bad_samples[0]['utc'])).total_seconds(),
        anomaly_note='First-to-last anomalous sample span, not continuous throttling duration.',
        incident_shapes=sorted({tuple(b['shape']) for b in incident_blocks}),
        incident_blocks_by_phase=dict(Counter(b['phase'] for b in incident_blocks)),
        reviewed_measurements=reviewed, segments=segments, sessions=sessions, exit_codes=exit_codes,
        restoration_commands_accepted=True,
        resume_note='Original DB stores PASS for the three clock-flagged rows. Existing resume skips them; --retry-failed does not target these rows. Do not blindly resume or edit raw status.',
        analysis_scope='Diagnosis only. No 200-shape aggregate, independent-round estimate, or new ceiling is claimed.')
    save('validation.json', summary)
    plot(samples, rows, metrics, bad_samples)
    verify_raw()
    print(json.dumps({k: summary[k] for k in ('evidence_verification', 'collection_decision',
        'telemetry_samples', 'sampled_core_counts', 'active_reason_counts', 'anomaly_first_kst',
        'anomaly_last_kst', 'incident_blocks_by_phase', 'reviewed_measurements', 'correctness_scopes')},
        ensure_ascii=False, indent=2))


def plot(samples, rows, metrics, bad):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180, 'pdf.fonttype': 42})
    times = [dt.datetime.fromisoformat(s['utc']).astimezone(KST) for s in samples]
    start = dt.datetime.fromisoformat(bad[0]['utc']).astimezone(KST)
    end = dt.datetime.fromisoformat(bad[-1]['utc']).astimezone(KST)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout='constrained')
    fields = [('clocks.sm', 'Core clock (MHz)', 780), ('power.draw', 'Power draw (W)', 70),
              ('temperature.gpu', 'GPU temperature (C)', None)]
    bad_times = [dt.datetime.fromisoformat(s['utc']).astimezone(KST) for s in bad]
    for j in range(2):
        for i, (field, label, ref) in enumerate(fields):
            ax = axes[i, j]
            ax.plot(times, [s[field] for s in samples], color='#246a92', lw=.9)
            ax.scatter(bad_times, [s[field] for s in bad], color='#cb4335', s=8, zorder=3)
            if ref is not None:
                ax.axhline(ref, color='#666666', ls='--', lw=.8)
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S', tz=KST))
            if j == 1:
                ax.set_xlim(start-dt.timedelta(seconds=3), end+dt.timedelta(seconds=3))
        axes[2, j].set_xlabel('2026-09-21, KST (UTC+9)')
    axes[0, 0].set_title('Entire partial run (including warm-up)')
    axes[0, 1].set_title('Incident: (3128, 680, 488), FP32')
    fig.suptitle('Stopped at 240 / 3600 records | sampled core 780 -> 765 MHz\n'
                 'Red: power-cap active or clock mismatch; 50 ms requested polling', fontsize=13)
    for ext in ('png', 'pdf'):
        fig.savefig(HERE / ('clock_incident.' + ext))
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, field, label in zip(axes, ('relative_iqr', 'first_last_drift'),
                                ('Relative IQR (%)', 'Absolute first-to-last drift (%)')):
        values = [100*abs(r['timing'][field]) for r in rows]
        ax.scatter(range(1, len(rows)+1), values, s=10, color='#246a92', label='Recorded timing')
        selected = [(i+1, v) for i, (r, v) in enumerate(zip(rows, values))
                    if metrics[r['task']['id']]['status'] != 'PASS']
        ax.scatter(*zip(*selected), s=30, facecolors='none', edgecolors='#cb4335',
                   label='Clock-flagged record', zorder=3)
        ax.set(xlabel='Record sequence', ylabel=label, ylim=(0, max(values)*1.25))
        ax.grid(alpha=.2)
        ax.text(.03, .94, 'Review threshold: 5% (outside zoomed y-range)', transform=ax.transAxes,
                va='top', fontsize=8)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1, .86), fontsize=8)
    fig.suptitle('Low within-record variation does not establish clock stability\n'
                 '40 / 200 shapes; first round only; no between-round estimate', fontsize=12)
    for ext in ('png', 'pdf'):
        fig.savefig(HERE / ('timing_quality_partial.' + ext))
    plt.close(fig)


if __name__ == '__main__':
    main()
