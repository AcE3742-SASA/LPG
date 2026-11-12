"""Verify the completed recovery on a copy using the frozen measurement analyzer."""
from collections import Counter, defaultdict
import csv
import datetime as dt
import json
import math
from pathlib import Path
import statistics as st
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'recover_720_v1'))
import prepare as recovery
run, analyze_sweep = recovery.run, recovery.analyze_sweep

RAW = HERE / 'raw/abstract_200_720_recovery_20260921'
DERIVED = HERE / 'recomputed'
ZIP = HERE / 'abstract_200_720_recovery_20260921_20260922_024735_e61123.zip'
ZIP_SHA = '71e19cf727a8ec7fd0caddbac9a21fd3c1d7ae23583c735d772071293c2488da'
NEW_SESSION = '20260922_003954_7a63c79b'
KST = dt.timezone(dt.timedelta(hours=9))


def verify_raw():
    assert run.sha(ZIP) == ZIP_SHA
    assert {str(p.relative_to(HERE / 'raw')): run.sha(p)
            for p in (HERE / 'raw').rglob('*') if p.is_file()} == run.read(HERE / 'raw_sha256.json')


def table(name, rows):
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    verify_raw()
    loaded = run.load(720)
    manifest = run.manifest_for(DERIVED, 'full', loaded)
    analyze_sweep.analyze(DERIVED, loaded)
    summary = run.read(DERIVED / 'summary.json')
    assert summary == run.read(RAW / 'summary.json')
    exports = ['raw_attempts.jsonl', 'metrics.csv', 'fresh_holdout_LOCKED.csv', 'blocks.csv',
               'blocks_fresh_holdout_LOCKED.csv', 'between_round.json', 'review_queue.csv',
               'review_fresh_holdout_LOCKED.csv', 'triplets.csv', 'triplets_fresh_holdout_LOCKED.csv',
               'shape_splits.csv', 'summary.json', 'report.md']
    exports += [str(p.relative_to(RAW)) for p in (RAW / 'sessions').glob('*/segment_*/quality.json')]
    assert all(run.sha(DERIVED / name) == run.sha(RAW / name) for name in exports)
    assert summary['gate'] == 'REVIEW' and summary['counts'] == {'PASS': 3600}
    assert summary['recorded'] == summary['attempts'] == 3600 and summary['missing'] == 0
    assert summary['timing_and_sampled_clocks_pass'] and summary['all_restorations_accepted']
    assert summary['round_groups_complete'] == summary['round_groups_expected'] == 1200

    entries = list(run.history(DERIVED))
    rows = [row for _, row in entries]
    assert [seq for seq, _ in entries] == list(range(1, 3601))
    assert [row['task'] for row in rows] == manifest['tasks']
    assert len({row['task']['id'] for row in rows}) == 3600
    assert len({tuple(row['task']['shape']) for row in rows}) == 200
    assert Counter(row['task']['round'] for row in rows) == {0: 1200, 1: 1200, 2: 1200}
    assert Counter(row['session'] for row in rows) == {recovery.SESSION: 1680, NEW_SESSION: 1920}
    with tempfile.TemporaryDirectory(prefix='verify-recovery-source-') as temporary:
        source = recovery.extract_reviewed(RAW / 'recovery/source_run.zip', Path(temporary))
        original = list(run.history(source))
        assert recovery.verify_import(DERIVED, source, original, loaded) == 0
        assert recovery.records_digest(original) == recovery.ORIGINAL_DIGEST
        assert entries[:1680] == original[:1680]
        assert [r['task'] for _, r in entries[1680:1800]] == [r['task'] for _, r in original[1680:]]
        for (_, fresh), (_, old) in zip(entries[1680:1800], original[1680:]):
            assert fresh['input_seed'] == old['input_seed']
            assert fresh['timing'].get('graph_nodes') == old['timing'].get('graph_nodes')
        assert not (DERIVED / f'sessions/{recovery.SESSION}/segment_00014').exists()
    receipt = run.read(RAW / 'recovery_provenance.json')
    assert receipt['imported_results_digest'] == recovery.records_digest(entries[:1680])
    assert receipt['review_note'] == recovery.REVIEW_NOTE
    release = run.read(ROOT / 'analysis/recover_720_v1_2026-09-21/validation.json')
    assert all(run.sha(RAW / 'recovery' / name) == wanted for name, wanted in release['files'].items())

    groups = defaultdict(list)
    for row in rows:
        task, timing = row['task'], row['timing']
        assert row['status'] == 'PASS' and row['correctness']['passed'] and row['correctness']['finite_all']
        assert row['input_seed'] == loaded['plan'].task_seed(task)
        assert len(timing['blocks']) == 7
        for block in timing['blocks']:
            assert math.isclose(block['latency_ms'], block['elapsed_ms'] /
                                (block['repeats'] * block['operations_per_call']), rel_tol=1e-12)
        calculated = loaded['measure'].summarize(timing['blocks'], manifest['protocol'])
        for key in ('median_ms', 'relative_iqr', 'first_last_drift'):
            assert math.isclose(timing[key], calculated[key], rel_tol=1e-12, abs_tol=1e-15)
        assert not calculated['flags']
        m, n, k = task['shape']
        assert row['flops'] == 2*m*n*k
        assert math.isclose(row['tflops'], row['flops'] / (timing['median_ms'] * 1e9), rel_tol=1e-12)
        groups[(m, n, k, task['dtype'], task['mode'])].append(row)
    aggregate = []
    for key, members in groups.items():
        assert [r['task']['round'] for r in members] == [0, 1, 2]
        values = [r['timing']['median_ms'] for r in members]
        median = st.median(values)
        aggregate.append(dict(zip(('M', 'N', 'K', 'dtype', 'mode'), key),
            round_0_ms=values[0], round_1_ms=values[1], round_2_ms=values[2], median_ms=median,
            round_min_ms=min(values), round_max_ms=max(values),
            relative_range=(max(values)-min(values))/median,
            flops=members[0]['flops'], ideal_bytes=members[0]['ideal_bytes'],
            ideal_oi=members[0]['ideal_oi'], tflops=members[0]['flops']/(median*1e9),
            sessions=';'.join(r['session'] for r in members)))
    assert len(aggregate) == 1200
    aggregate.sort(key=lambda r: r['relative_range'], reverse=True)
    assert math.isclose(aggregate[0]['relative_range'], summary['max_between_round_range'], rel_tol=1e-12)
    assert aggregate[0]['relative_range'] <= manifest['protocol']['between_round_limit']
    table('round_aggregates.csv', aggregate)
    table('largest_round_variation.csv', aggregate[:20])

    sensors = defaultdict(lambda: dict(samples=0, core=Counter(), memory=Counter(), active=Counter(),
        power_min=float('inf'), power_max=0., temp_min=float('inf'), temp_max=0., max_gap_s=0.))
    segments = []
    for directory in sorted((RAW / 'sessions').glob('*/segment_*')):
        metadata = run.read(directory / 'telemetry_metadata.json')
        samples, cadence = loaded['telemetry'].read(directory / 'telemetry.csv', metadata['timezone_offset_seconds'], 50)
        quality = run.read(directory / 'quality.json')
        assert not quality['flags'] and quality['collection_can_continue']
        assert all(s['clocks.sm'] == 720 and s['clocks.mem'] == 6801 and s['power.limit'] == 70 for s in samples)
        assert all(s[key] is False for s in samples for key in loaded['telemetry'].REASONS)
        for name in ('combined', 'imported' if directory.parent.name == recovery.SESSION else 'new'):
            value = sensors[name]
            value['samples'] += len(samples)
            value['core'].update(s['clocks.sm'] for s in samples)
            value['memory'].update(s['clocks.mem'] for s in samples)
            value['active'].update({key: sum(s[key] is True for s in samples) for key in loaded['telemetry'].REASONS})
            value['power_min'] = min(value['power_min'], min(s['power.draw'] for s in samples))
            value['power_max'] = max(value['power_max'], max(s['power.draw'] for s in samples))
            value['temp_min'] = min(value['temp_min'], min(s['temperature.gpu'] for s in samples))
            value['temp_max'] = max(value['temp_max'], max(s['temperature.gpu'] for s in samples))
            value['max_gap_s'] = max(value['max_gap_s'], cadence['observed_interval_seconds_max'])
        segments.append(dict(segment=str(directory.relative_to(RAW)), samples=len(samples), cadence=cadence,
                             flags=quality['flags'], counter_anomaly_blocks=quality['counter_anomaly_blocks']))
    assert len(segments) == 30
    assert sum(s['counter_anomaly_blocks'] for s in segments) == summary['counter_anomaly_blocks']

    new_launch = RAW / 'launches/20260922_003950_acb523'
    assert run.read(new_launch / 'status.json') == {'collection': 'COMPLETE', 'restoration': 'COMMANDS_ACCEPTED'}
    assert (new_launch / 'worker_0.exit_code.txt').read_text().strip() == '0'
    restored = run.read(new_launch / 'restore.json')
    assert restored['commands_accepted'] and all(restored['commands'][key]['exit_code'] == 0 for key in ('graphics', 'memory'))
    admission = run.read(new_launch / 'check_review.json')
    assert admission['manifest_sha256'] == recovery.CHECK_MANIFEST_SHA
    assert admission['results_digest'] == recovery.CHECK_DIGEST and admission['review_note'] == recovery.REVIEW_NOTE
    launcher = RAW / 'launcher/20260922_003928_o7zB9t'
    assert (launcher / 'exit_code.txt').read_text().strip() == '2'
    assert (launcher / 'tee_exit_code.txt').read_text().strip() == '0'
    session = run.read(RAW / f'sessions/{NEW_SESSION}/session.json')
    assert session['state'] == 'FINISHED' and session['new_records'] == 1920
    progress = run.read(RAW / 'progress.json')
    assert progress['state'] == 'FINISHED' and progress['recorded'] == progress['expected'] == 3600
    spread = [r['relative_range'] for r in aggregate]
    quality_distributions = {name: dict(median=st.median(values), p95=loaded['measure'].quantile(values, .95), max=max(values))
        for name, values in [('relative_iqr', [r['timing']['relative_iqr'] for r in rows]),
                             ('abs_drift', [abs(r['timing']['first_last_drift']) for r in rows]),
                             ('between_round_range', spread)]}
    result = dict(verification='PASS', decision='COLLECTION_COMPLETE_TIMING_AND_SAMPLED_CLOCKS_PASS_COUNTER_UNRESOLVED',
        zip_sha256=ZIP_SHA, raw_files=len(run.read(HERE / 'raw_sha256.json')),
        source_files_verified=len(manifest['source_hashes']), dependencies_verified=len(manifest['dependency_hashes']),
        recovery_source_files_verified=len(release['files']), reproduced_identical_outputs=len(exports),
        summary=summary, correctness_scopes=dict(Counter(r['correctness']['scope'] for r in rows)),
        recovery=dict(imported_records=1680, remeasured_flagged_segment=120, originally_missing_collected=1800,
                      new_records=1920, original_archive_preserved=True, original_prefix_identical=True),
        sensors=dict(sensors), quality_distributions=quality_distributions, largest_round_variation=aggregate[:5],
        session=session, end_kst=dt.datetime.fromisoformat(session['ended_utc']).astimezone(KST).isoformat(),
        segments=segments, new_gpu_execution=False,
        limitations=['Only the selected 200 shapes and tested configuration are verified; the 10000-condition remeasurement is not complete.',
                     'Rounds are repeated observations, not 3600 independent shapes; the recovered collection spans two workers.',
                     'Sampled clock quality does not establish continuous or effective kernel clock or explain cumulative counters.',
                     'Model accuracy, layer assembly error and 720 MHz Roofline ceilings are not measured by this run.'])
    (HERE / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    verify_raw()
    print(json.dumps({key: result[key] for key in ('verification', 'raw_files', 'reproduced_identical_outputs',
        'correctness_scopes', 'recovery', 'sensors', 'quality_distributions', 'largest_round_variation', 'end_kst')}, indent=2))


if __name__ == '__main__':
    main()
