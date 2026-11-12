"""Run after the unchanged v2 analyzer; report quality without holdout latencies."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

BASE = Path(__file__).resolve().parent
OUT = BASE / 'recomputed/full_720_remeasure3_20261001'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    archive = read(BASE / 'archive_validation.json')
    assert sha(Path(archive['archive'])) == archive['sha256'], 'Preserved archive changed'
    summary = read(OUT / 'summary.json')
    assert read(BASE / 'recompute.log') == summary, 'Analyzer did not finish successfully'
    assert summary == read(BASE / 'original_summary.json'), 'Recomputed summary differs'
    exports = {name: sha(OUT / name) == expected
               for name, expected in archive['original_export_sha256'].items()}
    assert all(exports.values()), exports
    assert sha(OUT / 'raw_attempts.jsonl') == summary['results_digest']
    manifest = read(OUT / 'manifest.json')
    expected = {task['id']: task for task in manifest['tasks']}
    assert len(expected) == len(manifest['tasks']) == summary['expected']
    with sqlite3.connect((OUT / 'results.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        assert db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
        attempts, distinct = db.execute('SELECT COUNT(*), COUNT(DISTINCT task_id) FROM attempts').fetchone()
        assert attempts == distinct == summary['recorded']
        assert {row[0] for row in db.execute('SELECT task_id FROM attempts')} == set(expected)
    sessions, raw_statuses, scopes = Counter(), Counter(), Counter()
    receipt = read(OUT / 'recovery_provenance.json')
    redo = set(read(OUT / 'recovery/remeasure_ids.json'))
    previous = BASE.parents[1] / 'analysis/remeasure_720_v3_result_2026-10-01/recomputed/full_720_remeasure2_20260930'
    assert redo == {r['task']['id'] for r in read(previous/'remeasure_tasks.json')}
    schedule = read(OUT / 'recovery/collection_schedule.json')
    new_ids, new_sessions, new_segments = [], set(), defaultdict(list)
    round_values = defaultdict(list)
    key = lambda t: json.dumps({k:v for k,v in t.items() if k not in ('id','round')}, sort_keys=True)
    raw_count = 0
    with (OUT / 'raw_attempts.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            assert expected[row['task']['id']] == row['task']
            assert row['correctness']['passed'] is True
            assert row['correctness']['finite_all'] is True
            assert len(row['timing']['blocks']) == manifest['protocol']['blocks']
            sessions[row['session']] += 1
            raw_statuses[row['status']] += 1
            scopes[row['correctness']['scope']] += 1
            round_values[key(row['task'])].append(row['timing']['median_ms'])
            if row['seq'] > receipt['imported_last_seq']:
                new_ids.append(row['task']['id'])
                new_sessions.add(row['session'])
                new_segments[row['segment']].append(row['task']['id'])
            raw_count += 1
    assert raw_count == attempts
    assert len(new_ids) == len(set(new_ids)) == 1800 and set(new_ids) == redo
    assert len(new_sessions) == 1 and len(new_segments) == 16
    assert new_ids == [tid for seg in schedule['segments'] for tid in seg['task_ids']]
    assert all(len(v)==3 for v in round_values.values())
    between_keys = {k for k,values in round_values.items()
                    if (max(values)-min(values))/statistics.median(values) > manifest['protocol']['between_round_limit']}
    assert len(between_keys) == summary['between_round_flags'] == 3
    assert between_keys == {key(r['task']) for r in read(OUT/'between_round.json')}
    between_ids = {tid for tid,task in expected.items() if key(task) in between_keys}
    assert len(between_ids) == 9
    review = read(OUT / 'remeasure_tasks.json')
    review_ids = {item['task']['id'] for item in review}
    assert len(review_ids) == len(review) == summary['review_queue_tasks']
    assert all(set(item) == {'task', 'flags', 'between_round_review'} for item in review)
    statuses, split_statuses, seen, flagged = Counter(), Counter(), set(), set()
    for filename in ('metrics.csv', 'fresh_holdout_LOCKED.csv'):
        with (OUT / filename).open(newline='') as stream:
            for row in csv.DictReader(stream):
                assert row['id'] not in seen
                seen.add(row['id'])
                statuses[row['status']] += 1
                split_statuses[(row['split'], row['status'])] += 1
                if row['status'] != 'PASS':
                    flagged.add(row['id'])
    assert seen == set(expected)
    assert dict(statuses) == summary['counts']
    assert review_ids == flagged | between_ids
    assert len(flagged) == 51 and len(between_ids-flagged) == 3
    assert all(item['between_round_review'] == (item['task']['id'] in between_ids) for item in review)
    assert len(review_ids) == 54 and summary['accepted_measurements'] == len(expected)-len(review_ids) == 113328
    quality_files = sorted((OUT/'sessions').glob('*/segment_*/quality.json'))
    assert len(quality_files) == 947
    all_statuses, imported_statuses, fresh_statuses, group_statuses = Counter(), Counter(), Counter(), Counter()
    direct_flags, seen, segments, ranges, active_samples = Counter(), set(), [], {}, Counter()
    direct_ids, new_samples, max_gap, counter_anomalies = set(), 0, 0, 0
    for path in quality_files:
        quality = read(path)
        assert quality['collection_can_continue']
        assert quality['telemetry_sha256'] == sha(path.parent/'telemetry.csv')
        fresh = path.parent.parent.name in new_sessions
        if fresh:
            group = read(path.parent/'collection_group.json')
            relative = str(path.parent.relative_to(OUT))
            assert group['task_ids'] == new_segments[relative]
            assert set(group['task_ids']) == set(quality['measurements'])
            segments.append(dict(segment=relative, group=group['group'], tasks=len(group['task_ids']), flags=quality['flags']))
            new_samples += quality['cadence']['samples']
            max_gap = max(max_gap, quality['cadence']['observed_interval_seconds_max'])
            counter_anomalies += quality['counter_anomaly_blocks']
            for name,bounds in quality['ranges'].items():
                old=ranges.get(name,bounds);ranges[name]=[min(old[0],bounds[0]),max(old[1],bounds[1])]
            for name,value in quality['counters'].items():active_samples[name]+=value['active_samples']
        for tid,value in quality['measurements'].items():
            assert tid not in seen
            seen.add(tid);all_statuses[value['status']]+=1
            (fresh_statuses if fresh else imported_statuses)[value['status']]+=1
            if fresh:
                group_statuses[(group['group'],value['status'])]+=1
                local=[f for f in value['flags'] if not f.startswith('segment:')]
                if local:direct_ids.add(tid);direct_flags.update(local)
    assert seen==set(expected) and all_statuses==statuses
    assert imported_statuses=={'PASS':111582} and fresh_statuses=={'PASS':1749,'REVIEW':51}
    assert group_statuses=={('other','PASS'):1749,('isolated_fp32','REVIEW'):51}
    assert len(direct_ids)==51 and direct_flags=={'observed_active:clocks_event_reasons.sw_power_cap':51,'observed_mismatch:clocks.sm':51}
    assert [read(OUT/s['segment']/'collection_group.json') for s in segments]==schedule['segments']
    assert review_ids-redo == between_ids-redo and len(review_ids-redo)==3
    assert all(expected[tid]['split']!='fresh_holdout' for tid in review_ids)
    breakdown=dict(quality_status_counts=dict(statuses), segments=len(quality_files),
        imported_status_counts=dict(imported_statuses), remeasured_status_counts=dict(fresh_statuses),
        group_status_counts=[dict(group=g,status=s,count=n) for (g,s),n in group_statuses.items()],
        new_segments=segments,new_ranges=ranges,new_telemetry_samples=new_samples,
        new_max_sample_gap_seconds=max_gap,new_counter_active_samples=dict(active_samples),
        new_counter_anomaly_blocks=counter_anomalies,direct_tasks=len(direct_ids),direct_flags=dict(direct_flags),
        between_round_groups=len(between_keys),between_round_tasks=len(between_ids),
        additional_between_round_tasks=len(between_ids-flagged),resolved_previous_review=len(redo-review_ids),
        newly_reviewed_imported_tasks=[expected[tid] for tid in sorted(review_ids-redo)],
        final_review_tasks=len(review_ids),accepted_measurements=summary['accepted_measurements'],
        fresh_holdout_review=0,holdout_latencies_disclosed=False)
    (BASE/'quality_breakdown.json').write_text(json.dumps(breakdown,indent=2)+'\n')
    result = dict(validation='PASS', summary_exact_match=True, export_byte_matches=exports,
                  sqlite_quick_check='ok', recorded=attempts, duplicate_task_ids=attempts-distinct,
                  missing_task_ids=0, correctness_passed=raw_count, correctness_scopes=dict(scopes),
                  raw_status_counts=dict(raw_statuses), sessions=dict(sessions),
                  final_status_counts=dict(statuses),
                  split_status_counts=[dict(split=split, status=status, count=count)
                                       for (split, status), count in sorted(split_statuses.items())],
                  accepted_measurements=summary['accepted_measurements'],
                  between_round_groups=len(between_keys), between_round_tasks=len(between_ids),
                  additional_between_round_tasks=len(between_ids-flagged),
                  remeasure_tasks=len(review), remeasure_list_has_no_latency=True,
                  telemetry_hashes_checked=len(quality_files), holdout_latencies_disclosed=False,
                  experiment_gate=summary['gate'], archive_sha256=archive['sha256'])
    (BASE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
