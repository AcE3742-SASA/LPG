"""Reuse the full-sweep verification and summarize remeasurement quality only."""
from collections import Counter
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / 'recomputed/full_720_remeasure_20260930'


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(OUT / 'manifest.json')
    expected = {task['id']: task for task in manifest['tasks']}
    receipt = read(OUT / 'recovery_provenance.json')
    redo = set(read(OUT / 'recovery/remeasure_ids.json'))
    original_review = read(ROOT / 'analysis/full_720_result_2026-09-30/recomputed/full_720/remeasure_tasks.json')
    assert redo == {item['task']['id'] for item in original_review}
    new_ids, new_sessions = set(), set()
    with (OUT / 'raw_attempts.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['seq'] > receipt['source_last_seq']:
                assert row['task']['id'] not in new_ids
                new_ids.add(row['task']['id'])
                new_sessions.add(row['session'])
    assert new_ids == redo and len(new_ids) == 5160
    all_statuses, new_statuses, old_statuses = Counter(), Counter(), Counter()
    flags, new_flags, ranges = Counter(), Counter(), {}
    flagged_segments, segment_counts = [], Counter()
    samples = Counter()
    split_status = Counter()
    active_samples = Counter()
    seen = set()
    new_counter_anomalies = 0
    max_gap = 0
    for path in sorted((OUT / 'sessions').glob('*/segment_*/quality.json')):
        quality = read(path)
        fresh = path.parent.parent.name in new_sessions
        kind = 'new' if fresh else 'imported'
        segment_counts[kind] += 1
        samples[kind] += quality['cadence']['samples']
        assert quality['collection_can_continue']
        flags.update(quality['flags'])
        for task_id, value in quality['measurements'].items():
            assert task_id not in seen
            seen.add(task_id)
            all_statuses[value['status']] += 1
            (new_statuses if fresh else old_statuses)[value['status']] += 1
            if fresh:
                assert task_id in redo
                split_status[(expected[task_id]['split'], value['status'])] += 1
        if fresh:
            new_flags.update(quality['flags'])
            new_counter_anomalies += quality['counter_anomaly_blocks']
            max_gap = max(max_gap, quality['cadence']['observed_interval_seconds_max'])
            for key, bounds in quality['ranges'].items():
                prior = ranges.get(key, bounds)
                ranges[key] = [min(prior[0], bounds[0]), max(prior[1], bounds[1])]
            for key, value in quality['counters'].items():
                active_samples[key] += value['active_samples']
        if quality['flags']:
            flagged_segments.append(dict(path=str(path.parent.relative_to(OUT)),
                records=len(quality['measurements']), flags=quality['flags'], new_session=fresh))
    assert seen == set(expected)
    assert old_statuses == {'PASS': 108222}
    current_review = read(OUT / 'remeasure_tasks.json')
    current_ids = {item['task']['id'] for item in current_review}
    assert current_ids <= redo
    breakdown = dict(quality_status_counts=dict(all_statuses), segments=sum(segment_counts.values()),
        segment_counts=dict(segment_counts), flagged_segments=flagged_segments,
        flags_by_segment=dict(flags), new_flags_by_segment=dict(new_flags),
        new_ranges=ranges, telemetry_samples=dict(samples), new_max_sample_gap_seconds=max_gap,
        new_counter_active_samples=dict(active_samples), new_counter_anomaly_blocks=new_counter_anomalies,
        imported_status_counts=dict(old_statuses), remeasured_status_counts=dict(new_statuses),
        resolved_review=len(redo-current_ids), remaining_review=len(current_ids),
        remaining_general_review=sum(expected[i]['split'] != 'fresh_holdout' for i in current_ids),
        remaining_fresh_holdout_review=sum(expected[i]['split'] == 'fresh_holdout' for i in current_ids),
        new_split_status_counts=[dict(split=split, status=status, count=count)
                                for (split, status), count in sorted(split_status.items())],
        new_sessions={name: read(OUT / 'sessions' / name / 'session.json') for name in sorted(new_sessions)},
        holdout_latencies_disclosed=False)
    (BASE / 'quality_breakdown.json').write_text(json.dumps(breakdown, indent=2) + '\n')
    spec = importlib.util.spec_from_file_location('full_result_validation',
        ROOT / 'analysis/full_720_result_2026-09-30/validate.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    previous.BASE, previous.OUT = BASE, OUT
    previous.main()
    print(json.dumps({k: v for k, v in breakdown.items() if k != 'flagged_segments'}, indent=2))


if __name__ == '__main__':
    main()
