"""Run after the unchanged v2 analyzer; report quality without holdout latencies."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

BASE = Path(__file__).resolve().parent
OUT = BASE / 'recomputed/full_720'


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
            raw_count += 1
    assert raw_count == attempts
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
    assert flagged == review_ids  # This run has no additional between-round flags.
    assert summary['between_round_flags'] == 0
    breakdown = read(BASE / 'quality_breakdown.json')
    assert breakdown['quality_status_counts'] == dict(statuses)
    quality_files = list((OUT / 'sessions').glob('*/segment_*/quality.json'))
    assert len(quality_files) == breakdown['segments']
    for path in quality_files:
        quality = read(path)
        assert quality['collection_can_continue']
        assert quality['telemetry_sha256'] == sha(path.parent / 'telemetry.csv')
    result = dict(validation='PASS', summary_exact_match=True, export_byte_matches=exports,
                  sqlite_quick_check='ok', recorded=attempts, duplicate_task_ids=attempts-distinct,
                  missing_task_ids=0, correctness_passed=raw_count, correctness_scopes=dict(scopes),
                  raw_status_counts=dict(raw_statuses), sessions=dict(sessions),
                  final_status_counts=dict(statuses),
                  split_status_counts=[dict(split=split, status=status, count=count)
                                       for (split, status), count in sorted(split_statuses.items())],
                  remeasure_tasks=len(review), remeasure_list_has_no_latency=True,
                  telemetry_hashes_checked=len(quality_files), holdout_latencies_disclosed=False,
                  experiment_gate=summary['gate'], archive_sha256=archive['sha256'])
    (BASE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
