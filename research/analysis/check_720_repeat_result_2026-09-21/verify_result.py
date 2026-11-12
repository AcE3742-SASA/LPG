"""Offline verification of the repeated 720 MHz check; no GPU execution."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / 'raw/check_720_repeat_20260921_190318'
sys.path.insert(0, str(ROOT / 'abstract_sweep_725_v1'))
import run
import analyze_sweep


def main():
    archive = run.read(HERE / 'archive_check.json')
    assert run.sha(HERE / Path(archive['original']).name) == archive['sha256']
    before = {str(p.relative_to(RAW)): run.sha(p) for p in RAW.rglob('*') if p.is_file()}
    manifest = run.read(RAW / 'manifest.json')
    loaded = run.load(720)
    assert run.manifest_for(RAW, 'check', loaded) == manifest
    copied = HERE / 'recomputed'
    if not copied.exists():
        shutil.copytree(RAW, copied)
    analyze_sweep.analyze(copied, loaded)
    summary = run.read(copied / 'summary.json')
    assert summary == run.read(RAW / 'summary.json')
    names = ['raw_attempts.jsonl', 'metrics.csv', 'fresh_holdout_LOCKED.csv', 'blocks.csv',
             'blocks_fresh_holdout_LOCKED.csv', 'between_round.json', 'review_queue.csv',
             'review_fresh_holdout_LOCKED.csv', 'triplets.csv', 'triplets_fresh_holdout_LOCKED.csv',
             'shape_splits.csv', 'reference_780_comparison.csv', 'summary.json', 'report.md']
    names += [str(p.relative_to(RAW)) for p in RAW.glob('sessions/*/segment_*/quality.json')]
    assert all(run.sha(copied / name) == run.sha(RAW / name) for name in names)
    rows = [r for _, r in run.history(copied)]
    assert len(rows) == 48 == len({r['task']['id'] for r in rows})
    assert [r['task'] for r in rows] == manifest['tasks']
    assert [dict(seq=i, **r) for i, r in run.history(copied)] == [json.loads(s) for s in (RAW / 'raw_attempts.jsonl').read_text().splitlines()]
    assert hashlib.sha256((RAW / 'raw_attempts.jsonl').read_bytes()).hexdigest() == summary['results_digest']
    grouped = defaultdict(list)
    for r in rows:
        t, tm, correct = r['task'], r['timing'], r['correctness']
        assert r['status'] == 'PASS' and correct['passed'] and correct['finite_all']
        assert correct['max_tolerance_ratio'] <= 1
        assert len(tm['blocks']) == 7
        values = []
        for b in tm['blocks']:
            value = b['elapsed_ms'] / (b['repeats'] * b['operations_per_call'])
            assert math.isclose(value, b['latency_ms'], rel_tol=1e-12)
            values.append(value)
        med = st.median(values)
        quartiles = st.quantiles(values, n=4, method='inclusive')
        recomputed = {'median_ms': med, 'relative_iqr': (quartiles[2] - quartiles[0]) / med,
                      'first_last_drift': (values[-1] - values[0]) / values[0]}
        assert all(math.isclose(v, tm[k], rel_tol=1e-10, abs_tol=1e-12) for k, v in recomputed.items())
        m, n, k = t['shape']
        assert r['flops'] == 2 * m * n * k
        assert r['ideal_bytes'] == (m * k + k * n + m * n) * (2 if t['dtype'] == 'fp16' else 4)
        assert math.isclose(r['tflops'], r['flops'] / (med * 1e9), rel_tol=1e-12)
        grouped[(tuple(t['shape']), t['dtype'], t['mode'])].append(r)
    assert len(grouped) == 24
    assert all(len(v) == 2 and len({r['session'] for r in v}) == 2 for v in grouped.values())
    spreads = [(max(r['timing']['median_ms'] for r in v) - min(r['timing']['median_ms'] for r in v)) /
               st.median(r['timing']['median_ms'] for r in v) for v in grouped.values()]
    assert math.isclose(max(spreads), summary['max_between_round_range'], rel_tol=1e-12)
    samples, sessions = [], []
    for folder in sorted(RAW.glob('sessions/*/segment_*')):
        metadata = run.read(folder / 'telemetry_metadata.json')
        data, cadence = loaded['telemetry'].read(folder / 'telemetry.csv', metadata['timezone_offset_seconds'], 50)
        assert all(s['clocks.sm'] == 720 and s['clocks.mem'] == 6801 and s['power.limit'] == 70 for s in data)
        assert all(s[k] is False for s in data for k in loaded['telemetry'].REASONS)
        session = run.read(folder.parent / 'session.json')
        assert session['new_records'] == 24
        sessions.append(dict(session=session, cadence=cadence, quality=run.read(folder / 'quality.json')['counters']))
        samples.extend(data)
    note = ('Offline review: all 48 check measurements, two workers, correctness within the configured scope, '
            'timing and sampled clocks verified; cumulative counter interpretation remains unresolved. '
            'This eligibility check does not start or authorize a GPU run.')
    admission = run.check_admission(copied, note, loaded)
    previous = ROOT / 'analysis/check_low_clock_result_2026-09-21/raw/check_low_clock/metrics.csv'
    comparison = []
    with previous.open() as f:
        for old in csv.DictReader(f):
            key = (tuple(json.loads(old['shape'])), old['dtype'], old['mode'])
            med = st.median(r['timing']['median_ms'] for r in grouped[key])
            comparison.append(dict(shape=list(key[0]), dtype=key[1], mode=key[2],
                                   previous_ms=float(old['median_ms']), current_ms=med,
                                   change_pct=100 * (med / float(old['median_ms']) - 1)))
    after = {str(p.relative_to(RAW)): run.sha(p) for p in RAW.rglob('*') if p.is_file()}
    assert before == after
    report = dict(verification='PASS', decision='CHECK_QUALITY_PASS_COUNTER_REVIEW_RETAINED',
                  archive=archive, raw_files=len(before), source_files_verified=len(manifest['source_hashes']),
                  dependencies_verified=len(manifest['dependency_hashes']), reproduced_identical_outputs=len(names),
                  result=summary, correctness_scopes=dict(Counter(r['correctness']['scope'] for r in rows)),
                  max_tolerance_ratio=max(r['correctness']['max_tolerance_ratio'] for r in rows),
                  telemetry_samples=len(samples), sessions=sessions,
                  core_counts=dict(Counter(s['clocks.sm'] for s in samples)),
                  memory_counts=dict(Counter(s['clocks.mem'] for s in samples)),
                  ranges={k: [min(s[k] for s in samples), max(s[k] for s in samples)] for k in ('power.draw', 'temperature.gpu')},
                  exit_codes={str(p.relative_to(RAW)): int(p.read_text()) for p in RAW.rglob('*exit_code.txt')},
                  admission=admission, previous_720_comparison=comparison,
                  prior_change_abs_max_pct=max(abs(r['change_pct']) for r in comparison),
                  gpu_executions=0, collection_code_or_threshold_changes=False, raw_hashes_unchanged=True)
    (HERE / 'raw_sha256.json').write_text(json.dumps(before, indent=2) + '\n')
    (HERE / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('verification', 'decision', 'raw_files', 'source_files_verified',
          'dependencies_verified', 'reproduced_identical_outputs', 'correctness_scopes', 'telemetry_samples',
          'core_counts', 'memory_counts', 'ranges', 'prior_change_abs_max_pct', 'exit_codes')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
