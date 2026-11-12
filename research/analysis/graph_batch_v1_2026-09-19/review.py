"""Verify the received Graph batch result without executing uploaded code."""
import csv
from collections import Counter
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / 'raw/graph_batch_20260919_211455_Y9h4lA'
sys.path.insert(0, str(ROOT / 'graph_batch_v1'))
import graph_batch as gb


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hashes = json.loads((HERE / 'raw_sha256.json').read_text())
    for name, digest in hashes['files'].items():
        assert sha(RAW / name) == digest, name
    release = ROOT / 'graph_batch_2026-09-19_v1.zip'
    assert sha(release) == 'd9e40e2afbb2d6a8d3715a60e9d9206eda295bbdd7b71bf78905bdf7f6345a29'
    with zipfile.ZipFile(release) as z:
        for name in z.namelist():
            short = Path(name).name
            assert z.read(name) == (RAW / 'source' / short).read_bytes() == (ROOT / 'graph_batch_v1' / short).read_bytes()
    expected, plan, measure, suite = gb.support.load_suite(ROOT / 'gpu_sweep')
    manifest = json.loads((RAW / 'manifest.json').read_text())
    assert manifest['tasks'] == gb.tasks() and manifest['baseline'] == expected
    assert manifest['profiler_enabled'] is False and manifest['plateau_margin'] == .01
    for name, digest in expected['code_hashes'].items():
        assert sha(RAW / 'baseline_source' / name) == digest
    for name, digest in manifest['source_hashes'].items():
        assert sha(RAW / 'source' / name) == digest
    rows = []
    recomputed = HERE / 'recomputed'
    recomputed.mkdir(exist_ok=True)
    for rd in range(4):
        env = json.loads((RAW / f'round_{rd}/environment.json').read_text())
        assert suite.signature(env) == expected['environment_signature']
        part = json.loads((RAW / f'round_{rd}/results.json').read_text())
        assert [r['task'] for r in part] == [t for t in gb.tasks() if t['round'] == rd]
        rows.extend(part)
        (recomputed / f'round_{rd}').mkdir(exist_ok=True)
        shutil.copy2(RAW / f'round_{rd}/results.json', recomputed / f'round_{rd}/results.json')
    a = gb.analyze(recomputed, measure, plan, suite)
    assert a == json.loads((RAW / 'analysis.json').read_text())
    for name in ('summary.csv', 'REPORT.md'):
        assert (recomputed / name).read_bytes() == (RAW / name).read_bytes()
    exit_codes = {str(p.relative_to(RAW)): int(p.read_text()) for p in RAW.rglob('*exit_code.txt')}
    assert exit_codes and all(code == 0 for code in exit_codes.values())

    # Independent checks on every denominator and saved statistic.
    independent = {}
    for row in rows:
        task, result = row['task'], row['result']
        batch = task['gemms_per_replay']
        timing = result['timing']
        blocks = timing['blocks']
        assert result['correctness']['passed'] is True
        assert result['correctness']['finite_all'] is True
        assert result['correctness']['max_tolerance_ratio'] <= 1
        assert len(blocks) == 7 and not timing['flags']
        values = [b['elapsed_ms'] / b['repeats'] / batch for b in blocks]
        assert all(math.isfinite(v) and v > 0 for v in values)
        assert all(math.isclose(v, b['latency_ms'], rel_tol=1e-12) for v, b in zip(values, blocks))
        q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
        med = statistics.median(values)
        assert math.isclose(med, timing['median_ms'], rel_tol=1e-12)
        assert math.isclose((q3-q1)/med, timing['relative_iqr'], abs_tol=1e-12)
        assert math.isclose((values[-1]-values[0])/values[0], timing['first_last_drift'], abs_tol=1e-12)
        assert sum(b['repeats'] * batch for b in blocks) == result['measured_gemm_calls']
        setup = timing['setup']
        assert math.isclose(sum(b['elapsed_ms'] for b in setup['settle_blocks']), setup['settle_event_ms'], rel_tol=1e-12)
        assert setup['settle_event_ms'] >= 3000
        assert all(b['operations_per_call'] == batch for b in setup['settle_blocks'] + setup['calibration'] + blocks)
        independent[(task['case_id'], batch, task['round'])] = med

    paired = {}
    for case in a['cases']:
        cid = case['case']['id']
        for pair in case['pairs']:
            changes = [independent[cid, pair['upper'], rd] / independent[cid, pair['lower'], rd] - 1 for rd in range(4)]
            assert changes == pair['paired_relative_changes']
            assert all(abs(v) <= .01 for v in changes) == pair['within_margin_all_rounds']
        candidate = next((b for b in (16, 32, 64) if all(
            abs(independent[cid, hi, rd] / independent[cid, lo, rd] - 1) <= .01
            for lo, hi in itertools.combinations([x for x in (16, 32, 64, 128) if x >= b], 2)
            for rd in range(4))), None)
        assert candidate == case['candidate_gemms_per_replay']
        p = next(p for p in case['pairs'] if (p['lower'], p['upper']) == (64, 128))
        paired[cid] = dict(median=p['median_change'], minimum=min(p['paired_relative_changes']),
                           maximum=max(p['paired_relative_changes']))

    classifications = list(RAW.rglob('*classification.json'))
    assert len(classifications) == 57
    assert all(json.loads(p.read_text())['blocked'] == [] for p in classifications)
    with (RAW / 'telemetry.csv').open() as f:
        telemetry = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    parse = lambda t: dt.datetime.strptime(t, '%Y/%m/%d %H:%M:%S.%f')
    moments = [parse(r['timestamp']) for r in telemetry]
    assert all(b > a for a, b in zip(moments, moments[1:]))
    metrics = {}
    for field in ('temperature.gpu', 'clocks.current.sm [MHz]', 'clocks.current.memory [MHz]', 'power.draw [W]'):
        values = [float(r[field].split()[0]) for r in telemetry]
        metrics[field] = dict(min=min(values), median=statistics.median(values), max=max(values))
    status = json.loads((RAW / 'status.json').read_text())
    assert status['collection'] == 'COMPLETE' and status['stability'] == 'PASS'
    assert status['candidates'] == {c['case']['id']: c['candidate_gemms_per_replay'] for c in a['cases']}
    scopes = Counter(r['result']['correctness']['scope'] for r in rows)
    assert scopes == {'full': 32, 'sampled': 16}
    assert all(r['result']['correctness']['points'] == (4096 if r['task']['shape'][0] == 64 else 256) for r in rows)
    verified = dict(zip_sha256=hashes['zip_sha256'], raw_files=len(hashes['files']),
        source_files=7, baseline_files=13, environments=4, analysis_exact_match=True,
        exit_codes=exit_codes, correctness_pass=48, full_output_checks=32, sampled_output_checks=16,
        measured_blocks=336, measured_gemms=sum(r['result']['measured_gemm_calls'] for r in rows),
        max_relative_iqr=max(r['result']['timing']['relative_iqr'] for r in rows),
        max_abs_drift=max(abs(r['result']['timing']['first_last_drift']) for r in rows),
        max_between_round_spread=max(s['between_round_spread'] for c in a['cases'] for s in c['sizes']),
        min_block_ms=min(b['elapsed_ms'] for r in rows for b in r['result']['timing']['blocks']),
        min_settle_ms=min(r['result']['timing']['setup']['settle_event_ms'] for r in rows),
        candidates=status['candidates'], paired_64_to_128=paired,
        process_snapshots=57, telemetry_samples=len(telemetry),
        telemetry_start=telemetry[0]['timestamp'], telemetry_end=telemetry[-1]['timestamp'],
        telemetry_duration_seconds=(moments[-1]-moments[0]).total_seconds(),
        telemetry_max_gap_seconds=max((b-a).total_seconds() for a,b in zip(moments,moments[1:])),
        telemetry_ranges=metrics, recommendation='64 GEMMs/replay for the three tested small conditions',
        limitations=['Graph stream time includes GPU work and internal gaps.',
                     'Observed 1% criterion is not a confidence interval or absolute error bound.',
                     'No simultaneous new prealloc measurement or whole-GPU interference exclusion.'])
    suite.atomic_json(HERE / 'verification.json', verified)
    for name, digest in hashes['files'].items():
        assert sha(RAW / name) == digest
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
