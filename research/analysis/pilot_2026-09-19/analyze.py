"""Recheck the uploaded pilot without modifying its raw files or executing its code."""
import csv
import datetime as dt
import hashlib
import json
import math
import sqlite3
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw/runs/pilot'
sys.path.insert(0, str(HERE.parents[1] / 'gpu_sweep'))
from measure import quantile, summarize
from plan import digest
from profiling import ncu_has_kernel_data


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    hashes = {str(p.relative_to(RAW)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in RAW.rglob('*') if p.is_file()}
    manifest = json.loads((RAW / 'manifest.json').read_text())
    old_summary = json.loads((RAW / 'summary.json').read_text())
    rows = [json.loads(line) for line in (RAW / 'raw_attempts.jsonl').read_text().splitlines()]
    con = sqlite3.connect(RAW.joinpath('results.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        db_rows = [json.loads(r[0]) for r in con.execute('SELECT record FROM attempts ORDER BY seq')]
    finally:
        con.close()
    assert rows == db_rows
    assert digest(rows) == old_summary['result_digest']
    assert len({r['task']['id'] for r in rows}) == len(rows)
    assert [r['task'] for r in rows] == manifest['tasks'][:len(rows)]
    for name, expected in manifest['code_hashes'].items():
        assert hashes['source/' + name] == expected
    measured = [r for r in rows if 'timing' in r]
    phases, pairs, reviews, chain = defaultdict(list), defaultdict(list), [], []
    for r in measured:
        t, tm = r['task'], r['timing']
        assert len(tm['blocks']) == manifest['protocol']['blocks']
        actual = summarize(tm['blocks'], manifest['protocol'])
        for key in ('median_ms', 'relative_iqr', 'first_last_drift', 'min_ms', 'max_ms'):
            assert math.isclose(actual[key], tm[key], rel_tol=1e-12, abs_tol=1e-12)
        for b in tm['blocks']:
            assert math.isclose(b['elapsed_ms'] / (b['repeats']*b['operations_per_call']),
                                b['latency_ms'], rel_tol=1e-12)
        if 'tflops' in r:
            assert math.isclose(r['flops'] / (tm['median_ms']*1e9), r['tflops'], rel_tol=1e-12)
        assert r['correctness']['passed']
        checks = r['correctness'].get('layers', [r['correctness']])
        assert all(c.get('max_tolerance_ratio', 0) <= 1 for c in checks)
        assert r['status'] == ('REVIEW' if tm['flags'] else 'PASS')
        phases[t.get('phase', t['kind'])].append(r)
        key = digest({k: v for k, v in t.items() if k not in ('round', 'id')})
        pairs[key].append(r)
        if tm['flags']:
            reviews.append({'id': t['id'], 'shape': t.get('shape'), 'dtype': t.get('dtype'),
                            'mode': t['mode'], 'round': t['round'], 'flags': ';'.join(tm['flags']),
                            'iqr_percent': tm['relative_iqr']*100,
                            'drift_percent': tm['first_last_drift']*100,
                            'block_elapsed_ms': [b['elapsed_ms'] for b in tm['blocks']]})
        if t['kind'] == 'chain':
            for b in r['paired_blocks']:
                assert math.isclose(b['chain_ms']-b['isolated_sum_ms'], b['composition_residual_ms'], abs_tol=1e-12)
                assert math.isclose(b['composition_residual_ms']/b['chain_ms'], b['composition_relative'], abs_tol=1e-12)
            chain.append({'dtype': t['dtype'], 'mode': t['mode'], 'round': t['round'],
                          'batch': t['batch'], 'widths': t['widths'],
                          'isolated_sum_us': stats.median(b['isolated_sum_ms'] for b in r['paired_blocks'])*1000,
                          'chain_us': tm['median_ms']*1000,
                          'paired_residual_percent': stats.median(b['composition_relative'] for b in r['paired_blocks'])*100,
                          'iqr_percent': tm['relative_iqr']*100})
    between = []
    for group in pairs.values():
        if len(group) < 2:
            continue
        values = [r['timing']['median_ms'] for r in group]
        spread = (max(values)-min(values))/stats.median(values)
        if spread > manifest['protocol']['between_round_limit']:
            t = group[0]['task']
            between.append({'id_round0': t['id'], 'phase': t.get('phase', t['kind']),
                            'shape': t.get('shape'), 'dtype': t.get('dtype'), 'mode': t['mode'],
                            'round0_us': values[0]*1000, 'round1_us': values[1]*1000,
                            'relative_range_percent': spread*100})
    assert len(between) == old_summary['between_round_flags']
    for actual, saved in zip(between, old_summary['between_round']):
        assert actual['id_round0'] == saved['task']['id']
        assert math.isclose(actual['relative_range_percent'], saved['relative_range']*100, abs_tol=1e-12)
    profile = RAW / rows[-1]['directory']
    text = profile.joinpath('output.log').read_text()
    parsed = list(csv.reader(line for line in text.splitlines() if line.startswith('"')))
    kernel = dict(zip(parsed[0], parsed[-1]))
    assert rows[-1]['exit_code'] == 0 and profile.joinpath('exit_code.txt').read_text().strip() == '0'
    assert profile.joinpath('capture.ncu-repz').stat().st_size > 0
    assert '"Metric Name"' not in text and ncu_has_kernel_data(text)
    telemetry = [{k.strip(): v.strip() for k, v in r.items()}
                 for r in csv.DictReader(next(RAW.glob('sessions/*/telemetry.csv')).open())]
    for r in telemetry:
        r['at'] = dt.datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
    observations = []
    for r in phases['main']:
        t, tm = r['task'], r['timing']
        first, last = tm['blocks'][0], tm['blocks'][-1]
        # Ubuntu telemetry uses local KST; event records use UTC.
        start = dt.datetime.fromisoformat(first['timestamp']).replace(tzinfo=None) + dt.timedelta(hours=9)
        end = dt.datetime.fromisoformat(last['timestamp']).replace(tzinfo=None) + dt.timedelta(hours=9, milliseconds=last['host_wall_ms'])
        samples = [x for x in telemetry if start <= x['at'] <= end]
        obs = {'shape': t['shape'], 'dtype': t['dtype'], 'round': t['round'],
               'latency_us': tm['median_ms']*1000, 'samples_inside_timing_window': len(samples)}
        for k in ('temperature.gpu', 'clocks.current.sm [MHz]', 'power.draw [W]'):
            obs[k] = stats.median(float(x[k].split()[0]) for x in samples) if samples else None
        observations.append(obs)
    phase_summary = {k: {'count': len(v), 'review': sum(r['status']=='REVIEW' for r in v),
                        'iqr_median_percent': stats.median(r['timing']['relative_iqr'] for r in v)*100,
                        'iqr_p95_percent': quantile([r['timing']['relative_iqr'] for r in v], .95)*100}
                     for k, v in phases.items()}
    output = {'raw_gate_unchanged': old_summary['gate'], 'counts': dict(Counter(r['status'] for r in rows)),
              'recorded': len(rows), 'missing': len(manifest['tasks'])-len(rows),
              'verified_measurement_rows': len(measured), 'between_round_flags': len(between),
              'phases': phase_summary, 'profile_exit_code': rows[-1]['exit_code'],
              'profile_report': 'capture.ncu-repz', 'profile_kernel': kernel['Kernel Name'],
              'profile_csv_columns': len(parsed[0]), 'profile_reclassified_by_new_checker': True,
              'elapsed_seconds': json.loads(next(RAW.glob('sessions/*/session.json')).read_text())['elapsed_seconds'],
              'iqr_median_percent': stats.median(r['timing']['relative_iqr'] for r in measured)*100,
              'iqr_p95_percent': quantile([r['timing']['relative_iqr'] for r in measured], .95)*100,
              'telemetry_temperature_min_max_C': [min(int(r['temperature.gpu']) for r in telemetry), max(int(r['temperature.gpu']) for r in telemetry)],
              'checks': ['SQLite integrity; DB/JSONL equality; task IDs/order; source hashes',
                         'summary digest; 7-block statistics; throughput; saved correctness checks',
                         'round flags; paired chain arithmetic; NCU output evidence']}
    (HERE/'verification.json').write_text(json.dumps(output, indent=2, ensure_ascii=False)+'\n')
    write_csv('review_conditions.csv', reviews)
    write_csv('between_round.csv', between)
    write_csv('chain_results.csv', chain)
    write_csv('main_telemetry_alignment.csv', observations)
    for path, expected in hashes.items():
        assert hashlib.sha256((RAW/path).read_bytes()).hexdigest() == expected
    (HERE/'raw_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
