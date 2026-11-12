"""Read-only verification of the uploaded v4 run; writes only derived artifacts."""
import csv
import datetime as dt
import hashlib
import json
import math
import re
import sqlite3
import statistics as stats
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw/pilot_v4'
sys.path.insert(0, str(HERE.parents[1] / 'gpu_sweep'))
from measure import quantile, summarize
from plan import digest
from profiling import ncu_has_kernel_data, nsys_has_kernel_data


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def table(name, rows):
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def read_db(path, sql):
    # The uploaded WAL must be empty before immutable reads can be used.
    wal = path.with_name(path.name + '-wal')
    assert not wal.exists() or wal.stat().st_size == 0
    con = sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        return con.execute(sql).fetchall()
    finally:
        con.close()


def main():
    hashes = {str(p.relative_to(RAW)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in RAW.rglob('*') if p.is_file()}
    source = json.loads((HERE / 'source_manifest.json').read_text())
    assert hashlib.sha256(Path(source['source']).read_bytes()).hexdigest() == source['sha256']
    with zipfile.ZipFile(source['source']) as z:
        expected_raw = {str(Path(n).relative_to('pilot_v4')): hashlib.sha256(z.read(n)).hexdigest()
                        for n in z.namelist() if not n.endswith('/')}
    assert hashes == expected_raw
    manifest = json.loads((RAW / 'manifest.json').read_text())
    saved = json.loads((RAW / 'summary.json').read_text())
    rows = [json.loads(x) for x in (RAW / 'raw_attempts.jsonl').read_text().splitlines()]
    assert rows == [json.loads(x[0]) for x in read_db(RAW / 'results.sqlite3', 'SELECT record FROM attempts ORDER BY seq')]
    assert digest(rows) == saved['result_digest']
    assert len(rows) == len({r['task']['id'] for r in rows}) == len(manifest['tasks']) == 253
    assert [r['task'] for r in rows] == manifest['tasks']
    with zipfile.ZipFile(HERE.parents[1] / 'gpu_sweep_2026-09-19_v4.zip') as z:
        for name, expected in manifest['code_hashes'].items():
            assert hashes['source/' + name] == expected
            assert hashlib.sha256(z.read('gpu_sweep/' + name)).hexdigest() == expected
    measured = [r for r in rows if 'timing' in r]
    pairs, phases = defaultdict(list), defaultdict(list)
    reviews, chain, blocks = [], [], []
    for r in measured:
        task, tm = r['task'], r['timing']
        assert len(tm['blocks']) == manifest['protocol']['blocks'] == 7
        actual = summarize(tm['blocks'], manifest['protocol'])
        for key in ('median_ms', 'relative_iqr', 'first_last_drift', 'min_ms', 'max_ms'):
            assert math.isclose(actual[key], tm[key], rel_tol=1e-12, abs_tol=1e-12)
        flags = actual['flags']
        if 'setup' in tm and task['mode'] != 'pressure' and any(b['elapsed_ms'] < .8 * manifest['protocol']['block_ms'] for b in tm['blocks']):
            flags.append('short_block')
        if task['kind'] == 'chain':
            for i in range(len(task['widths']) - 1):
                layer = summarize([b['measurements'][str(i)] for b in r['paired_blocks']], manifest['protocol'])
                flags.extend(f'layer{i}_{flag}' for flag in layer['flags'])
        assert flags == tm['flags']
        for i, b in enumerate(tm['blocks']):
            assert math.isclose(b['elapsed_ms'] / (b['repeats'] * b['operations_per_call']), b['latency_ms'], rel_tol=1e-12)
            blocks.append({'id': task['id'], 'block': i, **b})
        if 'tflops' in r:
            assert math.isclose(r['flops'] / (tm['median_ms'] * 1e9), r['tflops'], rel_tol=1e-12)
        if 'effective_copy_gbps' in r:
            assert math.isclose(r['payload_bytes_read_plus_write'] / (tm['median_ms'] * 1e6), r['effective_copy_gbps'], rel_tol=1e-12)
        assert r['correctness']['passed']
        assert all(c.get('max_tolerance_ratio', 0) <= 1 for c in r['correctness'].get('layers', [r['correctness']]))
        assert r['status'] == ('REVIEW' if flags else 'PASS')
        phases[task.get('phase', task['kind'])].append(r)
        pairs[digest({k: v for k, v in task.items() if k not in ('round', 'id')})].append(r)
        if flags:
            reviews.append({'id': task['id'], 'shape': task.get('shape'), 'dtype': task.get('dtype'),
                            'mode': task['mode'], 'round': task['round'], 'flags': ';'.join(flags),
                            'iqr_percent': tm['relative_iqr'] * 100, 'drift_percent': tm['first_last_drift'] * 100,
                            'latency_us': [b['latency_ms'] * 1000 for b in tm['blocks']],
                            'elapsed_ms': [b['elapsed_ms'] for b in tm['blocks']]})
        if task['kind'] == 'chain':
            for b in r['paired_blocks']:
                assert math.isclose(b['chain_ms'] - b['isolated_sum_ms'], b['composition_residual_ms'], abs_tol=1e-12)
                assert math.isclose(b['composition_residual_ms'] / b['chain_ms'], b['composition_relative'], abs_tol=1e-12)
            chain.append({'dtype': task['dtype'], 'mode': task['mode'], 'round': task['round'],
                          'batch': task['batch'], 'widths': task['widths'],
                          'chain_us': tm['median_ms'] * 1000,
                          'paired_residual_percent': stats.median(b['composition_relative'] for b in r['paired_blocks']) * 100,
                          'iqr_percent': tm['relative_iqr'] * 100})
    between = []
    for group in pairs.values():
        values = [r['timing']['median_ms'] for r in group]
        if len(values) < 2:
            continue
        spread = (max(values) - min(values)) / stats.median(values)
        task = group[0]['task']
        between.append({'id_round0': task['id'], 'phase': task.get('phase', task['kind']),
                        'shape': task.get('shape'), 'dtype': task.get('dtype'), 'mode': task['mode'],
                        'round0_us': values[0] * 1000, 'round1_us': values[1] * 1000,
                        'relative_range_percent': spread * 100,
                        'review': spread > manifest['protocol']['between_round_limit']})
    flagged = [b for b in between if b['review']]
    assert len(flagged) == saved['between_round_flags']
    for actual, expected in zip(flagged, saved['between_round']):
        assert actual['id_round0'] == expected['task']['id']
        assert math.isclose(actual['relative_range_percent'], expected['relative_range'] * 100, abs_tol=1e-12)
    profiles = []
    for r in rows:
        if r['task']['kind'] != 'profile':
            continue
        directory = RAW / r['directory']
        assert r['exit_code'] == int((directory / 'exit_code.txt').read_text()) == 0
        assert (RAW / r['report']).stat().st_size > 0
        item = {'id': r['task']['id'], 'tool': r['task']['tool'], 'mode': r['task']['mode'],
                'saved_status': r['status'], 'exit_code': r['exit_code']}
        if r['task']['tool'] == 'ncu':
            assert ncu_has_kernel_data((directory / 'output.log').read_text())
            item['kernel_metrics_present'] = True
        else:
            item['kernel_csv_passes_v5_checker'] = nsys_has_kernel_data((directory / 'kernel_summary.csv').read_text())
            db = directory / 'capture.sqlite'
            names = {x[0] for x in read_db(db, "SELECT name FROM sqlite_master WHERE type='table'")}
            for name in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE'):
                item[name] = read_db(db, f'SELECT COUNT(*) FROM "{name}"')[0][0] if name in names else 0
            if item['mode'] == 'graph':
                item['positive_time_graph_records'] = read_db(db, 'SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE start>=0 AND end>start')[0][0]
        profiles.append(item)
    telemetry = [{k.strip(): v.strip() for k, v in r.items()}
                 for r in csv.DictReader(next(RAW.glob('sessions/*/telemetry.csv')).open())]
    for r in telemetry:
        r['at'] = dt.datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone(dt.timedelta(hours=9)))
    observations = []
    for r in measured:
        task, tm = r['task'], r['timing']
        first, last = tm['blocks'][0], tm['blocks'][-1]
        # v4 pressure blocks have no timestamps; do not invent a timing window.
        if 'timestamp' not in first:
            continue
        start = dt.datetime.fromisoformat(first['timestamp'])
        end = dt.datetime.fromisoformat(last['timestamp']) + dt.timedelta(milliseconds=last['host_wall_ms'])
        samples = [x for x in telemetry if start <= x['at'] <= end]
        obs = {'id': task['id'], 'phase': task.get('phase', task['kind']), 'shape': task.get('shape'),
               'dtype': task.get('dtype'), 'mode': task['mode'], 'round': task['round'],
               'start_utc': start.isoformat(), 'end_utc': end.isoformat(), 'latency_us': tm['median_ms'] * 1000,
               'samples_inside_timing_window': len(samples)}
        for key in ('temperature.gpu', 'clocks.current.sm [MHz]', 'power.draw [W]'):
            obs[key] = stats.median(float(x[key].split()[0]) for x in samples) if samples else None
        observations.append(obs)
    counts = dict(Counter(r['status'] for r in rows))
    assert counts == saved['counts']
    before = json.loads((RAW / 'environment.json').read_text())['nvidia_smi']['stdout']
    after = json.loads(next(RAW.glob('sessions/*/gpu_after.json')).read_text())['stdout']
    counter_deltas = {}
    for name in ('SW Power Capping', 'SW Thermal Slowdown', 'HW Thermal Slowdown'):
        pattern = re.escape(name) + r'\s*:\s*(\d+) us'
        counter_deltas[name] = (int(re.search(pattern, after).group(1)) - int(re.search(pattern, before).group(1))) / 1e6
    output = {'raw_gate_unchanged': saved['gate'], 'recorded': len(rows), 'missing': 0, 'counts': counts,
              'verified_measurement_rows': len(measured), 'between_round_flags': len(flagged),
              'max_between_round_percent': max(b['relative_range_percent'] for b in between),
              'iqr_median_percent': stats.median(r['timing']['relative_iqr'] for r in measured) * 100,
              'iqr_p95_percent': quantile([r['timing']['relative_iqr'] for r in measured], .95) * 100,
              'phases': {k: {'count': len(v), 'review': sum(r['status'] == 'REVIEW' for r in v)} for k, v in phases.items()},
              'profiles': profiles, 'source_hash_count': len(manifest['code_hashes']),
              'protocol': manifest['protocol'],
              'session': json.loads(next(RAW.glob('sessions/*/session.json')).read_text()),
              'before_after_counter_delta_seconds': counter_deltas,
              'telemetry_temperature_min_max_C': [min(int(r['temperature.gpu']) for r in telemetry), max(int(r['temperature.gpu']) for r in telemetry)],
              'checks': ['ZIP/extracted hashes; SQLite integrity and empty WAL; DB/JSONL equality',
                         'manifest/task IDs/order; v4 distribution and source hashes; result digest',
                         '7-block statistics and flags; FLOP/s and copy GB/s; saved correctness',
                         'round flags; paired chain arithmetic; NCU metrics and NSYS SQLite evidence']}
    dump('verification.json', output)
    table('review_conditions.csv', reviews)
    table('between_round.csv', between)
    table('chain_results.csv', chain)
    table('telemetry_alignment.csv', observations)
    table('blocks.csv', blocks)
    assert hashes == {str(p.relative_to(RAW)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in RAW.rglob('*') if p.is_file()}
    dump('raw_sha256.json', hashes)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
