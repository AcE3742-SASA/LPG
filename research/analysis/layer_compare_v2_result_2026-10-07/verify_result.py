#!/usr/bin/env python3
"""Offline verification of the returned run. Never writes raw data or runs CUDA."""
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
from types import SimpleNamespace
import zipfile

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
RAW = HERE / 'raw/layer_compare_20261006_143314_5Z1acE'
ARCHIVE = RESEARCH / 'layer_compare_20261006_143314_5Z1acE_5e654885.zip'
ARCHIVE_SHA = '1bb1cb9989b179ebe408f25cca715f332ad99b3e6dc1f306ee1c342e3c2c0b7d'
sys.path.insert(0, str(RESEARCH / 'layer_compare_v2'))
import compare
from layers import indices, seed_for


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def layout(shape):
    stride, product = [], 1
    for size in reversed(shape):
        stride.insert(0, product)
        product *= size
    return stride


def output_shape(case):
    if case['layer'].startswith('conv'):
        return [case[k] for k in ('batch', 'cout', 'height', 'width')]
    return [case['rows'], case.get('output_width', case['input_width'])]


def main():
    require(sha(ARCHIVE) == ARCHIVE_SHA, 'Original archive SHA mismatch')
    with zipfile.ZipFile(ARCHIVE) as archive:
        require(archive.testzip() is None, 'ZIP CRC failure')
        archive_files = sum(not item.is_dir() for item in archive.infolist())
    config, cases = compare.verify()
    frozen, loaded = compare.load_frozen()
    manifest = read(RAW / 'manifest.json')
    require(manifest['source_hashes'] == read(RESEARCH / 'layer_compare_v2/release_hashes.json'),
            'Returned source differs from reviewed v2 release')
    require(manifest['protocol'] == config and manifest['cases'] == cases,
            'Returned protocol or shape list mismatch')
    hashes = read(RAW / 'result_hashes.json')
    for name, wanted in hashes.items():
        path = (RAW / name).resolve()
        require(RAW in path.parents and path.is_file() and sha(path) == wanted,
                'Result hash mismatch: ' + name)
    for name, wanted in manifest['source_hashes'].items():
        require(sha(RAW / 'source' / name) == wanted, 'Embedded source mismatch: ' + name)
    require(loaded['suite'].signature(read(RAW / 'worker/environment.json')) ==
            manifest['environment_signature'], 'Environment signature mismatch')
    precision = read(RAW / 'worker/layer_precision.json')
    require(precision['conv'] == {'fp32_precision': 'ieee'} and precision['enabled'] is True and
            precision['benchmark'] is True and precision['deterministic'] is False and
            precision['autocast'] is False, 'Returned layer precision settings mismatch')
    require(not (RAW / 'failure.json').exists(), 'Run has failure.json')
    require(read(RAW / 'worker/complete.json')['complete'] is True, 'Worker incomplete')
    require(read(RAW / 'status.json')['collection'] == 'COMPLETE', 'Collection incomplete')
    require(read(RAW / 'status.json')['restoration'] == 'COMMANDS_ACCEPTED',
            'Restoration commands not accepted')

    records = [read(path) for path in sorted((RAW / 'records').glob('*.json'))]
    expected = {compare.key(c, d, r, i) for c in cases for d in config['dtypes']
                for r in range(3) for i in ('standard', 'decomposed')}
    require(len(records) == 1920 and {r['id'] for r in records} == expected,
            'Measurement coverage mismatch')
    by_case = {c['id']: c for c in cases}
    checked_correctness, all_blocks = {}, []
    max_errors = {d: {'max_tolerance_ratio': 0, 'max_abs_error': 0} for d in config['dtypes']}
    for row in records:
        case, dtype, rd, implementation = row['case'], row['dtype'], row['round'], row['implementation']
        require(case == by_case[case['id']], 'Record case mismatch')
        require(row['id'] == compare.key(case, dtype, rd, implementation), 'Record ID mismatch')
        require(row['stage'] == ('health' if case['id'] in config['health_case_ids'] else 'pilot'),
                'Stage mismatch')
        require(set(row['pair_order']) == {'standard', 'decomposed'}, 'Pair order mismatch')
        timing = row['timing']
        require(len(timing['blocks']) == 7, 'Block count mismatch')
        recalc = loaded['measure'].summarize(timing['blocks'], config['timing'])
        if any(b['elapsed_ms'] < .8 * config['timing']['block_ms'] for b in timing['blocks']):
            recalc['flags'].append('short_block')
        require(all(timing[k] == v for k, v in recalc.items()), 'Timing summary mismatch')
        settle = timing['setup']['settle_blocks']
        require(sum(b['elapsed_ms'] for b in settle) == timing['setup']['settle_event_ms'] >= 3000,
                'Observed settling duration mismatch')
        blocks = settle + timing['setup']['calibration'] + timing['blocks']
        for block in blocks:
            repeats, ops = block['repeats'], block['operations_per_call']
            require(type(repeats) is int and 1 <= repeats <= config['timing']['max_repeats'] and ops == 1,
                    'Repeat count or operation count mismatch')
            require(math.isfinite(block['elapsed_ms']) and block['elapsed_ms'] > 0,
                    'Invalid elapsed time')
            require(block['latency_ms'] == block['elapsed_ms'] / (repeats * ops),
                    'Raw event per-call latency mismatch')
            require(block['host_latency_ms'] == block['host_wall_ms'] / (repeats * ops),
                    'Host per-call latency mismatch')
            require(datetime.fromisoformat(block['host_envelope_start_utc']) <
                    datetime.fromisoformat(block['host_envelope_end_utc']), 'Invalid host envelope')
        all_blocks.extend(timing['blocks'])
        shape = output_shape(case)
        memory = row['memory']
        require(memory['output_shape'] == shape and memory['output_stride'] == layout(shape),
                'Memory output layout mismatch')
        require(all(memory[k] >= 0 for k in ('baseline_allocated_bytes', 'extra_peak_allocated_bytes',
                                            'peak_reserved_bytes')), 'Negative memory count')
        identity = read(RAW / row['input_identity'])
        require(identity['seed'] == seed_for(case, dtype), 'Input seed mismatch')
        require(row['prepared_weight_bytes'] == sum(math.prod(t['shape']) for t in identity['prepared']) *
                (2 if dtype == 'fp16' else 4), 'Prepared weight size mismatch')
        require(all(len(t['sha256']) == 64 and t['stride'] == layout(t['shape']) and
                    t['dtype'] == ('torch.float16' if dtype == 'fp16' else 'torch.float32')
                    for t in [identity['input']] + identity['weights'] + identity['prepared']),
                'Saved tensor identity contract mismatch')
        for phase in ('before', 'after'):
            name = row['correctness_' + phase]
            require(name == f"correctness/{case['id']}_{dtype}_r{rd}_{phase}.json",
                    'Correctness reference mismatch')
            if name in checked_correctness:
                continue
            result = read(RAW / name)
            require(result['passed'] and result['full_pair_passed'], 'Stored correctness failed')
            require([result['atol'], result['rtol']] == config['tolerances'][dtype],
                    'Correctness tolerance mismatch')
            m = case['batch'] * case['height'] * case['width'] if case['layer'].startswith('conv') else case['rows']
            n = case['cout'] if case['layer'].startswith('conv') else case.get('output_width', case['input_width'])
            require(result['rows'] == indices(m, identity['seed']) and
                    result['columns'] == indices(n, identity['seed'] ^ 12345), 'Reference sample indices mismatch')
            require(set(result['implementations']) == {'standard', 'decomposed'}, 'Correctness path coverage mismatch')
            for value in result['implementations'].values():
                require(value['passed'] and value['finite_all'] and
                        math.isfinite(value['max_tolerance_ratio']) and 0 <= value['max_tolerance_ratio'] <= 1,
                        'Stored reference comparison failed')
                require(value['points'] == len(result['rows']) * len(result['columns']) and
                        value['output_shape'] == shape and value['output_stride'] == layout(shape),
                        'Correctness output contract mismatch')
                for key in ('max_tolerance_ratio', 'max_abs_error'):
                    max_errors[dtype][key] = max(max_errors[dtype][key], value[key])
            checked_correctness[name] = result
    require(len(checked_correctness) == len(list((RAW / 'correctness').glob('*.json'))) == 1920,
            'Correctness coverage mismatch')
    require(len(list((RAW / 'inputs').glob('*.json'))) == 320, 'Input identity coverage mismatch')
    print('RAW PASS: hashes, coverage, stored correctness and event statistics', flush=True)

    telemetry = loaded['telemetry']
    meta = read(RAW / 'worker/telemetry_metadata.json')
    rows, sensor_info = telemetry.read(RAW / 'worker/telemetry.csv', meta['timezone_offset_seconds'], 50)
    times = [datetime.fromisoformat(r['utc']) for r in rows]
    for row in rows:
        frozen.safe_row(row, manifest['environment_signature']['gpu_uuid'])

    def indexed_window(unused_rows, start, end):
        left, right = bisect_left(times, datetime.fromisoformat(start)), bisect_right(times, datetime.fromisoformat(end))
        return telemetry.window(rows[max(0, left-1):min(len(rows), right+1)], start, end)

    first_settle = min(b['host_envelope_start_utc'] for r in records for b in r['timing']['setup']['settle_blocks'])
    memory_deviations = [{'utc': r['utc'], 'memory_mhz': r['clocks.mem']} for r in rows if r['clocks.mem'] != 6801]
    require(all(r['utc'] < first_settle for r in memory_deviations),
            'Memory clock deviation after per-measurement settling began')

    # Preserve the original inclusive bounds, before/after samples and counter semantics.
    for j in range(32):
        block = all_blocks[j * (len(all_blocks)-1) // 31]
        start, end = block['host_envelope_start_utc'], block['host_envelope_end_utc']
        require(indexed_window(rows, start, end) == telemetry.window(rows, start, end),
                'Indexed sensor window differs from original full scan')
    print('SENSOR PASS: all rows checked; 32 full-scan window cross-checks', flush=True)
    indexed_loaded = dict(loaded, telemetry=SimpleNamespace(read=lambda *a: (rows, sensor_info), window=indexed_window))
    stage = HERE / 'recomputed'
    stage.mkdir(exist_ok=True)
    for name in ('records', 'correctness'):
        if not (stage / name).exists():
            shutil.copytree(RAW / name, stage / name, copy_function=os.link)
    (stage / 'worker').mkdir(exist_ok=True)
    for name in ('status.json', 'health_check.json', 'worker/complete.json', 'worker/telemetry_metadata.json'):
        shutil.copy2(RAW / name, stage / name)
    result = compare.analyze(stage, config, cases, indexed_loaded)
    require(result == read(RAW / 'summary.json'), 'Recomputed summary differs from returned summary')
    require((stage / 'comparison.csv').read_bytes() == (RAW / 'comparison.csv').read_bytes(),
            'Recomputed CSV differs from returned CSV')
    health = compare.analyze(stage, config, cases, indexed_loaded, health_only=True)
    saved_health = read(RAW / 'health_check.json')
    # The saved health snapshot predates full-run sensors, completion and restoration.
    health_keys = ('gate', 'scope', 'collection', 'expected', 'recorded', 'missing',
                   'counts', 'comparison_counts', 'comparisons', 'measurement_flags', 'interpretation')
    require(all(health[k] == saved_health[k] for k in health_keys),
            'Recomputed health measurements or gate differ from returned snapshot')
    require(result['counts'] == {'PASS': 1920} and result['comparison_counts'] == {'PASS': 319, 'REVIEW': 1},
            'Unexpected quality outcome')
    reviews = [c for c in result['comparisons'] if c['status'] != 'PASS']
    write('review_conditions.json', reviews)
    groups = []
    for layer in ('conv1', 'conv3', 'linear', 'ffn'):
        for dtype in config['dtypes']:
            selected = [c for c in result['comparisons'] if c['layer'] == layer and c['dtype'] == dtype and c['status'] == 'PASS']
            ratios = [c['ratio'] for c in selected]
            groups.append(dict(layer=layer, dtype=dtype, pass_comparisons=len(selected),
                median_ratio=statistics.median(ratios), min_ratio=min(ratios), max_ratio=max(ratios),
                standard_faster_count=sum(v > 1 for v in ratios), decomposed_faster_count=sum(v < 1 for v in ratios),
                within_5pct_count=sum(abs(v-1) <= .05 for v in ratios),
                median_standard_us=1000*statistics.median(c['standard_ms'] for c in selected),
                median_decomposed_us=1000*statistics.median(c['decomposed_ms'] for c in selected),
                median_difference_us=1000*statistics.median(c['difference_ms'] for c in selected),
                median_extra_peak_ratio=statistics.median(c['decomposed_extra_peak_bytes']/c['standard_extra_peak_bytes'] for c in selected)))
    write('group_summary.json', groups)
    with (HERE / 'group_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader(); writer.writerows(groups)
    for name, wanted in hashes.items():
        require(sha(RAW / name) == wanted, 'Raw data changed during analysis: ' + name)
    sensor_checks = dict(samples=len(rows), sm_mhz=sorted({r['clocks.sm'] for r in rows}),
        memory_mhz=sorted({r['clocks.mem'] for r in rows}),
        power_limit_w=sorted({r['power.limit'] for r in rows}),
        temperature_c=[min(r['temperature.gpu'] for r in rows), max(r['temperature.gpu'] for r in rows)],
        power_w=[min(r['power.draw'] for r in rows), max(r['power.draw'] for r in rows)],
        sw_power_cap_samples=sum(r['clocks_event_reasons.sw_power_cap'] for r in rows),
        thermal_event_samples=sum(r['clocks_event_reasons.sw_thermal_slowdown'] or r['clocks_event_reasons.hw_thermal_slowdown'] for r in rows),
        pre_measurement_memory_deviations=memory_deviations, first_settle_utc=first_settle,
        full_scan_window_cross_checks=32, main_windows_checked=len(all_blocks), **{k:v for k,v in sensor_info.items() if k != 'samples'})
    write('verification.json', dict(offline_verification='PASS', archive=ARCHIVE.name, archive_sha256=ARCHIVE_SHA,
        archive_crc='PASS', archive_files=archive_files, result_hashes_checked=len(hashes),
        embedded_source_hashes_checked=len(manifest['source_hashes']), raw_preservation='PASS',
        expected_measurements=len(expected), recorded_measurements=len(records), raw_timing_recalculation='PASS',
        correctness_jsons_checked=len(checked_correctness), correctness_maxima=max_errors,
        correctness_scope='Stored GPU checks: full pair and finite; sampled CPU FP64 reference. No new execution.',
        identity_files=320, environment_signature_match=True, independent_sensor_checks=sensor_checks,
        exact_summary_reproduction=True, exact_csv_reproduction=True, health_measurement_reproduction=True,
        collection=result['collection'], gate=result['gate'], measurement_counts=result['counts'],
        comparison_counts=result['comparison_counts'], restoration=result['restoration'],
        finished=read(RAW / 'worker/complete.json')))
    print('VERIFY PASS: 1920 measurements; 319 PASS comparisons; 1 REVIEW comparison', flush=True)
    for group in groups:
        print(f"{group['layer']} {group['dtype']} n={group['pass_comparisons']} median_ratio={group['median_ratio']:.6f}")


if __name__ == '__main__':
    main()
