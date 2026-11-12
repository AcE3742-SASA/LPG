"""Validate clock steps while keeping timing, sampled clocks and counters distinct."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import statistics as st

import steps


def elapsed(start, end):
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def range_of(samples, field):
    values = [r[field] for r in samples if r[field] is not None]
    return [min(values), max(values)] if values else None


def counter_summary(samples, telemetry):
    flags, counters = [], {}
    for counter, reason in zip(telemetry.COUNTERS, telemetry.REASONS):
        values = [r[counter] for r in samples]
        active = sum(r[reason] is True for r in samples)
        unknown = sum(r[reason] is None for r in samples)
        if None in values:
            delta, decreases = None, None
            flags.append(counter + ':unavailable')
        else:
            decreases = sum(b < a for a, b in zip(values, values[1:]))
            delta = values[-1] - values[0] if not decreases else None
            if decreases:
                flags.append(counter + ':decreased')
            if delta and active == 0:
                flags.append(counter + ':increase_without_active_sample')
        if unknown:
            flags.append(reason + ':unavailable')
        counters[counter] = dict(first=values[0], last=values[-1], delta_us=delta,
                                 decreases=decreases, active_samples=active, unknown_samples=unknown)
    return dict(samples=len(samples), span_s=elapsed(samples[0]['utc'], samples[-1]['utc']),
                counters=counters, flags=flags,
                note='First-to-last sample span only; not exact block throttling fractions or continuous status.')


def q_counters(snapshot):
    labels = ('SW Power Capping', 'SW Thermal Slowdown', 'HW Thermal Slowdown')
    text = snapshot.get('stdout', '')
    values = {}
    for label in labels:
        match = re.search(r'^\s*' + re.escape(label) + r'\s*:\s*(\d+) us', text, re.M)
        values[label] = int(match.group(1)) if match else None
    return values


def phase_data(directory, interval, gpu, telemetry):
    meta = steps.read(directory / 'telemetry_metadata.json')
    if meta['fields'] != list(telemetry.FIELDS) or meta['requested_interval_ms'] != interval:
        raise ValueError('Telemetry protocol mismatch: ' + str(directory))
    samples, cadence = telemetry.read(directory / 'telemetry.csv', meta['timezone_offset_seconds'], interval)
    if any(r['uuid'] != gpu for r in samples):
        raise ValueError('Wrong telemetry GPU')
    flags = []
    if cadence['unavailable_fields']:
        flags.append('telemetry_fields_unavailable')
    if cadence['observed_interval_seconds_max'] > max(.5, interval*.003):
        flags.append('telemetry_sampling_gap')
    if meta.get('process_exit_code') not in (0, -15):
        flags.append('telemetry_exit_code')
    result = counter_summary(samples, telemetry)
    result.update(cadence=cadence, ranges={key:range_of(samples, key) for key in
                  ('clocks.sm', 'clocks.mem', 'power.limit', 'power.draw', 'temperature.gpu')})
    result['flags'].extend(flags)
    return samples, result


def add_q_comparison(info, before, after):
    a, b = q_counters(before), q_counters(after)
    deltas = {}
    for key in a:
        deltas[key] = b[key]-a[key] if a[key] is not None and b[key] is not None and b[key] >= a[key] else None
        if deltas[key] is None:
            info['flags'].append('q_counter_unavailable_or_decreased:' + key)
    info['q_before_counters_us'], info['q_after_counters_us'] = a, b
    info['q_counter_deltas_us'] = deltas
    info['q_note'] = 'Separate snapshots bracket a slightly wider interval than the telemetry sample span.'


def analyze_stage(out, stage, loaded):
    _, audit, original, telemetry, baseline, _, config = loaded
    directory = out / stage['name']
    env = steps.read(directory / 'environment.json')
    if {key:env[key] for key in baseline['environment_signature']} != baseline['environment_signature']:
        raise ValueError('Environment differs from v6: ' + stage['name'])
    rows = steps.read(directory / 'results.json')
    if [row['task'] for row in rows] != steps.cases(config):
        raise ValueError('Missing, duplicate or changed tasks: ' + stage['name'])
    samples, info = phase_data(directory, stage['interval_ms'], env['gpu_uuid'], telemetry)
    add_q_comparison(info, env['nvidia_smi'], steps.read(directory / 'gpu_after.json'))
    stage_flags = list(info['flags'])
    targets = {'clocks.sm':stage['sm_mhz'], 'clocks.mem':config['memory_mhz'], 'power.limit':config['power_limit_w']}
    for field, wanted in targets.items():
        if any(row[field] != wanted for row in samples):
            stage_flags.append('worker_observed_mismatch:' + field)
    measurements, windows, groups = [], [], defaultdict(list)
    with steps.configured_audit(audit, config):
        for row in rows:
            task, result = row['task'], row['result']
            flags = original.check_timing(result, task, baseline)
            correct = result['correctness']
            if correct.get('scope') != 'sampled' or correct.get('points') != 256:
                raise ValueError('Changed correctness scope: ' + task['id'])
            inside = []
            for index, block in enumerate(result['timing']['blocks']):
                window = telemetry.window(samples, block['host_envelope_start_utc'], block['host_envelope_end_utc'])
                windows.append(dict(stage=stage['name'], task_id=task['id'], block=index, window=window))
                if not window['inside']:
                    flags.append(f'block_{index}:no_inside_clock_sample')
                inside.extend(window['inside'])
            for field, wanted in targets.items():
                if any(r[field] != wanted for r in inside):
                    flags.append('observed_mismatch:' + field)
            for reason in telemetry.REASONS:
                if any(r[reason] is True for r in inside):
                    flags.append('observed_active:' + reason)
                if any(r[reason] is None for r in inside):
                    flags.append('unavailable:' + reason)
            flags = sorted(set(flags))
            median = result['timing']['median_ms']*1000
            group_key = (tuple(task['shape']), task['dtype'], task['mode'])
            groups[group_key].append(median)
            measurements.append(dict(stage=stage['name'], sm_mhz=stage['sm_mhz'], interval_ms=stage['interval_ms'],
                task=task, median_us=median, relative_iqr=result['timing']['relative_iqr'],
                first_last_drift=result['timing']['first_last_drift'], inside_samples=len(inside),
                observed_values={key:sorted({r[key] for r in inside}) for key in targets},
                correctness=correct, flags=flags, status='REVIEW' if flags else 'PASS'))
    rounds = []
    for (shape, dtype, mode), values in groups.items():
        spread = (max(values)-min(values))/st.median(values)
        rounds.append(dict(shape=shape, dtype=dtype, mode=mode, round_us=values, relative_range=spread))
        if spread > baseline['protocol']['between_round_limit']:
            stage_flags.append(f'{shape}/{dtype}/{mode}:round_range_above_5pct')
    stage_flags.extend(item['task']['id'] + ':' + flag for item in measurements for flag in item['flags'])
    clock_timing_flags = [f for f in stage_flags if not f.startswith(('clocks_event_reasons_counters.', 'q_counter_'))]
    result = dict(stage=stage, gate='REVIEW' if stage_flags else 'PASS', flags=sorted(set(stage_flags)),
                  timing_and_sampled_clocks_pass=not clock_timing_flags, counters=info, rounds=rounds,
                  interpretation='No final clock is selected; sampled agreement is not continuous verification.')
    return result, measurements, windows


def compare_measurements(measurements, config):
    by_stage = defaultdict(dict)
    for stage in config['stages']:
        groups = defaultdict(list)
        for row in measurements:
            if row['stage'] == stage['name']:
                task = row['task']
                groups[(tuple(task['shape']), task['dtype'], task['mode'])].append(row['median_us'])
        by_stage[stage['name']] = {key:st.median(values) for key, values in groups.items()}
    comparisons = []
    pairs = [('polling_before', 's0_900_100_before', 's1_900_50_before'),
             ('polling_after', 's6_900_100_after', 's5_900_50_after'),
             ('repeat_900_50', 's1_900_50_before', 's5_900_50_after'),
             ('repeat_900_100', 's0_900_100_before', 's6_900_100_after')]
    for label, first, second in pairs:
        for key, a in by_stage[first].items():
            b = by_stage[second][key]
            comparisons.append(dict(comparison=label, first_stage=first, second_stage=second,
                shape=key[0], dtype=key[1], mode=key[2], first_us=a, second_us=b,
                second_over_first_minus_one=b/a-1, relative_range=abs(b-a)/st.median([a,b]),
                meaning='Includes process and time/order effects; not an isolated causal effect.'))
    return comparisons


def analyze(out, loaded):
    _, _, _, telemetry, baseline, suite, config = loaded
    manifest = steps.read(out / 'manifest.json')
    if manifest['config'] != config or manifest['baseline'] != baseline or manifest['cases'] != steps.cases(config):
        raise ValueError('Manifest/protocol differs')
    if manifest['source_hashes'] != steps.own_hashes() or manifest['dependency_hashes'] != steps.read(steps.HERE/'dependency_hashes.json'):
        raise ValueError('Source manifest differs')
    for folder, hashes in [('source', manifest['source_hashes']), ('dependencies', manifest['dependency_hashes'])]:
        for name, expected in hashes.items():
            if steps.digest(out / folder / name) != expected:
                raise ValueError('Source snapshot differs: ' + name)
    stage_results, measurements, windows, idle_results, flags = [], [], [], [], []
    for stage in config['stages']:
        if (out / (stage['name']+'.exit_code.txt')).read_text().strip() != '0':
            raise ValueError('Worker failed: ' + stage['name'])
        result, rows, observed = analyze_stage(out, stage, loaded)
        stage_results.append(result)
        measurements.extend(rows)
        windows.extend(observed)
        flags.extend(stage['name'] + ':' + f for f in result['flags'])
    idle_specs = [('auto_before', None), ('auto_after', None)]
    idle_specs += [(s['name']+'_'+when, s['sm_mhz']) for s in config['stages'] for when in ('before','after')]
    for name, sm_mhz in idle_specs:
        directory = out/'idle'/name
        phase = steps.read(directory/'phase.json')
        if phase['name'] != name or phase['sm_mhz'] != sm_mhz or phase['seconds'] != config['idle_seconds'] or phase['memory_mhz'] != config['memory_mhz']:
            raise ValueError('Idle phase differs: ' + name)
        samples, info = phase_data(directory, config['idle_interval_ms'], baseline['environment_signature']['gpu_uuid'], telemetry)
        add_q_comparison(info, steps.read(directory/'gpu_before.json'), steps.read(directory/'gpu_after.json'))
        for key, target in [('clocks.mem', config['memory_mhz']), ('power.limit', config['power_limit_w'])]:
            if any(r[key] != target for r in samples):
                info['flags'].append('idle_mismatch:' + key)
        if sm_mhz is not None and any(r['clocks.sm'] != sm_mhz for r in samples):
            info['flags'].append('idle_mismatch:clocks.sm')
        idle_results.append(dict(phase=phase, telemetry=info))
        flags.extend('idle/' + name + ':' + f for f in info['flags'])
    comparisons = compare_measurements(measurements, config)
    for comparison in comparisons:
        if comparison['relative_range'] > baseline['protocol']['between_round_limit']:
            flags.append(comparison['comparison'] + ':' + str(comparison['shape']) + '/' + comparison['dtype'] + '/' + comparison['mode'] + ':above_5pct')
    restore = steps.read(out/'restore.json')
    if not restore['commands_accepted']:
        flags.append('restoration_failed')
    counts = dict(Counter(row['status'] for row in measurements))
    summary = dict(gate='FAIL' if not restore['commands_accepted'] else ('REVIEW' if flags else 'PASS'),
                   expected=manifest['expected_measurements'], recorded=len(measurements),
                   missing=manifest['expected_measurements']-len(measurements), counts=counts,
                   timed_blocks=len(windows), idle_phases=len(idle_results), flags=sorted(set(flags)),
                   restoration='COMMANDS_ACCEPTED' if restore['commands_accepted'] else 'FAILED',
                   full_sweep='HOLD', selected_clock_mhz=None, interpretation='HUMAN_REVIEW_REQUIRED')
    suite.atomic_json(out/'analysis.json', dict(summary=summary, stages=stage_results, idle=idle_results,
        measurements=measurements, comparisons=comparisons,
        limitations=['Four development conditions only; not full-sweep or all-shape certification.',
                     'Higher candidates have one process each; before/after replication is at 900 MHz.',
                     'No new automatic-core GEMM load control; automatic-core phases have no diagnostic worker.',
                     'Polling request rate does not establish sensor resolution or zero measurement overhead.',
                     'Idle means no diagnostic GEMM worker; desktop GPU clients remain recorded.',
                     'Counter growth with no active sample remains unresolved, not labeled a driver bug.']))
    suite.atomic_json(out/'clock_windows.json', windows)
    lines = ['# Clock steps result', '', 'Overall: '+summary['gate']+'; full: HOLD; no final clock selected.', '',
             '| stage | timing and sampled clocks | status | maximum round range |', '|---|---|---|---:|']
    for result in stage_results:
        lines.append(f"| {result['stage']['name']} | {'PASS' if result['timing_and_sampled_clocks_pass'] else 'REVIEW'} | {result['gate']} | {max(r['relative_range'] for r in result['rounds']):.3%} |")
    lines += ['', 'Counter details and polling comparisons are in analysis.json.',
              'REVIEW flags are retained; no interpolation or removal of missing samples.', '', 'Flags:']
    lines.extend('- '+flag for flag in summary['flags'])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary
