"""Keep timing gates, sampled clock gates and unresolved counters separately visible."""
from collections import Counter, defaultdict
import statistics as st

import fixed


def analyze(out, loaded):
    _, _, audit, original, telemetry, baseline, suite, config, tasks = loaded
    from analyze_steps import phase_data, add_q_comparison
    from report import csv_write
    manifest = fixed.read(out/'manifest.json')
    if (manifest['config'], manifest['baseline'], manifest['cases']) != (config, baseline, tasks):
        raise ValueError('Manifest/protocol changed')
    if manifest['source_hashes'] != fixed.own_hashes() or manifest['dependency_hashes'] != fixed.read(fixed.HERE/'dependency_hashes.json'):
        raise ValueError('Source manifest changed')
    for folder, hashes in [('source', manifest['source_hashes']), ('dependencies', manifest['dependency_hashes'])]:
        for name, expected in hashes.items():
            if fixed.digest(out/folder/name) != expected:
                raise ValueError('Source snapshot changed: '+name)
    if (out/'worker.exit_code.txt').read_text().strip() != '0':
        raise ValueError('Worker did not complete')
    directory = out/'pilot'
    env = fixed.read(directory/'environment.json')
    if suite.signature(env) != baseline['environment_signature']:
        raise ValueError('Environment differs from v6')
    rows = fixed.read(directory/'results.json')
    if [r['task'] for r in rows] != tasks:
        raise ValueError('Missing, duplicate, reordered or changed tasks')
    samples, info = phase_data(directory, config['interval_ms'], env['gpu_uuid'], telemetry)
    add_q_comparison(info, env['nvidia_smi'], fixed.read(directory/'gpu_after.json'))
    flags = list(info['flags'])
    target = {'clocks.sm':config['sm_mhz'], 'clocks.mem':config['memory_mhz'], 'power.limit':config['power_limit_w']}
    for key, wanted in target.items():
        if any(s[key] != wanted for s in samples):
            flags.append('worker_observed_mismatch:'+key)
    for key in telemetry.REASONS:
        if any(s[key] is True for s in samples):
            flags.append('worker_observed_active:'+key)
    measurements, windows = [], []
    groups, pairs = defaultdict(list), defaultdict(dict)
    with fixed.configured_worker(loaded):
        for row in rows:
            task, result = row['task'], row['result']
            local = original.check_timing(result, task, baseline)
            spec = next(s for s in config['fixed_cases'] if s['shape']==task['shape'] and s['dtype']==task['dtype'])
            correct = result['correctness']
            if (correct.get('scope'), correct.get('points')) != (spec['correctness_scope'], spec['correctness_points']):
                raise ValueError('Correctness scope changed: '+task['id'])
            inside = []
            for index, block in enumerate(result['timing']['blocks']):
                window = telemetry.window(samples, block['host_envelope_start_utc'], block['host_envelope_end_utc'])
                windows.append(dict(task_id=task['id'], block=index, window=window))
                inside.extend(window['inside'])
                if not window['inside']:
                    local.append(f'block_{index}:no_inside_clock_sample')
            for key, wanted in target.items():
                if any(s[key] != wanted for s in inside):
                    local.append('observed_mismatch:'+key)
            for key in telemetry.REASONS:
                if any(s[key] is True for s in inside):
                    local.append('observed_active:'+key)
                if any(s[key] is None for s in inside):
                    local.append('unavailable:'+key)
            local = sorted(set(local))
            timing = result['timing']
            measured = dict(task=task, median_us=timing['median_ms']*1000,
                relative_iqr=timing['relative_iqr'], first_last_drift=timing['first_last_drift'],
                correctness=correct, graph_nodes=timing.get('graph_nodes'), inside_samples=len(inside),
                observed_values={key:sorted({s[key] for s in inside}) for key in target},
                flags=local, status='REVIEW' if local else 'PASS')
            measurements.append(measured)
            flags.extend(task['id']+':'+f for f in local)
            groups[(tuple(task['shape']),task['dtype'],task['mode'])].append(measured)
            pairs[(tuple(task['shape']),task['dtype'],task['round'])][task['mode']] = measured
    rounds = []
    round_flagged = set()
    for (shape, dtype, mode), members in groups.items():
        if len(members) != 2:
            raise ValueError('Expected two rounds')
        values = [r['median_us'] for r in members]
        spread = (max(values)-min(values))/st.median(values)
        rounds.append(dict(shape=shape, dtype=dtype, mode=mode, round_us=values, relative_range=spread))
        if spread > baseline['protocol']['between_round_limit']:
            flags.append(f'{shape}/{dtype}/{mode}:round_range_above_5pct')
            round_flagged.update(r['task']['id'] for r in members)
    paired = []
    for (shape, dtype, rd), modes in pairs.items():
        if set(modes) != {'prealloc','graph'}:
            raise ValueError('Missing execution mode')
        p, g = modes['prealloc'], modes['graph']
        paired.append(dict(shape=shape, dtype=dtype, round=rd, input_seed=next(
            s['input_seed'] for s in config['fixed_cases'] if tuple(s['shape'])==shape and s['dtype']==dtype),
            prealloc_us=p['median_us'], graph_us=g['median_us'], graph_nodes=g['graph_nodes'],
            graph_over_prealloc=g['median_us']/p['median_us'],
            prealloc_status=p['status'], graph_status=g['status'],
            prealloc_round_review=p['task']['id'] in round_flagged,
            graph_round_review=g['task']['id'] in round_flagged))
    restore = fixed.read(out/'restore.json')
    if not restore['commands_accepted']:
        flags.append('restoration_failed')
    flags = sorted(set(flags))
    clock_timing_flags = [f for f in flags if not f.startswith(('clocks_event_reasons_counters.', 'q_counter_'))]
    summary = dict(gate='FAIL' if not restore['commands_accepted'] else ('REVIEW' if flags else 'PASS'),
        expected=192, recorded=len(measurements), missing=192-len(measurements), timed_blocks=len(windows),
        paired_expected=96, paired_complete=len(paired), counts=dict(Counter(r['status'] for r in measurements)),
        correctness_scopes=dict(Counter(r['correctness']['scope'] for r in measurements)),
        timing_and_sampled_clocks_pass=not clock_timing_flags, flags=flags,
        restoration='COMMANDS_ACCEPTED' if restore['commands_accepted'] else 'FAILED',
        target=target, polling_ms=config['interval_ms'], full_sweep='HOLD', selected_clock_mhz=None,
        interpretation='HUMAN_REVIEW_REQUIRED; counter issue is retained separately, not waived')
    suite.atomic_json(out/'analysis.json', dict(summary=summary, measurements=measurements, rounds=rounds, telemetry=info,
        limitations=['Same-process rounds are not independent sessions.',
                     'Sampling is not continuous observation; counters are not GEMM slowdown fractions.',
                     'Fixed-clock performance is a separate condition from automatic-core results.',
                     'No new automatic-core load control or profiler run.']))
    suite.atomic_json(out/'clock_windows.json', windows)
    csv_write(out/'paired_metrics.csv', paired)
    flat = [dict(**r['task'], **{k:v for k,v in r.items() if k!='task'},
                 between_round_review=r['task']['id'] in round_flagged) for r in measurements]
    csv_write(out/'measurements.csv', flat)
    csv_write(out/'review_queue.csv', [r for r in flat if r['status']!='PASS' or r['between_round_review']])
    lines = ['# Fixed clock pilot', '', 'Overall: '+summary['gate']+'; full remains HOLD.', '',
        f"Measurements {len(measurements)}/192; blocks {len(windows)}/1344; pairs {len(paired)}/96.",
        'Timing and sampled clocks: '+('PASS' if summary['timing_and_sampled_clocks_pass'] else 'REVIEW'),
        'Correctness scopes: '+str(summary['correctness_scopes']), '',
        '900/6801 MHz requested; 70 W existing limit; 50 ms polling. No final clock selected.',
        'Counter inconsistency remains unresolved. No conversion to a short-window throttling fraction.',
        '', 'Flags:'] + ['- '+f for f in flags]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary
