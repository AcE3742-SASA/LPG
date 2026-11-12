"""Keep observed clock gates, timing gates and counter limitations separate."""
from collections import Counter, defaultdict
import statistics as st

import lower


def analyze_stage(out, stage, loaded):
    _, _, _, original, telemetry, baseline, suite, config = loaded
    from analyze_steps import phase_data, add_q_comparison
    name = stage['name']
    directory = out/name
    if (out/(name+'.exit_code.txt')).read_text().strip() != '0':
        raise ValueError('Worker did not complete: '+name)
    env = lower.read(directory/'environment.json')
    if suite.signature(env) != baseline['environment_signature']:
        raise ValueError('Environment differs from v6')
    records = lower.read(directory/'results.json')
    if [r['task'] for r in records] != lower.cases(config):
        raise ValueError('Missing, duplicate, reordered or changed tasks')
    samples, info = phase_data(directory, stage['interval_ms'], env['gpu_uuid'], telemetry)
    add_q_comparison(info, env['nvidia_smi'], lower.read(directory/'gpu_after.json'))
    flags = list(info['flags'])
    target = {'clocks.sm':stage['sm_mhz'], 'clocks.mem':config['memory_mhz'], 'power.limit':config['power_limit_w']}
    for key, wanted in target.items():
        if any(s[key] != wanted for s in samples):
            flags.append('worker_observed_mismatch:'+key)
    for key in telemetry.REASONS:
        if any(s[key] is True for s in samples):
            flags.append('worker_observed_active:'+key)
    measurements, windows, anomalies = [], [], []
    groups, triplets = defaultdict(list), defaultdict(dict)
    specs = {(tuple(s['shape']),s['dtype']):s for s in config['fixed_cases']}
    with lower.configured_worker(loaded):
        for row in records:
            task, result = row['task'], row['result']
            local = original.check_timing(result, task, baseline)
            correct, timing = result['correctness'], result['timing']
            spec = specs[(tuple(task['shape']),task['dtype'])]
            if (correct.get('scope'),correct.get('points')) != (spec['correctness_scope'],spec['correctness_points']):
                raise ValueError('Correctness scope changed: '+task['id'])
            inside = []
            for index, block in enumerate(timing['blocks']):
                window = telemetry.window(samples, block['host_envelope_start_utc'], block['host_envelope_end_utc'])
                windows.append(dict(stage=name, task_id=task['id'], block=index, window=window))
                inside.extend(window['inside'])
                if not window['inside']:
                    local.append(f'block_{index}:no_inside_clock_sample')
                for counter, reason in zip(telemetry.COUNTERS, telemetry.REASONS):
                    delta = window['counter_deltas_us'][counter]
                    span = window['counter_interval_seconds']
                    issues = []
                    if delta and all(s[reason] is False for s in window['inside']):
                        issues.append('increase_without_active_inside_sample')
                    if delta is not None and span is not None and delta > span*1e6+1:
                        issues.append('delta_exceeds_inside_sample_span')
                    if issues:
                        anomalies.append(dict(stage=name,task_id=task['id'],block=index,counter=counter,
                                              delta_us=delta,span_s=span,issues=issues))
                        flags.extend(counter+':block_'+issue for issue in issues)
            for key, wanted in target.items():
                if any(s[key] != wanted for s in inside):
                    local.append('observed_mismatch:'+key)
            for key in telemetry.REASONS:
                if any(s[key] is True for s in inside):
                    local.append('observed_active:'+key)
                if any(s[key] is None for s in inside):
                    local.append('unavailable:'+key)
            local = sorted(set(local))
            measured = dict(stage=name, task=task, median_us=timing['median_ms']*1000,
                relative_iqr=timing['relative_iqr'],first_last_drift=timing['first_last_drift'],
                correctness=correct,graph_nodes=timing.get('graph_nodes'),inside_samples=len(inside),
                observed_values={key:sorted({s[key] for s in inside if s[key] is not None}) for key in target},
                flags=local,status='REVIEW' if local else 'PASS')
            measurements.append(measured)
            flags.extend(task['id']+':'+f for f in local)
            groups[(tuple(task['shape']),task['dtype'],task['mode'])].append(measured)
            triplets[(tuple(task['shape']),task['dtype'],task['round'])][task['mode']] = measured
    rounds, round_flagged = [], set()
    for (shape,dtype,mode), members in groups.items():
        if len(members) != 2:
            raise ValueError('Expected two rounds')
        values = [r['median_us'] for r in members]
        spread = (max(values)-min(values))/st.median(values)
        rounds.append(dict(stage=name,shape=shape,dtype=dtype,mode=mode,round_us=values,relative_range=spread))
        if spread > baseline['protocol']['between_round_limit']:
            flags.append(f'{shape}/{dtype}/{mode}:round_range_above_5pct')
            round_flagged.update(r['task']['id'] for r in members)
    paired = []
    for (shape,dtype,rd), modes in triplets.items():
        if set(modes) != set(config['modes']):
            raise ValueError('Missing execution mode')
        values = {m:modes[m]['median_us'] for m in config['modes']}
        pair = dict(stage=name,shape=shape,dtype=dtype,round=rd,input_seed=specs[(shape,dtype)]['input_seed'],
                    graph_nodes=modes['graph']['graph_nodes'],
                    **{m+'_us':values[m] for m in config['modes']},
                    allocate_over_prealloc=values['allocate']/values['prealloc'],
                    graph_over_prealloc=values['graph']/values['prealloc'])
        for mode, measured in modes.items():
            pair[mode+'_status'] = measured['status']
            pair[mode+'_round_review'] = measured['task']['id'] in round_flagged
        paired.append(pair)
    for measured in measurements:
        measured['between_round_review'] = measured['task']['id'] in round_flagged
    flags = sorted(set(flags))
    clock_timing = [f for f in flags if not f.startswith(('clocks_event_reasons_counters.','q_counter_'))]
    result = dict(stage=stage,gate='REVIEW' if flags else 'PASS',flags=flags,
        timing_and_sampled_clocks_pass=not clock_timing,counts=dict(Counter(r['status'] for r in measurements)),
        telemetry=info,rounds=rounds,max_relative_iqr=max(r['relative_iqr'] for r in measurements),
        max_abs_drift=max(abs(r['first_last_drift']) for r in measurements),
        max_between_round_range=max(r['relative_range'] for r in rounds),
        counter_anomaly_blocks=len({(a['task_id'],a['block']) for a in anomalies}))
    return result, measurements, windows, paired, anomalies


def analyze(out, loaded):
    _, _, _, _, _, baseline, suite, config = loaded
    from report import csv_write
    manifest = lower.read(out/'manifest.json')
    if (manifest['config'],manifest['baseline'],manifest['cases']) != (config,baseline,lower.cases(config)):
        raise ValueError('Manifest/protocol changed')
    if manifest['source_hashes'] != lower.own_hashes() or manifest['dependency_hashes'] != lower.read(lower.HERE/'dependency_hashes.json'):
        raise ValueError('Source manifest changed')
    for folder, hashes in [('source',manifest['source_hashes']),('dependencies',manifest['dependency_hashes'])]:
        for name, wanted in hashes.items():
            if lower.digest(out/folder/name) != wanted:
                raise ValueError('Source snapshot changed: '+name)
    stages, measurements, windows, paired, anomalies, flags = [], [], [], [], [], []
    for stage in config['stages']:
        result, rows, ws, ps, cs = analyze_stage(out,stage,loaded)
        stages.append(result); measurements.extend(rows); windows.extend(ws); paired.extend(ps); anomalies.extend(cs)
        flags.extend(stage['name']+':'+f for f in result['flags'])
    first, last = stages[0], stages[-1]
    indexed = {(tuple(r['shape']),r['dtype'],r['mode']):r for r in first['rounds']}
    repeats = []
    for row in last['rounds']:
        old = indexed[(tuple(row['shape']),row['dtype'],row['mode'])]
        a,b = st.median(old['round_us']),st.median(row['round_us'])
        spread = abs(b-a)/st.median([a,b])
        repeats.append(dict(shape=row['shape'],dtype=row['dtype'],mode=row['mode'],before_us=a,after_us=b,
            after_over_before_minus_one=b/a-1,relative_range=spread,
            review=spread>baseline['protocol']['between_round_limit']))
        if repeats[-1]['review']:
            flags.append(f"repeat_780/{row['shape']}/{row['dtype']}/{row['mode']}:range_above_5pct")
    restored = lower.read(out/'restore.json')['commands_accepted']
    if not restored:
        flags.append('restoration_failed')
    expected = len(lower.cases(config))*len(config['stages'])
    expected_blocks = expected*baseline['protocol']['blocks']
    expected_triplets = expected//len(config['modes'])
    if (len(measurements),len(windows),len(paired)) != (expected,expected_blocks,expected_triplets):
        raise ValueError('Incomplete result cardinality')
    summary = dict(gate='FAIL' if not restored else ('REVIEW' if flags else 'PASS'),
        expected=expected,recorded=len(measurements),missing=expected-len(measurements),timed_blocks=len(windows),
        mode_triplets=len(paired),counts=dict(Counter(r['status'] for r in measurements)),
        correctness_scopes=dict(Counter(r['correctness']['scope'] for r in measurements)),
        stage_timing_and_sampled_clocks={r['stage']['name']:r['timing_and_sampled_clocks_pass'] for r in stages},
        repeat_780_time_pass=not any(r['review'] for r in repeats),
        counter_anomaly_blocks=len({(a['stage'],a['task_id'],a['block']) for a in anomalies}),
        historical_counter_issue='UNRESOLVED; per-stage and per-block observations retained',
        restoration='COMMANDS_ACCEPTED' if restored else 'FAILED',flags=sorted(set(flags)),
        full_sweep='HOLD',selected_clock_mhz=None,reference_clock_mhz=config['reference_clock_mhz'],
        validated_for_full=False,
        interpretation='Screened conditions only. Review before a broader pilot. Mode ratios are not causal overhead estimates.')
    suite.atomic_json(out/'analysis.json',dict(summary=summary,stages=stages,measurements=measurements,
                                             repeat_780=repeats,counter_anomalies=anomalies))
    suite.atomic_json(out/'clock_windows.json',windows)
    flat = [dict(**r['task'],**{k:v for k,v in r.items() if k!='task'}) for r in measurements]
    csv_write(out/'measurements.csv',flat)
    csv_write(out/'review_queue.csv',[r for r in flat if r['status']!='PASS' or r['between_round_review']])
    csv_write(out/'paired_metrics.csv',paired)
    csv_write(out/'repeat_780.csv',repeats)
    lines = ['# Core clock candidate comparison','',f"Overall: {summary['gate']}; full remains HOLD.",
             f'{expected} measurements, {expected_blocks} blocks, {expected_triplets} matched three-mode groups expected.',
             '780 MHz reference; no replacement clock selected. Counter interpretation remains unresolved.','',
             '| Stage | Timing and sampled clocks | Max IQR % | Max drift % | Max round range % |',
             '|---|---|---:|---:|---:|']
    for s in stages:
        lines.append(f"| {s['stage']['name']} | {'PASS' if s['timing_and_sampled_clocks_pass'] else 'REVIEW'} | "
                     f"{100*s['max_relative_iqr']:.4f} | {100*s['max_abs_drift']:.4f} | {100*s['max_between_round_range']:.4f} |")
    lines += ['', '780 MHz before/after timing: '+('PASS' if summary['repeat_780_time_pass'] else 'REVIEW'),
              'Sampling is not continuous verification. Host envelopes are not exact GPU event boundaries.',
              'Allocate uses the existing caching allocator; it does not force a fresh cudaMalloc per call.',
              'Separate processes and order effects remain. No mode is automatically omitted from full collection.','',
              'Flags:']+['- '+f for f in summary['flags']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary
