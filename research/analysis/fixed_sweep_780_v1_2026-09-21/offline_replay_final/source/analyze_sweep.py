"""Bounded-memory verification and three-mode export; raw SQLite/telemetry are read-only."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import contextlib
import csv
import datetime as dt
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import statistics as st

from run import history, read, sha, choices_and_done, manifest_for, MODES


def segment_quality(directory, rows, loaded):
    telemetry, protocol = loaded['telemetry'], loaded['plan'].PROTOCOL
    flags, measurements, counter_blocks = [], {}, 0
    target = {'clocks.sm':780, 'clocks.mem':6801, 'power.limit':70}
    try:
        meta = read(directory/'telemetry_metadata.json')
        if meta['fields']!=list(telemetry.FIELDS) or meta['requested_interval_ms']!=50:
            raise ValueError('Telemetry configuration mismatch')
        samples,cadence = telemetry.read(directory/'telemetry.csv',meta['timezone_offset_seconds'],50)
        gpu = loaded['baseline']['environment_signature']['gpu_uuid']
        if any(s['uuid']!=gpu for s in samples):
            raise ValueError('Telemetry GPU mismatch')
        if cadence['observed_interval_seconds_max']>.5:
            flags.append('telemetry_sampling_gap')
        if cadence['unavailable_fields']:
            flags.append('telemetry_unavailable_fields')
        for key,value in target.items():
            if any(s[key]!=value for s in samples):
                flags.append('worker_mismatch:'+key)
        for reason in telemetry.REASONS:
            if any(s[reason] is True for s in samples):
                flags.append('worker_active:'+reason)
        times = [dt.datetime.fromisoformat(s['utc']).timestamp() for s in samples]
    except Exception as e:
        samples,times,cadence = [],[],dict(error=str(e))
        flags.append('telemetry_invalid_or_missing')
    counter_info = {}
    for counter,reason in zip(telemetry.COUNTERS,telemetry.REASONS):
        values=[s[counter] for s in samples if s[counter] is not None]
        counter_info[counter]=dict(delta_us=values[-1]-values[0] if len(values)>1 else None,
            decreases=sum(b<a for a,b in zip(values,values[1:])),
            active_samples=sum(s[reason] is True for s in samples),
            increase_without_active=bool(len(values)>1 and values[-1]>values[0]
                                         and not any(s[reason] is True for s in samples)))
    for row in rows:
        task=row['task']; local=[]; windows=[]; fatal=False
        try:
            result=row
            if row['status']=='FAIL':
                raise ValueError(row.get('error','Stored correctness failure'))
            m,n,_=task['shape']
            scope,points=('full',m*n) if m*n<=4096 else ('sampled',min(m,16)*min(n,16))
            correct=result['correctness']
            if (correct['scope'],correct['points'])!=(scope,points):
                raise ValueError('Correctness scope changed')
            if (correct['atol'],correct['rtol'])!=tuple(protocol['tolerances'][task['dtype']]):
                raise ValueError('Correctness tolerances changed')
            nodes=result['timing'].get('graph_nodes') if task['mode']=='graph' else None
            if nodes is not None and (type(nodes) is not int or not 1<=nodes<=64):
                raise ValueError('Invalid Graph batch')
            baseline=dict(protocol=protocol,fixed_cases=[dict(shape=task['shape'],dtype=task['dtype'],
                input_seed=loaded['plan'].task_seed(task))])
            old=loaded['audit'].graph_nodes
            try:
                loaded['audit'].graph_nodes=lambda _: nodes
                local.extend(loaded['original'].check_timing(result,task,baseline))
            finally:
                loaded['audit'].graph_nodes=old
            for i,b in enumerate(result['timing']['blocks']):
                factor=b['repeats']*b['operations_per_call']
                if not math.isclose(b['host_latency_ms'],b['host_wall_ms']/factor,rel_tol=1e-10):
                    raise ValueError('Host denominator mismatch')
                start,end=[dt.datetime.fromisoformat(b[k]).timestamp() for k in
                           ('host_envelope_start_utc','host_envelope_end_utc')]
                lo,hi=bisect_left(times,start),bisect_right(times,end)
                window=telemetry.window(samples[max(0,lo-1):min(len(samples),hi+1)],
                    b['host_envelope_start_utc'],b['host_envelope_end_utc'])
                inside=window['inside']
                if not inside:
                    local.append(f'block_{i}:no_inside_sample')
                for key,value in target.items():
                    if any(s[key]!=value for s in inside):
                        local.append('observed_mismatch:'+key)
                for reason in telemetry.REASONS:
                    if any(s[reason] is True for s in inside):
                        local.append('observed_active:'+reason)
                    if any(s[reason] is None for s in inside):
                        local.append('unavailable:'+reason)
                anomaly=[]
                for counter,reason in zip(telemetry.COUNTERS,telemetry.REASONS):
                    delta=window['counter_deltas_us'][counter]
                    span=window['counter_interval_seconds']
                    if delta and (not any(s[reason] is True for s in inside)
                                  or (span is not None and delta>span*1e6)):
                        anomaly.append(counter)
                counter_blocks += bool(anomaly)
                windows.append(dict(block=i,samples_inside=len(inside),
                    observed={k:sorted({s[k] for s in inside}) for k in target},
                    counter_deltas_us=window['counter_deltas_us'],inside_sample_span_s=window['counter_interval_seconds'],
                    counter_anomalies=anomaly))
        except Exception as e:
            fatal=True;local.append('validation_error:'+str(e))
        measurements[task['id']]=dict(status='FAIL' if fatal else 'REVIEW' if local else 'PASS',
                                      flags=sorted(set(local)),windows=windows)
    return dict(timing_and_sampled_clocks_pass=not flags and
                all(r['status']=='PASS' for r in measurements.values()),
                collection_can_continue=not flags and all(r['status']!='FAIL' and
                    set(r['flags']) <= {'relative_iqr','first_last_drift','short_block'}
                    for r in measurements.values()),
                flags=flags,measurements=measurements,cadence=cadence,counters=counter_info,
                ranges={key:[min(s[key] for s in samples),max(s[key] for s in samples)]
                        for key in (*target,'power.draw','temperature.gpu')
                        if samples and all(s[key] is not None for s in samples)},
                counter_anomaly_blocks=counter_blocks,
                telemetry_sha256=sha(directory/'telemetry.csv') if (directory/'telemetry.csv').exists() else None)


def analyze(out, loaded):
    manifest=read(out/'manifest.json'); suite=loaded['suite']; protocol=manifest['protocol']
    manifest_for(out,manifest['size'],loaded)
    choices_and_done(out,manifest,loaded)
    expected={t['id']:t for t in manifest['tasks']}
    counts=Counter(); rounds=defaultdict(list); triples=defaultdict(dict)
    all_flags=[]; num_blocks=0; recorded=0; attempts=0; wall=[]
    maximum_iqr=maximum_drift=0.; counter_anomalies=0
    digest=hashlib.sha256()
    # Stream raw attempts: full collection contains several GiB of preparation blocks.
    with (out/'raw_attempts.jsonl').open('w') as f:
        for seq,row in history(out):
            value=json.dumps(dict(seq=seq,**row),ensure_ascii=False,sort_keys=True,allow_nan=False)
            f.write(value+'\n');digest.update((value+'\n').encode());attempts+=1
    metric_fields=['id','phase','shape','dtype','mode','round','split','group','session','status',
                   'median_ms','relative_iqr','first_last_drift','input_seed','graph_nodes','flags']
    block_fields=['id','split','block','timestamp','repeats','operations_per_call','elapsed_ms','latency_ms',
                  'host_wall_ms','host_latency_ms','host_envelope_start_utc','host_envelope_end_utc']
    # Latest attempts arrive in segment order; only one segment of timing/telemetry is in RAM.
    with contextlib.ExitStack() as stack:
        writers={}
        for name,fields in [('metrics.csv',metric_fields),('fresh_holdout_LOCKED.csv',metric_fields),
                            ('blocks.csv',block_fields),('blocks_fresh_holdout_LOCKED.csv',block_fields)]:
            f=stack.enter_context((out/name).open('w',newline=''))
            writers[name]=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writers[name].writeheader()
        segment_groups=groupby(history(out,latest=True),key=lambda pair:pair[1]['segment'])
        for relative,group in segment_groups:
            directory=out/relative
            if directory.resolve().parent.parent != (out/'sessions').resolve():
                raise ValueError('Invalid segment path')
            rows=[row for _,row in group]
            env=read(directory.parent/'environment.json')
            if suite.signature(env)!=manifest['environment_signature']:
                raise ValueError('Saved environment mismatch')
            quality=segment_quality(directory,rows,loaded)
            suite.atomic_json(directory/'quality.json',quality)
            all_flags.extend(relative+':'+f for f in quality['flags'])
            counter_anomalies+=quality['counter_anomaly_blocks']
            for row in rows:
                t=row['task']; tm=row.get('timing',{}); q=quality['measurements'][t['id']]
                counts[q['status']]+=1;recorded+=1;num_blocks+=len(tm.get('blocks',[]))
                flat={**t,'session':row['session'],'status':q['status'],'median_ms':tm.get('median_ms'),
                      'relative_iqr':tm.get('relative_iqr'),'first_last_drift':tm.get('first_last_drift'),
                      'input_seed':row.get('input_seed'),'graph_nodes':tm.get('graph_nodes'),'flags':q['flags']}
                locked=t['split']=='fresh_holdout'
                writers['fresh_holdout_LOCKED.csv' if locked else 'metrics.csv'].writerow(flat)
                for i,b in enumerate(tm.get('blocks',[])):
                    writers['blocks_fresh_holdout_LOCKED.csv' if locked else 'blocks.csv'].writerow(
                        dict(id=t['id'],split=t['split'],block=i,**b))
                key=loaded['plan'].digest({k:v for k,v in t.items() if k not in ('id','mode')})
                triples[key][t['mode']]=flat
                if q['status']!='FAIL' and tm:
                    rkey=loaded['plan'].digest({k:v for k,v in t.items() if k not in ('id','round')})
                    rounds[rkey].append(dict(task=t,median_ms=tm['median_ms'],session=row['session']))
                    maximum_iqr=max(maximum_iqr,tm['relative_iqr'])
                    maximum_drift=max(maximum_drift,abs(tm['first_last_drift']))
                if 'task_wall_seconds' in row:
                    wall.append(row['task_wall_seconds'])
    between=[];max_round=0.
    for values in rounds.values():
        medians=[r['median_ms'] for r in values]
        spread=(max(medians)-min(medians))/st.median(medians)
        max_round=max(max_round,spread)
        if spread>protocol['between_round_limit']:
            between.append(dict(task=values[0]['task'],relative_range=spread,
                round_median_ms='withheld' if values[0]['task']['split']=='fresh_holdout' else medians))
    suite.atomic_json(out/'between_round.json',between)
    between_keys={loaded['plan'].digest({k:v for k,v in b['task'].items() if k not in ('id','round')})
                  for b in between}
    for source,destination in [('metrics.csv','review_queue.csv'),
                               ('fresh_holdout_LOCKED.csv','review_fresh_holdout_LOCKED.csv')]:
        with (out/source).open() as src,(out/destination).open('w',newline='') as dest:
            reader=csv.DictReader(src)
            writer=csv.DictWriter(dest,fieldnames=reader.fieldnames+['between_round_review'])
            writer.writeheader()
            for row in reader:
                task=expected[row['id']]
                key=loaded['plan'].digest({k:v for k,v in task.items() if k not in ('id','round')})
                row['between_round_review']=key in between_keys
                if row['status']!='PASS' or row['between_round_review']:
                    writer.writerow(row)
    triplet_rows=[];triplets_complete=0
    for members in triples.values():
        example=next(iter(members.values()))
        row={k:example.get(k) for k in ('phase','shape','dtype','round','split','group','input_seed')}
        complete=set(members)==set(MODES) and all(m['status']!='FAIL' for m in members.values())
        row['complete']=complete;triplets_complete+=complete
        if len({m['input_seed'] for m in members.values() if m['input_seed'] is not None})>1:
            raise ValueError('Mode input seeds differ')
        for mode in MODES:
            r=members.get(mode,{})
            row[mode+'_ms']=r.get('median_ms');row[mode+'_status']=r.get('status','MISSING')
        row['graph_nodes']=members.get('graph',{}).get('graph_nodes')
        if complete:
            row['allocate_over_prealloc']=row['allocate_ms']/row['prealloc_ms']
            row['graph_over_prealloc']=row['graph_ms']/row['prealloc_ms']
        triplet_rows.append(row)
    import report
    report.csv_write(out/'triplets.csv',[r for r in triplet_rows if r['split']!='fresh_holdout'])
    report.csv_write(out/'triplets_fresh_holdout_LOCKED.csv',[r for r in triplet_rows if r['split']=='fresh_holdout'])
    report.csv_write(out/'shape_splits.csv',manifest['shape_splits'])
    comparison=[]
    if manifest['size']=='check':
        reference=loaded['config']['reference_780_pilot']['conditions']
        for ref in reference:
            values=[v['median_ms']*1000 for group in rounds.values() for v in group
                    if v['task']['shape']==ref['shape'] and v['task']['dtype']==ref['dtype']
                    and v['task']['mode']==ref['mode']]
            if values:
                current=st.median(values)
                comparison.append(dict(**ref,check_round_us=values,check_median_us=current,
                    change_pct=100*(current/ref['median_us']-1)))
        report.csv_write(out/'reference_780_comparison.csv',comparison)
    resets=[]
    for path in (out/'launches').glob('*/status.json'):
        state=read(path)
        if state['restoration']=='COMMANDS_ACCEPTED':
            restore=read(path.parent/'restore.json')
            gpu=manifest['environment_signature']['gpu_uuid']
            if not restore['commands_accepted'] or any(
                restore['commands'][name].get('exit_code')!=0 or
                restore['commands'][name]['command'][-1]!=option or
                gpu not in restore['commands'][name]['command']
                for name,option in [('graphics','-rgc'),('memory','-rmc')]):
                raise ValueError('Reset evidence mismatch')
        resets.append(state)
    attempted=[s for s in resets if s['restoration']!='NOT_ATTEMPTED']
    restored=bool(attempted) and all(s['restoration']=='COMMANDS_ACCEPTED' for s in attempted)
    missing=len(expected)-recorded
    quality_pass=not missing and not counts['FAIL'] and not counts['REVIEW'] and not between and not all_flags
    gate='FAIL' if counts['FAIL'] or any(s['restoration']=='FAILED' for s in resets) else (
        'INCOMPLETE' if missing else 'REVIEW')
    # Historical cumulative-counter ambiguity stays visible even when timing and clocks pass.
    result=dict(gate=gate,size=manifest['size'],expected=len(expected),recorded=recorded,missing=missing,
        attempts=attempts,counts=dict(counts),timed_blocks=num_blocks,triplets_expected=len(expected)//3,
        triplets_complete=triplets_complete,timing_and_sampled_clocks_pass=quality_pass,
        max_relative_iqr=maximum_iqr,max_abs_drift=maximum_drift,max_between_round_range=max_round,
        between_round_flags=len(between),clock_flags=sorted(set(all_flags)),
        counter_anomaly_blocks=counter_anomalies,known_counter_issue=loaded['config']['known_counter_issue'],
        all_restorations_accepted=restored,results_digest=digest.hexdigest(),
        task_wall_seconds_median=st.median(wall) if wall else None,
        reference_change_abs_max_pct=max((abs(r['change_pct']) for r in comparison),default=None),
        full_sweep='HOLD_PENDING_CHECK_REVIEW' if manifest['size']=='check' else 'COLLECTION_ONLY',
        sampling_note='Host-envelope samples are not continuous/effective kernel clocks.')
    suite.atomic_json(out/'summary.json',result)
    (out/'report.md').write_text('# '+manifest['size']+' 결과\n\n'+
        f"수집 {recorded}/{len(expected)}, 판정 {gate}\n\n"+
        '```json\n'+json.dumps(result,indent=2,ensure_ascii=False)+'\n```\n\n'+
        'allocate/prealloc/Graph는 triplets.csv의 별도 시간이다. 같은 조건의 round는 독립 학습 표본이 아니다.\n'
        'check의 두 round는 별도 worker이며 동일 머신·인접 시간의 실행이다. full은 재시작 전까지 같은 worker다.\n'
        'fresh_holdout 시간은 LOCKED CSV와 원자료에 보존한다. 모델 선택 전 성능값을 열어보지 않는다.\n'
        '카운터 문제는 미해결이며 제한 시간 비율로 환산하지 않는다. 본수집 자동 전환은 없다.\n')
    return result
