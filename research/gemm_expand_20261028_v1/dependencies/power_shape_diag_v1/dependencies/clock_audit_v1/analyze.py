"""Compare polling regimes and Nsight, retaining every timing and warning."""
import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import json
import math
from pathlib import Path
import statistics as st

import audit
import telemetry
import trace_tools as trace


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def table(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)


def check_timing(result, task, baseline):
    spec=next(s for s in baseline['fixed_cases'] if s['shape']==task['shape'] and s['dtype']==task['dtype'])
    correct=result['correctness']
    if not (correct['passed'] and correct['finite_all'] and correct['max_tolerance_ratio']<=1):
        raise ValueError('Correctness failed')
    if result['input_seed']!=spec['input_seed']:
        raise ValueError('Input seed mismatch')
    timing=result['timing'];factor=audit.graph_nodes(task) or 1
    if task['mode']=='graph' and timing['graph_nodes']!=factor:
        raise ValueError('Graph batch changed')
    if len(timing['blocks'])!=baseline['protocol']['blocks']:
        raise ValueError('Missing timing block')
    setup=timing['setup']
    if setup['settle_event_ms']<baseline['protocol']['settle_ms']:
        raise ValueError('Insufficient observed stabilization')
    if not math.isclose(sum(b['elapsed_ms'] for b in setup['settle_blocks']),setup['settle_event_ms'],rel_tol=1e-10):
        raise ValueError('Stabilization sum mismatch')
    for block in setup['settle_blocks']+setup['calibration']+timing['blocks']:
        if type(block['repeats']) is not int or block['repeats']<1 or block['operations_per_call']!=factor:
            raise ValueError('Invalid GEMM denominator')
        normalized=trace.finite_positive(block['elapsed_ms'])/(block['repeats']*factor)
        if not math.isclose(normalized,block['latency_ms'],rel_tol=1e-10):
            raise ValueError('Latency denominator mismatch')
        if dt.datetime.fromisoformat(block['host_envelope_end_utc'])<=dt.datetime.fromisoformat(block['host_envelope_start_utc']):
            raise ValueError('Invalid host envelope')
    vals=[b['latency_ms'] for b in timing['blocks']]
    q1,_,q3=st.quantiles(vals,n=4,method='inclusive');med=st.median(vals)
    actual=dict(median_ms=med,relative_iqr=(q3-q1)/med,first_last_drift=(vals[-1]-vals[0])/vals[0])
    for key,value in actual.items():
        if not math.isclose(timing[key],value,rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('Stored timing statistic mismatch: '+key)
    flags=[];protocol=baseline['protocol']
    if actual['relative_iqr']>protocol['relative_iqr_limit']:flags.append('relative_iqr')
    if abs(actual['first_last_drift'])>protocol['drift_limit']:flags.append('first_last_drift')
    if any(b['elapsed_ms']<.8*protocol['block_ms'] for b in timing['blocks']):flags.append('short_block')
    if timing['flags']!=flags:raise ValueError('Stored timing flags mismatch')
    return flags


def analyze(out):
    manifest=json.loads((out/'manifest.json').read_text())
    baseline=json.loads((audit.HERE/'baseline_v6.json').read_text())
    if manifest['baseline']!=baseline or manifest['cases']!=audit.cases() or manifest['passes']!=audit.PASSES:
        raise ValueError('Diagnostic manifest mismatch')
    for name,wanted in baseline['code_hashes'].items():
        if audit.sha256_file(out/'baseline_source'/name)!=wanted:raise ValueError('Baseline source mismatch: '+name)
    local_sources={p.name:audit.sha256_file(p) for p in audit.HERE.iterdir() if p.suffix in ('.py','.sh','.json','.md')}
    if local_sources!=manifest['source_hashes']:raise ValueError('Diagnostic source manifest mismatch')
    for name,wanted in local_sources.items():
        if audit.sha256_file(out/'source'/name)!=wanted:raise ValueError('Diagnostic source snapshot mismatch')
    records={};observations=[];expected={};sample_info={};flags=[];counts=Counter()
    for pass_name,interval in audit.PASSES.items():
        directory=out/pass_name
        env=json.loads((directory/'environment.json').read_text())
        if {k:env[k] for k in baseline['environment_signature']}!=baseline['environment_signature']:
            raise ValueError('Environment mismatch: '+pass_name)
        metadata=json.loads((directory/'telemetry_metadata.json').read_text())
        if metadata['requested_interval_ms']!=interval or metadata['fields']!=list(telemetry.FIELDS):
            raise ValueError('Telemetry configuration mismatch')
        samples,info=telemetry.read(directory/'telemetry.csv',metadata['timezone_offset_seconds'],interval)
        if any(r['uuid']!=env['gpu_uuid'] for r in samples):raise ValueError('Telemetry GPU mismatch')
        sample_info[pass_name]=info
        if info['unavailable_fields']:flags.append(pass_name+':some_clock_fields_unavailable')
        if info['observed_interval_seconds_max']>max(.5,3*interval/1000):flags.append(pass_name+':telemetry_sampling_gap')
        rows=json.loads((directory/'results.json').read_text())
        if [r['task'] for r in rows]!=manifest['cases']:raise ValueError('Missing/duplicate/out-of-order tasks: '+pass_name)
        records[pass_name]={r['task']['id']:r for r in rows}
        for row in rows:
            task,result=row['task'],row['result'];case_id=task['id'];tm=result['timing']
            local_flags=check_timing(result,task,baseline)
            counts['REVIEW' if local_flags else 'PASS']+=1
            flags.extend(f'{pass_name}/{case_id}:{x}' for x in local_flags)
            if pass_name=='nsys' and not tm['setup'].get('capture_started_before_prepare'):
                raise ValueError('Profiler did not start before stabilization')
            for phase,blocks in (('settle',tm['setup']['settle_blocks']),('calibration',tm['setup']['calibration']),('timed',tm['blocks'])):
                for i,block in enumerate(blocks):
                    window=telemetry.window(samples,block['host_envelope_start_utc'],block['host_envelope_end_utc'])
                    observations.append(dict(pass_name=pass_name,task=task,phase=phase,block=i,
                                             gemm_us=block['latency_ms']*1000,clock_window=window))
                    if phase=='timed' and pass_name=='nsys':
                        label=f'audit/{case_id}/block/{i}'
                        if block.get('trace_label')!=label:raise ValueError('Trace label mismatch')
                        expected[label]=block
    traced={};trace_info={};trace_hashes={}
    for path in sorted((out/'traces').glob('*.sqlite')):
        part,digest=trace.read_trace(path,expected)
        if set(part)&set(traced):raise ValueError('Duplicate traced blocks')
        traced.update(part);trace_hashes[path.name]=digest
        trace_info[path.name]=trace.read_trace_metadata(path)
    if set(traced)!=set(expected):raise ValueError('Missing NVTX blocks')
    summaries=[];groups=defaultdict(list)
    for task in manifest['cases']:
        groups[(tuple(task['shape']),task['dtype'],task['mode'])].append(task)
        case_id=task['id'];blocks=[traced[f'audit/{case_id}/block/{i}'] for i in range(7)]
        metadata=trace_info[blocks[0]['trace']]
        trace.validate_preparation(blocks[0],metadata,case_id,baseline['protocol']['settle_ms'])
        if metadata['warnings'] or not metadata['diagnostic_table_present']:
            flags.append(case_id+':nsys_warning_or_diagnostics_unavailable')
        for i,b in enumerate(blocks):
            if not b['activity_gap_breakdown']['supported']:flags.append(case_id+':gap_attribution_unavailable')
            if b['kernel_count']%b['expected_gemm_calls']:flags.append(case_id+':noninteger_kernel_count_per_gemm')
            if b['unassociated_kernel_count_in_span']:flags.append(case_id+':unassociated_kernels')
            # Kernel multiplicity was not independently established for these shapes.
            b['kernel_name_counts_per_gemm']={k:v/b['expected_gemm_calls'] for k,v in b['kernel_names'].items()}
            b['kernel_union_us_per_gemm']=b['kernel_union_ms']*1000/b['expected_gemm_calls']
    for (shape,dtype,mode),tasks in groups.items():
        by_pass={p:[records[p][t['id']]['result']['timing']['median_ms']*1000 for t in tasks] for p in audit.PASSES}
        sparse=by_pass['sparse_before']+by_pass['sparse_after']
        dense=by_pass['dense_before']+by_pass['dense_after']
        prof=by_pass['nsys']
        spreads={key:(max(v)-min(v))/st.median(v) for key,v in [('sparse',sparse),('dense',dense),('nsys',prof)]}
        sampler_ratio=st.median(dense)/st.median(sparse)
        nsys_ratio=st.median(prof)/st.median(dense)
        group_flags=[key+'_relative_range_above_5pct' for key,value in spreads.items() if value>.05]
        if abs(sampler_ratio-1)>.05:group_flags.append('dense_vs_sparse_difference_above_5pct')
        if abs(nsys_ratio-1)>.05:group_flags.append('nsys_vs_dense_difference_above_5pct')
        paths=[traced[f'audit/{t["id"]}/block/{i}']['kernel_name_counts_per_gemm'] for t in tasks for i in range(7)]
        if any(p!=paths[0] for p in paths[1:]):group_flags.append('observed_kernel_path_differs')
        flags.extend(f'{shape}/{dtype}/{mode}:{f}' for f in group_flags)
        summaries.append(dict(shape=shape,dtype=dtype,mode=mode,round_us_by_pass=by_pass,
                              sparse_us=st.median(sparse),dense_us=st.median(dense),nsys_us=st.median(prof),
                              dense_over_sparse=sampler_ratio,nsys_over_dense=nsys_ratio,
                              relative_ranges=spreads,flags=group_flags))
    flags=sorted(set(flags))
    summary=dict(gate='REVIEW' if flags else 'PASS',expected=40,recorded=40,missing=0,counts=dict(counts),
                 traced_blocks=len(traced),correctness_passed=40,interpretation='NOT_REVIEWED',flags=flags)
    save(out/'summary.json',summary)
    save(out/'analysis.json',dict(summary=summary,conditions=summaries,telemetry=sample_info,
                                trace_metadata=trace_info,trace_hashes=trace_hashes,trace_blocks=traced,
                                caveats=['Five separate processes; polling/profiling ratios include session variation.',
                                         '100 ms polling is not verified 100 ms sensor resolution.',
                                         'Clock windows are host envelopes around CUDA event blocks, not exact GPU event boundaries.',
                                         'Counter deltas cover inside samples only; empty windows remain empty.',
                                         'The full 192-task pilot workload history is not replayed; non-reproduction does not clear its REVIEW.',
                                         'Kernel counts/path consistency do not guarantee complete system-wide capture.']))
    save(out/'clock_windows.json',observations)
    table(out/'conditions.csv',summaries)
    lines=['# Clock audit results','',f"Collection: 40/40; traced blocks: {len(traced)}/56; quality: {summary['gate']}.",
           '', '모든 시간은 GEMM당 μs다. 각 값은 해당 방식·기록 간격의 round 중앙값들을 요약한 중앙값이다.',
           '서로 다른 프로세스의 차이도 포함하므로 두 비율을 순수 계측 비용으로 단정하지 않는다.',
           '', '| 조건 | 1초 기록 | 0.1초 기록 | Nsight+0.1초 | 촘촘/기존 | Nsight/촘촘 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['shape']} {r['dtype']} {r['mode']} | {r['sparse_us']:.3g} | {r['dense_us']:.3g} | {r['nsys_us']:.3g} | {r['dense_over_sparse']:.3g} | {r['nsys_over_dense']:.3g} |")
    lines+=['','## 검토 표식','']+[('- '+f) for f in flags]
    if not flags:lines.append('자동 기준 초과 없음. 원인 해석과 full 판단은 별도 검토한다.')
    lines+=['','`clock_windows.json`은 안정화·보정·측정 블록마다 내부 표본과 직전·직후 표본을 구분한다.',
            '카운터 차이는 첫 내부 표본부터 마지막 내부 표본까지만 계산한다. 표본이 없는 블록은 보간하지 않는다.',
            '`analysis.json`은 Nsight 경고, 커널 경로와 GEMM당 GPU 작업·간격을 보존한다.',
            '두 신규 형상의 커널 수는 사전 확정하지 않았으므로 개수 일치만으로 수집 완전성을 주장하지 않는다.',
            '원래 192작업의 앞선 부하 순서를 모두 재현하지 않는다. 변동이 재현되지 않아도 원래 REVIEW가 해소된 것으로 처리하지 않는다.',
            'GPU 설정·측정 기준을 바꾸지 않았으며 자동으로 full을 시작하지 않는다.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    args=parser.parse_args();print(json.dumps(analyze(args.out.resolve()),indent=2))
