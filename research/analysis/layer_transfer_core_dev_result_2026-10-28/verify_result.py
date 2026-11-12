#!/usr/bin/env python3
"""Offline verification, reusing the reviewed layer and telemetry calculations."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RAW=HERE/'raw/core_dev_20261028_143530_VEnIY5'
sys.path.insert(0,str(ROOT/'layer_transfer_20261028_v1'))
import compare
from layers import indices, seed_for


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2)+'\n')


def require(value,message):
    if not value:raise ValueError(message)


def layout(shape):
    stride,product=[],1
    for size in reversed(shape):stride.insert(0,product);product*=size
    return stride


def main(all_records=False):
    manifest=read(RAW/'manifest.json')
    require(manifest['phase']=='core_dev','Only development results permitted')
    if all_records:
        choice=ROOT/'analysis/layer_transfer_fit_2026-10-28/runs/matched_v1'
        frozen=read(choice/'selection_frozen.json')
        require(frozen['status']=='RQ3_SELECTION_FROZEN_VALIDATION_ONLY' and frozen['explicit_values_read']==0,'Model selection must precede explicit diagnostics')
        require(sha(choice/'selected_configurations.csv')==frozen['selected_configurations_sha256'],'Frozen selection changed')
        write('explicit_diagnostics_access.json',dict(opened_utc=datetime.now().astimezone().isoformat(),
            selection_sha256=sha(choice/'selection_frozen.json'),locked_test_opened=False))
    config,cases=compare.verify();frozen,loaded=compare.load_frozen()
    require(manifest['protocol']==config and manifest['cases']==cases,'Manifest differs from reviewed release')
    require(manifest['source_hashes']==read(ROOT/'layer_transfer_20261028_v1/release_hashes.json'),'Source release mismatch')
    hashes=read(RAW/'result_hashes.json')
    for name,digest in hashes.items():
        path=(RAW/name).resolve()
        require(path.is_relative_to(RAW) and sha(path)==digest,'Result hash mismatch: '+name)
    for name,digest in manifest['source_hashes'].items():require(sha(RAW/'source'/name)==digest,'Embedded source changed')
    require(loaded['suite'].signature(read(RAW/'worker/environment.json'))==manifest['environment_signature'],'Environment signature mismatch')
    precision=read(RAW/'worker/layer_precision.json')
    require(precision['conv']=={'fp32_precision':'ieee'} and precision['enabled'] and precision['benchmark'] and not precision['deterministic'] and not precision['autocast'],'Layer precision mismatch')
    require(not (RAW/'failure.json').exists() and read(RAW/'worker/complete.json')['complete'],'Worker incomplete or failed')
    status=read(RAW/'status.json');require(status['collection']=='COMPLETE' and status['restoration']=='COMMANDS_ACCEPTED','Collection or restoration incomplete')
    paths=sorted((RAW/'records').glob('*.json'))
    if all_records:paths+=sorted((RAW/'restricted_explicit/records').glob('*.json'))
    records=[read(p) for p in paths]
    implementations=['standard','decomposed'] if all_records else ['standard']
    expected={compare.key(c,d,r,i) for c in cases for d in config['dtypes'] for r in range(3) for i in implementations}
    require(len(records)==len(expected) and {r['id'] for r in records}==expected,'Record coverage mismatch')
    by_case={c['id']:c for c in cases};correctness={};blocks=[]
    for row in records:
        case,dtype,rd,implementation=(row[k] for k in ['case','dtype','round','implementation'])
        require(case==by_case[case['id']] and row['id']==compare.key(case,dtype,rd,implementation),'Record identity mismatch')
        require(case['split'] in {'train','validation'},'Unexpected locked group')
        require(row['stage']==('health' if case['id'] in config['health_case_ids'] else 'pilot'),'Stage mismatch')
        require(set(row['pair_order'])=={'standard','decomposed'},'Pair order mismatch')
        timing=row['timing'];require(len(timing['blocks'])==7,'Block count mismatch')
        recalculated=loaded['measure'].summarize(timing['blocks'],config['timing'])
        if any(b['elapsed_ms']<.8*config['timing']['block_ms'] for b in timing['blocks']):recalculated['flags'].append('short_block')
        require(all(timing[k]==v for k,v in recalculated.items()),'Event summary mismatch')
        settle=timing['setup']['settle_blocks']
        require(sum(b['elapsed_ms'] for b in settle)==timing['setup']['settle_event_ms']>=3000,'Settling duration mismatch')
        for block in settle+timing['setup']['calibration']+timing['blocks']:
            repeat,ops=block['repeats'],block['operations_per_call']
            require(type(repeat) is int and 1<=repeat<=config['timing']['max_repeats'] and ops==1,'Operation or repeat count mismatch')
            require(math.isfinite(block['elapsed_ms']) and block['elapsed_ms']>0,'Invalid elapsed time')
            require(block['latency_ms']==block['elapsed_ms']/(repeat*ops) and block['host_latency_ms']==block['host_wall_ms']/(repeat*ops),'Raw per-call timing mismatch')
            require(datetime.fromisoformat(block['host_envelope_start_utc'])<datetime.fromisoformat(block['host_envelope_end_utc']),'Invalid host envelope')
        blocks+=timing['blocks']
        shape=[case['rows'],case.get('output_width',case['input_width'])];memory=row['memory']
        require(memory['output_shape']==shape and memory['output_stride']==layout(shape),'Memory output layout mismatch')
        require(all(memory[k]>=0 for k in ['baseline_allocated_bytes','extra_peak_allocated_bytes','peak_reserved_bytes']),'Negative memory count')
        require(row['input_identity']==f"inputs/{case['id']}_{dtype}.json",'Input identity reference mismatch')
        identity=read(RAW/row['input_identity'])
        require(identity['seed']==seed_for(case,dtype),'Input seed mismatch')
        require(row['prepared_weight_bytes']==sum(math.prod(t['shape']) for t in identity['prepared'])*(2 if dtype=='fp16' else 4),'Prepared weight size mismatch')
        for tensor in [identity['input']]+identity['weights']+identity['prepared']:
            require(len(tensor['sha256'])==64 and tensor['stride']==layout(tensor['shape']) and tensor['dtype']==('torch.float16' if dtype=='fp16' else 'torch.float32'),'Tensor identity contract mismatch')
        for phase in ['before','after']:
            name=row['correctness_'+phase]
            require(name==f"correctness/{case['id']}_{dtype}_r{rd}_{phase}.json",'Correctness reference mismatch')
            if name in correctness:continue
            result=read(RAW/name)
            require(result['passed'] and result['full_pair_passed'],'Stored correctness failed')
            require([result['atol'],result['rtol']]==config['tolerances'][dtype],'Correctness tolerance mismatch')
            require(result['rows']==indices(case['rows'],identity['seed']) and result['columns']==indices(shape[1],identity['seed']^12345),'Correctness sample indices mismatch')
            require(set(result['implementations'])=={'standard','decomposed'},'Correctness path coverage mismatch')
            for result_row in result['implementations'].values():
                require(result_row['passed'] and result_row['finite_all'] and math.isfinite(result_row['max_tolerance_ratio']) and 0<=result_row['max_tolerance_ratio']<=1,'FP64 reference check failed')
                require(result_row['points']==len(result['rows'])*len(result['columns']) and result_row['output_shape']==shape and result_row['output_stride']==layout(shape),'Correctness output contract mismatch')
            correctness[name]=result
    require(len(correctness)==len(list((RAW/'correctness').glob('*.json')))==960,'Correctness coverage mismatch')
    require(len(list((RAW/'inputs').glob('*.json')))==160,'Input coverage mismatch')
    telemetry=loaded['telemetry'];meta=read(RAW/'worker/telemetry_metadata.json')
    rows,sensor_info=telemetry.read(RAW/'worker/telemetry.csv',meta['timezone_offset_seconds'],50)
    timestamps=[datetime.fromisoformat(r['utc']) for r in rows]
    require(all(0<=(b-a).total_seconds()<=5 for a,b in zip(timestamps,timestamps[1:])),'Sensor reversal or gap')
    for row in rows:frozen.safe_row(row,manifest['environment_signature']['gpu_uuid'])
    def window(unused,start,end):
        left=bisect_left(timestamps,datetime.fromisoformat(start));right=bisect_right(timestamps,datetime.fromisoformat(end))
        return telemetry.window(rows[max(0,left-1):min(len(rows),right+1)],start,end)
    for j in range(32):
        b=blocks[j*(len(blocks)-1)//31];a,z=b['host_envelope_start_utc'],b['host_envelope_end_utc']
        require(window(rows,a,z)==telemetry.window(rows,a,z),'Indexed sensor window mismatch')
    indexed=SimpleNamespace(window=window,read=lambda *args:(rows,sensor_info))
    summary=read(RAW/'summary.json')
    for row in records:
        flags=sorted(set(row['timing']['flags']+compare.sensor_flags(indexed,rows,row['timing']['blocks'])))
        require(flags==summary['measurement_flags'][row['id']] and not flags,'Recomputed sensor or timing flag differs')
    for case in cases:
        for dtype in config['dtypes']:
            for implementation in implementations:
                rr=[r['timing']['median_ms'] for r in records if r['case']['id']==case['id'] and r['dtype']==dtype and r['implementation']==implementation]
                require((max(rr)-min(rr))/statistics.median(rr)<=.05,'Between-round range exceeds 5 percent')
    if all_records:
        stage=HERE/'recomputed';stage.mkdir(exist_ok=False)
        for name in ['records','restricted_explicit/records','correctness']:
            shutil.copytree(RAW/name,stage/name,copy_function=os.link)
        (stage/'worker').mkdir()
        for name in ['status.json','health_check.json','worker/complete.json','worker/telemetry_metadata.json']:shutil.copy2(RAW/name,stage/name)
        recalculated=compare.analyze(stage,config,cases,dict(loaded,telemetry=indexed))
        require(recalculated==summary,'Recomputed summary differs')
        require((stage/'restricted_explicit/comparison.csv').read_bytes()==(RAW/'restricted_explicit/comparison.csv').read_bytes(),'Recomputed comparison CSV differs')
        require(read(stage/'restricted_explicit/comparison.json')==read(RAW/'restricted_explicit/comparison.json'),'Recomputed comparison JSON differs')
        health=compare.analyze(stage,config,cases,dict(loaded,telemetry=indexed),health_only=True);saved=read(RAW/'health_check.json')
        for name in ['gate','scope','collection','expected','recorded','missing','counts','comparison_counts','comparisons','measurement_flags','interpretation']:
            require(health[name]==saved[name],'Health snapshot differs: '+name)
    for name,digest in hashes.items():require(sha(RAW/name)==digest,'Raw data changed during verification')
    sensors=dict(samples=len(rows),sm_mhz=sorted({r['clocks.sm'] for r in rows}),memory_mhz=sorted({r['clocks.mem'] for r in rows}),
        temperature_c=[min(r['temperature.gpu'] for r in rows),max(r['temperature.gpu'] for r in rows)],
        power_w=[min(r['power.draw'] for r in rows),max(r['power.draw'] for r in rows)],
        sw_power_cap_samples=sum(r['clocks_event_reasons.sw_power_cap'] for r in rows),
        thermal_samples=sum(bool(r['clocks_event_reasons.sw_thermal_slowdown'] or r['clocks_event_reasons.hw_thermal_slowdown']) for r in rows),
        windows_checked=len(blocks),full_scan_cross_checks=32,**{k:v for k,v in sensor_info.items() if k!='samples'})
    report=dict(status='PASS',scope='all_development' if all_records else 'standard_only',records=len(records),
        source_hashes_verified=len(manifest['source_hashes']),result_hashes_verified=len(hashes),correctness_files=len(correctness),
        input_identity_files=160,timing_summary_exact=True,raw_preserved=True,sensors=sensors,
        explicit_values_read=480 if all_records else 0,locked_test_opened=False,
        collection_wall_seconds=read(RAW/'cost_ledger.json')['collection_wall_seconds'],
        exact_public_summary_reproduction=all_records,exact_explicit_comparison_reproduction=all_records)
    write('verification_all.json' if all_records else 'verification_standard.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--all-development',action='store_true')
    main(p.parse_args().all_development)
