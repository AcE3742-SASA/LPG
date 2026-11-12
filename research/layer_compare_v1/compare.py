#!/usr/bin/env python3
"""Paired ordinary-call pilot. Reuses frozen timing, telemetry and clock cleanup."""
import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import sys
import time
import traceback
from types import SimpleNamespace

from layers import correctness, make_pair, tensor_identity

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def components(case):
    if case['layer'].startswith('conv'):
        return [(case['batch']*case['height']*case['width'], case['cout'],
                 case['cin']*case['kernel']**2)]
    m, d = case['rows'], case['input_width']
    return [(m, case['output_width'], d)] if case['layer']=='linear' else [
        (m, case['hidden_width'], d), (m, d, case['hidden_width'])]


def forbidden_overlap(shape, forbidden):
    return any(tuple(v+(delta if j==axis else 0) for j,v in enumerate(shape)) in forbidden
               for axis in range(3) for delta in range(-8, 9))


def verify():
    for name, wanted in read(HERE/'release_hashes.json').items():
        path = (HERE/name).resolve()
        if HERE not in path.parents or not path.is_file() or sha(path)!=wanted:
            raise RuntimeError('Package checksum mismatch: '+name)
    config, cases = read(HERE/'protocol.json'), read(HERE/'cases.json')
    forbidden = {tuple(s) for s in read(HERE/'forbidden_shapes.json')}
    if (config['rounds'], config['sm_mhz'], config['memory_mhz'], config['power_limit_w']) != (3,720,6801,70):
        raise ValueError('Unapproved hardware or round settings')
    if Counter(c['layer'] for c in cases) != {'conv1':64,'conv3':64,'linear':16,'ffn':16}:
        raise ValueError('Unexpected layer coverage')
    ids = set()
    for case in cases:
        contract = {k:v for k,v in case.items() if k!='id'}
        if case['id']!='lc_'+digest(contract)[:16] or case['id'] in ids:
            raise ValueError('Invalid or duplicate case ID')
        ids.add(case['id'])
        if case['split']!='development' or any(min(s)<1 for s in components(case)):
            raise ValueError('Invalid case or evaluation split')
        if any(forbidden_overlap(s,forbidden) for s in components(case)):
            raise ValueError('Pilot overlaps or neighbours a reserved evaluation shape')
    if any(name not in ids for name in config['health_case_ids']) or len(config['health_case_ids'])!=4:
        raise ValueError('Health check must contain four listed cases')
    if {c['layer'] for c in cases if c['id'] in config['health_case_ids']} != {'conv1','conv3','linear','ffn'}:
        raise ValueError('Health check must cover all four layers')
    return config, cases


def load_frozen():
    path = HERE/'dependencies/power_shape_diag_v1/diagnose.py'
    spec = importlib.util.spec_from_file_location('frozen_layer_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded, _ = module.load()
    return module, loaded


def precision(torch):
    if hasattr(torch.backends.cudnn, 'conv') and hasattr(torch.backends.cudnn.conv, 'fp32_precision'):
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        conv = {'fp32_precision':torch.backends.cudnn.conv.fp32_precision}
    else:
        torch.backends.cudnn.allow_tf32 = False
        conv = {'allow_tf32':torch.backends.cudnn.allow_tf32}
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    return dict(conv=conv, enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,
                benchmark_limit=getattr(torch.backends.cudnn,'benchmark_limit',None),
                cudnn_version=torch.backends.cudnn.version(),
                autocast=False, execution='inference_mode ordinary allocation, no compile/Graph')


def memory(torch, fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    result = dict(baseline_allocated_bytes=base,
                  extra_peak_allocated_bytes=torch.cuda.max_memory_allocated()-base,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  output_shape=list(output.shape), output_stride=list(output.stride()),
                  scope='one warmed forward outside timing; ready inputs and both weight layouts resident')
    del output
    return result


def key(case, dtype, rd, implementation):
    return f'{case["id"]}_{dtype}_r{rd}_{implementation}'


def sensor_flags(telemetry, rows, blocks):
    flags = set()
    for block in blocks:
        window = telemetry.window(rows, dt.datetime.fromisoformat(block['host_envelope_start_utc']),
                                  dt.datetime.fromisoformat(block['host_envelope_end_utc']))
        inside = window['inside']
        if not inside:
            flags.add('no_sensor_sample_in_block')
        for row in inside:
            if row['clocks.sm']!=720 or row['clocks.mem']!=6801:
                flags.add('clock_deviation')
            if row['clocks_event_reasons.sw_power_cap']:
                flags.add('sw_power_cap')
    return sorted(flags)


def analyze(out, config, cases, loaded, health_only=False):
    suite, telemetry = loaded['suite'], loaded['telemetry']
    meta = read(out/'worker/telemetry_metadata.json') if (out/'worker/telemetry_metadata.json').exists() else None
    rows, sensor_info = [], {}
    if meta:
        rows, sensor_info = telemetry.read(out/'worker/telemetry.csv',meta['timezone_offset_seconds'],50)
    selection = [c for c in cases if not health_only or c['id'] in config['health_case_ids']]
    wanted = {key(c,d,r,i) for c in selection for d in config['dtypes']
              for r in range(3) for i in ('standard','decomposed')}
    saved, flags, comparisons, counts = {}, {}, [], Counter()
    for path in sorted((out/'records').glob('*.json')):
        row = read(path)
        if health_only and row['id'] not in wanted:
            continue
        if row['id'] not in wanted or row['id'] in saved:
            raise RuntimeError('Unexpected or duplicate saved measurement')
        saved[row['id']] = row
        problems = row['timing']['flags'] + sensor_flags(telemetry,rows,row['timing']['blocks'])
        correct = [out/row['correctness_before'], out/row['correctness_after']]
        failed = any(not p.is_file() or not read(p)['passed'] for p in correct)
        flags[row['id']] = sorted(set(problems))
        counts['FAIL' if failed else 'REVIEW' if problems else 'PASS'] += 1
    for case in selection:
        for dtype in config['dtypes']:
            groups = {i:[saved.get(key(case,dtype,r,i)) for r in range(3)]
                      for i in ('standard','decomposed')}
            if not all(all(v is not None for v in group) for group in groups.values()):
                continue
            medians, ranges = {}, {}
            for name, group in groups.items():
                values = [row['timing']['median_ms'] for row in group]
                medians[name] = statistics.median(values)
                ranges[name] = (max(values)-min(values))/medians[name]
            pairs = [groups['decomposed'][r]['timing']['median_ms']/groups['standard'][r]['timing']['median_ms'] for r in range(3)]
            quality = any(flags[row['id']] for group in groups.values() for row in group)
            quality |= any(v>config['timing']['between_round_limit'] for v in ranges.values())
            valid = all(all((out/row[p]).is_file() and read(out/row[p])['passed']
                            for p in ('correctness_before','correctness_after'))
                        for group in groups.values() for row in group)
            comparisons.append(dict(case_id=case['id'], layer=case['layer'], dtype=dtype,
                selection_family=case['selection_family'], selection_reason=case['selection_reason'],
                **{k:case.get(k) for k in ('batch','height','width','cin','cout','kernel',
                                          'rows','input_width','output_width','hidden_width')},
                standard_ms=medians['standard'], decomposed_ms=medians['decomposed'],
                ratio=medians['decomposed']/medians['standard'],
                paired_ratios=pairs, paired_ratio_median=statistics.median(pairs),
                difference_ms=medians['decomposed']-medians['standard'],
                between_round_relative_range=ranges,
                standard_extra_peak_bytes=max(row['memory']['extra_peak_allocated_bytes'] for row in groups['standard']),
                decomposed_extra_peak_bytes=max(row['memory']['extra_peak_allocated_bytes'] for row in groups['decomposed']),
                prepared_weight_bytes=groups['decomposed'][0]['prepared_weight_bytes'],
                status='FAIL' if not valid else 'REVIEW' if quality else 'PASS'))
    complete = len(saved)==len(wanted) and len(comparisons)==len(selection)*2
    bad = counts['FAIL'] or any(c['status']=='FAIL' for c in comparisons)
    review = counts['REVIEW'] or any(c['status']=='REVIEW' for c in comparisons)
    status = read(out/'status.json') if (out/'status.json').exists() else {}
    stopped = (out/'health_check.json').exists() and read(out/'health_check.json')['gate']!='PASS'
    restored = status.get('restoration')=='COMMANDS_ACCEPTED'
    finished = (out/'worker/complete.json').exists() and read(out/'worker/complete.json').get('complete') is True
    execution_ok = (status.get('collection')=='COMPLETE' and restored
                    and not (out/'failure.json').exists())
    accepted = health_only or execution_ok and (finished or stopped)
    gate = ('FAIL' if bad or not accepted else 'PASS' if complete and not review else
            'REVIEW' if complete or stopped else 'FAIL')
    summary = dict(gate=gate, scope='health' if health_only else 'pilot',
                   collection='COMPLETE' if complete else 'PRECHECK_STOPPED' if stopped else 'INCOMPLETE',
                   expected=len(wanted), recorded=len(saved), missing=len(wanted)-len(saved),
                   counts=dict(counts), comparison_counts=dict(Counter(c['status'] for c in comparisons)),
                   restoration=status.get('restoration'), execution_state=status.get('collection'),
                   worker_complete=finished, sensor_info=sensor_info,
                   comparisons=comparisons, measurement_flags=flags,
                   interpretation='Implementation-path gap; not prediction error or Delta_assembly.')
    if not health_only:
        suite.atomic_json(out/'summary.json',summary)
        if comparisons:
            with (out/'comparison.csv').open('w',newline='') as stream:
                fields=[k for k in comparisons[0] if k not in ('paired_ratios','between_round_relative_range')]
                writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
                writer.writeheader(); writer.writerows(comparisons)
    return summary


def worker(out, config, cases, frozen, loaded):
    import torch
    suite, measure, audit = (loaded[k] for k in ('suite','measure','audit'))
    directory = out/'worker'
    manifest = read(out/'manifest.json')
    if manifest['source_hashes'] != read(HERE/'release_hashes.json'):
        raise RuntimeError('Worker source snapshot mismatch')
    env = suite.environment(torch)
    suite.atomic_json(directory/'environment.json',env)
    if suite.signature(env)!=manifest['environment_signature']:
        raise RuntimeError('GPU/software/base precision differs from reviewed baseline')
    if env['environment'].get('CUDA_LAUNCH_BLOCKING') not in (None,'','0'):
        raise RuntimeError('CUDA_LAUNCH_BLOCKING is enabled')
    settings = precision(torch)
    suite.atomic_json(directory/'layer_precision.json',settings)
    guard = frozen.Guard(directory,loaded['telemetry'],env['gpu_uuid'])
    observed = audit.observed_timer(measure.Timer)
    class CheckedTimer(observed):
        def block(self, fn, repeats, operations=1):
            guard.check()
            result = super().block(fn,repeats,operations)
            guard.check()
            return result
    timer = CheckedTimer(torch,config['timing'])
    with torch.inference_mode():
        audit.check_processes(suite,env['gpu_uuid'],directory/'processes_before.json')
        warm_a,warm_b,warm_out,warm_fn = measure.make_gemm(
            torch,dict(shape=[1024,1024,1024],dtype='fp32',mode='prealloc'))
        started = time.monotonic()
        while time.monotonic()-started < config['preheat_seconds']:
            timer.block(warm_fn,32)
        del warm_a,warm_b,warm_out,warm_fn
        torch.cuda.synchronize()
        suite.atomic_json(directory/'preheat.json',dict(seconds=time.monotonic()-started))
        identities = {}
        health = [c for c in cases if c['id'] in config['health_case_ids']]
        rest = [c for c in cases if c['id'] not in config['health_case_ids']]
        for stage, selection in [('health',health),('pilot',rest)]:
            for rd in range(3):
                ordered = list(selection)
                if rd==1:
                    ordered.reverse()
                elif rd==2:
                    random.Random(config['seed']).shuffle(ordered)
                for pos, case in enumerate(ordered):
                    dtypes = config['dtypes'] if rd!=1 else list(reversed(config['dtypes']))
                    for dtype in dtypes:
                        guard.check()
                        if torch.cuda.mem_get_info()[0] < 4*1024**3:
                            raise RuntimeError('Less than 4 GiB free GPU memory')
                        created = time.perf_counter()
                        pair = make_pair(torch,case,dtype)
                        torch.cuda.synchronize()
                        creation_ms = (time.perf_counter()-created)*1000
                        pair_id = case['id']+'_'+dtype
                        identity = dict(input=tensor_identity(pair['input']),
                                        weights=[tensor_identity(t) for t in pair['weights']],
                                        prepared=[tensor_identity(t) for t in pair['prepared']],seed=pair['seed'])
                        if identities.setdefault(pair_id,identity)!=identity:
                            raise RuntimeError('Input/weight identity changed between rounds')
                        suite.atomic_json(out/'inputs'/f'{pair_id}.json',identity)
                        before = f'correctness/{pair_id}_r{rd}_before.json'
                        after = f'correctness/{pair_id}_r{rd}_after.json'
                        checked = time.perf_counter()
                        result = correctness(torch,case,pair,dtype,config['tolerances'])
                        suite.atomic_json(out/before,result)
                        correctness_ms = (time.perf_counter()-checked)*1000
                        if not result['passed']:
                            raise RuntimeError('Correctness failed; inspect '+before)
                        order = ('standard','decomposed') if (rd+pos)%2==0 else ('decomposed','standard')
                        for implementation in order:
                            fn = pair['functions'][implementation]
                            timing = timer.run(fn)
                            mem = memory(torch,fn)
                            row_id = key(case,dtype,rd,implementation)
                            suite.atomic_json(out/'records'/f'{row_id}.json',
                                dict(id=row_id,case=case,dtype=dtype,round=rd,implementation=implementation,
                                     pair_order=list(order),stage=stage,timing=timing,memory=mem,
                                     correctness_before=before,correctness_after=after,
                                     prepared_weight_bytes=pair['prepared_weight_bytes'],
                                     preparation=dict(pair_creation_wall_ms=creation_ms,
                                                      before_correctness_wall_ms=correctness_ms),
                                     input_identity=f'inputs/{pair_id}.json'))
                        suite.atomic_json(out/after,correctness(torch,case,pair,dtype,config['tolerances']))
                        if not read(out/after)['passed']:
                            raise RuntimeError('Post-timing correctness failed')
                        audit.check_processes(suite,env['gpu_uuid'],directory/'processes_latest.json')
                        print(f'{stage}: round={rd+1}/3 case={pos+1}/{len(ordered)} {case["layer"]} {dtype} saved',flush=True)
                        del pair, fn
                        torch.cuda.synchronize()
            if stage=='health':
                gate = analyze(out,config,cases,loaded,health_only=True)
                suite.atomic_json(out/'health_check.json',gate)
                if gate['gate']!='PASS':
                    print('HEALTH REVIEW/FAIL: full pilot was not started; inspect health_check.json.',flush=True)
                    return
                print('HEALTH PASS: continuing the remaining development cases.',flush=True)
        suite.atomic_json(directory/'complete.json',dict(complete=True,ended_utc=suite.now()))


def run(out, config, cases, frozen, loaded):
    suite, telemetry = loaded['suite'], loaded['telemetry']
    root = Path('/home/eu/gemm-research').resolve()
    if root not in out.parents or os.geteuid()==0:
        raise RuntimeError('Run as eu with output under /home/eu/gemm-research')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        raise RuntimeError('Preserve existing run; choose a fresh output directory')
    if shutil.disk_usage(out).free < 1024**3:
        raise RuntimeError('Need at least 1 GiB free disk')
    gpu = loaded['baseline']['environment_signature']['gpu_uuid']
    identity = suite.command_text(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'])
    suite.atomic_json(out/'gpu_identity.json',identity)
    if identity.get('exit_code')!=0 or identity['stdout'].strip()!=gpu:
        raise RuntimeError('GPU UUID differs from the reviewed baseline')
    manifest = dict(kind='layer_compare_v1',protocol=config,cases=cases,
                    environment_signature=loaded['baseline']['environment_signature'],
                    source_hashes=read(HERE/'release_hashes.json'),expected_measurements=len(cases)*12,
                    split='development_only',no_original_measurement_replacement=True)
    suite.atomic_json(out/'manifest.json',manifest)
    for name in manifest['source_hashes']:
        target=out/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(HERE/name,target)
    shutil.copy2(HERE/'release_hashes.json',out/'source/release_hashes.json')
    for name in ('records','correctness','inputs','worker'):
        (out/name).mkdir()
    def execute(credentials):
        directory = out/'worker'
        child = [sys.executable,'-u','-B',str(HERE/'compare.py'),'worker','--output',str(out)]
        with telemetry.collect(directory,gpu,50,suite) as monitor:
            guard = frozen.Guard(directory,telemetry,gpu,monitor)
            last_progress = 0
            def check():
                nonlocal last_progress
                credentials.check()
                sensor=guard.check()
                if time.monotonic()-last_progress>=30:
                    print(f'Saved={len(list((out/"records").glob("*.json")))}/{len(cases)*12} '
                          f'SM={sensor["clocks.sm"]} MHz temperature={sensor["temperature.gpu"]} C',flush=True)
                    last_progress=time.monotonic()
            frozen.frozen.run_worker(child,directory,'worker',config['worker_timeout_s'],SimpleNamespace(check=check))
            guard.check()
        verify()
    with suite.lock(out/'.run.lock'):
        try:
            frozen.controlled_run(out,loaded,config,execute)
        except BaseException as error:
            suite.atomic_json(out/'failure.json',dict(type=type(error).__name__,message=str(error)))
            raise
        finally:
            result=analyze(out,config,cases,loaded)
    print('PILOT '+result['gate']+': '+result['collection'],flush=True)
    return 0 if result['gate']=='PASS' else 2 if result['gate']=='REVIEW' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','run','worker','analyze','pack'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='pack':
        import fcntl
        import uuid
        import zipfile
        out=args.output.resolve()
        with (out/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            paths=[p for p in sorted(out.rglob('*')) if p.is_file()
                   and p not in (out/'.run.lock',out/'result_hashes.json')]
            (out/'result_hashes.json').write_text(json.dumps({str(p.relative_to(out)):sha(p)
                for p in paths},indent=2)+'\n')
            target=out.with_name(out.name+'_'+uuid.uuid4().hex[:8]+'.zip')
            with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as archive:
                for p in paths+[out/'result_hashes.json']:
                    archive.write(p,str(Path(out.name)/p.relative_to(out)))
            target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n')
            print('RESULT_ZIP='+str(target),flush=True)
        return 0
    config,cases=verify()
    frozen,loaded=load_frozen()
    if args.action=='verify':
        print(f'VERIFY PASS: {len(cases)} development cases, 1920 measurements, no reserved shape neighbours.')
        return 0
    if args.output is None:
        parser.error('--output is required')
    out=args.output.resolve()
    if args.action=='run':
        return run(out,config,cases,frozen,loaded)
    if args.action=='worker':
        worker(out,config,cases,frozen,loaded)
        return 0
    result=analyze(out,config,cases,loaded)
    print(result['gate'])
    return 0 if result['gate']=='PASS' else 2 if result['gate']=='REVIEW' else 1


if __name__=='__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt('Stop requested')
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,stop)
    try:
        raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as error:
        print('FAIL: '+str(error),file=sys.stderr,flush=True)
        traceback.print_exc()
        raise SystemExit(130 if isinstance(error,KeyboardInterrupt) else 1)
