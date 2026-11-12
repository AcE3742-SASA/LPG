#!/usr/bin/env python3
"""Controlled two-shape diagnostic. Results never replace sweep measurements."""
import argparse
import contextlib
import csv
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_remeasure', HERE/'dependencies/remeasure_720_v4/run.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
read, sha = frozen.read, frozen.sha


def hashes():
    return {p.name: sha(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.md')}


def load():
    for name, expected in read(HERE/'dependency_hashes.json').items():
        if sha(HERE/'dependencies'/name) != expected:
            raise RuntimeError('Frozen dependency changed: '+name)
    loaded = frozen.load(720)
    config = read(HERE/'protocol.json')
    if ([t['shape'] for t in config['tasks']] != [[4096,4096,72], [4096,4096,127]]
            or any(t['dtype'] != 'fp32' or t['mode'] != 'prealloc' or t['split'] == 'holdout'
                   for t in config['tasks'])
            or (config['sm_mhz'], config['memory_mhz'], config['power_limit_w']) != (720,6801,70)
            or config['case_orders'] != [[0,1],[1,0],[1,0],[0,1]]
            or config['rounds'] != 4 or config['continuous_trials'] != 16
            or not config['diagnostic_only'] or config['replace_measurements']):
        raise ValueError('Diagnostic scope or controls changed')
    return loaded, config


def batches(config):
    return [(r, position, case) for r, order in enumerate(config['case_orders'])
            for position, case in enumerate(order)]


def safe_row(row, gpu):
    required = ('temperature.gpu', 'utilization.gpu', 'clocks.sm', 'clocks.mem',
                'power.draw', 'power.limit', 'clocks_event_reasons.active',
                'clocks_event_reasons.sw_power_cap', 'clocks_event_reasons.sw_thermal_slowdown',
                'clocks_event_reasons.hw_thermal_slowdown')
    if row['uuid'] != gpu or any(row[k] is None for k in required):
        raise RuntimeError('Missing telemetry or wrong GPU')
    if abs(row['power.limit']-70) > .1:
        raise RuntimeError('Power limit changed from 70 W')
    if row['temperature.gpu'] >= 85 or row['clocks_event_reasons.active'] & ~7:
        raise RuntimeError('Thermal or uncontrolled clock event; inspect telemetry')
    if row['clocks_event_reasons.sw_thermal_slowdown'] or row['clocks_event_reasons.hw_thermal_slowdown']:
        raise RuntimeError('Thermal clock event; inspect telemetry')


class Guard:
    """Incremental CSV reader shared by the worker and its supervising process."""
    def __init__(self, directory, telemetry, gpu, process=None):
        self.directory, self.telemetry, self.gpu, self.process = directory, telemetry, gpu, process
        self.offset = read(directory/'telemetry_metadata.json')['timezone_offset_seconds']
        self.position, self.pending, self.latest, self.last_time = 0, '', None, None

    def check(self):
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError('Telemetry process stopped')
        with (self.directory/'telemetry.csv').open() as stream:
            stream.seek(self.position)
            chunk = stream.read()
            self.position = stream.tell()
        self.new_rows = []
        parts = (self.pending+chunk).split('\n')
        self.pending = parts.pop()
        for line in parts:
            if not line.strip():
                continue
            row = self.telemetry.parse_row(next(csv.reader([line])), self.offset)
            safe_row(row, self.gpu)
            stamp = dt.datetime.fromisoformat(row['utc']).timestamp()
            if self.last_time is not None and not 0 <= stamp-self.last_time <= 5:
                raise RuntimeError('Telemetry timestamp reversal or gap over 5 seconds')
            self.new_rows.append(row)
            self.latest, self.last_time = row, stamp
        if self.latest is None or not -1 <= time.time()-self.last_time <= 5:
            raise RuntimeError('Telemetry missing or stale for over 5 seconds')
        return self.latest


def recover(guard, config):
    begin, stable, reported = time.monotonic(), None, 0
    earliest = time.time()
    print('Recovery: wait for <= '+str(config['cooldown_max_c'])+' C for '+str(config['cooldown_stable_s'])+' sampled seconds.',flush=True)
    while time.monotonic()-begin < config['cooldown_timeout_s']:
        row = guard.check()
        for sample in guard.new_rows:
            stamp = dt.datetime.fromisoformat(sample['utc']).timestamp()
            if stamp < earliest:
                continue
            if sample['temperature.gpu'] <= config['cooldown_max_c']:
                stable = stamp if stable is None else stable
            else:
                stable = None
        stamp = dt.datetime.fromisoformat(row['utc']).timestamp()
        if stable is not None and stamp-stable >= config['cooldown_stable_s']:
            return dict(wait_seconds=time.monotonic()-begin, start_sample=row,
                        below_threshold_seconds=stamp-stable)
        if time.monotonic()-reported >= 30:
            print('Recovery: temperature='+str(row['temperature.gpu'])+' C',flush=True)
            reported = time.monotonic()
        time.sleep(.2)
    raise TimeoutError('Cooling target not reached; no trial started')


def sensor_fields(out, loaded, config, gpu):
    telemetry, suite = loaded['telemetry'], loaded['suite']
    fields, evidence = list(telemetry.FIELDS), {}
    for field in config['optional_sensor_fields']:
        result = suite.command_text(['nvidia-smi','-i',gpu,'--query-gpu='+field,
                                     '--format=csv,noheader,nounits'])
        try:
            available = result.get('exit_code') == 0 and math.isfinite(float(result['stdout'].strip()))
        except (ValueError, KeyError):
            available = False
        evidence[field] = dict(available=available, query=result)
        if available:
            fields.append(field)
    suite.atomic_json(out/'sensor_capabilities.json', evidence)
    telemetry.FIELDS = tuple(fields)
    return fields


def profile_command(tool, child, directory, config):
    exe = shutil.which(tool)
    if exe is None:
        raise RuntimeError(tool+' unavailable; use the existing research installation')
    if tool == 'ncu':
        cmd = [exe,'--profile-from-start','off','--target-processes','all',
               '--clock-control','none','--cache-control','none']
        for section in config['ncu_sections']:
            cmd += ['--section',section]
        return cmd+['--csv','--page','raw','--export',str(directory/'capture'),*child]
    return [exe,'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop',
            '--output',str(directory/'capture'),*child]


def verify_profile(tool, directory, loaded, credentials):
    from profiling import ncu_has_kernel_data, nsys_has_kernel_data
    suffixes = ('ncu-rep','ncu-repz') if tool == 'ncu' else ('nsys-rep',)
    report = next((directory/('capture.'+s) for s in suffixes
                   if (directory/('capture.'+s)).is_file() and (directory/('capture.'+s)).stat().st_size), None)
    if report is None:
        raise RuntimeError('Profiler did not produce a nonempty report')
    if tool == 'ncu':
        valid = ncu_has_kernel_data((directory/'worker.log').read_text(errors='replace'))
    else:
        command = [shutil.which('nsys'),'stats','--report','cuda_gpu_kern_sum','--format','csv',str(report)]
        frozen.run_worker(command,directory,'kernel_summary',300,credentials)
        shutil.copy2(directory/'kernel_summary.log',directory/'kernel_summary.csv')
        valid = nsys_has_kernel_data((directory/'kernel_summary.csv').read_text(errors='replace'))
    if not valid:
        raise RuntimeError('Profiler report contains no verified kernel metrics')
    loaded['suite'].atomic_json(directory/'profile_status.json',
                               dict(status='COMPLETE',report=report.name,timing_use='diagnostic_only'))


def worker(args, loaded, config):
    import torch
    out, directory = args.output, args.directory
    manifest, suite, measure, telemetry = read(out/'manifest.json'), loaded['suite'], loaded['measure'], loaded['telemetry']
    if manifest['source_hashes'] != hashes() or manifest['protocol'] != config:
        raise RuntimeError('Source or protocol differs from saved manifest')
    telemetry.FIELDS = tuple(manifest['telemetry_fields'])
    env = suite.environment(torch)
    suite.atomic_json(directory/'environment.json',env)
    if suite.signature(env) != manifest['environment_signature']:
        raise RuntimeError('GPU/software/precision differs from the reviewed baseline')
    if torch.cuda.mem_get_info()[0] < 4*1024**3:
        raise RuntimeError('Less than 4 GiB free GPU memory')
    task = config['tasks'][args.case]
    guard = Guard(directory,telemetry,env['gpu_uuid'])
    observed, all_blocks = loaded['audit'].observed_timer(measure.Timer), []
    class CheckedTimer(observed):
        def block(self, fn, repeats, operations=1):
            guard.check()
            result = super().block(fn,repeats,operations)
            all_blocks.append(result)
            guard.check()
            return result
    def save_correctness(name, a, b, c):
        value = measure.check_mm(torch,a,b,c,'fp32',loaded['plan'].task_seed(task),loaded['plan'].PROTOCOL)
        suite.atomic_json(directory/name,value)
        if not value['passed']:
            raise RuntimeError('Correctness check failed')
        return value
    try:
        with torch.inference_mode():
            a,b,output,fn = measure.make_gemm(torch,task)
            save_correctness('correctness_before.json',a,b,output[0])
            torch.cuda.synchronize()
            loaded['audit'].check_processes(suite,env['gpu_uuid'],directory/'processes_before.json')
            recovery = recover(guard,config)
            suite.atomic_json(directory/'recovery.json',recovery)
            timer = CheckedTimer(torch,loaded['plan'].PROTOCOL)
            if args.tool:
                repeats, setup = timer.prepare(fn)
                suite.atomic_json(directory/'profile_setup.json',dict(repeats=repeats,setup=setup))
                torch.cuda.synchronize()
                start = suite.now()
                torch.cuda.cudart().cudaProfilerStart()
                try:
                    with torch.cuda.nvtx.range('diagnostic_K'+str(task['shape'][2])):
                        for _ in range(1 if args.tool == 'ncu' else 20):
                            fn()
                        torch.cuda.synchronize()
                finally:
                    torch.cuda.cudart().cudaProfilerStop()
                suite.atomic_json(directory/'capture_window.json',dict(start_utc=start,end_utc=suite.now()))
                guard.check()
                save_correctness('correctness_after.json',a,b,output[0])
            else:
                for trial in range(config['continuous_trials']):
                    start = suite.now()
                    timing = timer.run(fn)
                    end = suite.now()
                    correctness = save_correctness(f'correctness_{trial:02d}.json',a,b,output[0])
                    guard.check()
                    loaded['audit'].check_processes(suite,env['gpu_uuid'],directory/f'processes_{trial:02d}.json')
                    suite.atomic_json(directory/f'trial_{trial:02d}.json',
                        dict(trial=trial,condition='recovered_first' if trial == 0 else 'continued',
                             task=task,input_seed=loaded['plan'].task_seed(task),start_utc=start,end_utc=end,
                             timing=timing,correctness=correctness,diagnostic_only=True))
                    print(f"Trial {trial+1}/{config['continuous_trials']}: K={task['shape'][2]} saved",flush=True)
            suite.atomic_json(directory/'worker_complete.json',dict(complete=True,ended_utc=suite.now()))
    finally:
        suite.atomic_json(directory/'all_event_blocks.json',all_blocks)


def window_stats(rows, start, end):
    selected = [r for r in rows if start <= r['utc'] <= end]
    result = dict(samples=len(selected),sample_fraction_is_not_time_fraction=True)
    for key in ('clocks.sm','clocks.mem','power.draw','temperature.gpu','utilization.gpu',
                'power.draw.average','power.draw.instant','fan.speed','voltage.graphics','utilization.memory'):
        values = [r[key] for r in selected if r.get(key) is not None]
        result[key] = dict(min=min(values),median=statistics.median(values),max=max(values)) if values else None
    capped = [r for r in selected if r['clocks_event_reasons.sw_power_cap']]
    drops = [r for r in selected if r['clocks.sm'] < 720]
    result.update(sw_power_cap_samples=len(capped),below_target_samples=len(drops),
                  first_cap_utc=capped[0]['utc'] if capped else None,
                  first_drop_utc=drops[0]['utc'] if drops else None,
                  memory_clock_mismatch=any(r['clocks.mem'] != 6801 for r in selected))
    return result


def analyze(out, loaded, config):
    telemetry, suite = loaded['telemetry'], loaded['suite']
    manifest = read(out/'manifest.json')
    telemetry.FIELDS = tuple(manifest['telemetry_fields'])
    trials, pairs, issues = [], [], []
    for rd,position,case in batches(config):
        directory = out/f'r{rd+1}_p{position+1}_k{config["tasks"][case]["shape"][2]}'
        if not (directory/'worker_complete.json').exists():
            issues.append(directory.name+': incomplete worker')
        if not (directory/'telemetry_metadata.json').exists():
            continue
        try:
            meta = read(directory/'telemetry_metadata.json')
            rows,cadence = telemetry.read(directory/'telemetry.csv',meta['timezone_offset_seconds'],config['interval_ms'])
            suite.atomic_json(directory/'telemetry_quality.json',cadence)
            recovery = read(directory/'recovery.json')
        except (OSError,ValueError,KeyError) as error:
            issues.append(directory.name+': '+str(error))
            continue
        previous_end = None
        for path in sorted(directory.glob('trial_*.json')):
            row = read(path)
            blocks = row['timing']['blocks']
            timing_rows = [r for r in rows if any(b['host_envelope_start_utc'] <= r['utc'] <= b['host_envelope_end_utc'] for b in blocks)]
            setup = row['timing']['setup']['settle_blocks'] + row['timing']['setup']['calibration']
            item = dict(round=rd+1,position=position+1,k=row['task']['shape'][2],trial=row['trial'],
                        condition=row['condition'],latency_ms=row['timing']['median_ms'],
                        timing_flags=row['timing']['flags'],correctness_passed=row['correctness']['passed'],
                        start_temperature_c=recovery['start_sample']['temperature.gpu'],
                        entire_trial=window_stats(rows,row['start_utc'],row['end_utc']),
                        timing=window_stats(timing_rows,blocks[0]['host_envelope_start_utc'],blocks[-1]['host_envelope_end_utc']),
                        settling=window_stats(rows,setup[0]['host_envelope_start_utc'],setup[-1]['host_envelope_end_utc']))
            item['gap_from_previous_trial_ms'] = ((dt.datetime.fromisoformat(row['start_utc'])-
                dt.datetime.fromisoformat(previous_end)).total_seconds()*1000 if previous_end else None)
            cap_at = item['entire_trial']['first_cap_utc']
            item['first_cap_after_trial_start_s'] = ((dt.datetime.fromisoformat(cap_at)-
                dt.datetime.fromisoformat(row['start_utc'])).total_seconds() if cap_at else None)
            previous_end = row['end_utc']
            trials.append(item)
    for rd in range(1,config['rounds']+1):
        first = [r for r in trials if r['round']==rd and r['trial']==0]
        delta = abs(first[0]['start_temperature_c']-first[1]['start_temperature_c']) if len(first)==2 else None
        matched = delta is not None and delta <= config['start_temperature_tolerance_c']
        pairs.append(dict(round=rd,temperature_difference_c=delta,matched=matched))
        if not matched:
            issues.append(f'Round {rd}: START_TEMPERATURE_UNMATCHED')
    profile_count = sum((out/f'{tool}_k{k}'/'profile_status.json').exists()
                        for tool in ('nsys','ncu') for k in (72,127))
    status = read(out/'status.json') if (out/'status.json').exists() else {}
    expected = len(batches(config))*config['continuous_trials']
    failed = sum(not t['correctness_passed'] for t in trials)
    complete = len(trials)==expected and profile_count==4 and status.get('collection')=='COMPLETE'
    restored = status.get('restoration')=='COMMANDS_ACCEPTED'
    result = dict(gate='REVIEW' if complete and restored and failed==0 else 'FAIL',
                  collection='COMPLETE' if complete else 'INCOMPLETE',recorded=len(trials),expected=expected,
                  missing=expected-len(trials),counts={'PASS':len(trials)-failed,'FAIL':failed},
                  profiles_complete=profile_count,profiles_expected=4,restoration=status.get('restoration'),
                  pair_controls=pairs,issues=issues,diagnostic_only=True,replace_measurements=False,
                  interpretation='REVIEW means evidence collected for causal review, not automatic root-cause proof.',
                  trials=trials)
    suite.atomic_json(out/'summary.json',result)
    return result


def controlled_run(out, loaded, config, operation):
    suite, steps, hold, audit = [loaded[k] for k in ('suite','steps','hold','audit')]
    gpu = loaded['baseline']['environment_signature']['gpu_uuid']
    smi = shutil.which('nvidia-smi')
    if not smi:
        raise RuntimeError('nvidia-smi unavailable')
    state = dict(collection='NOT_STARTED',restoration='NOT_ATTEMPTED')
    with suite.lock(Path('/tmp')/f'gemm-sweep-{gpu}.lock'):
        try:
            audit.check_processes(suite,gpu,out/'processes_before.json')
            steps.require_power(suite,out,smi,gpu,'power_before',loaded['config'])
            steps.snapshot(suite,out,smi,gpu,'gpu_before_settings')
            with steps.Credentials(suite,out) as credentials:
                try:
                    steps.require_clock(hold,suite,out,smi,gpu,'set_memory','-lmc','6801,6801')
                    steps.require_clock(hold,suite,out,smi,gpu,'set_graphics','-lgc','720,720')
                    state['collection']='RUNNING'
                    operation(credentials)
                    credentials.check()
                    state['collection']='COMPLETE'
                finally:
                    with frozen.cleanup_signals():
                        restored = hold.restore(suite,out,smi,gpu)
                    state['restoration']='COMMANDS_ACCEPTED' if restored else 'FAILED'
                if not restored:
                    raise RuntimeError('Clock reset failed; inspect restore.json')
        except BaseException:
            state['collection']='FAILED_OR_INTERRUPTED'
            raise
        finally:
            suite.atomic_json(out/'status.json',state)


def run(out, loaded, config):
    out.mkdir(parents=True,exist_ok=True)
    with loaded['suite'].lock(out/'.run.lock'):
        return run_locked(out,loaded,config)


def run_locked(out, loaded, config):
    suite, telemetry = loaded['suite'], loaded['telemetry']
    out.mkdir(parents=True,exist_ok=True)
    gpu = loaded['baseline']['environment_signature']['gpu_uuid']
    if (out/'manifest.json').exists():
        raise RuntimeError('Output already has a manifest; use a fresh output directory')
    for tool in ('ncu','nsys','nvidia-smi'):
        if shutil.which(tool) is None:
            raise RuntimeError(tool+' unavailable')
    if shutil.disk_usage(out).free < 20*1024**3:
        raise RuntimeError('Need at least 20 GiB free disk')
    fields = sensor_fields(out,loaded,config,gpu)
    manifest = dict(protocol=config,source_hashes=hashes(),dependency_hashes=read(HERE/'dependency_hashes.json'),
                    environment_signature=loaded['baseline']['environment_signature'],telemetry_fields=fields,
                    timing_protocol=loaded['plan'].PROTOCOL,
                    input_seeds=[loaded['plan'].task_seed(t) for t in config['tasks']])
    suite.atomic_json(out/'manifest.json',manifest)
    for source,folder,names in [(HERE,'source',manifest['source_hashes']),
                                (HERE/'dependencies','dependencies',manifest['dependency_hashes'])]:
        for name in names:
            target = out/folder/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,target)
    def execute(credentials):
        schedule = [(f'r{r+1}_p{position+1}_k{config["tasks"][case]["shape"][2]}',case,None)
                    for r,position,case in batches(config)]
        schedule += [(f'{tool}_k{config["tasks"][case]["shape"][2]}',case,tool)
                     for tool in ('nsys','ncu') for case in (0,1)]
        for index,(name,case,tool) in enumerate(schedule):
            directory = out/name
            directory.mkdir()
            print(f'Stage {index+1}/{len(schedule)}: {name}; log={directory / "worker.log"}',flush=True)
            child = [sys.executable,'-u','-B',str(HERE/'diagnose.py'),'worker','--output',str(out),
                     '--directory',str(directory),'--case',str(case)]
            if tool:
                child += ['--tool',tool]
                command = profile_command(tool,child,directory,config)
            else:
                command = child
            with telemetry.collect(directory,gpu,config['interval_ms'],suite) as monitor:
                guard = Guard(directory,telemetry,gpu,monitor)
                last_progress = 0
                def check():
                    nonlocal last_progress
                    credentials.check()
                    row = guard.check()
                    if time.monotonic()-last_progress >= 30:
                        saved = len(list(directory.glob('trial_*.json')))
                        print(f"  {name}: trials_saved={saved}, temperature={row['temperature.gpu']} C, SM={row['clocks.sm']} MHz",flush=True)
                        last_progress = time.monotonic()
                checked = SimpleNamespace(check=check)
                frozen.run_worker(command,directory,'worker',
                    config['profiler_timeout_s' if tool else 'worker_timeout_s'],checked)
                guard.check()
            if tool:
                verify_profile(tool,directory,loaded,credentials)
            if not read(directory/'worker_complete.json')['complete']:
                raise RuntimeError('Worker completion marker missing')
        if hashes()!=manifest['source_hashes']:
            raise RuntimeError('Sources changed while running')
        load()
    try:
        controlled_run(out,loaded,config,execute)
    finally:
        if (out/'manifest.json').exists():
            analyze(out,loaded,config)
    print('COLLECTION COMPLETE: 128 diagnostic trials and 4 profiles; inspect summary.json.',flush=True)
    return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','run','worker','analyze','pack'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--case',type=int,choices=[0,1])
    parser.add_argument('--tool',choices=['nsys','ncu'])
    args = parser.parse_args()
    if args.action == 'pack':
        frozen.pack(args.output.resolve())
        return 0
    loaded,config = load()
    if args.action == 'verify':
        print(f'VERIFY PASS: {len(read(HERE / "dependency_hashes.json"))} frozen dependencies, 2 shapes, 128 trials, 4 profiles.')
        return 0
    if args.output is None:
        parser.error('--output is required')
    args.output = args.output.resolve()
    if args.action == 'run':
        root = Path('/home/eu/gemm-research').resolve()
        if root not in args.output.parents or os.geteuid()==0:
            raise RuntimeError('Run as the normal user with output under /home/eu/gemm-research')
        return run(args.output,loaded,config)
    if args.action == 'worker':
        if args.directory is None or args.case is None:
            parser.error('worker requires --directory and --case')
        worker(args,loaded,config)
        return 0
    summary = analyze(args.output,loaded,config)
    print('DIAGNOSTIC '+summary['gate']+': '+summary['collection'])
    return 2 if summary['gate']=='REVIEW' else 1


if __name__ == '__main__':
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
