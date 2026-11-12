#!/usr/bin/env python3
"""Check the 900/6801 MHz setting using the unchanged clock-audit worker."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
TARGET = {'clocks.sm': 900.0, 'clocks.mem': 6801.0, 'power.limit': 70.0}


def progress(message):
    try:
        print(message, flush=True)
    except BrokenPipeError:
        # Ctrl-C can close an outer tee before the GPU reset finishes.
        # Log files remain authoritative; a closed UI pipe must not abort cleanup.
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, sys.stdout.fileno())
        finally:
            os.close(null)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def load(base, directory):
    expected = read(HERE/'dependency_hashes.json')
    for name, wanted in expected.items():
        if digest(base/name) != wanted:
            raise RuntimeError('clock_audit_v1 dependency changed: '+name)
    sys.path.insert(0, str(base))
    import audit
    import analyze
    import telemetry
    for module in (audit, analyze, telemetry):
        if Path(module.__file__).resolve().parent != base:
            raise RuntimeError('Wrong dependency imported: '+module.__name__)
    baseline, _, _, suite = audit.load_suite(directory)
    return audit, analyze, telemetry, baseline, suite


def summarize(out, audit, analyze, telemetry, baseline):
    directory = out/'dense_before'
    rows = read(directory/'results.json')
    if [row['task'] for row in rows] != audit.cases():
        raise ValueError('Missing, duplicate or changed tasks')
    env = read(directory/'environment.json')
    if {k:env[k] for k in baseline['environment_signature']} != baseline['environment_signature']:
        raise ValueError('Environment differs from v6')
    meta = read(directory/'telemetry_metadata.json')
    if meta['fields'] != list(telemetry.FIELDS) or meta['requested_interval_ms'] != 100:
        raise ValueError('Telemetry configuration changed')
    samples, cadence = telemetry.read(directory/'telemetry.csv', meta['timezone_offset_seconds'], 100)
    if any(s['uuid'] != baseline['environment_signature']['gpu_uuid'] for s in samples):
        raise ValueError('Telemetry UUID mismatch')
    flags = []
    measurements, windows = [], []
    groups = defaultdict(list)
    counts = Counter()
    for row in rows:
        task, result = row['task'], row['result']
        local_flags = analyze.check_timing(result, task, baseline)
        collected = []
        for i, block in enumerate(result['timing']['blocks']):
            window = telemetry.window(samples, block['host_envelope_start_utc'], block['host_envelope_end_utc'])
            windows.append(dict(task_id=task['id'], block=i, gemm_us=block['latency_ms']*1000, window=window))
            if not window['inside']:
                local_flags.append(f'block_{i}:no_inside_clock_sample')
            collected.extend(window['inside'])
        for field, expected in TARGET.items():
            if any(s[field] != expected for s in collected):
                local_flags.append('observed_mismatch:'+field)
        for reason in telemetry.REASONS:
            if any(s[reason] is True for s in collected):
                local_flags.append('observed_active:'+reason)
            if any(s[reason] is None for s in collected):
                local_flags.append('unavailable:'+reason)
        local_flags = sorted(set(local_flags))
        flags.extend(task['id']+':'+f for f in local_flags)
        counts['REVIEW' if local_flags else 'PASS'] += 1
        med = result['timing']['median_ms']*1000
        groups[(tuple(task['shape']), task['dtype'], task['mode'])].append(med)
        measurements.append(dict(task=task, median_us=med,
            relative_iqr=result['timing']['relative_iqr'], first_last_drift=result['timing']['first_last_drift'],
            inside_samples=len(collected), flags=local_flags,
            observed_values={field:sorted({s[field] for s in collected}) for field in TARGET},
            correctness=result['correctness']))
    rounds = []
    for (shape, dtype, mode), values in groups.items():
        if len(values) != 2:
            raise ValueError('Expected two rounds')
        spread = (max(values)-min(values))/statistics.median(values)
        rounds.append(dict(shape=shape, dtype=dtype, mode=mode, round_us=values, relative_range=spread))
        if spread > baseline['protocol']['between_round_limit']:
            flags.append(f'{shape}/{dtype}/{mode}:round_range_above_5pct')
    summary = dict(gate='REVIEW' if flags else 'PASS', expected=8, recorded=8, missing=0,
                   counts=dict(counts), timed_blocks=len(windows), flags=sorted(set(flags)),
                   target=TARGET, full_sweep='HOLD', interpretation='NOT_REVIEWED')
    save(out/'measurement_summary.json', summary)
    save(out/'analysis.json', dict(summary=summary, measurements=measurements, rounds=rounds, cadence=cadence,
        caveats=['Clock match is checked at recorded host-envelope samples, not continuously.',
                 '100 ms polling does not establish sensor refresh resolution.',
                 'This single process has no fresh matched automatic-core control; it cannot isolate causality.',
                 'Counter deltas are not converted to short-window throttling fractions.']))
    save(out/'clock_windows.json', windows)
    lines = ['# 900/6801 MHz hold check', '', 'Measurement quality: '+summary['gate'], '',
             '| shape / dtype / mode | round 0 µs | round 1 µs | range / median |',
             '|---|---:|---:|---:|']
    for row in rounds:
        lines.append(f"| {row['shape']} {row['dtype']} {row['mode']} | {row['round_us'][0]:.3f} | {row['round_us'][1]:.3f} | {row['relative_range']:.2%} |")
    lines += ['', 'Flags:'] + ['- '+f for f in summary['flags']]
    lines += ['', '명령 수락·관측 클럭 일치·측정 품질·복원 명령 수락은 서로 구분한다.',
              '실제 기록 표본의 일치이며 연속 고정이나 다른 형상의 안정성을 보증하지 않는다.',
              '새 자동 코어 대조군이 없어 원인 확정 실험은 아니다. full은 보류한다.',
              '최종 복원 결과는 restore.json, 최종 상태는 summary.json을 확인한다.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def clock_command(suite, out, smi, gpu_uuid, name, *options):
    result = suite.command_text(['sudo', '-n', smi, '-i', gpu_uuid, *options])
    save(out/(name+'.json'), result)
    return result


def restore(suite, out, smi, gpu_uuid):
    # Both resets are attempted, even if the first fails. Keep the shared lock held.
    previous = {sig:signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        results = {}
        for name, option in [('graphics','-rgc'), ('memory','-rmc')]:
            try:
                results[name] = clock_command(suite, out, smi, gpu_uuid, 'reset_'+name, option)
            except Exception as error:
                results[name] = {'error':str(error)}
        accepted = all(r.get('exit_code') == 0 for r in results.values())
        state = dict(commands_accepted=accepted, commands=results,
                     note='Reset command acceptance; not proof of a specific idle clock.')
        save(out/'restore.json', state)
        save(out/'gpu_after_restore.json', suite.command_text([smi,'-i',gpu_uuid,'-q']))
        progress('RESTORE: '+('COMMANDS ACCEPTED' if accepted else 'FAILED; inspect restore.json'))
        if not accepted:
            progress(f'MANUAL RESTORE: sudo nvidia-smi -i {gpu_uuid} -rgc')
            progress(f'MANUAL RESTORE: sudo nvidia-smi -i {gpu_uuid} -rmc')
        return accepted
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run(args):
    audit, analyze, telemetry, baseline, suite = load(args.base, args.suite)
    out = args.output
    if (out/'manifest.json').exists():
        raise RuntimeError('Use a fresh output directory')
    gpu_uuid = baseline['environment_signature']['gpu_uuid']
    smi = shutil.which('nvidia-smi')
    if not smi:
        raise RuntimeError('nvidia-smi unavailable')
    snapshot = suite.command_text([smi,'--query-gpu=uuid','--format=csv,noheader'])
    save(out/'gpu_identity.json', snapshot)
    if snapshot.get('exit_code') != 0 or snapshot['stdout'].strip() != gpu_uuid:
        raise RuntimeError('Target GPU UUID does not match')
    dependencies = read(HERE/'dependency_hashes.json')
    own = {p.name:digest(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh','.json','.md')}
    save(out/'manifest.json', dict(target=TARGET, cases=audit.cases(), baseline=baseline,
        dependency_hashes=dependencies, source_hashes=own, kind='fixed_clock_hold_check',
        worker_pass='dense_before', new_automatic_control=False, restore_to='default GPU and memory clocks'))
    for source, names, destination in [(HERE,own,'source'),(args.base,dependencies,'audit_source'),
                                       (args.suite,baseline['code_hashes'],'baseline_source')]:
        for name in names:
            target = out/destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,target)
    status = dict(collection='NOT_STARTED', restoration='NOT_ATTEMPTED', full_sweep='HOLD')
    try:
        with suite.lock(Path('/tmp')/f'gemm-sweep-{gpu_uuid}.lock'):
            # Do not change or reset clocks if another workload prevents admission.
            audit.check_processes(suite,gpu_uuid,out/'processes_before.json')
            summary = None
            try:
                save(out/'gpu_before_settings.json',suite.command_text([smi,'-i',gpu_uuid,'-q']))
                for name, options in [('set_graphics',['-lgc','900,900']),('set_memory',['-lmc','6801,6801'])]:
                    if clock_command(suite,out,smi,gpu_uuid,name,*options).get('exit_code') != 0:
                        raise RuntimeError('Clock request failed: '+name)
                status['collection'] = 'RUNNING'
                progress('180 s preheat + 8 measurements; progress: '+str(out/'worker.log'))
                audit.run_logged(audit.worker_command(args.suite,out,'dense_before'),out,'worker',timeout=900)
                summary = summarize(out,audit,analyze,telemetry,baseline)
                load(args.base,args.suite)
                status['collection'] = 'COMPLETE'
            finally:
                restored = restore(suite,out,smi,gpu_uuid)
                status['restoration'] = 'COMMANDS_ACCEPTED' if restored else 'FAILED'
            if not restored:
                summary = dict(summary,gate='FAIL',restoration='FAILED')
            else:
                summary['restoration'] = 'COMMANDS_ACCEPTED'
            save(out/'summary.json',summary)
            progress('HOLD CHECK: '+summary['gate']+'; no automatic full sweep.')
            return {'PASS':0,'REVIEW':2,'FAIL':1}[summary['gate']]
    except BaseException as error:
        status.update(collection='FAILED_OR_INTERRUPTED',error=str(error))
        raise
    finally:
        save(out/'status.json',status)
        if status['restoration']=='NOT_ATTEMPTED':
            progress('RESTORE NOT ATTEMPTED: admission failed before this run changed settings. Your earlier manual locks may remain.')


def package(out):
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path,str(Path(out.name)/path.relative_to(out)))
    archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
    print('RESULT_ZIP='+str(archive),flush=True)


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','run','pack'])
    parser.add_argument('--base',type=Path,default=HERE.parent/'clock_audit_v1')
    parser.add_argument('--suite',type=Path,default=HERE.parent/'gpu_sweep_v6')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    args.base,args.suite=args.base.resolve(),args.suite.resolve()
    if args.output:
        args.output=args.output.resolve()
    if args.action=='verify':
        load(args.base,args.suite)
        print('DEPENDENCY CHECK: PASS')
    elif args.action=='pack':
        package(args.output)
    else:
        return run(args)


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
