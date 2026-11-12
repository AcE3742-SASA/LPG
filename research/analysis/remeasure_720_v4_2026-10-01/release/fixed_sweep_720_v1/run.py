#!/usr/bin/env python3
"""720 MHz full GEMM collection with the verified measurement and cleanup implementation."""
import argparse
from collections import Counter
from itertools import groupby
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import tempfile
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
MODES = ('allocate', 'prealloc', 'graph')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def own_hashes():
    return {p.name: sha(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.md')}


def load(sm_mhz=None):
    for name, wanted in read(HERE/'dependency_hashes.json').items():
        if sha(HERE.parent/name) != wanted:
            raise RuntimeError('Frozen dependency changed: '+name)
    sys.path.insert(0, str(HERE.parent/'fixed_pilot_780_v1'))
    fixed = importlib.import_module('fixed')
    if Path(fixed.__file__).resolve().parent != HERE.parent/'fixed_pilot_780_v1':
        raise RuntimeError('Wrong fixed module')
    steps, hold, audit, original, telemetry, baseline, suite, pinned, _ = fixed.load()
    config=read(HERE/'protocol.json')
    if (config['sm_mhz'],config['requested_sm_mhz'],config['fallback_sm_mhz'],config['memory_mhz'],config['power_limit_w']) != (720,720,None,6801,70):
        raise ValueError('Frozen 720 / 6801 MHz / 70 W changed')
    if sm_mhz is not None:
        if sm_mhz != 720:
            raise ValueError('Unapproved selected core clock')
        config['sm_mhz']=sm_mhz
    if (config['full_rounds'],config['interval_ms'],config['segment_tasks']) != (3,50,120):
        raise ValueError('Rounds, telemetry interval or segment size changed')
    if config['reviewed_graph_cases'][:6] != read(HERE.parent/'clock_recheck_v1/protocol.json')['fixed_cases']:
        raise ValueError('Reviewed Graph cases changed')
    if config['segment_tasks']<1 or any(sorted(m)!=sorted(MODES) for m in config['mode_order']):
        raise ValueError('Invalid segment size or mode coverage')
    return dict(steps=steps, hold=hold, audit=audit, original=original, telemetry=telemetry,
                baseline=baseline, suite=suite, pinned=pinned, plan=sys.modules['plan'],
                measure=sys.modules['measure'], config=config)


def make_plan(size, loaded):
    old, config = loaded['plan'], loaded['config']
    if size == 'check':
        bases = [dict(kind='gemm', phase='check', shape=s['shape'], dtype=s['dtype'],
                      round=r, split='development') for r in range(config['check_rounds'])
                 for s in (config['check_cases'] if r == 0 else config['check_cases'][::-1])]
        splits = []
    else:
        original = old.make_plan('full')
        bases = [{k:v for k,v in t.items() if k not in ('id','mode')}
                 for t in original['tasks'] if t['kind']=='gemm' and t['mode']=='prealloc'
                 and t['phase'] in config['scope']]
        splits = original['shape_splits']
    tasks = []
    for base in bases:
        for mode in config['mode_order'][base['round']]:
            t = dict(base, mode=mode)
            tasks.append(dict(t, id=old.digest(t)[:24]))
    tasks = json.loads(json.dumps(tasks))
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate tasks')
    return dict(size=size, protocol=old.PROTOCOL, config=config, tasks=tasks,
                shape_splits=splits, modes=MODES, selected_clock_mhz=config['sm_mhz'],
                rounds=config['check_rounds' if size=='check' else 'full_rounds'])


def history(out, latest=False):
    path = out/'results.sqlite3'
    if not path.exists():
        return
    with contextlib.closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as con:
        query = ('SELECT a.seq,a.record FROM attempts a JOIN '
                 '(SELECT task_id,MAX(seq) seq FROM attempts GROUP BY task_id) b ON a.seq=b.seq '
                 'ORDER BY a.seq') if latest else 'SELECT seq,record FROM attempts ORDER BY seq'
        for seq, value in con.execute(query):
            yield seq, json.loads(value)


def pinned_choices(loaded):
    choices = {loaded['plan'].graph_key(s): s['graph_gemms_per_replay']
               for s in loaded['pinned']['fixed_cases']}
    for spec in loaded['config']['reviewed_graph_cases']:
        key, nodes = loaded['plan'].graph_key(spec), spec['graph_gemms_per_replay']
        if choices.setdefault(key, nodes) != nodes:
            raise ValueError('Reviewed Graph batch conflicts with the frozen pilot')
    return choices


def choices_and_done(out, manifest, loaded):
    choices, done = pinned_choices(loaded), {}
    expected = {t['id']: t for t in manifest['tasks']}
    for _, row in history(out):
        t = row['task']
        if expected.get(t['id']) != t:
            raise ValueError('Saved task changed or is unplanned')
        if t['mode']=='graph' and 'timing' in row:
            n = row['timing']['graph_nodes']
            if type(n) is not int or not 1 <= n <= 64:
                raise ValueError('Invalid saved Graph batch')
            if choices.setdefault(loaded['plan'].graph_key(t), n) != n:
                raise ValueError('Saved Graph batch changed')
        done[t['id']] = row['status']
    return choices, done


def manifest_for(out, size, loaded):
    suite = loaded['suite']
    expected = dict(make_plan(size, loaded), kind='fixed_sweep_720_v1',
                    source_hashes=own_hashes(), dependency_hashes=read(HERE/'dependency_hashes.json'),
                    environment_signature=loaded['baseline']['environment_signature'],
                    pinned_graph_choices=pinned_choices(loaded))
    expected = json.loads(json.dumps(expected))
    path = out/'manifest.json'
    if path.exists():
        if read(path) != expected:
            raise RuntimeError('Manifest/code/protocol mismatch; use a new run directory')
        for folder,hashes in [('source',expected['source_hashes']),('dependencies',expected['dependency_hashes'])]:
            if any(sha(out/folder/name)!=wanted for name,wanted in hashes.items()):
                raise RuntimeError('Saved source snapshot changed: '+folder)
    else:
        if (out/'results.sqlite3').exists():
            raise RuntimeError('Database without a manifest; preserve and inspect it')
        suite.atomic_json(path, expected)
        for source, folder, hashes in [(HERE, 'source', expected['source_hashes']),
                                       (HERE.parent, 'dependencies', expected['dependency_hashes'])]:
            for name in hashes:
                dest = out/folder/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/name, dest)
    return expected


def check_admission(path, note, loaded):
    if not path or not note or not note.strip():
        raise RuntimeError('Full requires the reviewed 200-shape ZIP and a review note')
    path = path.resolve()
    expected = loaded['config']['reviewed_200_shape_result']
    if not path.is_file() or sha(path) != expected['zip_sha256']:
        raise RuntimeError('Not the reviewed completed 720 MHz 200-shape ZIP; no GPU settings changed')
    prefix = expected['archive_directory'] + '/'
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Reviewed ZIP CRC failed')
        manifest_bytes = archive.read(prefix + 'manifest.json')
        manifest = json.loads(manifest_bytes)
        result = json.loads(archive.read(prefix + 'summary.json'))
    if (hashlib.sha256(manifest_bytes).hexdigest() != expected['manifest_sha256']
            or result['results_digest'] != expected['results_digest']
            or manifest['selected_clock_mhz'] != 720
            or manifest['protocol'] != loaded['plan'].PROTOCOL
            or manifest['environment_signature'] != loaded['baseline']['environment_signature']
            or result['recorded'] != 3600 or result['missing'] != 0
            or result['counts'] != {'PASS': 3600}
            or not result['timing_and_sampled_clocks_pass'] or not result['all_restorations_accepted']):
        raise RuntimeError('Reviewed 200-shape evidence mismatch')
    return dict(check_directory=str(path), check_archive_sha256=expected['zip_sha256'],
                manifest_sha256=expected['manifest_sha256'], results_digest=result['results_digest'],
                check_gate=result['gate'], reviewed_measurements=3600, review_note=note,
                known_counter_issue=loaded['config']['known_counter_issue'])


def require_clean_resume(out, loaded):
    # Raw PASS excludes telemetry; recheck saved samples before trusting resume decisions.
    from analyze_sweep import segment_quality
    for relative, group in groupby(history(out, latest=True), key=lambda pair: pair[1]['segment']):
        rows = [row for _,row in group if row['status'] != 'FAIL']
        if not rows:
            continue  # Explicit --retry-failed still controls raw computation failures.
        directory = out/relative
        if directory.resolve().parent.parent != (out/'sessions').resolve():
            raise ValueError('Invalid saved segment path')
        env = read(directory.parent/'environment.json')
        if loaded['suite'].signature(env) != loaded['baseline']['environment_signature']:
            raise RuntimeError('Saved environment differs; preserve and inspect it')
        quality = segment_quality(directory, rows, loaded)
        if not quality.get('collection_can_continue', quality['timing_and_sampled_clocks_pass']):
            raise RuntimeError('Saved clock/telemetry/data problem: resume refused. '
                               'Preserve this run; review it before using a fresh output directory.')


def worker(args, loaded):
    suite, audit, measure, telemetry = [loaded[k] for k in ('suite','audit','measure','telemetry')]
    manifest = read(args.output/'manifest.json')
    if manifest != manifest_for(args.output, manifest['size'], loaded):
        raise RuntimeError('Worker manifest differs')
    choices, done = choices_and_done(args.output, manifest, loaded)
    require_clean_resume(args.output, loaded)
    if 'FAIL' in done.values() and not args.retry_failed:
        raise RuntimeError('Earlier FAIL retained; inspect before --retry-failed')
    pending = [t for t in manifest['tasks'] if t['id'] not in done or
               (args.retry_failed and done[t['id']]=='FAIL')]
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        return 0
    import torch
    env = suite.environment(torch)
    if suite.signature(env) != manifest['environment_signature']:
        raise RuntimeError('GPU/software/precision differs from the reviewed baseline')
    if torch.cuda.mem_get_info()[0] < 4*1024**3:
        raise RuntimeError('Less than 4 GiB free GPU memory')
    session = dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    session_dir = args.output/'sessions'/session
    session_dir.mkdir(parents=True)
    suite.atomic_json(session_dir/'environment.json', env)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print('STOP: finish and save this task; no next task.', flush=True)
    old_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    for sig in old_handlers:
        signal.signal(sig, request_stop)
    store = suite.Store(args.output)
    original_timer = measure.Timer
    measure.Timer = audit.observed_timer(original_timer)
    started, count = time.monotonic(), 0
    state = 'ERROR'
    try:
        step = loaded['config']['segment_tasks']
        for offset in range(0, len(pending), step):
            if stop:
                break
            segment = session_dir/f'segment_{offset//step:05d}'
            segment.mkdir()
            batch = []
            try:
                with telemetry.collect(segment, env['gpu_uuid'], loaded['config']['interval_ms'], suite) as monitor:
                    audit.check_processes(suite, env['gpu_uuid'], segment/'processes_before.json')
                    if offset == 0:
                        with torch.inference_mode():
                            x = torch.randn((1024,1024), device='cuda')*.01
                            y = torch.empty_like(x)
                            timer = measure.Timer(torch, manifest['protocol'])
                            start = time.monotonic()
                            print('Preheating 180 seconds', flush=True)
                            while time.monotonic()-start < manifest['protocol']['preheat_seconds'] and not stop:
                                timer.block(lambda: torch.mm(x,x,out=y),100)
                            del x,y
                    for task in pending[offset:offset+step]:
                        if stop:
                            break
                        if monitor.poll() is not None:
                            raise RuntimeError('Telemetry stopped')
                        audit.check_processes(suite, env['gpu_uuid'], segment/('processes_'+task['id']+'.json'))
                        begin = time.monotonic()
                        row = dict(task=task, session=session, segment=str(segment.relative_to(args.output)),
                                   timestamp=suite.now())
                        try:
                            data = measure.measure(torch, task, manifest['protocol'],
                                graph_nodes=suite.graph_nodes_for(task, choices))
                            if task['mode']=='graph':
                                key,n = loaded['plan'].graph_key(task),data['timing']['graph_nodes']
                                if choices.setdefault(key,n) != n:
                                    raise RuntimeError('Graph batch changed during collection')
                            row.update(data, status='FAIL' if not data['correctness']['passed'] else
                                       'REVIEW' if data['timing']['flags'] else 'PASS')
                        except Exception as e:
                            row.update(status='FAIL', error=str(e), traceback=traceback.format_exc())
                        row['task_wall_seconds'] = time.monotonic()-begin
                        store.append(row)
                        batch.append(row)
                        done[task['id']] = row['status']
                        count += 1
                        suite.atomic_json(args.output/'progress.json', dict(state='RUNNING', session=session,
                            recorded=len(done), expected=len(manifest['tasks']), current_task=task,
                            updated_utc=suite.now()))
                        print(f"[{len(done)}/{len(manifest['tasks'])}] {task['phase']} {task['shape']} "
                              f"{task['dtype']} {task['mode']} r{task['round']}: {row['status']} "
                              f"({row['task_wall_seconds']:.2f}s)", flush=True)
                        if row['status']=='FAIL':
                            raise RuntimeError('Failed task preserved; stop collection')
            finally:
                # Chunking bounds telemetry RAM; measurement protocol and preheat stay unchanged.
                if batch:
                    from analyze_sweep import segment_quality
                    quality = segment_quality(segment, batch, loaded)
                    suite.atomic_json(segment/'quality.json', quality)
            if batch and not quality.get('collection_can_continue',quality['timing_and_sampled_clocks_pass']):
                raise RuntimeError('Clock/telemetry/data problem in this segment; stop before further collection')
        state = 'STOPPED' if stop else 'FINISHED'
    finally:
        measure.Timer = original_timer
        store.close()
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)
        suite.atomic_json(session_dir/'session.json', dict(state=state, new_records=count,
            elapsed_seconds=time.monotonic()-started, ended_utc=suite.now()))
        suite.atomic_json(args.output/'progress.json', dict(state=state, session=session,
            recorded=len(done), expected=len(manifest['tasks']), updated_utc=suite.now()))
    return 130 if stop else 0


def run_worker(command, directory, name, timeout, credentials):
    (directory/(name+'.command.json')).write_text(json.dumps(command)+'\n')
    with (directory/(name+'.log')).open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.monotonic()
        try:
            while True:
                credentials.check()
                if time.monotonic()-start > timeout:
                    raise TimeoutError('Worker time limit reached')
                try:
                    code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            with cleanup_signals():
                if process.poll() is None:
                    # Let the worker commit its task before terminating its owned group.
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if process.poll() is None:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            (directory/(name+'.exit_code.txt')).write_text(str(process.returncode)+'\n')
        if code != 0:
            raise RuntimeError(f'Worker exited {code}; inspect {directory/(name+".log")}')


@contextlib.contextmanager
def cleanup_signals():
    previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        for s in previous:
            signal.signal(s,signal.SIG_IGN)
        yield
    finally:
        for s,handler in previous.items():
            signal.signal(s,handler)


def choose_supported_clock(stdout, preferred=None):
    values=set()
    for line in stdout.splitlines():
        value=line.strip()
        if not value:
            continue
        if not value.isdecimal():
            raise RuntimeError('Supported graphics clock query is unavailable or malformed')
        values.add(int(value))
    candidates=(720,)
    if preferred is not None and preferred != 720:
        raise RuntimeError('Saved/check clock is not authorized')
    for value in candidates:
        if value in values:
            return value
    raise RuntimeError('No exact authorized clock is supported; no clock settings changed')


def resolve_clock(args, loaded):
    # Preserve the reviewed 720 MHz target; reject unsupported or changed saved clocks.
    saved=args.output/'manifest.json'
    if args.size=='full' and (not args.check or not args.review_note or not args.review_note.strip()):
        raise RuntimeError('Full requires a reviewed check ZIP and explicit --review-note')
    preferred=read(saved)['selected_clock_mhz'] if saved.exists() else 720
    smi=shutil.which('nvidia-smi')
    if not smi:
        raise RuntimeError('nvidia-smi unavailable')
    gpu=loaded['baseline']['environment_signature']['gpu_uuid']
    result=loaded['suite'].command_text([smi,'-i',gpu,'--query-supported-clocks=graphics','--format=csv,noheader,nounits'])
    name='clock_selection_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]+'.json'
    evidence=dict(requested_mhz=720,authorized_fallback_mhz=None,preferred_saved_or_check_mhz=preferred,query=result)
    try:
        if result.get('exit_code')!=0:
            raise RuntimeError('Supported clock query failed; no clock settings changed')
        selected=choose_supported_clock(result.get('stdout',''),preferred)
        evidence.update(selected_clock_mhz=selected,selection='requested')
    finally:
        loaded['suite'].atomic_json(args.output/name,evidence)
    print(f'Requested: 720 MHz; selected exact supported target: {selected} MHz',flush=True)
    return load(selected)


def run(args, loaded):
    suite, steps, hold, audit = [loaded[k] for k in ('suite','steps','hold','audit')]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with suite.lock(out/'.run.lock'):
        approval = check_admission(args.check, args.review_note, loaded) if args.size=='full' else None
        loaded=resolve_clock(args,loaded)
        manifest = manifest_for(out, args.size, loaded)
        _, done = choices_and_done(out, manifest, loaded)
        require_clean_resume(out, loaded)
        if 'FAIL' in done.values() and not args.retry_failed:
            raise RuntimeError('Earlier FAIL retained; review it before --retry-failed')
        if len(done)==len(manifest['tasks']) and 'FAIL' not in done.values():
            from analyze_sweep import analyze
            summary = analyze(out, loaded)
            print('Already collected; no GPU settings changed.', flush=True)
            return 0 if summary['gate']=='PASS' else 2
        if shutil.disk_usage(out).free < loaded['config']['minimum_free_disk_gib']*1024**3:
            raise RuntimeError('Need at least 20 GiB free disk before starting/resuming')
        attempt = out/'launches'/(dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        attempt.mkdir(parents=True)
        if approval:
            suite.atomic_json(attempt/'check_review.json', approval)
        state = dict(collection='NOT_STARTED', restoration='NOT_ATTEMPTED')
        error = None
        gpu = manifest['environment_signature']['gpu_uuid']
        smi = shutil.which('nvidia-smi')
        if not smi:
            raise RuntimeError('nvidia-smi unavailable')
        with suite.lock(Path('/tmp')/f'gemm-sweep-{gpu}.lock'):
            try:
                identity = suite.command_text([smi,'--query-gpu=uuid','--format=csv,noheader'])
                suite.atomic_json(attempt/'gpu_identity.json',identity)
                if identity.get('exit_code') != 0 or identity.get('stdout','').strip()!=gpu:
                    raise RuntimeError('Target GPU UUID mismatch')
                audit.check_processes(suite,gpu,attempt/'processes_before.json')
                steps.require_power(suite,attempt,smi,gpu,'power_before',loaded['config'])
                steps.snapshot(suite,attempt,smi,gpu,'gpu_before_settings')
                with steps.Credentials(suite,attempt) as credentials:
                    try:
                        steps.require_clock(hold,suite,attempt,smi,gpu,'set_memory','-lmc','6801,6801')
                        steps.require_clock(hold,suite,attempt,smi,gpu,'set_graphics','-lgc',f"{loaded['config']['sm_mhz']},{loaded['config']['sm_mhz']}")
                        state['collection']='RUNNING'
                        limits = loaded['config']['check_worker_limits'] if args.size=='check' else [None]
                        for i,limit in enumerate(limits):
                            # On check resume, never collect both rounds in one fresh worker.
                            if args.size=='check':
                                _, current = choices_and_done(out,manifest,loaded)
                                remaining_first = sum(t['round']==0 and (t['id'] not in current or
                                    (args.retry_failed and current[t['id']]=='FAIL')) for t in manifest['tasks'])
                                if i==0 and not remaining_first:
                                    continue
                                if i==0:
                                    limit=remaining_first
                            command=[sys.executable,'-u','-B',str(HERE/'run.py'),'worker','--output',str(out)]
                            if limit is not None:
                                command += ['--limit',str(limit)]
                            if args.retry_failed:
                                command += ['--retry-failed']
                            print('Worker log: '+str(attempt/f'worker_{i}.log'),flush=True)
                            run_worker(command,attempt,f'worker_{i}',
                                loaded['config'][args.size+'_worker_timeout_s'],credentials)
                        credentials.check()
                        state['collection']='COMPLETE'
                    finally:
                        with cleanup_signals():
                            restored=hold.restore(suite,attempt,smi,gpu)
                        state['restoration']='COMMANDS_ACCEPTED' if restored else 'FAILED'
                    if not restored:
                        raise RuntimeError('Clock reset failed; inspect restore.json')
            except BaseException as e:
                state['collection']='FAILED_OR_INTERRUPTED'
                error=e
            finally:
                suite.atomic_json(attempt/'status.json',state)
        from analyze_sweep import analyze
        summary=analyze(out,loaded)
        if own_hashes()!=manifest['source_hashes']:
            raise RuntimeError('Source changed during collection')
        load()
        if error:
            raise error
        return 0 if summary['gate']=='PASS' else 2 if summary['gate']=='REVIEW' else 1


def pack(out):
    # Packaging must also work after dependency/preflight failures.
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target=out.with_name(out.name+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]+'.zip')
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.rglob('*')):
                if path.is_file() and path.name!='.run.lock':
                    z.write(path,str(Path(out.name)/path.relative_to(out)))
        target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n')
        print('RESULT_ZIP='+str(target),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','plan','run','worker','analyze','status','pack'])
    parser.add_argument('--size',choices=['check','full'],default='check')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',type=Path)
    parser.add_argument('--review-note')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.output:
        args.output=args.output.resolve()
    if args.action not in ('verify','plan') and not args.output:
        parser.error('--output required')
    if args.limit is not None and args.limit<1:
        parser.error('--limit must be positive')
    if args.action=='pack':
        pack(args.output)
        return 0
    saved=args.output/'manifest.json' if args.output else None
    selected=read(saved)['selected_clock_mhz'] if saved and saved.exists() else None
    loaded=load(selected)
    if args.action in ('verify','plan'):
        value=make_plan(args.size,loaded)
        print(json.dumps(dict(status='PASS',size=args.size,tasks=len(value['tasks']),
            phases=dict(Counter(t['phase'] for t in value['tasks'])),
            triplets=len(value['tasks'])//3,task_digest=loaded['plan'].digest(value['tasks'])),indent=2))
    elif args.action=='run':
        return run(args,loaded)
    elif args.action=='worker':
        return worker(args,loaded)
    elif args.action=='status':
        for name in ['progress.json','summary.json']:
            if (args.output/name).exists():
                print(name, json.dumps(read(args.output/name),ensure_ascii=False))
    else:
        from analyze_sweep import analyze
        with loaded['suite'].lock(args.output/'.run.lock'):
            print(json.dumps(analyze(args.output,loaded),indent=2))


if __name__=='__main__':
    def stop(signum,frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,stop)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
