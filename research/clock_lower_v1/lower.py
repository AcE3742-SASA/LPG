#!/usr/bin/env python3
"""Screen 750/780/810 MHz with the existing v6 allocate/prealloc/Graph worker."""
import argparse
import contextlib
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import signal
import sys
import traceback
import zipfile

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def own_hashes():
    return {p.name: digest(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.sh', '.json', '.md')}


def cases(config):
    tasks = []
    for rd in range(config['rounds']):
        indices = range(len(config['fixed_cases']))
        if rd:
            indices = reversed(indices)
        for index in indices:
            spec = config['fixed_cases'][index]
            offset = index % len(config['modes'])
            modes = config['modes'][offset:] + config['modes'][:offset]
            for mode in (reversed(modes) if rd else modes):
                tasks.append(dict(id=f'c{index}_{mode}_r{rd}', kind='gemm', shape=spec['shape'],
                    dtype=spec['dtype'], mode=mode, round=rd, phase='clock_lower', split='diagnostic_only'))
    return tasks


def load():
    for name, wanted in read(HERE/'dependency_hashes.json').items():
        if digest(HERE.parent/name) != wanted:
            raise RuntimeError('Dependency changed: '+name)
    sys.path.insert(0, str(HERE.parent/'fixed_pilot_v1'))
    fixed = importlib.import_module('fixed')
    if Path(fixed.__file__).resolve().parent != HERE.parent/'fixed_pilot_v1':
        raise RuntimeError('Wrong fixed module')
    steps, hold, audit, original, telemetry, baseline, suite, previous, _ = fixed.load()
    config = read(HERE/'protocol.json')
    if config['modes'] != ['allocate','prealloc','graph'] or config['rounds'] != 2:
        raise RuntimeError('Execution modes/rounds changed')
    if len(config['fixed_cases']) != 10 or len({(tuple(s['shape']),s['dtype']) for s in config['fixed_cases']}) != 10:
        raise RuntimeError('Condition coverage changed')
    if config['memory_mhz'] != 6801 or config['power_limit_w'] != 70:
        raise RuntimeError('Memory or power protocol changed')
    expected_stages = [('s0_750_before',750),('s1_780',780),('s2_810',810),('s3_750_after',750)]
    if config['stages'] != [dict(name=n,sm_mhz=c,interval_ms=50) for n,c in expected_stages]:
        raise RuntimeError('Candidate schedule changed')
    for spec in config['fixed_cases']:
        if spec not in previous['fixed_cases']:
            raise RuntimeError('Pinned seed, Graph batch or correctness scope changed')
    extended = copy.deepcopy(baseline)
    extended['fixed_cases'] = config['fixed_cases']
    return steps, hold, audit, original, telemetry, extended, suite, config


@contextlib.contextmanager
def configured_worker(loaded):
    steps, _, audit, _, _, _, _, config = loaded
    previous = audit.cases, audit.graph_nodes, audit.PASSES
    audit.cases = lambda: cases(config)
    audit.graph_nodes = lambda task: steps.graph_nodes(config, task)
    audit.PASSES = {s['name']:s['interval_ms'] for s in config['stages']}
    try:
        yield
    finally:
        audit.cases, audit.graph_nodes, audit.PASSES = previous


def make_manifest(out, loaded):
    _, _, _, _, _, baseline, suite, config = loaded
    manifest = dict(kind='clock_lower_v1', config=config, baseline=baseline, cases=cases(config),
        source_hashes=own_hashes(), dependency_hashes=read(HERE/'dependency_hashes.json'),
        expected_measurements=240, expected_blocks=1680, expected_mode_triplets=80,
        started_utc=suite.now(), full_sweep='HOLD', selected_clock_mhz=None,
        new_automatic_core_load_control=False, new_profiler_capture=False,
        restore_to='default clocks, not any pre-existing manual locks')
    suite.atomic_json(out/'manifest.json', manifest)
    for folder, source, hashes in [('source', HERE, manifest['source_hashes']),
                                  ('dependencies', HERE.parent, manifest['dependency_hashes'])]:
        for name in hashes:
            target = out/folder/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/name, target)
    return manifest


def run(args):
    loaded = load()
    steps, hold, audit, _, _, baseline, suite, config = loaded
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise RuntimeError('Use a fresh result directory')
    manifest = make_manifest(out, loaded)
    status = dict(collection='NOT_STARTED', restoration='NOT_ATTEMPTED', full_sweep='HOLD')
    try:
        smi = shutil.which('nvidia-smi')
        if not smi:
            raise RuntimeError('nvidia-smi unavailable')
        gpu = baseline['environment_signature']['gpu_uuid']
        identity = suite.command_text([smi, '--query-gpu=uuid', '--format=csv,noheader'])
        suite.atomic_json(out/'gpu_identity.json', identity)
        if identity.get('exit_code') != 0 or identity['stdout'].strip() != gpu:
            raise RuntimeError('GPU UUID mismatch')
        with suite.lock(Path('/tmp')/f'gemm-sweep-{gpu}.lock'):
            audit.check_processes(suite, gpu, out/'processes_before.json')
            steps.require_power(suite, out, smi, gpu, 'power_before', config)
            steps.snapshot(suite, out, smi, gpu, 'gpu_before_settings')
            with steps.Credentials(suite, out) as credentials:
                try:
                    status['collection'] = 'RUNNING'
                    for index, stage in enumerate(config['stages']):
                        credentials.check()
                        name = stage['name']
                        audit.check_processes(suite, gpu, out/(name+'_processes.json'))
                        steps.require_power(suite, out, smi, gpu, name+'_power', config)
                        steps.require_clock(hold, suite, out, smi, gpu, name+'_memory', '-lmc', '6801,6801')
                        value = str(stage['sm_mhz'])
                        steps.require_clock(hold, suite, out, smi, gpu, name+'_core', '-lgc', value+','+value)
                        hold.progress(f"STAGE {index+1}/4: {name}; 180 s preheat + 60 measurements; {out/(name+'.log')}")
                        command = [sys.executable, '-u', '-B', str(HERE/'lower.py'), 'worker',
                                   '--output', str(out), '--stage', name]
                        audit.run_logged(command, out, name, timeout=config['worker_timeout_s'])
                        credentials.check()
                        suite.atomic_json(out/'progress.json', dict(stage_completed=index+1, total_stages=4))
                    status['collection'] = 'COMPLETE'
                finally:
                    # Worker termination is handled by run_logged before both resets under this lock.
                    restored = hold.restore(suite, out, smi, gpu)
                    status['restoration'] = 'COMMANDS_ACCEPTED' if restored else 'FAILED'
            from analyze_lower import analyze
            result = analyze(out, loaded)
            if not restored or credentials.failed:
                result.update(gate='FAIL', restoration=status['restoration'])
                result['flags'].append('restoration_or_authorization_failed')
            if own_hashes() != manifest['source_hashes']:
                raise RuntimeError('Source changed while running')
            load()
            suite.atomic_json(out/'summary.json', result)
            hold.progress('LOWER CLOCKS: '+result['gate']+'; full remains HOLD')
            return {'PASS':0, 'REVIEW':2, 'FAIL':1}[result['gate']]
    except BaseException as error:
        status.update(collection='FAILED_OR_INTERRUPTED', error=str(error))
        suite.atomic_json(out/'summary.json', dict(gate='FAIL', full_sweep='HOLD', error=str(error),
                                                  restoration=status['restoration']))
        raise
    finally:
        suite.atomic_json(out/'status.json', status)
        if status['restoration'] == 'NOT_ATTEMPTED':
            hold.progress('RESTORE NOT ATTEMPTED: admission failed before this run changed settings.')


def package(out):
    # Packaging must also work when dependency verification fails.
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, str(Path(out.name)/path.relative_to(out)))
    archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
    print('RESULT_ZIP='+str(archive), flush=True)


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify','run','worker','analyze','pack'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage')
    args = parser.parse_args()
    if args.output:
        args.output = args.output.resolve()
    if args.action == 'pack':
        package(args.output)
    elif args.action == 'run':
        return run(args)
    else:
        loaded = load()
        if args.action == 'verify':
            print('DEPENDENCY CHECK: PASS; 4 stages / 240 measurements / 1680 blocks / 80 mode triplets')
        elif args.action == 'worker':
            if args.stage not in {s['name'] for s in loaded[-1]['stages']}:
                raise ValueError('Unknown stage')
            with configured_worker(loaded):
                loaded[2].worker(argparse.Namespace(suite=HERE.parent/'gpu_sweep_v6',
                                                    output=args.output, pass_name=args.stage))
        else:
            from analyze_lower import analyze
            loaded[6].atomic_json(args.output/'summary.json', analyze(args.output, loaded))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
