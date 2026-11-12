#!/usr/bin/env python3
"""Run the unchanged 192-task v6 pilot worker at requested 900/6801 MHz."""
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
    return {p.name: digest(p) for p in sorted(HERE.iterdir())
            if p.suffix in ('.py', '.sh', '.json', '.md')}


def load():
    for name, expected in read(HERE / 'dependency_hashes.json').items():
        if digest(HERE.parent / name) != expected:
            raise RuntimeError('Dependency changed: ' + name)
    sys.path.insert(0, str(HERE.parent / 'clock_steps_v1'))
    steps = importlib.import_module('steps')
    if Path(steps.__file__).resolve().parent != HERE.parent / 'clock_steps_v1':
        raise RuntimeError('Wrong steps module')
    hold, audit, original, telemetry, baseline, suite, _ = steps.load()
    config = read(HERE / 'protocol.json')
    plan = sys.modules['plan']
    tasks = json.loads(json.dumps(plan.make_plan('pilot')['tasks']))
    if len(tasks) != 192 or plan.digest(tasks) != config['task_plan_sha256']:
        raise RuntimeError('Pilot plan changed')
    keys = {(tuple(t['shape']), t['dtype']) for t in tasks}
    if keys != {(tuple(s['shape']), s['dtype']) for s in config['fixed_cases']} or len(config['fixed_cases']) != 48:
        raise RuntimeError('Pinned condition coverage changed')
    for spec in config['fixed_cases']:
        m, n, _ = spec['shape']
        scope, points = ('full', m*n) if m*n <= 4096 else ('sampled', min(m,16)*min(n,16))
        if spec['input_seed'] != plan.task_seed(spec) or not 1 <= spec['graph_gemms_per_replay'] <= 64:
            raise RuntimeError('Invalid pinned seed or Graph batch')
        if (spec['correctness_scope'], spec['correctness_points']) != (scope, points):
            raise RuntimeError('Correctness scope changed')
    extended = copy.deepcopy(baseline)
    extended['fixed_cases'] = config['fixed_cases']
    return steps, hold, audit, original, telemetry, extended, suite, config, tasks


@contextlib.contextmanager
def configured_worker(loaded):
    steps, _, audit, _, _, _, _, config, tasks = loaded
    original = audit.cases, audit.graph_nodes, audit.PASSES
    audit.cases = lambda: tasks
    audit.graph_nodes = lambda task: steps.graph_nodes(config, task)
    audit.PASSES = {'pilot': config['interval_ms']}
    try:
        yield
    finally:
        audit.cases, audit.graph_nodes, audit.PASSES = original


def make_manifest(out, loaded):
    _, _, _, _, _, baseline, suite, config, tasks = loaded
    manifest = dict(kind='fixed_pilot_v1', config=config, baseline=baseline, cases=tasks,
        source_hashes=own_hashes(), dependency_hashes=read(HERE/'dependency_hashes.json'),
        started_utc=suite.now(), expected_measurements=len(tasks), full_sweep='HOLD',
        new_automatic_core_load_control=False, restore_to='default clocks, not prior manual locks')
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
    steps, hold, audit, _, _, baseline, suite, config, _ = loaded
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise RuntimeError('Use a fresh result directory; existing attempts stay unchanged')
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
                    steps.require_clock(hold, suite, out, smi, gpu, 'set_memory', '-lmc', '6801,6801')
                    steps.require_clock(hold, suite, out, smi, gpu, 'set_graphics', '-lgc', '900,900')
                    status['collection'] = 'RUNNING'
                    hold.progress('FIXED PILOT: 180 s preheat + 192 measurements; progress: '+str(out/'worker.log'))
                    command = [sys.executable, '-u', '-B', str(HERE/'fixed.py'), 'worker', '--output', str(out)]
                    audit.run_logged(command, out, 'worker', timeout=config['worker_timeout_s'])
                    credentials.check()
                    status['collection'] = 'COMPLETE'
                finally:
                    # The GPU lock and noninteractive sudo refresh remain held through both resets.
                    restored = hold.restore(suite, out, smi, gpu)
                    status['restoration'] = 'COMMANDS_ACCEPTED' if restored else 'FAILED'
            from analyze_pilot import analyze
            result = analyze(out, loaded)
            if not restored or credentials.failed:
                result.update(gate='FAIL', restoration=status['restoration'])
                result['flags'].append('restoration_or_authorization_failed')
            if own_hashes() != manifest['source_hashes']:
                raise RuntimeError('Source changed during run')
            load()
            suite.atomic_json(out/'summary.json', result)
            hold.progress('FIXED PILOT: '+result['gate']+'; full remains HOLD')
            return {'PASS':0, 'REVIEW':2, 'FAIL':1}[result['gate']]
    except BaseException as error:
        status.update(collection='FAILED_OR_INTERRUPTED', error=str(error))
        # Preserve a useful top-level state even if collection/analysis never completes.
        suite.atomic_json(out/'summary.json', dict(gate='FAIL', full_sweep='HOLD', error=str(error),
                                                  restoration=status['restoration']))
        raise
    finally:
        suite.atomic_json(out/'status.json', status)
        if status['restoration'] == 'NOT_ATTEMPTED':
            hold.progress('RESTORE NOT ATTEMPTED: admission failed before this run changed settings.')


def package(out):
    # No dependency import: failed preflight and interrupted runs must still produce a ZIP.
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
    parser.add_argument('action', choices=['verify', 'run', 'worker', 'analyze', 'pack'])
    parser.add_argument('--output', type=Path)
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
            print('DEPENDENCY CHECK: PASS; 192 measurements / 1344 blocks / 96 pairs; clocks 900/6801 MHz')
        elif args.action == 'worker':
            with configured_worker(loaded):
                loaded[2].worker(argparse.Namespace(suite=HERE.parent/'gpu_sweep_v6',
                                                    output=args.output, pass_name='pilot'))
        else:
            from analyze_pilot import analyze
            loaded[6].atomic_json(args.output/'summary.json', analyze(args.output, loaded))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
