#!/usr/bin/env python3
"""Run a bounded clock ladder with the verified, unchanged v6 measurement worker."""
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
import threading
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def own_hashes():
    return {p.name: digest(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.sh', '.json', '.md')}


def load():
    dependencies = read(HERE / 'dependency_hashes.json')
    for relative, expected in dependencies.items():
        if digest(HERE.parent / relative) != expected:
            raise RuntimeError('Dependency changed: ' + relative)
    sys.path.insert(0, str(HERE.parent / 'clock_hold_v1'))
    hold = importlib.import_module('hold')
    if Path(hold.__file__).resolve().parent != HERE.parent / 'clock_hold_v1':
        raise RuntimeError('Wrong hold module imported')
    audit, analyze, telemetry, baseline, suite = hold.load(
        HERE.parent / 'clock_audit_v1', HERE.parent / 'gpu_sweep_v6')
    config = read(HERE / 'protocol.json')
    if config['fixed_cases'][:2] != baseline['fixed_cases']:
        raise RuntimeError('Existing diagnostic cases changed')
    for spec in config['fixed_cases']:
        if sys.modules['plan'].task_seed(spec) != spec['input_seed']:
            raise RuntimeError('Case seed mismatch')
    extended = copy.deepcopy(baseline)
    extended['fixed_cases'] = config['fixed_cases']
    return hold, audit, analyze, telemetry, extended, suite, config


def cases(config):
    result = []
    for rd in range(2):
        indices = range(len(config['fixed_cases'])) if rd == 0 else reversed(range(len(config['fixed_cases'])))
        for index in indices:
            spec = config['fixed_cases'][index]
            for mode in (('prealloc', 'graph') if rd == 0 else ('graph', 'prealloc')):
                result.append(dict(id=f'c{index}_{mode}_r{rd}', kind='gemm', shape=spec['shape'],
                                   dtype=spec['dtype'], mode=mode, round=rd,
                                   phase='clock_steps', split='diagnostic_only'))
    return result


def graph_nodes(config, task):
    if task['mode'] != 'graph':
        return None
    return next(spec['graph_gemms_per_replay'] for spec in config['fixed_cases']
                if spec['shape'] == task['shape'] and spec['dtype'] == task['dtype'])


@contextlib.contextmanager
def configured_audit(audit, config):
    """Adapt only case selection and polling; worker/Timer/source files stay intact."""
    original = audit.cases, audit.graph_nodes, audit.PASSES
    audit.cases = lambda: cases(config)
    audit.graph_nodes = lambda task: graph_nodes(config, task)
    audit.PASSES = {stage['name']: stage['interval_ms'] for stage in config['stages']}
    try:
        yield
    finally:
        audit.cases, audit.graph_nodes, audit.PASSES = original


class Credentials:
    """Renew existing sudo authorization only while this diagnostic owns the GPU."""
    def __init__(self, suite, out, interval=45):
        self.suite, self.out, self.interval = suite, out, interval
        self.stopped = threading.Event()
        self.failed = False
        self.records = []

    def refresh(self):
        result = self.suite.command_text(['sudo', '-n', '-v'])
        self.records.append(dict(timestamp=self.suite.now(), **result))
        self.suite.atomic_json(self.out / 'sudo_refresh.json', self.records)
        if result.get('exit_code') != 0:
            self.failed = True

    def check(self):
        if self.failed:
            raise RuntimeError('sudo authorization refresh failed; inspect sudo_refresh.json')

    def loop(self):
        try:
            while not self.stopped.wait(self.interval):
                self.refresh()
                if self.failed:
                    break
        except Exception:
            self.failed = True

    def __enter__(self):
        self.refresh()
        self.check()
        self.thread = threading.Thread(target=self.loop, name='sudo-refresh', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join(timeout=18)  # command_text itself has a 15-second timeout.
        if self.thread.is_alive():
            raise RuntimeError('sudo refresh did not terminate')


def require_clock(hold, suite, out, smi, gpu, name, *options):
    result = hold.clock_command(suite, out, smi, gpu, name, *options)
    if result.get('exit_code') != 0:
        raise RuntimeError('Clock command failed: ' + name)


def snapshot(suite, out, smi, gpu, name):
    result = suite.command_text([smi, '-i', gpu, '-q'])
    suite.atomic_json(out / (name + '.json'), result)
    if result.get('exit_code') != 0:
        raise RuntimeError('GPU snapshot failed: ' + name)


def require_power(suite, out, smi, gpu, name, config):
    result = suite.command_text([smi, '-i', gpu, '--query-gpu=power.limit', '--format=csv,noheader,nounits'])
    suite.atomic_json(out / (name + '.json'), result)
    if result.get('exit_code') != 0 or float(result['stdout'].strip()) != config['power_limit_w']:
        raise RuntimeError('Expected existing 70 W limit; power settings were not changed')


def idle_probe(out, name, sm_mhz, loaded, smi, gpu):
    hold, audit, _, telemetry, _, suite, config = loaded
    directory = out / 'idle' / name
    directory.mkdir(parents=True)
    audit.check_processes(suite, gpu, directory / 'processes_before.json')
    snapshot(suite, directory, smi, gpu, 'gpu_before')
    suite.atomic_json(directory / 'phase.json', dict(name=name, sm_mhz=sm_mhz,
        memory_mhz=config['memory_mhz'], seconds=config['idle_seconds'],
        meaning='No diagnostic GEMM worker; desktop clients may still use the GPU.'))
    hold.progress('IDLE PROBE: ' + name)
    with telemetry.collect(directory, gpu, config['idle_interval_ms'], suite):
        time.sleep(config['idle_seconds'])
    snapshot(suite, directory, smi, gpu, 'gpu_after')
    audit.check_processes(suite, gpu, directory / 'processes_after.json')


def make_manifest(out, loaded):
    _, _, _, _, baseline, suite, config = loaded
    dependencies = read(HERE / 'dependency_hashes.json')
    manifest = dict(kind='clock_steps_v1', config=config, baseline=baseline,
                    cases=cases(config), source_hashes=own_hashes(), dependency_hashes=dependencies,
                    started_utc=suite.now(), full_sweep='HOLD', new_automatic_core_load_control=False,
                    expected_measurements=len(cases(config))*len(config['stages']),
                    restore_to='default GPU and memory clocks, not any pre-existing manual locks')
    suite.atomic_json(out / 'manifest.json', manifest)
    for name in manifest['source_hashes']:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / name, target)
    for name in dependencies:
        target = out / 'dependencies' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE.parent / name, target)
    return manifest


def run(args):
    loaded = load()
    hold, audit, _, _, baseline, suite, config = loaded
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Use a fresh result directory')
    manifest = make_manifest(out, loaded)
    smi = shutil.which('nvidia-smi')
    if not smi:
        raise RuntimeError('nvidia-smi unavailable')
    gpu = baseline['environment_signature']['gpu_uuid']
    identity = suite.command_text([smi, '--query-gpu=uuid', '--format=csv,noheader'])
    suite.atomic_json(out / 'gpu_identity.json', identity)
    if identity.get('exit_code') != 0 or identity['stdout'].strip() != gpu:
        raise RuntimeError('GPU UUID mismatch')
    status = dict(collection='NOT_STARTED', restoration='NOT_ATTEMPTED', full_sweep='HOLD')
    try:
        with suite.lock(Path('/tmp') / f'gemm-sweep-{gpu}.lock'):
            audit.check_processes(suite, gpu, out / 'processes_before.json')
            require_power(suite, out, smi, gpu, 'power_before', config)
            snapshot(suite, out, smi, gpu, 'gpu_before_settings')
            with Credentials(suite, out) as credentials:
                try:
                    status['collection'] = 'RUNNING'
                    require_clock(hold, suite, out, smi, gpu, 'memory_initial', '-lmc', '6801,6801')
                    require_clock(hold, suite, out, smi, gpu, 'core_auto_before', '-rgc')
                    idle_probe(out, 'auto_before', None, loaded, smi, gpu)
                    for index, stage in enumerate(config['stages']):
                        credentials.check()
                        audit.check_processes(suite, gpu, out / (stage['name'] + '_processes.json'))
                        require_power(suite, out, smi, gpu, stage['name'] + '_power', config)
                        require_clock(hold, suite, out, smi, gpu, stage['name'] + '_memory', '-lmc', '6801,6801')
                        value = str(stage['sm_mhz'])
                        require_clock(hold, suite, out, smi, gpu, stage['name'] + '_core', '-lgc', value + ',' + value)
                        idle_probe(out, stage['name'] + '_before', stage['sm_mhz'], loaded, smi, gpu)
                        hold.progress(f"STAGE {index+1}/{len(config['stages'])}: {stage['name']} (180 s preheat + 16 measurements)")
                        command = [sys.executable, '-u', '-B', str(HERE / 'steps.py'), 'worker',
                                   '--output', str(out), '--stage', stage['name']]
                        audit.run_logged(command, out, stage['name'], timeout=900)
                        idle_probe(out, stage['name'] + '_after', stage['sm_mhz'], loaded, smi, gpu)
                        credentials.check()
                        suite.atomic_json(out / 'progress.json', dict(stage_completed=index+1, total_stages=len(config['stages'])))
                    require_clock(hold, suite, out, smi, gpu, 'core_auto_after', '-rgc')
                    idle_probe(out, 'auto_after', None, loaded, smi, gpu)
                    credentials.check()
                    status['collection'] = 'COMPLETE'
                finally:
                    # Cleanup is under the same GPU lock; sudo refresh stays alive until both resets finish.
                    restored = hold.restore(suite, out, smi, gpu)
                    status['restoration'] = 'COMMANDS_ACCEPTED' if restored else 'FAILED'
            from analyze_steps import analyze
            summary = analyze(out, loaded)
            if not restored or credentials.failed:
                summary.update(gate='FAIL', restoration=status['restoration'])
                summary['flags'].append('restoration_or_authorization_failed')
            if own_hashes() != manifest['source_hashes']:
                raise RuntimeError('Diagnostic source changed while running')
            load()
            suite.atomic_json(out / 'summary.json', summary)
            hold.progress('CLOCK STEPS: ' + summary['gate'] + '; full remains HOLD')
            return {'PASS': 0, 'REVIEW': 2, 'FAIL': 1}[summary['gate']]
    except BaseException as error:
        status.update(collection='FAILED_OR_INTERRUPTED', error=str(error))
        raise
    finally:
        suite.atomic_json(out / 'status.json', status)
        if status['restoration'] == 'NOT_ATTEMPTED':
            hold.progress('RESTORE NOT ATTEMPTED: admission failed before this run changed settings.')


def package(out):
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                output.write(path, str(Path(out.name) / path.relative_to(out)))
    archive.with_suffix('.zip.sha256').write_text(digest(archive) + '  ' + archive.name + '\n')
    print('RESULT_ZIP=' + str(archive), flush=True)


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'run', 'worker', 'analyze', 'pack'])
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
        hold, audit, _, _, _, _, config = loaded
        if args.action == 'verify':
            print('DEPENDENCY CHECK: PASS; 7 stages / 112 measurements / 784 blocks')
        elif args.action == 'worker':
            if args.stage not in {stage['name'] for stage in config['stages']}:
                raise ValueError('Unknown stage')
            with configured_audit(audit, config):
                audit.worker(argparse.Namespace(suite=HERE.parent/'gpu_sweep_v6', output=args.output, pass_name=args.stage))
        else:
            from analyze_steps import analyze
            hold.save(args.output / 'summary.json', analyze(args.output, loaded))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
