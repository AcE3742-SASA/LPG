#!/usr/bin/env python3
"""Collect all 30 batches; resume skips verified complete archives and preserves attempts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import zipfile
import inherit_campaign

import compare

HERE = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def batch_order():
    phases = compare.read(HERE/'cases_by_phase.json')
    return [phase for prefix in ('train_', 'validation_', 'test_')
            for phase in sorted(phases) if phase.startswith(prefix)]


def completed(out, phase, source_hashes):
    required = ('manifest.json', 'summary.json', 'archive_receipt.json')
    if not all((out/name).is_file() for name in required):
        return False
    manifest = compare.read(out/'manifest.json')
    if manifest['source_hashes'] != source_hashes or manifest['phase'] != phase:
        raise RuntimeError('Completed batch source or phase mismatch')
    summary = compare.read(out/'summary.json')
    cases = compare.read(HERE/'cases_by_phase.json')[phase]
    config = dict(compare.read(HERE/'protocol.json'), phase=phase)
    exclusions = compare.excluded_conditions(cases, config)
    expected = compare.expected_records(cases, config)
    if not (summary['collection'] == 'COMPLETE' and summary['gate'] in ('PASS', 'REVIEW')
            and summary['missing'] == 0 and summary['recorded'] == summary['expected'] == expected
            and summary.get('planned', 1080) == 1080
            and summary.get('excluded', 0) == 1080 - expected
            and summary.get('excluded_conditions', []) == exclusions
            and (not exclusions or summary['gate'] == 'REVIEW')
            and summary['counts'].get('FAIL', 0) == 0
            and summary['restoration'] == 'COMMANDS_ACCEPTED' and summary['worker_complete']):
        return False
    receipt = compare.read(out/'archive_receipt.json')
    filename = receipt['filename']
    if Path(filename).name != filename or not filename.startswith(out.name + '_') or not filename.endswith('.zip'):
        raise RuntimeError('Invalid archive receipt path')
    archive = out.parent/filename
    if not archive.is_file() or compare.sha(archive) != receipt['sha256']:
        raise RuntimeError('Completed batch archive is missing or damaged')
    with zipfile.ZipFile(archive) as package:
        if (json.loads(package.read(out.name + '/manifest.json')) != manifest or
                json.loads(package.read(out.name + '/summary.json')) != summary):
            raise RuntimeError('Completed batch metadata differs from its archive')
    return True


def resolve_attempt(campaign, name, phase):
    path = (campaign/name).resolve()
    if path.parent != campaign.resolve() or not path.name.startswith(phase + '_attempt_'):
        raise RuntimeError('Invalid attempt path in campaign state')
    return path


def run_batch(command):
    # Keep the terminal attached so sudo can authenticate in the existing session.
    child = subprocess.Popen(command)
    previous = {}
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        if child.poll() is None:
            listing = subprocess.run(['ps', '-axo', 'pid=,ppid='], capture_output=True, text=True, check=True)
            parents = {int(pid): int(parent) for pid, parent in
                       (line.split() for line in listing.stdout.splitlines())}
            descendants = {child.pid}
            while True:
                expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
                if expanded == descendants:
                    break
                descendants = expanded
            for pid in sorted(descendants, reverse=True):
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, stop)
        return child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def collect(campaign, execute=run_batch):
    sources = compare.read(HERE/'release_hashes.json')
    state_path = campaign/'campaign.json'
    if state_path.exists():
        state = compare.read(state_path)
        if state['source_hashes'] != sources or state['batch_order'] != batch_order():
            raise RuntimeError('Campaign release mismatch; use the original release')
        if 'inherited_batches' in state:
            inherit_campaign.validate_state(campaign, state)
    else:
        state = dict(source_hashes=sources, batch_order=batch_order(), attempts={},
                     sealed_test_policy='collect and quality-check only; no prediction evaluation')
        save(state_path, state)
    review_batches = []
    completed_summaries = {}
    for index, phase in enumerate(state['batch_order'], 1):
        if phase in state.get('inherited_batches', {}):
            summary = inherit_campaign.check_archive(campaign, phase)
            completed_summaries[phase] = summary
            if summary['gate'] == 'REVIEW':
                review_batches.append(phase)
            print(f'SKIP INHERITED: {phase} ({index}/30), gate={summary["gate"]}', flush=True)
            continue
        attempts = state['attempts'].setdefault(phase, [])
        previous = resolve_attempt(campaign, attempts[-1], phase) if attempts else None
        if previous is not None and completed(previous, phase, sources):
            summary = compare.read(previous/'summary.json')
            completed_summaries[phase] = summary
            if summary['gate'] == 'REVIEW':
                review_batches.append(phase)
            print(f'SKIP COMPLETE: {phase} ({index}/30)', flush=True)
            continue
        attempt = campaign/f'{phase}_attempt_{len(attempts) + 1:03d}'
        if attempt.exists():
            raise RuntimeError('Attempt directory already exists; preserve campaign state')
        attempts.append(attempt.name)
        save(state_path, state)
        command = ['bash', str(HERE/'launch.sh'), phase, str(attempt)]
        if phase.startswith('test_'):
            command.append('--collect-sealed-test')
        config = dict(compare.read(HERE/'protocol.json'), phase=phase)
        expected = compare.expected_records(compare.read(HERE/'cases_by_phase.json')[phase], config)
        print(f'START: {phase} ({index}/30), {expected} measurements, {1080-expected} excluded', flush=True)
        code = execute(command)
        if code not in (0, 2) or not completed(attempt, phase, sources):
            print(f'STOP: {phase}, exit={code}; preserve this attempt and resume the campaign', flush=True)
            return code if code else 1
        completed_summaries[phase] = compare.read(attempt/'summary.json')
        if completed_summaries[phase]['gate'] == 'REVIEW':
            review_batches.append(phase)
        state['completed_batches'] = index
        save(state_path, state)
    recorded = sum(s['recorded'] for s in completed_summaries.values())
    excluded = sum(s.get('excluded', 0) for s in completed_summaries.values())
    save(campaign/'collection_summary.json', dict(collection='COMPLETE',
         measurements=recorded, planned_measurements=32400, excluded_measurements=excluded,
         batches=30, review_batches=review_batches,
         gate='REVIEW' if review_batches else 'PASS', timing_values_exposed=False))
    print(f'COLLECTION COMPLETE: {recorded} measurements; {excluded} excluded; review_batches={len(review_batches)}', flush=True)
    return 2 if review_batches else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--resume', type=Path)
    choice.add_argument('--import-campaign', type=Path)
    args = parser.parse_args()
    if HERE != Path('/home/eu/gemm-research/layer_full_20261110_v4') or os.geteuid() == 0:
        parser.error('Run as eu from /home/eu/gemm-research/layer_full_20261110_v4')
    compare.verify()
    runs = HERE/'runs'
    runs.mkdir(exist_ok=True)
    if runs.resolve().parent != HERE:
        parser.error('Keep runs inside this release directory')
    if args.resume:
        campaign = args.resume.resolve()
        if campaign.parent != runs.resolve() or not (campaign/'campaign.json').is_file():
            parser.error('Resume requires an existing full collection directory inside this release runs directory')
    else:
        with (runs/'.import.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            campaign = inherit_campaign.prepare(args.import_campaign, runs)
    if 'inherited_batches' not in compare.read(campaign/'campaign.json'):
        parser.error('This release resumes the audited v2 campaign; use --import-campaign first')
    print('CAMPAIGN_DIR=' + str(campaign), flush=True)
    print('RESUME: python3 -B ' + str(HERE/'collect_all.py') + ' --resume ' + str(campaign), flush=True)
    with (campaign/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return collect(campaign)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print('FAIL: ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
