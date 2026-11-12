#!/usr/bin/env python3
"""One CPU-only regression check using the reviewed archive and completed check."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

import prepare as recovery
import progress


def rejected(call, fragment):
    try:
        call()
    except RuntimeError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError('Expected rejection: ' + fragment)


def main(archive, check):
    checks = []
    run = recovery.run
    before = run.sha(archive)
    with tempfile.TemporaryDirectory(prefix='test-gemm720-recovery-') as folder:
        root = Path(folder)
        local_check = root / 'check'
        shutil.copytree(check, local_check)
        out = root / 'recovered'
        assert recovery.prepare(archive, out, local_check) == 1920
        summary = run.read(out / 'summary.json')
        assert summary['counts'] == {'PASS': 1680} and summary['missing'] == 1920
        loaded = run.load(720)
        manifest = run.manifest_for(out, 'full', loaded)
        choices, done = run.choices_and_done(out, manifest, loaded)
        assert len(choices) >= 52 and len(done) == 1680
        assert [row['task'] for _, row in run.history(out)] == manifest['tasks'][:1680]
        assert [t for t in manifest['tasks'] if t['id'] not in done] == manifest['tasks'][1680:]
        assert not (out / f'sessions/{recovery.SESSION}/segment_00014').exists()
        assert run.read(out / f'sessions/{recovery.SESSION}/session.json')['state'] == 'ERROR'
        checks.append('exact clean 1680 import, same 1920 pending tasks, original history retained')

        digest = recovery.records_digest(run.history(out))
        assert recovery.prepare(archive, out, local_check) == 1920
        assert recovery.records_digest(run.history(out)) == digest
        checks.append('repeat preparation is idempotent and does not duplicate records')

        marker = out / 'recovery_provenance.json'
        saved = marker.read_bytes()
        marker.write_text('{}')
        rejected(lambda: recovery.prepare(archive, out, local_check), 'different recovery')
        marker.write_bytes(saved)
        unrelated = root / 'unrelated'
        unrelated.mkdir()
        (unrelated / 'keep.txt').write_text('preserve me')
        rejected(lambda: recovery.prepare(archive, unrelated, local_check), 'completed recovery receipt')
        assert (unrelated / 'keep.txt').read_text() == 'preserve me'
        bad = root / 'bad.zip'
        bad.write_bytes(b'not the reviewed ZIP')
        rejected(lambda: recovery.prepare(bad, root / 'bad-output', local_check), 'differs')
        assert not (root / 'bad-output').exists()
        checks.append('wrong archive, unrelated/partial output and invalid receipt rejected')

        evidence = out / f'sessions/{recovery.SESSION}/environment.json'
        saved = evidence.read_bytes()
        evidence.write_bytes(saved + b'\n')
        rejected(lambda: recovery.prepare(archive, out, local_check), 'evidence changed')
        evidence.write_bytes(saved)
        with tempfile.TemporaryDirectory(prefix='test-source-') as temporary:
            source = recovery.extract_reviewed(archive, Path(temporary))
            rejected(lambda: run.require_clean_resume(source, loaded), 'resume refused')
        checks.append('changed imported evidence and original dirty-run resume remain blocked')

        assert '1,680/3,600 (46.7%)' in progress.line(out)
        for count, state, expected in [(1800, 'RUNNING', '50.0%'), (3600, 'FINISHED', '100.0%')]:
            (out / 'progress.json').write_text(json.dumps(dict(recorded=count, expected=3600, state=state)))
            assert expected in progress.line(out) and state in progress.line(out)
        checks.append('progress starts at 46.7 percent and shows saved totals through completion')

        # Exercise the shell wrapper with harmless stand-ins, including exit-code propagation.
        wrapper = root / 'recover_720_v1'
        wrapper.mkdir()
        shutil.copy2(recovery.HERE / 'launch.sh', wrapper / 'launch.sh')
        (wrapper / 'prepare.py').write_text("REVIEW_NOTE = 'review-test'\n")
        (wrapper / 'progress.py').write_text('import sys, time\nif len(sys.argv) > 2: time.sleep(60)\n')
        runner = root / 'abstract_sweep_725_v1'
        runner.mkdir()
        (runner / 'launch.sh').write_text('printf "%s\\n" "$@"\nexit 2\n')
        result = subprocess.run(['bash', str(wrapper / 'launch.sh')], capture_output=True,
                                text=True, timeout=10, env={**os.environ, 'TMUX': 'existing-test-session'})
        assert result.returncode == 2 and result.stdout.splitlines()[0] == 'full', result
        assert '--check' in result.stdout and '--output' in result.stdout and 'review-test' in result.stdout
        assert 'tmux' not in (wrapper / 'launch.sh').read_text()
        checks.append('wrapper delegates full run, preserves REVIEW exit 2 and stops its monitor')

        (runner / 'launch.sh').write_text(
            "trap 'touch \"$TEST_ROOT/cleaned\"; exit 130' INT\n"
            'touch "$TEST_ROOT/ready"\nsleep 30\n')
        process = subprocess.Popen(['bash', str(wrapper / 'launch.sh')], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True,
                                   env={**os.environ, 'TEST_ROOT': str(root)})
        try:
            deadline = time.monotonic() + 10
            while not (root / 'ready').exists() and time.monotonic() < deadline:
                time.sleep(.05)
            assert (root / 'ready').exists(), 'Signal test launcher did not start'
            os.killpg(process.pid, signal.SIGINT)
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 130 and (root / 'cleaned').exists(), (stdout, stderr)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        checks.append('Ctrl+C reaches the existing launcher cleanup and retains exit 130')
    assert run.sha(archive) == before == recovery.ARCHIVE_SHA
    print(json.dumps(dict(status='PASS', gpu_execution=False, checks=checks), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--check', type=Path, required=True)
    args = parser.parse_args()
    main(args.archive.resolve(), args.check.resolve())
