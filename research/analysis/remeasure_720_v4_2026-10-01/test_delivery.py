"""Check the ZIP and launcher with harmless command stand-ins; never use a GPU."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOURCE = ROOT / 'remeasure_720_v4'
ZIP = ROOT / 'remeasure_720_review_2026-10-01_v4_isolated_ascii.zip'


def main():
    assert not ZIP.exists(), 'Do not overwrite an existing delivery ZIP'
    for name in ('analyze_sweep.py', 'dependency_hashes.json', 'progress.py', 'run_research_checked.sh'):
        assert (SOURCE / name).read_bytes() == (ROOT / 'remeasure_720_v3' / name).read_bytes(), name
    for name in ('run.py', 'prepare.py', 'progress.py', 'launch.sh', 'run_research_checked.sh'):
        assert (SOURCE / name).read_text().isascii(), name
    for name in ('launch.sh', 'run_research_checked.sh'):
        subprocess.run(['bash', '-n', str(SOURCE / name)], check=True)
    # Console messages must remain readable with an ASCII terminal encoding.
    with tempfile.TemporaryDirectory(prefix='ascii-progress-') as temporary:
        out = Path(temporary)
        for value in (None, dict(recorded=111582, expected=113382, flagged=0, state='RUNNING'),
                      dict(recorded=113382, expected=113382, flagged=120, state='FINISHED')):
            if value is not None:
                (out / 'progress.json').write_text(json.dumps(value))
                (out / 'recovery_provenance.json').write_text(json.dumps(dict(imported_records=111582, redo_records=1800)))
            result = subprocess.run(['python3', '-B', str(SOURCE / 'progress.py'), str(out), 'full'],
                                    capture_output=True, env={**os.environ, 'LC_ALL': 'C', 'PYTHONIOENCODING': 'ascii'}, check=True)
            line = result.stdout.decode('ascii')
            assert '[Total]' in line and 'Remaining' in line
            if value is not None:
                assert '[Remeasure]' in line and value['state'] in line
        wrapper = out / 'run_research_checked.sh'
        shutil.copy2(SOURCE / wrapper.name, wrapper)
        usage = subprocess.run(['bash', str(wrapper)], capture_output=True)
        assert usage.returncode == 2 and usage.stderr.startswith(b'Usage: ')
        usage.stderr.decode('ascii')
    assert {k:v for k,v in json.loads((SOURCE/'protocol.json').read_text()).items() if k != 'remeasurement_schedule'} == json.loads((ROOT/'remeasure_720_v3/protocol.json').read_text())
    files = sorted(p for p in SOURCE.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.json','.md'))
    assert len(files) == 10
    with zipfile.ZipFile(ZIP, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, 'remeasure_720_v4/' + path.name)
    release = BASE / 'release'
    release.mkdir()
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        archive.extractall(release)
    package = release / 'remeasure_720_v4'
    assert all(p.read_bytes() == (package / p.name).read_bytes() for p in files)
    for name, expected in json.loads((SOURCE / 'dependency_hashes.json').read_text()).items():
        original = ROOT / name
        assert hashlib.sha256(original.read_bytes()).hexdigest() == expected
        destination = release / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    plan = subprocess.run(['python3', '-B', str(package / 'run.py'), 'verify', '--size', 'full'],
                          check=True, capture_output=True, text=True)
    assert json.loads(plan.stdout)['tasks'] == 113382
    subprocess.run(['bash', '-n', str(package / 'launch.sh')], check=True)
    with tempfile.TemporaryDirectory(prefix='remeasure launcher ') as temporary:
        root = Path(temporary)
        script, bins = root / 'package', root / 'bin'
        script.mkdir(); bins.mkdir(); (root / 'gpu_sweep_v6').mkdir()
        shutil.copy2(package / 'launch.sh', script / 'launch.sh')
        bodies = {
            'python3': 'printf "%s\\n" "$*" >> "$CALLS"\n'
                       'if [[ "$*" == *prepare.py* && "${FAIL_PREPARE:-0}" == 1 ]]; then exit 7; fi\nexit 0',
            'sudo': 'echo sudo >> "$CALLS"\nexit 0',
            'flock': 'exit 0',
            'tmux': 'echo unexpected-tmux >> "$CALLS"\nexit 99',
        }
        for name, body in bodies.items():
            path = bins / name
            path.write_text('#!/usr/bin/env bash\n' + body + '\n')
            path.chmod(0o755)
        (script / 'run_research_checked.sh').write_text(
            'printf "%s\\n" "$@" > "$ARGUMENTS"\nexit 2\n')
        env = dict(os.environ, PATH=str(bins) + os.pathsep + os.environ['PATH'],
                   CALLS=str(root / 'calls'), ARGUMENTS=str(root / 'arguments'), TMUX='existing-test-session')
        run = subprocess.run(['bash', str(script / 'launch.sh')], env=env,
                             capture_output=True, text=True, timeout=10)
        assert run.returncode == 2, run.stderr
        assert run.stdout.isascii() and run.stderr.isascii()
        assert 'Running in the current tmux session' in run.stdout
        args = (root / 'arguments').read_text().splitlines()
        assert args[args.index('--output') + 1] == str(script / 'runs/full_720_remeasure3_20261001')
        calls = (root / 'calls').read_text()
        assert 'prepare.py --archive ' in calls and ' pack --output ' in calls
        assert 'full_720_remeasure2_20260930_20260930_232816_56d6fb.zip' in calls
        assert calls.index('prepare.py') < calls.index('sudo') and 'unexpected-tmux' not in calls
        (root / 'calls').write_text('')
        run = subprocess.run(['bash', str(script / 'launch.sh')], env={**env, 'FAIL_PREPARE': '1'},
                             capture_output=True, text=True, timeout=10)
        assert run.returncode == 7 and 'sudo' not in (root / 'calls').read_text()
    digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    ZIP.with_suffix('.zip.sha256').write_text(digest + '  ' + ZIP.name + '\n')
    result = dict(status='PASS', gpu_execution=False, zip=str(ZIP), sha256=digest,
                  source_files=len(files), dependency_hashes=100, crc_and_extracted_bytes_match=True,
                  extracted_full_plan=json.loads(plan.stdout), shell_syntax=True,
                  console_ascii=True, measurement_and_analysis_unchanged=True, collection_schedule_changed=True,
                  prepare_before_gpu=True, preparation_failure_stops_launch=True,
                  review_exit_2_and_pack_preserved=True, existing_tmux_used=True)
    (BASE / 'delivery_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
