"""CPU-only launcher check; stub measurement/sudo/flock, never touch a GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'launch_fixed_sweep_780.sh'
subprocess.run(['bash', '-n', str(LAUNCHER)], check=True)
with tempfile.TemporaryDirectory(prefix='gemm launcher ') as folder:
    root = Path(folder)
    (root/'fixed_sweep_780_v1').mkdir()
    (root/'gpu_sweep_v6').mkdir()
    bins = root/'bin'
    bins.mkdir()
    shutil.copy2(LAUNCHER, root/LAUNCHER.name)
    for name, body in {
        'python3': 'printf "%s\\n" "$*" >> "$CALLS"; exit 0',
        'sudo': 'exit 0',
        'flock': 'exit 0',
        'tmux': 'echo unexpected-tmux >> "$CALLS"; exit 99',
    }.items():
        path = bins/name
        path.write_text('#!/usr/bin/env bash\n'+body+'\n')
        path.chmod(0o755)
    (root/'gpu_sweep_v6/run_research_checked.sh').write_text(
        'printf "%s\\n" "$@" > "$ARGUMENTS"\nexit "$MEASURE_EXIT"\n')
    for inside in (False, True):
        for size, code in (('check', 0), ('full', 2)):
            out = root/f'output {inside} {size}'
            calls, args_file = root/'calls', root/'args'
            calls.write_text('')
            env = dict(os.environ, PATH=str(bins)+os.pathsep+os.environ['PATH'],
                       CALLS=str(calls), ARGUMENTS=str(args_file), MEASURE_EXIT=str(code))
            env.pop('TMUX', None)
            if inside:
                env['TMUX'] = '/tmp/existing-tmux,123,0'
            extra = ['--check', '/tmp/check with spaces', '--review-note', 'review with spaces'] if size == 'full' else []
            command = ['bash', str(root/LAUNCHER.name), size, '--output', str(out), *extra]
            with tempfile.TemporaryFile() as log:
                process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
                try:
                    actual = process.wait(timeout=5)  # stdin stays open: an Enter prompt must fail this check.
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    process.stdin.close()
                log.seek(0)
                output = log.read().decode()
            assert actual == code, (command, actual, output)
            assert 'unexpected-tmux' not in calls.read_text(), calls.read_text()
            assert ' pack --output ' in calls.read_text(), calls.read_text()
            assert ('현재 tmux 세션에서' if inside else 'SSH 연결을 유지') in output, output
            invocation, = (out/'launcher').iterdir()
            assert (invocation/'exit_code.txt').read_text().strip() == str(code)
            assert (invocation/'launcher_source.sh').read_bytes() == LAUNCHER.read_bytes()
            assert args_file.exists(), output
            args = args_file.read_text().splitlines()
            assert args[args.index('--output')+1] == str(out)
            if extra:
                assert args[-len(extra):] == extra, args
            print(f'PASS: tmux={inside}, size={size}, exit={code}, no session/window creation or Enter wait')
