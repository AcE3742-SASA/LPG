"""Offline runner check: fake external commands only; never launches CUDA."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / 'cutlass_780_kernel_compare.sh').read_text()


def main():
    subprocess.run(['bash', '-n', str(ROOT / 'cutlass_780_kernel_compare.sh')], check=True)
    with tempfile.TemporaryDirectory(prefix='cutlass-smoke-test-') as directory:
        base = Path(directory)
        bin_dir = base / 'bin'; bin_dir.mkdir()
        external = '''#!/usr/bin/env python3
import csv, json, os, sys, time
from pathlib import Path
name=Path(sys.argv[0]).name
case=os.environ['FAKE_CASE']
if name=='sudo':
 if sys.argv[1]=='-v': sys.exit(0)
 assert sys.argv[1]=='-n'
 os.execvp(sys.argv[2],sys.argv[2:])
elif name=='flock':
 sys.exit(2 if case=='lock' else 0)
elif name=='nvidia-smi':
 if '--query-gpu=uuid' in sys.argv: print('GPU-ffff')
 elif '-lgc' in sys.argv or '-lmc' in sys.argv:
  Path(os.environ['SET_FILE']).open('a').write(' '.join(sys.argv[1:])+'\\n')
  if case=='setfail': sys.exit(9)
 elif '--format=csv,noheader,nounits' in sys.argv:
  print('210, 405, 65' if case=='power' else Path(os.environ['IDLE_SNAPSHOT']).read_text().strip())
 elif any('--query-compute-apps' in s for s in sys.argv):
  if case=='busy': print('123')
 elif '--loop-ms=200' in sys.argv:
  Path(os.environ['PID_FILE']).open('a').write(str(os.getpid())+'\\n')
  if case=='monitor': print('fake telemetry warning',file=sys.stderr)
  print('timestamp, utilization.gpu [%], clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], power.limit [W], temperature.gpu',flush=True)
  while True:
   mem=7001 if case=='clock' else 6801
   sm=1470 if case=='smclock' else 780
   print(f'2026/09/21 14:00:00.000, 100 %, {sm} MHz, {mem} MHz, 45 W, 70 W, 45',flush=True)
   time.sleep(.01)
 else: print('fake device snapshot')
elif name=='git':
 if 'rev-parse' in sys.argv: print('f'*40)
elif name=='sha256sum': print('f'*64+'  fake-binary')
elif name=='date': print('2026-09-21T14:00:00.000000000+09:00')
elif name=='cutlass_profiler':
 args=dict(s[2:].split('=',1) for s in sys.argv[1:])
 assert args['verification-providers']=='device' and args['epsilon']=='0'
 manifest={r['kernel']:r for r in json.loads(Path(os.environ['KERNELS']).read_text())}
 assert [args[k] for k in ('A','B','C','D')]==[manifest[args['kernels']][k] for k in ('A','B','C','D')]
 Path(os.environ['CALL_FILE']).open('a').write(args['kernels']+'\\n')
 print('fake profiler log')
 time.sleep(.10)
 if case=='exit': sys.exit(7)
 row={'Provider':'CUTLASS','Status':'success','Disposition':'not_verified' if case=='verify' else 'passed',
      'm':args['m'],'n':args['n'],'k':args['k'],'op_class':args['op_class'],'accum':args['accum'],
      'A':args['A'],'B':args['B'],'C':args['C'],'D':args['D'],'Operation':args['kernels'],'Runtime':'50'}
 if case=='layout': row['A']=args['A'].split(':')[0]+(':row' if ':column' in args['A'] else ':column')
 with open(args['output']+'.gemm.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
'''
        for name in ('nvidia-smi', 'git', 'sha256sum', 'date', 'cutlass_profiler', 'sudo', 'flock'):
            p = bin_dir / name; p.write_text(external); p.chmod(0o755)
        for case, expected in [('ok', 0), ('verify', 1), ('layout', 1)]:
            work = base / case; work.mkdir()
            script = work / 'runner.sh'
            script.write_text(SOURCE.replace('/home/eu/cutlass_780_kernel_compare.XXXXXX', str(work / 'out.XXXXXX'))
                              .replace('/home/eu/cutlass/build/tools/profiler/cutlass_profiler', str(bin_dir / 'cutlass_profiler'))
                              .replace('/tmp/gemm-sweep-', str(work / 'gemm-sweep-')))
            env = dict(os.environ, PATH=str(bin_dir) + ':' + os.environ['PATH'],
                       FAKE_CASE=case, PID_FILE=str(work / 'pid'), CALL_FILE=str(work / 'calls'), SET_FILE=str(work / 'settings'), IDLE_SNAPSHOT=str(ROOT.parent / 'cutlass_smoke_v3_2026-09-21/raw_v2_failure/fp32_simt.before.csv'), KERNELS=str(ROOT / 'kernels.json'))
            result = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True, timeout=10)
            assert result.returncode == expected, (case, result.stdout, result.stderr)
            out = next(p for p in work.glob('out.*') if p.is_dir())
            assert (out / 'runner.exit_code').read_text().strip() == str(expected)
            assert Path(str(out) + '.tar.gz').is_file()
            if (work / 'pid').exists():
                for pid in (work / 'pid').read_text().splitlines():
                    try:
                        os.kill(int(pid), 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise AssertionError((case, 'monitor still alive'))
            calls = (work / 'calls').read_text().splitlines() if (work / 'calls').exists() else []
            if case == 'ok':
                summary = json.loads((out / 'fp32_simt_01.check.json').read_text())
                assert summary['fp32_simt_01']['tflops_2mnk'] == 2 * 8192**3 / 50 / 1e9
                assert summary['correctness_pass'] and summary['clocks_pass']
                assert len(calls) == 16
                assert set(calls) == {r['kernel'] for r in json.loads((ROOT/'kernels.json').read_text())}
            elif case in ('busy', 'lock', 'power', 'setfail'):
                assert not calls
                if case in ('busy', 'lock'): assert not (work / 'settings').exists()
            else:
                assert len(calls) == 1, (case, calls)
            if case in ('verify', 'clock', 'smclock', 'monitor'):
                summary = json.loads((out / 'fp32_simt_01.check.json').read_text())
                assert not summary[{'verify':'correctness_pass','clock':'clocks_pass','smclock':'clocks_pass','monitor':'monitor_pass'}[case]]
    print('PASS: 16 exact inventory kernels/layouts once each; first verification or layout failure stops subsequent kernels; archives and monitor cleanup. No GPU execution.')


if __name__ == '__main__':
    main()
