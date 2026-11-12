"""Offline runner check: fake external commands only; never launches CUDA."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / 'cutlass_780_smoke.sh').read_text()


def main():
    subprocess.run(['bash', '-n', str(ROOT / 'cutlass_780_smoke.sh')], check=True)
    with tempfile.TemporaryDirectory(prefix='cutlass-smoke-test-') as directory:
        base = Path(directory)
        bin_dir = base / 'bin'; bin_dir.mkdir()
        external = '''#!/usr/bin/env python3
import csv, json, os, sys, time
from pathlib import Path
name=Path(sys.argv[0]).name
case=os.environ['FAKE_CASE']
if name=='nvidia-smi':
 if any('--query-compute-apps' in s for s in sys.argv):
  if case=='busy': print('123')
 elif '--loop-ms=200' in sys.argv:
  Path(os.environ['PID_FILE']).write_text(str(os.getpid()))
  print('timestamp, utilization.gpu [%], clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], power.limit [W], temperature.gpu',flush=True)
  while True:
   mem=7001 if case=='clock' else 6801
   print(f'2026/09/21 14:00:00.000, 100 %, 780 MHz, {mem} MHz, 45 W, 70 W, 45',flush=True)
   time.sleep(.01)
 else: print('fake device snapshot')
elif name=='git':
 if 'rev-parse' in sys.argv: print('f'*40)
elif name=='sha256sum': print('f'*64+'  fake-binary')
elif name=='date': print('2026-09-21T14:00:00.000000000+09:00')
elif name=='cutlass_profiler':
 args=dict(s[2:].split('=',1) for s in sys.argv[1:])
 Path(os.environ['CALL_FILE']).open('a').write(args['kernels']+'\\n')
 print('fake profiler log')
 time.sleep(.10)
 if case=='exit': sys.exit(7)
 row={'Provider':'CUTLASS','Status':'success','Disposition':'not_verified' if case=='verify' else 'passed',
      'm':args['m'],'n':args['n'],'k':args['k'],'op_class':args['op_class'],'accum':args['accum'],
      'A':args['A'],'B':args['B'],'C':args['C'],'D':args['D'],'Operation':args['kernels'],'Runtime':'50'}
 with open(args['output']+'.gemm.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
'''
        for name in ('nvidia-smi', 'git', 'sha256sum', 'date', 'cutlass_profiler'):
            p = bin_dir / name; p.write_text(external); p.chmod(0o755)
        for case, expected in [('ok', 0), ('verify', 1), ('clock', 1), ('exit', 7), ('busy', 2)]:
            work = base / case; work.mkdir()
            script = work / 'runner.sh'
            script.write_text(SOURCE.replace('/home/eu/cutlass_780_smoke.XXXXXX', str(work / 'out.XXXXXX'))
                              .replace('/home/eu/cutlass/build/tools/profiler/cutlass_profiler', str(bin_dir / 'cutlass_profiler')))
            env = dict(os.environ, PATH=str(bin_dir) + ':' + os.environ['PATH'],
                       FAKE_CASE=case, PID_FILE=str(work / 'pid'), CALL_FILE=str(work / 'calls'))
            result = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True, timeout=10)
            assert result.returncode == expected, (case, result.stdout, result.stderr)
            out = next(p for p in work.glob('out.*') if p.is_dir())
            assert (out / 'runner.exit_code').read_text().strip() == str(expected)
            assert Path(str(out) + '.tar.gz').is_file()
            if (work / 'pid').exists():
                try:
                    os.kill(int((work / 'pid').read_text()), 0)
                except ProcessLookupError:
                    pass
                else:
                    raise AssertionError((case, 'monitor still alive'))
            if case == 'ok':
                summary = json.loads((out / 'smoke_summary.json').read_text())
                assert summary['fp32_simt']['tflops_2mnk'] == 2 * 8192**3 / 50 / 1e9
                assert len((work / 'calls').read_text().splitlines()) == 2
            elif case == 'busy':
                assert not (work / 'calls').exists()
            elif case == 'exit':
                assert len((work / 'calls').read_text().splitlines()) == 1
            else:
                assert not (out / 'smoke_summary.json').exists()
    print('PASS: successful run, not_verified rejection, clock mismatch, profiler failure, busy GPU guard, archive and monitor cleanup. No GPU execution.')


if __name__ == '__main__':
    main()
