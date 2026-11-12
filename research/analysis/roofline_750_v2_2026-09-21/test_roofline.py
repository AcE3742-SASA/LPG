"""Offline end-to-end test with fake commands; never runs CUDA or sudo."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess as sp
import sys
import tarfile
import tempfile
import signal
import time
import shutil

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'roofline_750_v2'
spec = importlib.util.spec_from_file_location('roofline', PACKAGE / 'roofline.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
kernels = json.loads((PACKAGE/'kernels.json').read_text())
p = m.plan(kernels)
assert len(p['jobs']) == 306 and len({j['tag'] for j in p['jobs']}) == 306
assert sum(j['kind']=='gemm' for j in p['jobs']) == 288
assert sum(j['round']>0 and j['kind']=='bw' for j in p['jobs']) == 15
assert p == m.plan(kernels)
for r in range(6):
    assert len({(j['spec']['kernel'],j['size']) for j in p['jobs'] if j['round']==r and j['kind']=='gemm'})==48
for j in p['jobs']:
    cmd=m.command(j,Path('/tmp/results'))
    if j['kind']=='gemm':
        args=dict(s[2:].split('=',1) for s in cmd[1:])
        assert [args[x] for x in ('A','B','C','D')]==[j['spec'][x] for x in ('A','B','C','D')]
        assert all(args[x]==str(j['size']) for x in ('m','n','k'))
    else:
        assert '--gigabytes' in cmd and cmd[cmd.index('--numtimes')+1]=='100'

FAKE = r'''#!/usr/bin/env python3
import csv,datetime as dt,json,os,sys,time
from pathlib import Path
name=Path(sys.argv[0]).name
case=os.environ['CASE']
with open(os.environ['CALLS'],'a') as f: f.write(json.dumps([name,*sys.argv[1:]])+'\n')
if name=='sudo':
 if sys.argv[1]=='-v': sys.exit(0)
 assert sys.argv[1]=='-n'
 if case=='setfail': sys.exit(9)
 os.execvp(sys.argv[2],sys.argv[2:])
elif name=='git':
 if 'rev-parse' in sys.argv:
  print('17ab377b0e919e14fd3df2b67268761fdac8abb3' if sys.argv[2].endswith('BabelStream') else '147295a3d4b75f3aeff247c25b8927cea9a7006a')
 elif 'status' in sys.argv:
  if case=='build_untracked' and '--untracked-files=no' not in sys.argv and sys.argv[2].endswith('BabelStream'):
   print(Path(os.environ['STATUS_FIXTURE']).read_text(),end='')
  if case=='tracked_dirty': print(' M src/main.cpp')
  if case=='staged_dirty': print('M  src/main.cpp')
elif name=='nvidia-smi':
 args=' '.join(sys.argv)
 if '--query-gpu=uuid' in args: print('GPU-abcd-750')
 elif '--query-compute-apps=pid' in args:
  if case=='busy': print('12345')
 elif '--query-gpu=power.limit,memory.free' in args: print('70, 19000')
 elif '-lgc' in sys.argv: assert sys.argv[-1]=='750,750'
 elif '-lmc' in sys.argv: assert sys.argv[-1]=='6801,6801'
 elif '--query-gpu=timestamp' in args:
  looping=any('--loop-ms' in a for a in sys.argv)
  if looping:
   with open(os.environ['PIDS'],'a') as f:f.write(str(os.getpid())+'\n')
  while True:
   stamp=dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
   sm=780 if case=='clock' and looping else (750 if looping else 210)
   mem=7001 if case=='memory' and looping else (6801 if looping else 405)
   reason='Active' if case=='reason' and looping else 'Not Active'
   if case=='unsupported':reason='[N/A]'
   print(f'{stamp}, {100 if looping else 0}, {sm}, {mem}, 45, 70, 45, {reason}, Not Active, Not Active, 100000',flush=True)
   if not looping:break
   if case=='monitor':print('sensor warning',file=sys.stderr,flush=True)
   time.sleep(.005)
 else: print('fake GPU snapshot')
elif name in ('cutlass_profiler','cuda-stream'):
 with open(os.environ['PIDS'],'a') as f:f.write(str(os.getpid())+'\n')
 time.sleep(10 if case=='signal' else .13)
 if case=='exit':sys.exit(7)
 if name=='cuda-stream':
  print(Path(os.environ['BW_FIXTURE']).read_text(),end='')
  if case=='bw':print('FAILED validation of sum',file=sys.stderr)
 else:
  args=dict(a[2:].split('=',1) for a in sys.argv[1:])
  n=int(args['m']);flops=2*n**3+2*n**2
  row={'Provider':'CUTLASS','Operation':args['kernels'],'Status':'success','Disposition':'not_verified' if case=='verify' else 'passed','Runtime':'50','Flops':str(flops),'GFLOPs':str(flops/50/1e6)}
  row.update({k:args[k] for k in ('m','n','k','A','B','C','D','op_class','accum','alpha','beta','batch_count','split_k_slices')})
  if case=='layout':row['A']='f32:wrong'
  with open(args['output']+'.gemm.csv','w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
  print('reference_device: Passed')
'''
with tempfile.TemporaryDirectory(prefix='roofline-offline-') as tmp:
    base=Path(tmp);bin_dir=base/'bin';bin_dir.mkdir()
    for name in ('sudo','git','nvidia-smi','cutlass_profiler','cuda-stream'):
        f=bin_dir/name;f.write_text(FAKE);f.chmod(0o755)
    driver=base/'driver.py'
    driver.write_text('''import importlib.util,json,os,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('roofline',Path(os.environ['PACKAGE'])/'roofline.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.REMOTE=Path(os.environ['WORK'])
m.CUTLASS=Path(os.environ['BIN'])/'cutlass_profiler'
m.BABEL=Path(os.environ['BIN'])/'cuda-stream'
m.MATRIX_SIZES=[8192];m.ARRAY_SIZES=[67108864];m.ROUNDS=1
original_sha=m.sha
m.sha=lambda p: '1e0608eb3eda54d329fd2cc017a28057aca952ae5732f8242aa5073abd7e8a4d' if p==m.CUTLASS else ('c465f7855e87a0a3de062ecb18812dc3d9b97569b94f041200902c45fc7790c5' if p==m.BABEL else original_sha(p))
if os.environ['CASE']=='lock':
 def locked(*args):raise BlockingIOError('test: GPU lock busy')
 m.fcntl.flock=locked
sys.exit(m.run())
''')
    cases={'ok':0,'reason':2,'clock':1,'memory':1,'verify':1,'layout':1,'bw':1,'exit':1,'busy':1,'lock':1,'setfail':1,'unsupported':1,'monitor':1,'signal':143,'build_untracked':0,'tracked_dirty':1,'staged_dirty':1}
    for case,code in cases.items():
        work=base/case;work.mkdir()
        env=dict(os.environ,PATH=str(bin_dir)+':'+os.environ['PATH'],PACKAGE=str(PACKAGE),CASE=case,
                 WORK=str(work),BIN=str(bin_dir),CALLS=str(work/'calls'),PIDS=str(work/'pids'),
                 STATUS_FIXTURE=str(ROOT/'analysis/roofline_750_v2_2026-09-21/raw_v1_failure/BabelStream_source_status.txt'),
                 BW_FIXTURE=str(ROOT/'analysis/babelstream_780_bw_2026-09-21/raw/extracted/r1_n67108864.log'))
        if case=='signal':
            process=sp.Popen([sys.executable,str(driver)],env=env,text=True,stdout=sp.PIPE,stderr=sp.PIPE)
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                if (work/'pids').exists() and len((work/'pids').read_text().splitlines())>=2:break
                time.sleep(.02)
            else:raise AssertionError('mock workload did not start')
            process.send_signal(signal.SIGTERM)
            stdout,stderr=process.communicate(timeout=20)
            result=sp.CompletedProcess(process.args,process.returncode,stdout,stderr)
        else:
            result=sp.run([sys.executable,str(driver)],env=env,text=True,capture_output=True,timeout=45)
        assert result.returncode==code,(case,result.stdout,result.stderr)
        out=next(p for p in work.glob('roofline_750_*') if p.is_dir())
        archive=Path(str(out)+'.tar.gz')
        assert archive.is_file() and (out/'runner.exit_code').read_text().strip()==str(code)
        with tarfile.open(archive) as t:
            assert t.extractfile(out.name+'/runner.exit_code').read().decode().strip()==str(code)
        calls=[json.loads(line) for line in (work/'calls').read_text().splitlines()]
        workloads=[c for c in calls if c[0] in ('cutlass_profiler','cuda-stream')]
        if case in ('ok','reason','build_untracked'):
            assert len(workloads)==34
            summary=json.loads((out/'analysis.json').read_text())
            assert summary['collection_complete'] and summary['valid_runs']==34
            assert len(summary['summaries'])==21
            assert all(r['repeats']==1 for r in summary['summaries'])
            assert summary['status']==('REVIEW' if case=='reason' else 'PASS')
        elif case in ('busy','lock','unsupported','tracked_dirty','staged_dirty'):
            assert not workloads and not any(c[0]=='sudo' for c in calls)
        elif case=='setfail':assert not workloads
        else:
            assert len(workloads)==(2 if case in ('verify','layout') else 1)
            assert all(x is None for x in json.loads((out/'analysis.json').read_text())['ceiling_candidates'].values())
        if (work/'pids').exists():
            for pid in (work/'pids').read_text().splitlines():
                try:os.kill(int(pid),0)
                except ProcessLookupError:pass
                else:raise AssertionError((case,'child still alive',pid))
        if case=='ok':
            # Replay five rounds from the captured mock run, without rerunning programs.
            replay=base/'replay';shutil.copytree(out,replay)
            m.MATRIX_SIZES=[8192];m.ARRAY_SIZES=[67108864]
            replay_plan=m.plan(kernels)
            m.save(replay/'plan.json',replay_plan)
            for job in replay_plan['jobs']:
                if job['round']<=1:continue
                source='r1_'+job['tag'].split('_',1)[1]
                for file in out.glob(source+'.*'):
                    shutil.copyfile(file,replay/(job['tag']+file.name[len(source):]))
            summary=m.analyze(replay)
            assert summary['status']=='PASS' and len(summary['summaries'])==21
            assert all(r['repeats']==5 and r['iqr']==0 for r in summary['summaries'])
            def change_runtime(tag,ms):
                f=replay/(tag+'.gemm.csv')
                row=list(csv.DictReader(f.open()))[0]
                row['Runtime']=str(ms);row['GFLOPs']=str(int(row['Flops'])/ms/1e6)
                with f.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            change_runtime('r0_fp32_simt_01_n8192',80)
            assert m.analyze(replay)['summaries']==summary['summaries'], 'warmup leaked into medians'
            change_runtime('r5_fp32_simt_01_n8192',80)
            changed=m.analyze(replay)
            assert changed['status']=='REVIEW' and changed['collection_complete']
            group=next(r for r in changed['summaries'] if r['function']==kernels[0]['kernel'])
            assert group['range_pct']>5 and group['median']==2*8192**3/50/1e9
            (replay/'r5_fp32_simt_02_n8192.exit_code').unlink()
            incomplete=m.analyze(replay)
            assert not incomplete['collection_complete'] and all(v is None for v in incomplete['ceiling_candidates'].values())
            print('five-round statistics, excluded warmup, variation REVIEW, incomplete-result gate PASS',flush=True)
        print(case,'PASS',flush=True)
print('PASS: full 306-job plan and 17 offline end-to-end cases plus five-round aggregation; no GPU execution.')
