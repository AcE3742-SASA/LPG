import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import hold

BASE = (hold.HERE.parent/'clock_audit_v1').resolve()
SUITE = (hold.HERE.parent/'gpu_sweep_v6').resolve()
audit, analyze, telemetry, baseline, suite = hold.load(BASE, SUITE)
from test_clock_audit import ClockAuditTests


class HoldTests(unittest.TestCase):
    def fixture(self, out):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            ClockAuditTests().fixture(source)
            shutil.copytree(source/'dense_before',out/'dense_before')
        path=out/'dense_before/telemetry.csv'
        lines=[]
        for line in path.read_text().splitlines():
            fields=[s.strip() for s in line.split(',')]
            fields[5]='900'; fields[9]='0'; fields[10]='Not Active'
            lines.append(','.join(fields))
        path.write_text('\n'.join(lines)+'\n')

    def summary(self, out):
        return hold.summarize(out,audit,analyze,telemetry,baseline)

    def test_complete_clock_match_and_sampled_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out)
            result=self.summary(out)
            self.assertEqual((result['gate'],result['recorded'],result['timed_blocks']),('PASS',8,56))
            self.assertEqual(result['full_sweep'],'HOLD')
            self.assertEqual(len(hold.read(out/'analysis.json')['rounds']),4)
            self.assertEqual(hold.read(out/'analysis.json')['measurements'][0]['observed_values']['clocks.mem'],[6801.0])

    def test_clock_memory_cap_and_missing_samples_are_review(self):
        variants=[('900','885'),('6801','7001'),('Not Active,Not Active,Not Active','Active,Not Active,Not Active')]
        for old,new in variants:
            with self.subTest(new=new),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);self.fixture(out)
                path=out/'dense_before/telemetry.csv'
                path.write_text(path.read_text().replace(old,new))
                self.assertEqual(self.summary(out)['gate'],'REVIEW')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out)
            path=out/'dense_before/telemetry.csv';path.write_text('\n'.join(path.read_text().splitlines()[:2])+'\n')
            self.assertTrue(any('no_inside_clock_sample' in f for f in self.summary(out)['flags']))

    def test_round_spread_and_missing_or_bad_results(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out)
            path=out/'dense_before/results.json';rows=hold.read(path)
            timing=rows[-1]['result']['timing']
            for b in timing['blocks']:
                b['repeats']*=2;b['latency_ms']/=2
            timing['median_ms']/=2
            hold.save(path,rows)
            self.assertTrue(any('round_range' in f for f in self.summary(out)['flags']))
            rows.pop();hold.save(path,rows)
            with self.assertRaisesRegex(ValueError,'Missing'):
                self.summary(out)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out)
            path=out/'dense_before/results.json';rows=hold.read(path)
            rows[0]['result']['correctness']['passed']=False;hold.save(path,rows)
            with self.assertRaisesRegex(ValueError,'Correctness'):
                self.summary(out)

    def test_restore_attempts_both_after_first_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);calls=[]
            def command(args):
                calls.append(args)
                return {'exit_code':1 if '-rgc' in args else 0,'stdout':'','stderr':''}
            fake=SimpleNamespace(command_text=command)
            self.assertFalse(hold.restore(fake,out,'/nvidia-smi','GPU-test'))
            self.assertEqual([c[-1] for c in calls],['-rgc','-rmc','-q'])
            self.assertFalse(hold.read(out/'restore.json')['commands_accepted'])

    def test_run_restores_under_shared_lock_success_failure_and_interrupt(self):
        for outcome in ['success','worker_error','interrupt','restore_error']:
            with self.subTest(outcome=outcome),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);events=[];locked=[]
                @contextlib.contextmanager
                def lock(path):
                    locked.append(True);events.append('lock')
                    try:yield
                    finally:locked.pop();events.append('unlock')
                def command(args):
                    if args[0]=='sudo':
                        self.assertTrue(locked);events.append(args[-2] if args[-2].startswith('-l') else args[-1])
                    code=1 if outcome=='restore_error' and '-rgc' in args else 0
                    return {'exit_code':code,'stdout':baseline['environment_signature']['gpu_uuid']+'\n','stderr':''}
                def worker(*args,**kwargs):
                    if outcome=='worker_error':raise RuntimeError('worker failed')
                    if outcome=='interrupt':raise KeyboardInterrupt
                    self.fixture(out)
                args=SimpleNamespace(base=BASE,suite=SUITE,output=out)
                with patch.object(suite,'lock',lock),patch.object(suite,'command_text',command),\
                     patch.object(audit,'check_processes'),patch.object(audit,'run_logged',worker),\
                     patch.object(shutil,'which',return_value='/nvidia-smi'):
                    if outcome in ['success','restore_error']:
                        self.assertEqual(hold.run(args),0 if outcome=='success' else 1)
                    elif outcome=='worker_error':
                        with self.assertRaises(RuntimeError):hold.run(args)
                    else:
                        with self.assertRaises(KeyboardInterrupt):hold.run(args)
                self.assertEqual(events,['lock','-lgc','-lmc','-rgc','-rmc','unlock'])
                self.assertTrue((out/'status.json').exists())

    def test_dependency_mutation_rejected_and_package_has_closed_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'base';shutil.copytree(BASE,base,ignore=shutil.ignore_patterns('__pycache__','runs'))
            (base/'audit.py').write_text('# changed\n')
            with self.assertRaisesRegex(RuntimeError,'dependency changed'):
                hold.load(base,SUITE)
            out=root/'result';out.mkdir();(out/'output.log').write_text('closed\n')
            hold.package(out)
            import zipfile
            with zipfile.ZipFile(out.with_suffix('.zip')) as z:
                self.assertEqual(z.read('result/output.log'),b'closed\n')

    def test_launcher_waits_for_interrupted_cleanup_before_packaging(self):
        with tempfile.TemporaryDirectory(prefix='hold shell ') as temp:
            root=Path(temp);package=root/'clock_hold_v1';package.mkdir()
            fake_suite=root/'gpu_sweep_v6';fake_suite.mkdir();bins=root/'bin';bins.mkdir()
            for name in ['hold.py','launch.sh']:
                shutil.copy2(hold.HERE/name,package/name)
            (fake_suite/'run_research_checked.sh').write_text('#!/bin/bash\n"$@"\n')
            stubs={
                'sudo':'#!/bin/sh\nexit 0\n',
                'python3':f'#!{sys.executable}\nimport sys,os\nif len(sys.argv)>3 and sys.argv[3]=="verify": sys.exit(0)\nos.execv({sys.executable!r},[{sys.executable!r}]+sys.argv[1:])\n',
                'python':f'''#!{sys.executable}
import sys,signal,time,pathlib
out=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
def stop(sig,frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
(out/'started').write_text('yes')
try: time.sleep(30)
except KeyboardInterrupt: pass
finally:
 time.sleep(.2)
 (out/'restored.marker').write_text('complete')
'''}
            for name,content in stubs.items():
                path=bins/name;path.write_text(content);path.chmod(0o755)
            env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ['PATH'])
            p=subprocess.Popen(['bash',str(package/'launch.sh')],env=env,start_new_session=True,
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                deadline=time.monotonic()+5
                while not list(package.glob('runs/*/started')) and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertTrue(list(package.glob('runs/*/started')))
                os.killpg(p.pid,signal.SIGINT)
                p.communicate(timeout=5)
                archives=list(package.glob('runs/*.zip'))
                self.assertEqual(len(archives),1)
                import zipfile
                with zipfile.ZipFile(archives[0]) as z:
                    self.assertTrue(any(n.endswith('restored.marker') for n in z.namelist()))
            finally:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGKILL);p.communicate()


if __name__=='__main__':
    unittest.main()
