"""CPU-only regression checks; synthetic telemetry does not validate GPU performance."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
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
import zipfile

import fixed
import analyze_pilot

LOADED = fixed.load()
steps, hold, audit, original, telemetry, baseline, suite, config, tasks = LOADED
from test_steps import fixture_phase, q_result


def fixture(out, growth=False):
    directory = out/'pilot'
    fixture_phase(directory, config['sm_mhz'], config['interval_ms'], growth)
    hold.save(directory/'environment.json', dict(baseline['environment_signature'], nvidia_smi=q_result()))
    start = datetime(2026,9,20,tzinfo=timezone.utc)
    def block(ms, ops, index=0):
        at = start+timedelta(seconds=3.1+.11*index)
        return dict(repeats=1, operations_per_call=ops, elapsed_ms=ms, latency_ms=ms/ops,
            host_envelope_start_utc=at.isoformat(), host_envelope_end_utc=(at+timedelta(milliseconds=ms+5)).isoformat())
    rows = []
    for task in tasks:
        spec = next(s for s in config['fixed_cases'] if s['shape']==task['shape'] and s['dtype']==task['dtype'])
        ops = spec['graph_gemms_per_replay'] if task['mode']=='graph' else 1
        timing = dict(blocks=[block(100,ops,i) for i in range(7)], median_ms=100/ops,
            relative_iqr=0, first_last_drift=0, flags=[],
            setup=dict(settle_event_ms=3000, settle_blocks=[block(3000,ops)], calibration=[block(100,ops)]))
        if task['mode']=='graph': timing['graph_nodes']=ops
        correct=dict(passed=True, finite_all=True, max_tolerance_ratio=.01,
                     scope=spec['correctness_scope'], points=spec['correctness_points'])
        rows.append(dict(task=task,result=dict(timing=timing,correctness=correct,input_seed=spec['input_seed'])))
    hold.save(directory/'results.json', rows)
    (out/'worker.exit_code.txt').write_text('0\n')
    hold.save(out/'restore.json', dict(commands_accepted=True))


class FixedTests(unittest.TestCase):
    def test_exact_pilot_order_seeds_batches_and_adapter_restoration(self):
        self.assertEqual(len(tasks),192)
        self.assertEqual(len({tuple(t['shape']) for t in tasks}),24)
        self.assertEqual(sum(s['correctness_scope']=='full' for s in config['fixed_cases']),4)
        before=audit.cases,audit.graph_nodes,audit.PASSES
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with fixed.configured_worker(LOADED):
                self.assertEqual(audit.cases(),tasks)
                self.assertEqual(audit.PASSES,{'pilot':50})
                for task in tasks:
                    spec=next(s for s in config['fixed_cases'] if s['shape']==task['shape'] and s['dtype']==task['dtype'])
                    self.assertEqual(audit.graph_nodes(task),spec['graph_gemms_per_replay'] if task['mode']=='graph' else None)
                raise RuntimeError('deliberate')
        self.assertEqual((audit.cases,audit.graph_nodes,audit.PASSES),before)

    def test_complete_pilot_and_counter_review_remain_distinct(self):
        for growth in (False,True):
            with self.subTest(growth=growth),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);fixed.make_manifest(out,LOADED);fixture(out,growth)
                result=analyze_pilot.analyze(out,LOADED)
                self.assertEqual(result['gate'],'REVIEW' if growth else 'PASS')
                self.assertTrue(result['timing_and_sampled_clocks_pass'])
                self.assertEqual((result['recorded'],result['timed_blocks'],result['paired_complete']),(192,1344,96))
                self.assertEqual(result['correctness_scopes'],{'sampled':176,'full':16})
                self.assertEqual(result['counts'],{'PASS':192})
                self.assertEqual(result['full_sweep'],'HOLD')
                self.assertEqual(result['selected_clock_mhz'],780)
                self.assertEqual(result['clock_selection'],'USER_SELECTED')
                self.assertFalse(result['validated_for_full'])
                if growth:self.assertTrue(any('increase_without_active_sample' in f for f in result['flags']))

    def test_incorrect_clocks_active_reason_or_missing_inside_sample_cannot_pass(self):
        for field,value in [(5,'885'),(6,'7001'),(8,'60'),(10,'Active'),(11,'Active'),(14,'N/A')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);fixed.make_manifest(out,LOADED);fixture(out)
                path=out/'pilot/telemetry.csv';lines=[]
                for line in path.read_text().splitlines():
                    row=line.split(',');row[field]=value;lines.append(','.join(row))
                path.write_text('\n'.join(lines)+'\n')
                result=analyze_pilot.analyze(out,LOADED)
                self.assertEqual(result['gate'],'REVIEW')
                self.assertFalse(result['timing_and_sampled_clocks_pass'])
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);fixed.make_manifest(out,LOADED);fixture(out)
            path=out/'pilot/telemetry.csv';path.write_text('\n'.join(path.read_text().splitlines()[:2])+'\n')
            result=analyze_pilot.analyze(out,LOADED)
            self.assertTrue(any('no_inside_clock_sample' in f for f in result['flags']))

    def test_changed_timing_seed_batch_correctness_missing_duplicate_tasks_rejected(self):
        for variant in ['denominator','seed','batch','scope','points','correctness','missing','duplicate','statistics']:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);fixed.make_manifest(out,LOADED);fixture(out)
                path=out/'pilot/results.json';rows=fixed.read(path);result=rows[0]['result']
                if variant=='denominator':result['timing']['blocks'][0]['repeats']=2
                elif variant=='seed':result['input_seed']=0
                elif variant=='batch':rows[1]['result']['timing']['graph_nodes']=999
                elif variant=='scope':result['correctness']['scope']='full'
                elif variant=='points':result['correctness']['points']=1
                elif variant=='correctness':result['correctness']['passed']=False
                elif variant=='statistics':result['timing']['median_ms']*=2
                elif variant=='missing':rows.pop()
                else:rows[-1]=copy.deepcopy(rows[0])
                hold.save(path,rows)
                with self.assertRaises(ValueError):analyze_pilot.analyze(out,LOADED)

    def test_round_variation_is_not_hidden_by_stable_individual_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);fixed.make_manifest(out,LOADED);fixture(out)
            path=out/'pilot/results.json';rows=fixed.read(path);timing=rows[0]['result']['timing']
            for block in timing['blocks']:
                block['repeats']*=2;block['latency_ms']/=2
            timing['median_ms']/=2;hold.save(path,rows)
            result=analyze_pilot.analyze(out,LOADED)
            self.assertEqual(result['counts'],{'PASS':192})
            self.assertFalse(result['timing_and_sampled_clocks_pass'])
            self.assertTrue(any('round_range_above_5pct' in f for f in result['flags']))
            self.assertIn('True',(out/'review_queue.csv').read_text())

    def test_sources_environment_and_worker_exit_checked(self):
        for variant in ['source','environment','worker']:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);fixed.make_manifest(out,LOADED);fixture(out)
                if variant=='source':(out/'source/protocol.json').write_text('{}')
                elif variant=='worker':(out/'worker.exit_code.txt').write_text('1')
                else:
                    path=out/'pilot/environment.json';env=fixed.read(path);env['torch']='changed';hold.save(path,env)
                with self.assertRaises(ValueError):analyze_pilot.analyze(out,LOADED)

    def test_controller_restores_under_gpu_lock_and_preserves_failure(self):
        for outcome in ['pass','review','setup_error','worker_error','interrupt','restore_error','wrong_power','busy','analysis_error']:
            with self.subTest(outcome=outcome),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);locked=[];commands=[]
                @contextlib.contextmanager
                def lock(path):
                    locked.append(True)
                    try:yield
                    finally:locked.pop()
                def command(cmd):
                    commands.append(cmd)
                    if cmd[0]=='sudo':self.assertTrue(locked)
                    if '--query-gpu=uuid' in cmd:return dict(exit_code=0,stdout=baseline['environment_signature']['gpu_uuid'],stderr='')
                    if '--query-gpu=power.limit' in cmd:return dict(exit_code=0,stdout='30' if outcome=='wrong_power' else '70',stderr='')
                    fail=(outcome=='setup_error' and '-lmc' in cmd) or (outcome=='restore_error' and '-rmc' in cmd)
                    return q_result(code=1 if fail else 0)
                def worker(cmd,directory,name,timeout):
                    self.assertEqual(timeout,2400)
                    if outcome=='worker_error':raise RuntimeError('worker failed')
                    if outcome=='interrupt':raise KeyboardInterrupt
                    fixture(out,outcome=='review')
                    if outcome=='analysis_error':(out/'pilot/results.json').write_text('[]')
                credentials=SimpleNamespace(failed=False,check=lambda:None)
                with patch.object(suite,'lock',lock),patch.object(suite,'command_text',command),\
                     patch.object(audit,'check_processes',side_effect=RuntimeError('busy') if outcome=='busy' else None),\
                     patch.object(audit,'run_logged',worker),patch.object(fixed.shutil,'which',return_value='/nvidia-smi'),\
                     patch.object(steps,'Credentials',return_value=contextlib.nullcontext(credentials)):
                    if outcome in ['pass','review','restore_error']:
                        self.assertEqual(fixed.run(SimpleNamespace(output=out)),{'pass':0,'review':2,'restore_error':1}[outcome])
                    elif outcome=='interrupt':
                        with self.assertRaises(KeyboardInterrupt):fixed.run(SimpleNamespace(output=out))
                    else:
                        with self.assertRaises((RuntimeError,ValueError)):fixed.run(SimpleNamespace(output=out))
                status=fixed.read(out/'status.json')
                if outcome in ['wrong_power','busy']:
                    self.assertEqual(status['restoration'],'NOT_ATTEMPTED')
                    self.assertFalse(any(c[0]=='sudo' for c in commands))
                else:
                    self.assertTrue((out/'reset_graphics.json').exists() and (out/'reset_memory.json').exists())
                    self.assertEqual(status['restoration'],'FAILED' if outcome=='restore_error' else 'COMMANDS_ACCEPTED')
                self.assertFalse(any('-pl' in c for c in commands))
                if outcome in ['pass','review']:
                    self.assertEqual([c[-1] for c in commands if '-lgc' in c],['780,780'])
                    self.assertEqual([c[-1] for c in commands if '-lmc' in c],['6801,6801'])
                if outcome not in ['pass','review']:
                    self.assertEqual(fixed.read(out/'summary.json')['gate'],'FAIL')

    def test_launcher_preserves_review_and_waits_for_cleanup_before_packaging(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=root/'fixed_pilot_780_v1';package.mkdir();bins=root/'bin';bins.mkdir()
            for name in ['fixed.py','launch.sh']:shutil.copy2(fixed.HERE/name,package/name)
            suite_dir=root/'gpu_sweep_v6';suite_dir.mkdir()
            (suite_dir/'run_research_checked.sh').write_text('#!/bin/bash\n"$@"\n')
            stubs={'sudo':'#!/bin/sh\nexit 0\n',
                'python3':f'#!{sys.executable}\nimport sys,os\nif len(sys.argv)>3 and sys.argv[3]=="verify":sys.exit(0)\nos.execv({sys.executable!r},[{sys.executable!r}]+sys.argv[1:])\n',
                'python':f'#!{sys.executable}\nimport sys\nprint("RESTORE CLOSED")\nsys.exit(2)\n'}
            for name,text in stubs.items():
                path=bins/name;path.write_text(text);path.chmod(0o755)
            env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ['PATH'])
            result=subprocess.run(['bash',str(package/'launch.sh')],env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            archive=next(package.glob('runs/*.zip'))
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('RESTORE CLOSED',z.read(next(n for n in z.namelist() if n.endswith('/launcher.log'))).decode())
                self.assertEqual(z.read(next(n for n in z.namelist() if n.endswith('/exit_code.txt'))).strip(),b'2')
            (bins/'python').write_text(f'''#!{sys.executable}
import pathlib,signal,sys,time
out=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
def stop(sig,frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT,stop)
signal.signal(signal.SIGTERM,stop)
(out/'started.marker').write_text('yes')
try:time.sleep(30)
except KeyboardInterrupt:pass
finally:
 time.sleep(.2)
 (out/'restored.marker').write_text('complete')
sys.exit(130)
''')
            proc=subprocess.Popen(['bash',str(package/'launch.sh')],env=env,start_new_session=True,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                deadline=time.monotonic()+5
                while not list(package.glob('runs/*/started.marker')) and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue(list(package.glob('runs/*/started.marker')))
                os.killpg(proc.pid,signal.SIGINT);proc.communicate(timeout=5)
                interrupted=[p for p in package.glob('runs/*.zip') if p!=archive]
                self.assertEqual(len(interrupted),1)
                with zipfile.ZipFile(interrupted[0]) as z:
                    self.assertTrue(any(n.endswith('/restored.marker') for n in z.namelist()))
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGKILL);proc.communicate()


if __name__=='__main__':unittest.main()
