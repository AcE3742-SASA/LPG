"""CPU-only checks of admission, result integrity, mode coverage and cleanup."""
import contextlib
from datetime import datetime, timedelta, timezone
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

import lower
import analyze_lower

LOADED = lower.load()
steps, hold, audit, original, telemetry, baseline, suite, config = LOADED
from test_steps import fixture_phase, q_result


def fixture_stage(out, stage, growth=False):
    directory = out/stage['name']
    fixture_phase(directory, stage['sm_mhz'], 50, growth)
    hold.save(directory/'environment.json',dict(baseline['environment_signature'],nvidia_smi=q_result()))
    start = datetime(2026,9,20,tzinfo=timezone.utc)
    def block(ms, ops, index=0):
        at = start+timedelta(seconds=3.1+.11*index)
        return dict(repeats=1,operations_per_call=ops,elapsed_ms=ms,latency_ms=ms/ops,
            host_envelope_start_utc=at.isoformat(),host_envelope_end_utc=(at+timedelta(milliseconds=ms+5)).isoformat())
    rows = []
    for task in lower.cases(config):
        spec = next(s for s in config['fixed_cases'] if s['shape']==task['shape'] and s['dtype']==task['dtype'])
        ops = steps.graph_nodes(config,task) or 1
        timing = dict(blocks=[block(100,ops,i) for i in range(7)],median_ms=100/ops,
            relative_iqr=0,first_last_drift=0,flags=[],
            setup=dict(settle_event_ms=3000,settle_blocks=[block(3000,ops)],calibration=[block(100,ops)]))
        if task['mode']=='graph':timing['graph_nodes']=ops
        correct = dict(passed=True,finite_all=True,max_tolerance_ratio=.01,
                       scope=spec['correctness_scope'],points=spec['correctness_points'])
        rows.append(dict(task=task,result=dict(timing=timing,correctness=correct,input_seed=spec['input_seed'])))
    hold.save(directory/'results.json',rows)
    (out/(stage['name']+'.exit_code.txt')).write_text('0\n')


def fixture(out, growth=False):
    lower.make_manifest(out,LOADED)
    for stage in config['stages']:fixture_stage(out,stage,growth)
    hold.save(out/'restore.json',dict(commands_accepted=True))


class LowerTests(unittest.TestCase):
    def test_three_modes_same_inputs_batches_and_balanced_reverse_order(self):
        tasks = lower.cases(config)
        self.assertEqual(len(tasks),60)
        self.assertEqual(len({t['id'] for t in tasks}),60)
        first = [(t['shape'],t['dtype'],t['mode']) for t in tasks[:30]]
        self.assertEqual(first,list(reversed([(t['shape'],t['dtype'],t['mode']) for t in tasks[30:]])))
        previous = audit.cases,audit.graph_nodes,audit.PASSES
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with lower.configured_worker(LOADED):
                for task in audit.cases():
                    spec=next(s for s in config['fixed_cases'] if s['shape']==task['shape'] and s['dtype']==task['dtype'])
                    self.assertEqual(sys.modules['plan'].task_seed(task),spec['input_seed'])
                    self.assertEqual(audit.graph_nodes(task),spec['graph_gemms_per_replay'] if task['mode']=='graph' else None)
                raise RuntimeError('stop')
        self.assertEqual((audit.cases,audit.graph_nodes,audit.PASSES),previous)

    def test_complete_and_counter_only_review_have_separate_gates(self):
        for growth in (False,True):
            with self.subTest(growth=growth),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);fixture(out,growth)
                result=analyze_lower.analyze(out,LOADED)
                self.assertEqual(result['gate'],'REVIEW' if growth else 'PASS')
                self.assertEqual((result['recorded'],result['timed_blocks'],result['mode_triplets']),(240,1680,80))
                self.assertEqual(result['correctness_scopes'],{'sampled':192,'full':48})
                self.assertTrue(all(result['stage_timing_and_sampled_clocks'].values()))
                self.assertEqual(result['counter_anomaly_blocks'],1680 if growth else 0)
                self.assertIsNone(result['selected_clock_mhz'])
                self.assertEqual(result['full_sweep'],'HOLD')

    def test_block_counter_issue_not_hidden_by_active_sample_elsewhere(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);stage=config['stages'][0];fixture_stage(out,stage,True)
            path=out/stage['name']/'telemetry.csv';lines=path.read_text().splitlines()
            fields=lines[0].split(',');fields[10]='Active';lines[0]=','.join(fields)
            path.write_text('\n'.join(lines)+'\n')
            result,_,_,_,anomalies=analyze_lower.analyze_stage(out,stage,LOADED)
            self.assertEqual(result['counter_anomaly_blocks'],420)
            self.assertTrue(anomalies)
            self.assertTrue(any('block_increase_without_active_inside_sample' in f for f in result['flags']))
            self.assertFalse(result['timing_and_sampled_clocks_pass'])

    def test_bad_clocks_power_cap_unavailable_and_missing_samples_are_review(self):
        for field,value in [(5,'735'),(6,'7001'),(8,'60'),(10,'Active'),(11,'Not Supported')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);stage=config['stages'][0];fixture_stage(out,stage)
                path=out/stage['name']/'telemetry.csv';lines=[]
                for line in path.read_text().splitlines():
                    fields=line.split(',');fields[field]=value;lines.append(','.join(fields))
                path.write_text('\n'.join(lines)+'\n')
                result,*_=analyze_lower.analyze_stage(out,stage,LOADED)
                self.assertFalse(result['timing_and_sampled_clocks_pass'])
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);stage=config['stages'][0];fixture_stage(out,stage)
            path=out/stage['name']/'telemetry.csv';path.write_text('\n'.join(path.read_text().splitlines()[:2])+'\n')
            result,*_=analyze_lower.analyze_stage(out,stage,LOADED)
            self.assertTrue(any('no_inside_clock_sample' in f for f in result['flags']))

    def test_corrupted_results_and_incomplete_allocate_coverage_rejected(self):
        for variant in ('denominator','seed','graph','correctness','scope','missing','duplicate','environment'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);stage=config['stages'][0];fixture_stage(out,stage)
                path=out/stage['name']/'results.json';rows=lower.read(path)
                if variant=='denominator':rows[0]['result']['timing']['blocks'][0]['repeats']=2
                elif variant=='seed':rows[0]['result']['input_seed']=1
                elif variant=='graph':next(r for r in rows if r['task']['mode']=='graph')['result']['timing']['graph_nodes']=1
                elif variant=='correctness':rows[0]['result']['correctness']['passed']=False
                elif variant=='scope':rows[-1]['result']['correctness']['points']=1
                elif variant=='missing':rows=[r for r in rows if r['task']['mode']!='allocate']
                elif variant=='duplicate':rows[-1]=rows[0]
                else:
                    ep=out/stage['name']/'environment.json';env=lower.read(ep);env['gpu_uuid']='wrong';hold.save(ep,env)
                hold.save(path,rows)
                with self.assertRaises(ValueError):analyze_lower.analyze_stage(out,stage,LOADED)

    def test_stable_blocks_cannot_hide_round_and_before_after_variation(self):
        for variant in ('round','before_after'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);fixture(out)
                stage=config['stages'][-1];path=out/stage['name']/'results.json';rows=lower.read(path)
                targets=[rows[-1]] if variant=='round' else rows
                for row in targets:
                    timing=row['result']['timing']
                    for block in timing['blocks']:
                        block['repeats']*=2;block['latency_ms']/=2
                    timing['median_ms']/=2
                hold.save(path,rows)
                result=analyze_lower.analyze(out,LOADED)
                self.assertEqual(result['gate'],'REVIEW')
                if variant=='round':self.assertFalse(result['stage_timing_and_sampled_clocks'][stage['name']])
                else:
                    self.assertTrue(all(result['stage_timing_and_sampled_clocks'].values()))
                    self.assertFalse(result['repeat_750_time_pass'])

    def test_source_manifest_and_dependency_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);fixture(out);(out/'source/protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):analyze_lower.analyze(out,LOADED)
        original_read=lower.read
        def changed(path):
            data=original_read(path)
            if path==lower.HERE/'dependency_hashes.json':data[next(iter(data))]='0'*64
            return data
        with patch.object(lower,'read',changed),self.assertRaisesRegex(RuntimeError,'Dependency changed'):lower.load()

    def test_run_restores_under_lock_on_success_failure_and_interrupt(self):
        for outcome in ('success','setup_error','worker_error','interrupt','restore_error','wrong_power','wrong_gpu'):
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
                    if '--query-gpu=uuid' in cmd:
                        return dict(exit_code=0,stdout='wrong' if outcome=='wrong_gpu' else baseline['environment_signature']['gpu_uuid'],stderr='')
                    if '--query-gpu=power.limit' in cmd:return dict(exit_code=0,stdout='30' if outcome=='wrong_power' else '70',stderr='')
                    fail=(outcome=='setup_error' and '-lgc' in cmd) or (outcome=='restore_error' and '-rmc' in cmd)
                    return q_result(code=1 if fail else 0)
                def worker(cmd,directory,name,timeout):
                    if outcome=='worker_error':raise RuntimeError('worker failed')
                    if outcome=='interrupt':raise KeyboardInterrupt
                    fixture_stage(out,next(s for s in config['stages'] if s['name']==name))
                credentials=SimpleNamespace(failed=False,check=lambda:None)
                with patch.object(suite,'lock',lock),patch.object(suite,'command_text',command),\
                     patch.object(audit,'check_processes'),patch.object(audit,'run_logged',worker),\
                     patch.object(lower.shutil,'which',return_value='/nvidia-smi'),\
                     patch.object(steps,'Credentials',return_value=contextlib.nullcontext(credentials)):
                    args=SimpleNamespace(output=out)
                    if outcome in ('success','restore_error'):self.assertEqual(lower.run(args),0 if outcome=='success' else 1)
                    elif outcome=='interrupt':
                        with self.assertRaises(KeyboardInterrupt):lower.run(args)
                    else:
                        with self.assertRaises(RuntimeError):lower.run(args)
                status=lower.read(out/'status.json')
                if outcome in ('wrong_power','wrong_gpu'):
                    self.assertEqual(status['restoration'],'NOT_ATTEMPTED')
                    self.assertFalse(any(c[0]=='sudo' for c in commands))
                else:
                    self.assertTrue((out/'reset_graphics.json').exists())
                    self.assertTrue((out/'reset_memory.json').exists())
                    self.assertEqual(status['restoration'],'FAILED' if outcome=='restore_error' else 'COMMANDS_ACCEPTED')

    def test_launcher_preserves_review_exit_and_waits_for_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pkg=root/'clock_lower_v1';pkg.mkdir();bins=root/'bin';bins.mkdir()
            for name in ('lower.py','launch.sh'):shutil.copy2(lower.HERE/name,pkg/name)
            sd=root/'gpu_sweep_v6';sd.mkdir();(sd/'run_research_checked.sh').write_text('#!/bin/bash\n"$@"\n')
            stubs={'sudo':'#!/bin/sh\nexit 0\n',
                'python3':f'#!{sys.executable}\nimport sys,os\nif len(sys.argv)>3 and sys.argv[3]=="verify": sys.exit(0)\nos.execv({sys.executable!r},[{sys.executable!r}]+sys.argv[1:])\n',
                'python':f'#!{sys.executable}\nimport sys\nprint("CLEANUP COMPLETE")\nsys.exit(2)\n'}
            for name,content in stubs.items():
                p=bins/name;p.write_text(content);p.chmod(0o755)
            env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ['PATH'])
            result=subprocess.run(['bash',str(pkg/'launch.sh')],env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            archive=next(pkg.glob('runs/*.zip'))
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('CLEANUP COMPLETE',z.read(next(n for n in z.namelist() if n.endswith('/launcher.log'))).decode())
                self.assertEqual(z.read(next(n for n in z.namelist() if n.endswith('/exit_code.txt'))).strip(),b'2')
            (bins/'python').write_text(f'''#!{sys.executable}
import pathlib,signal,sys,time
out=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
def stop(sig,frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT,stop)
signal.signal(signal.SIGTERM,stop)
(out/'started.marker').write_text('yes')
try: time.sleep(30)
except KeyboardInterrupt: pass
finally:
 time.sleep(.2)
 (out/'restored.marker').write_text('complete')
sys.exit(130)
''')
            proc=subprocess.Popen(['bash',str(pkg/'launch.sh')],env=env,start_new_session=True,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                deadline=time.monotonic()+5
                while not list(pkg.glob('runs/*/started.marker')) and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue(list(pkg.glob('runs/*/started.marker')))
                os.killpg(proc.pid,signal.SIGINT);proc.communicate(timeout=5)
                extra=[p for p in pkg.glob('runs/*.zip') if p!=archive]
                self.assertEqual(len(extra),1)
                with zipfile.ZipFile(extra[0]) as z:self.assertTrue(any(n.endswith('/restored.marker') for n in z.namelist()))
            finally:
                if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.communicate()


if __name__=='__main__':unittest.main()
