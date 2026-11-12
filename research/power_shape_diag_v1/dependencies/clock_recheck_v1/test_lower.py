"""CPU-only checks of admission, result integrity, mode coverage and cleanup."""
import contextlib
import copy
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
    def test_incident_inputs_graph_sizes_and_condition_order_are_frozen(self):
        self.assertEqual(config['fixed_cases'][0]['input_seed'],1456857178)
        self.assertEqual(config['fixed_cases'][0]['graph_gemms_per_replay'],5)
        self.assertEqual(config['fixed_cases'][1]['input_seed'],2682971897)
        self.assertEqual(config['fixed_cases'][1]['graph_gemms_per_replay'],20)
        original_read=lower.read
        for variant in ('seed','graph','order'):
            altered=copy.deepcopy(config)
            if variant=='seed':altered['fixed_cases'][0]['input_seed']+=1
            elif variant=='graph':altered['fixed_cases'][1]['graph_gemms_per_replay']=5
            else:altered['fixed_cases'].reverse()
            def changed(path):
                return altered if path==lower.HERE/'protocol.json' else original_read(path)
            with patch.object(lower,'read',changed),self.assertRaises(RuntimeError):lower.load()

    def test_candidate_schedule_and_reference_cannot_silently_change(self):
        original_read = lower.read
        for variant in ('candidate', 'missing_return', 'reference'):
            with self.subTest(variant=variant):
                altered = copy.deepcopy(config)
                if variant == 'candidate':
                    altered['stages'][1]['sm_mhz'] = 945
                elif variant == 'missing_return':
                    altered['stages'].pop()
                else:
                    altered['reference_clock_mhz'] = 900
                def changed(path):
                    return altered if path == lower.HERE/'protocol.json' else original_read(path)
                with patch.object(lower, 'read', changed), self.assertRaises(RuntimeError):
                    lower.load()

    def test_three_modes_same_inputs_batches_and_balanced_reverse_order(self):
        tasks = lower.cases(config)
        self.assertEqual(len(tasks),36)
        self.assertEqual(len({t['id'] for t in tasks}),36)
        first = [(t['shape'],t['dtype'],t['mode']) for t in tasks[:18]]
        self.assertEqual(first,list(reversed([(t['shape'],t['dtype'],t['mode']) for t in tasks[18:]])))
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
                self.assertEqual((result['recorded'],result['timed_blocks'],result['mode_triplets']),(108,756,36))
                self.assertEqual(result['correctness_scopes'],{'sampled':72,'full':36})
                self.assertTrue(all(result['stage_timing_and_sampled_clocks'].values()))
                self.assertEqual(result['counter_anomaly_blocks'],756 if growth else 0)
                self.assertIsNone(result['selected_clock_mhz'])
                self.assertEqual(result['reference_clock_mhz'],780)
                self.assertFalse(result['validated_for_full'])
                self.assertEqual(result['full_sweep'],'HOLD')

    def test_block_counter_issue_not_hidden_by_active_sample_elsewhere(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);stage=config['stages'][0];fixture_stage(out,stage,True)
            path=out/stage['name']/'telemetry.csv';lines=path.read_text().splitlines()
            fields=lines[0].split(',');fields[10]='Active';lines[0]=','.join(fields)
            path.write_text('\n'.join(lines)+'\n')
            result,_,_,_,anomalies=analyze_lower.analyze_stage(out,stage,LOADED)
            self.assertEqual(result['counter_anomaly_blocks'],252)
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
                    self.assertFalse(result['repeat_780_time_pass'])

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
        for outcome in ('success','clock_review','setup_error','worker_error','later_worker_error',
                        'interrupt','restore_error','wrong_power','wrong_gpu'):
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
                    if outcome=='later_worker_error' and name=='s1_750':raise RuntimeError('later worker failed')
                    if outcome=='interrupt':raise KeyboardInterrupt
                    fixture_stage(out,next(s for s in config['stages'] if s['name']==name))
                    if outcome=='clock_review' and name=='s0_780_before':
                        path=out/name/'telemetry.csv';lines=[]
                        for line in path.read_text().splitlines():
                            fields=line.split(',');fields[5]='765';lines.append(','.join(fields))
                        path.write_text('\n'.join(lines)+'\n')
                credentials=SimpleNamespace(failed=False,check=lambda:None)
                with patch.object(suite,'lock',lock),patch.object(suite,'command_text',command),\
                     patch.object(audit,'check_processes'),patch.object(audit,'run_logged',worker),\
                     patch.object(lower.shutil,'which',return_value='/nvidia-smi'),\
                     patch.object(steps,'Credentials',return_value=contextlib.nullcontext(credentials)):
                    args=SimpleNamespace(output=out)
                    if outcome in ('success','clock_review','restore_error'):
                        self.assertEqual(lower.run(args),{'success':0,'clock_review':2,'restore_error':1}[outcome])
                    elif outcome=='interrupt':
                        with self.assertRaises(KeyboardInterrupt):lower.run(args)
                    else:
                        with self.assertRaises(RuntimeError):lower.run(args)
                if outcome in ('success','clock_review','restore_error'):
                    self.assertEqual([c[-1] for c in commands if '-lgc' in c],
                                     [f'{clock},{clock}' for clock in [780,750,780]])
                    self.assertEqual([c[-1] for c in commands if '-lmc' in c],['6801,6801']*3)
                    self.assertEqual(lower.read(out/'progress.json')['total_stages'],3)
                if outcome=='clock_review':
                    result=lower.read(out/'summary.json')
                    self.assertEqual(result['recorded'],108)
                    self.assertFalse(result['stage_timing_and_sampled_clocks']['s0_780_before'])
                    self.assertTrue(result['stage_timing_and_sampled_clocks']['s2_780_after'])
                if outcome=='later_worker_error':
                    self.assertEqual(lower.read(out/'progress.json')['stage_completed'],1)
                    self.assertTrue((out/'s0_780_before/results.json').exists())
                    self.assertFalse((out/'s2_780_after').exists())
                    self.assertEqual(lower.read(out/'summary.json')['gate'],'FAIL')
                self.assertFalse(any('-pl' in c for c in commands))
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
            root=Path(temp);pkg=root/'clock_recheck_v1';pkg.mkdir();bins=root/'bin';bins.mkdir()
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

    def test_launcher_reuses_terminal_with_open_stdin_and_no_tmux_commands(self):
        with tempfile.TemporaryDirectory(prefix='clock recheck ') as temp:
            root=Path(temp);pkg=root/'clock_recheck_v1';pkg.mkdir();bins=root/'bin';bins.mkdir()
            shutil.copy2(lower.HERE/'launch.sh',pkg/'launch.sh')
            sd=root/'gpu_sweep_v6';sd.mkdir()
            (sd/'run_research_checked.sh').write_text('exit "$MEASURE_EXIT"\n')
            for name,body in {'sudo':'exit 0','python3':'exit 0',
                              'tmux':'echo unexpected-tmux >> "$CALLS"; exit 99'}.items():
                p=bins/name;p.write_text('#!/bin/sh\n'+body+'\n');p.chmod(0o755)
            for inside in (False,True):
                for code in (0,2):
                    calls=root/'calls';calls.write_text('')
                    env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ['PATH'],
                             MEASURE_EXIT=str(code),CALLS=str(calls))
                    env.pop('TMUX',None)
                    if inside:env['TMUX']='/tmp/existing,123,0'
                    with tempfile.TemporaryFile() as log:
                        proc=subprocess.Popen(['bash',str(pkg/'launch.sh')],env=env,
                                              stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
                        try:actual=proc.wait(timeout=5)
                        finally:
                            if proc.poll() is None:proc.kill();proc.wait()
                            proc.stdin.close()
                        log.seek(0);output=log.read().decode()
                    self.assertEqual(actual,code,output)
                    self.assertEqual(calls.read_text(),'')
                    self.assertIn('현재 tmux 세션에서' if inside else 'SSH 연결을 유지',output)
                    self.assertIn('현재 터미널로 돌아갑니다',output)
                    for out in (pkg/'runs').iterdir():
                        self.assertEqual((out/'launcher_source.sh').read_bytes(),(pkg/'launch.sh').read_bytes())


if __name__=='__main__':unittest.main()
