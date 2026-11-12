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

import steps
import analyze_steps

LOADED = steps.load()
hold, audit, original, telemetry, baseline, suite, config = LOADED


def q_result(counter=0, code=0):
    return dict(exit_code=code, stdout=f'SW Power Capping : {counter} us\nSW Thermal Slowdown : 0 us\nHW Thermal Slowdown : 0 us\n', stderr='')


def fixture_phase(directory, sm=900, interval=50, counter_growth=False):
    directory.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 9, 20, 0, 0, tzinfo=timezone.utc)
    csv = []
    for index in range(int(11_000/interval)+1):
        when = start + timedelta(milliseconds=index*interval)
        fields = [when.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3], baseline['environment_signature']['gpu_uuid'],
                  'P2', '65', '100', str(sm), '6801', '60', '70', '0',
                  'Not Active', 'Not Active', 'Not Active', str(index*interval*1000 if counter_growth else 0), '0', '0']
        csv.append(','.join(fields))
    (directory/'telemetry.csv').write_text('\n'.join(csv)+'\n')
    hold.save(directory/'telemetry_metadata.json', dict(fields=telemetry.FIELDS, requested_interval_ms=interval,
              timezone_offset_seconds=0, process_exit_code=0))
    hold.save(directory/'gpu_before.json', q_result())
    hold.save(directory/'gpu_after.json', q_result(11_000_000 if counter_growth else 0))


def fixture_stage(out, stage, counter_growth=False):
    directory = out/stage['name']
    fixture_phase(directory, stage['sm_mhz'], stage['interval_ms'], counter_growth)
    hold.save(directory/'environment.json', dict(baseline['environment_signature'], nvidia_smi=q_result()))
    start = datetime(2026, 9, 20, 0, 0, tzinfo=timezone.utc)
    def block(ms, ops, index=0):
        at = start+timedelta(seconds=3.1+index*.11)
        return dict(repeats=1, operations_per_call=ops, elapsed_ms=ms, latency_ms=ms/ops,
                    host_envelope_start_utc=at.isoformat(), host_envelope_end_utc=(at+timedelta(milliseconds=ms+5)).isoformat())
    rows = []
    for task in steps.cases(config):
        ops = steps.graph_nodes(config, task) or 1
        timing = dict(blocks=[block(100, ops, i) for i in range(7)], median_ms=100/ops,
            relative_iqr=0, first_last_drift=0, flags=[],
            setup=dict(settle_event_ms=3000, settle_blocks=[block(3000, ops)], calibration=[block(100, ops)]))
        if task['mode'] == 'graph':
            timing['graph_nodes'] = ops
        spec = next(s for s in config['fixed_cases'] if s['shape']==task['shape'] and s['dtype']==task['dtype'])
        result = dict(timing=timing, input_seed=spec['input_seed'],
                      correctness=dict(passed=True, finite_all=True, max_tolerance_ratio=.01, scope='sampled', points=256))
        rows.append(dict(task=task, result=result))
    hold.save(directory/'results.json', rows)
    (out/(stage['name']+'.exit_code.txt')).write_text('0\n')


def fixture_idle(out, name, sm, *unused):
    directory = out/'idle'/name
    fixture_phase(directory, sm or 1200, config['idle_interval_ms'])
    hold.save(directory/'phase.json', dict(name=name, sm_mhz=sm, seconds=config['idle_seconds'], memory_mhz=6801))


class StepsTests(unittest.TestCase):
    def test_plan_preserves_inputs_batches_and_adapter_restores(self):
        self.assertEqual(len(steps.cases(config)), 16)
        self.assertEqual(len({t['id'] for t in steps.cases(config)}), 16)
        self.assertEqual([s['sm_mhz'] for s in config['stages']], [900,900,1050,1200,1350,900,900])
        old = audit.cases, audit.graph_nodes, audit.PASSES
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with steps.configured_audit(audit, config):
                self.assertEqual(audit.cases(), steps.cases(config))
                self.assertEqual([audit.graph_nodes(t) for t in audit.cases()][:8], [None,29,None,56,None,1,None,1])
                raise RuntimeError('stop')
        self.assertEqual((audit.cases, audit.graph_nodes, audit.PASSES), old)

    def test_complete_synthetic_summary_and_polling_orientation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            steps.make_manifest(out, LOADED)
            for stage in config['stages']:
                fixture_stage(out, stage)
                for side in ('before','after'):
                    fixture_idle(out, stage['name']+'_'+side, stage['sm_mhz'])
            fixture_idle(out, 'auto_before', None)
            fixture_idle(out, 'auto_after', None)
            hold.save(out/'restore.json', dict(commands_accepted=True))
            result = analyze_steps.analyze(out, LOADED)
            self.assertEqual((result['gate'], result['recorded'], result['timed_blocks'], result['idle_phases']), ('PASS',112,784,16))
            self.assertEqual(result['full_sweep'], 'HOLD')
            self.assertIsNone(result['selected_clock_mhz'])
            comparisons = hold.read(out/'analysis.json')['comparisons']
            self.assertEqual(len(comparisons),32)
            for row in comparisons:
                if row['comparison'].startswith('polling'):
                    self.assertIn('_100_', row['first_stage'])
                    self.assertIn('_50_', row['second_stage'])

    def test_inactive_counter_growth_remains_review_without_rejecting_clock_match(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp);stage = config['stages'][1]
            fixture_stage(out, stage, counter_growth=True)
            result, rows, _ = analyze_steps.analyze_stage(out, stage, LOADED)
            self.assertEqual(result['gate'], 'REVIEW')
            self.assertTrue(result['timing_and_sampled_clocks_pass'])
            self.assertTrue(any('increase_without_active_sample' in f for f in result['flags']))
            self.assertTrue(all(row['status']=='PASS' for row in rows))
            self.assertEqual(result['counters']['q_counter_deltas_us']['SW Power Capping'],11_000_000)
            samples,_=telemetry.read(out/stage['name']/'telemetry.csv',0,50)
            samples[-1][telemetry.COUNTERS[0]]=0
            counters=analyze_steps.counter_summary(samples,telemetry)
            self.assertIsNone(counters['counters'][telemetry.COUNTERS[0]]['delta_us'])
            self.assertTrue(any('decreased' in f for f in counters['flags']))

    def test_clock_memory_cap_and_missing_samples_do_not_pass(self):
        for field, value in [(5,'885'), (6,'7001'), (8,'60'), (10,'Active')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                out = Path(temp);stage = config['stages'][1];fixture_stage(out,stage)
                path=out/stage['name']/'telemetry.csv'
                changed=[]
                for line in path.read_text().splitlines():
                    values=line.split(',');values[field]=value;changed.append(','.join(values))
                path.write_text('\n'.join(changed)+'\n')
                result, _, _=analyze_steps.analyze_stage(out,stage,LOADED)
                self.assertFalse(result['timing_and_sampled_clocks_pass'])
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);stage=config['stages'][1];fixture_stage(out,stage)
            path=out/stage['name']/'telemetry.csv';path.write_text('\n'.join(path.read_text().splitlines()[:2])+'\n')
            result,_,_=analyze_steps.analyze_stage(out,stage,LOADED)
            self.assertTrue(any('no_inside_clock_sample' in f for f in result['flags']))

    def test_corrupt_denominator_seed_graph_correctness_and_missing_rows_rejected(self):
        for variant in ('denominator','seed','graph','correctness','missing'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                out=Path(temp);stage=config['stages'][0];fixture_stage(out,stage)
                path=out/stage['name']/'results.json';rows=hold.read(path)
                if variant=='denominator':rows[0]['result']['timing']['blocks'][0]['repeats']=2
                elif variant=='seed':rows[0]['result']['input_seed']=1
                elif variant=='graph':rows[5]['result']['timing']['graph_nodes']=2
                elif variant=='correctness':rows[0]['result']['correctness']['passed']=False
                else:rows.pop()
                hold.save(path,rows)
                with self.assertRaises(ValueError):analyze_steps.analyze_stage(out,stage,LOADED)

    def test_round_variation_is_review_even_when_each_round_stable(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);stage=config['stages'][0];fixture_stage(out,stage)
            path=out/stage['name']/'results.json';rows=hold.read(path)
            timing=rows[-1]['result']['timing']
            for block in timing['blocks']:
                block['repeats']*=2;block['latency_ms']/=2
            timing['median_ms']/=2
            hold.save(path,rows)
            result,_,_=analyze_steps.analyze_stage(out,stage,LOADED)
            self.assertTrue(any('round_range_above_5pct' in f for f in result['flags']))

    def test_run_restores_under_lock_after_setup_worker_interrupt_and_success(self):
        for outcome in ('success','setup_error','worker_error','interrupt','restore_error','wrong_power'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temp:
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
                        return dict(exit_code=0,stdout=baseline['environment_signature']['gpu_uuid']+'\n',stderr='')
                    if '--query-gpu=power.limit' in cmd:
                        return dict(exit_code=0,stdout='30' if outcome=='wrong_power' else '70',stderr='')
                    fail=(outcome=='setup_error' and '-lmc' in cmd) or (outcome=='restore_error' and '-rmc' in cmd)
                    return q_result(code=1 if fail else 0)
                def worker(cmd, directory, name, timeout):
                    if outcome=='worker_error':raise RuntimeError('worker failed')
                    if outcome=='interrupt':raise KeyboardInterrupt
                    fixture_stage(out,next(s for s in config['stages'] if s['name']==name))
                credentials=SimpleNamespace(failed=False,check=lambda:None)
                with patch.object(suite,'lock',lock),patch.object(suite,'command_text',command),\
                     patch.object(audit,'check_processes'),patch.object(audit,'run_logged',worker),\
                     patch.object(steps,'idle_probe',fixture_idle),patch.object(steps.shutil,'which',return_value='/nvidia-smi'),\
                     patch.object(steps,'Credentials',return_value=contextlib.nullcontext(credentials)):
                    args=SimpleNamespace(output=out)
                    if outcome in ('success','restore_error'):
                        self.assertEqual(steps.run(args),0 if outcome=='success' else 1)
                    elif outcome=='interrupt':
                        with self.assertRaises(KeyboardInterrupt):steps.run(args)
                    else:
                        with self.assertRaises(RuntimeError):steps.run(args)
                status=hold.read(out/'status.json')
                if outcome=='wrong_power':
                    self.assertEqual(status['restoration'],'NOT_ATTEMPTED')
                    self.assertFalse(any(c[0]=='sudo' for c in commands))
                else:
                    self.assertTrue((out/'reset_graphics.json').exists())
                    self.assertTrue((out/'reset_memory.json').exists())
                    self.assertEqual(status['restoration'],'FAILED' if outcome=='restore_error' else 'COMMANDS_ACCEPTED')

    def test_credentials_noninteractive_failure_and_thread_shutdown(self):
        with tempfile.TemporaryDirectory() as temp:
            fake=SimpleNamespace(command_text=lambda cmd:dict(exit_code=0,stdout='',stderr='',command=cmd),
                                 now=lambda:'test',atomic_json=hold.save)
            guard=steps.Credentials(fake,Path(temp),interval=.005)
            with guard:
                time.sleep(.025)
            self.assertFalse(guard.thread.is_alive())
            self.assertGreater(len(guard.records),1)
            self.assertTrue(all(r['command']==['sudo','-n','-v'] for r in guard.records))
            fake.command_text=lambda cmd:dict(exit_code=1,stdout='',stderr='expired')
            with self.assertRaisesRegex(RuntimeError,'refresh failed'):
                with steps.Credentials(fake,Path(temp)):
                    self.fail('Entered despite missing authorization')

    def test_manifest_tamper_and_changed_dependency_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);steps.make_manifest(out,LOADED)
            (out/'source/protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'snapshot differs'):
                analyze_steps.analyze(out,LOADED)
        original_read=steps.read
        def bad_hash(path):
            data=original_read(path)
            if path.name=='dependency_hashes.json':data[next(iter(data))]='0'*64
            return data
        with patch.object(steps,'read',bad_hash),self.assertRaisesRegex(RuntimeError,'Dependency changed'):
            steps.load()

    def test_launch_preserves_review_exit_and_packages_closed_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=root/'clock_steps_v1';package.mkdir();bins=root/'bin';bins.mkdir()
            for name in ('steps.py','launch.sh'):
                shutil.copy2(steps.HERE/name,package/name)
            suite_dir=root/'gpu_sweep_v6';suite_dir.mkdir()
            (suite_dir/'run_research_checked.sh').write_text('#!/bin/bash\n"$@"\n')
            stubs={
                'sudo':'#!/bin/sh\nexit 0\n',
                'python3':f'#!{sys.executable}\nimport sys,os\nif len(sys.argv)>3 and sys.argv[3]=="verify": sys.exit(0)\nos.execv({sys.executable!r},[{sys.executable!r}]+sys.argv[1:])\n',
                'python':f'#!{sys.executable}\nimport sys\nprint("RESTORE CLOSED")\nsys.exit(2)\n'}
            for name,content in stubs.items():
                path=bins/name;path.write_text(content);path.chmod(0o755)
            result=subprocess.run(['bash',str(package/'launch.sh')],env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ['PATH']),
                                  capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            archive=next(package.glob('runs/*.zip'))
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                logs=[n for n in z.namelist() if n.endswith('/launcher.log')]
                self.assertEqual(len(logs),1)
                self.assertIn('RESTORE CLOSED',z.read(logs[0]).decode())
                self.assertEqual(z.read(next(n for n in z.namelist() if n.endswith('/exit_code.txt'))).strip(),b'2')
            # Real process-group SIGINT must not package before the child's cleanup ends.
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
            proc=subprocess.Popen(['bash',str(package/'launch.sh')],
                env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ['PATH']),start_new_session=True,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                deadline=time.monotonic()+5
                while not list(package.glob('runs/*/started.marker')) and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertTrue(list(package.glob('runs/*/started.marker')))
                os.killpg(proc.pid,signal.SIGINT)
                proc.communicate(timeout=5)
                interrupted=[p for p in package.glob('runs/*.zip') if p!=archive]
                self.assertEqual(len(interrupted),1)
                with zipfile.ZipFile(interrupted[0]) as z:
                    self.assertTrue(any(n.endswith('/restored.marker') for n in z.namelist()))
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.communicate()


if __name__ == '__main__':
    unittest.main()
