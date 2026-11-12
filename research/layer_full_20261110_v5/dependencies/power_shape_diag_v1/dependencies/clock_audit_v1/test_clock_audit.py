import copy
import datetime as dt
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import audit
import telemetry
import analyze


class ClockAuditTests(unittest.TestCase):
    def sample(self, when="2026/09/20 11:00:00.100"):
        return [when, "GPU-test", "P2", "72", "100", "1200", "6801", "69.5", "70",
                "0x0000000000000004", "Active", "Not Active", "Not Active", "123000", "0", "0"]

    def test_plan_fixed_batches_balanced_pairs_and_passes(self):
        self.assertEqual(list(audit.PASSES.values()), [1000, 100, 100, 100, 1000])
        tasks = audit.cases()
        self.assertEqual(len(tasks), 8)
        self.assertEqual(len({t['id'] for t in tasks}), 8)
        self.assertEqual([t['mode'] for t in tasks], ['prealloc','graph','prealloc','graph','graph','prealloc','graph','prealloc'])
        self.assertEqual([audit.graph_nodes(t) for t in tasks], [None,29,None,56,56,None,29,None])
        _, plan, _, _ = audit.load_suite((audit.HERE.parent/'gpu_sweep_v6').resolve())
        for a, b in zip(tasks[::2], tasks[1::2]):
            self.assertEqual(plan.task_seed(a), plan.task_seed(b))
        baseline=json.loads((audit.HERE/'baseline_v6.json').read_text())
        self.assertEqual(plan.PROTOCOL, baseline['protocol'])

    def test_telemetry_fields_timestamp_mask_and_counters(self):
        row = telemetry.parse_row(self.sample(), 32400)
        self.assertEqual(row['utc'], '2026-09-20T02:00:00.100000+00:00')
        self.assertEqual(row['clocks_event_reasons.active'], 4)
        self.assertEqual(row['clocks_event_reasons_counters.sw_power_cap'], 123000)
        self.assertEqual(row['clocks_event_reasons.sw_power_cap'], True)
        bad=self.sample();bad[5]='N/A'
        with self.assertRaises(ValueError):telemetry.parse_row(bad,32400)
        partial=self.sample();partial[13]='[N/A]'
        self.assertIsNone(telemetry.parse_row(partial,32400)['clocks_event_reasons_counters.sw_power_cap'])
        for bad in [self.sample()[:-1], ['Xid error'], ['timestamp']*16]:
            with self.assertRaises(ValueError):telemetry.parse_row(bad,32400)

    def test_window_keeps_external_samples_separate_and_counters_bounded(self):
        rows=[telemetry.parse_row(self.sample('2026/09/20 11:00:00.'+s),32400) for s in ('000','100','200','300')]
        key='clocks_event_reasons_counters.sw_power_cap'
        for i,r in enumerate(rows):r[key]=i*10000
        start='2026-09-20T02:00:00.050000+00:00';end='2026-09-20T02:00:00.250000+00:00'
        result=telemetry.window(rows,start,end)
        self.assertEqual(result['samples_inside'],2)
        self.assertEqual(result['before']['utc'],rows[0]['utc'])
        self.assertEqual(result['after']['utc'],rows[-1]['utc'])
        self.assertEqual(result['counter_deltas_us'][key],10000)
        self.assertEqual(result['counter_interval_seconds'],.1)
        empty=telemetry.window(rows,'2026-09-20T02:00:00.210000+00:00','2026-09-20T02:00:00.250000+00:00')
        self.assertEqual(empty['samples_inside'],0)
        self.assertIsNone(empty['sm_mhz_median'])
        self.assertIsNone(empty['counter_deltas_us'][key])
        rows[2][key]=0
        self.assertIsNone(telemetry.window(rows,start,end)['counter_deltas_us'][key])

    def test_sampler_metadata_and_cadence_do_not_claim_sensor_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'telemetry.csv'
            p.write_text('\n'.join(', '.join(self.sample('2026/09/20 11:00:00.'+s)) for s in ('000','100','200'))+'\n')
            rows,info=telemetry.read(p,32400,100)
            self.assertEqual(len(rows),3)
            self.assertAlmostEqual(info['observed_interval_seconds_median'],.1)
            self.assertFalse(info['sensor_resolution_verified'])
            p.write_text(p.read_text()+', '.join(self.sample('2026/09/20 10:59:59.900'))+'\n')
            with self.assertRaisesRegex(ValueError,'monotonic'):telemetry.read(p,32400,100)

    def test_nsys_command_captures_eight_ranges_after_start_api(self):
        command=audit.worker_command(Path('/suite'),Path('/output'),'nsys')
        self.assertIn('--capture-range-end=repeat:8',command)
        self.assertIn('--cuda-graph-trace=node',command)
        self.assertIn('--capture-range=cudaProfilerApi',command)
        self.assertEqual(command[-1],'nsys')

    def test_clock_envelopes_and_profiler_preserve_original_call_order(self):
        from types import SimpleNamespace
        events=[]
        class Timer:
            def __init__(self):self.p={'blocks':2}
            def block(self,fn,repeats,operations=1):
                for _ in range(repeats):fn()
                return dict(elapsed_ms=10,operations_per_call=operations)
            def prepare(self,fn,operations=1):
                self.block(fn,3,operations)
                return 2, {'original':True}
        torch=SimpleNamespace(cuda=SimpleNamespace(
            cudart=lambda:SimpleNamespace(cudaProfilerStart=lambda:events.append('start'),cudaProfilerStop=lambda:events.append('stop')),
            nvtx=SimpleNamespace(range_push=lambda x:events.append(x),range_pop=lambda:events.append('pop'))))
        timer=audit.traced_timer(audit.observed_timer(Timer),torch,'c')()
        repeats,setup=timer.prepare(lambda:events.append('gemm'))
        self.assertEqual(events[0:2],['start','prepare/c'])
        self.assertTrue(setup['capture_started_before_prepare'])
        results=[timer.block(lambda:events.append('gemm'),repeats) for _ in range(2)]
        self.assertEqual(events.count('gemm'),7)
        self.assertEqual(events[-1],'stop')
        self.assertEqual(results[-1]['trace_label'],'audit/c/block/1')
        for r in results:
            self.assertLessEqual(r['host_envelope_start_utc'],r['host_envelope_end_utc'])

    def test_worker_passes_fixed_batches_and_restores_timer_on_failure(self):
        import contextlib
        from types import SimpleNamespace
        class Tensor:
            def __mul__(self,x):return self
        torch=SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda:(5*1024**3,10*1024**3)),
                              inference_mode=contextlib.nullcontext,randn=lambda *a,**k:Tensor(),empty_like=lambda x:Tensor())
        class Timer:
            def __init__(self,*a):pass
        env={'gpu_uuid':'GPU-test'}
        suite=SimpleNamespace(environment=lambda t:env,signature=lambda e:e,
                              atomic_json=lambda p,v:p.write_text(json.dumps(v)),now=lambda:'time',
                              command_text=lambda c:{'exit_code':0,'stdout':''})
        for fail in (False,True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp:
                calls=[];closed=[]
                def measure_fn(torch,task,protocol,graph_nodes):
                    calls.append(graph_nodes)
                    return {'correctness':{'passed':not fail}}
                measure=SimpleNamespace(Timer=Timer,measure=measure_fn)
                @contextlib.contextmanager
                def monitor(*a):
                    try:yield SimpleNamespace(poll=lambda:None)
                    finally:closed.append(True)
                args=SimpleNamespace(suite=Path('/unused'),output=Path(temp),pass_name='dense_before')
                loaded=({'environment_signature':env},SimpleNamespace(PROTOCOL={'preheat_seconds':0}),measure,suite)
                with patch.object(audit,'load_suite',return_value=loaded),patch.object(audit,'check_processes'),patch.object(telemetry,'collect',monitor),patch.dict(sys.modules,torch=torch):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'Correctness'):audit.worker(args)
                    else:
                        audit.worker(args)
                        self.assertEqual(calls,[None,29,None,56,56,None,29,None])
                self.assertIs(measure.Timer,Timer)
                self.assertEqual(closed,[True])

    def fixture(self,out):
        baseline=json.loads((audit.HERE/'baseline_v6.json').read_text())
        sources={p.name:audit.sha256_file(p) for p in audit.HERE.iterdir() if p.suffix in ('.py','.sh','.json','.md')}
        manifest=dict(cases=audit.cases(),passes=audit.PASSES,baseline=baseline,source_hashes=sources)
        (out/'manifest.json').write_text(json.dumps(manifest))
        for source,names,dest in [(audit.HERE,sources,out/'source'),(audit.HERE.parent/'gpu_sweep_v6',baseline['code_hashes'],out/'baseline_source')]:
            for name in names:
                p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,p)
        env=baseline['environment_signature']
        start=dt.datetime(2026,9,20,2,0,tzinfo=dt.timezone.utc)
        def block(elapsed,ops,index=0):
            a=start+dt.timedelta(seconds=3.1+index*.11)
            return dict(repeats=1,operations_per_call=ops,elapsed_ms=elapsed,latency_ms=elapsed/ops,
                        host_envelope_start_utc=a.isoformat(),host_envelope_end_utc=(a+dt.timedelta(milliseconds=elapsed+5)).isoformat())
        for name,interval in audit.PASSES.items():
            directory=out/name;directory.mkdir()
            (directory/'environment.json').write_text(json.dumps(env))
            (directory/'telemetry_metadata.json').write_text(json.dumps(dict(fields=telemetry.FIELDS,requested_interval_ms=interval,timezone_offset_seconds=32400)))
            csv=[]
            for i in range(71):
                values=self.sample((start+dt.timedelta(hours=9,seconds=i/10)).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]);values[1]=env['gpu_uuid']
                csv.append(','.join(values))
            (directory/'telemetry.csv').write_text('\n'.join(csv)+'\n')
            rows=[]
            for task in audit.cases():
                ops=audit.graph_nodes(task) or 1
                blocks=[block(100,ops,i) for i in range(7)]
                if name=='nsys':
                    for i,b in enumerate(blocks):b['trace_label']=f'audit/{task["id"]}/block/{i}'
                tm=dict(blocks=blocks,median_ms=100/ops,relative_iqr=0,first_last_drift=0,flags=[],
                        setup=dict(settle_event_ms=3000,settle_blocks=[block(3000,ops)],calibration=[block(100,ops)],capture_started_before_prepare=True))
                if task['mode']=='graph':tm['graph_nodes']=ops
                spec=next(x for x in baseline['fixed_cases'] if x['shape']==task['shape'])
                rows.append(dict(task=task,result=dict(input_seed=spec['input_seed'],timing=tm,
                                correctness=dict(passed=True,finite_all=True,max_tolerance_ratio=.01))))
            (directory/'results.json').write_text(json.dumps(rows))
        (out/'traces').mkdir()
        for task in audit.cases():
            con=sqlite3.connect(out/'traces'/(task['id']+'.sqlite'))
            con.executescript('''
              CREATE TABLE StringIds(id INTEGER,value TEXT);
              CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
              CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
              CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER,demangledName INTEGER);
              CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT);
            ''')
            pid=123<<24;tid=pid+7;ops=audit.graph_nodes(task) or 1
            con.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'gemm'),(2,'cudaLaunchKernel')])
            con.execute('INSERT INTO NVTX_EVENTS VALUES (0,3000000000,?,NULL,?)',('prepare/'+task['id'],tid))
            con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000,2000,?,0,0,1,7,1)',(pid,))
            for i in range(7):
                t=3100000000+i*110000000
                con.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,NULL,?)',(t,t+105000000,f'audit/{task["id"]}/block/{i}',tid))
                con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,2)',(t+1000000,t+2000000,tid,i+1))
                for k in range(ops):
                    a=t+3000000+k*1000000
                    con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,0,1,7,1)',(a,a+500000,pid,i+1))
            con.commit();con.close()
        return manifest

    def test_full_analysis_retains_all_blocks_and_rejects_missing_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out)
            result=analyze.analyze(out)
            self.assertEqual((result['recorded'],result['traced_blocks'],result['gate']),(40,56,'PASS'))
            data=json.loads((out/'analysis.json').read_text())
            self.assertEqual(len(data['conditions']),4)
            self.assertTrue(all(r['dense_over_sparse']==r['nsys_over_dense']==1 for r in data['conditions']))
            observations=json.loads((out/'clock_windows.json').read_text())
            self.assertEqual(sum(r['phase']=='timed' for r in observations),280)
            path=next((out/'traces').glob('*.sqlite'));path.rename(path.with_suffix('.bak'))
            with self.assertRaisesRegex(ValueError,'Missing NVTX'):analyze.analyze(out)

    def test_analysis_rejects_changed_batch_and_preserves_profiler_warning(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out)
            path=next((out/'traces').glob('*.sqlite'))
            con=sqlite3.connect(path);con.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (2,'Incomplete CUDA event capture')");con.commit();con.close()
            self.assertEqual(analyze.analyze(out)['gate'],'REVIEW')
            self.assertIn('Incomplete CUDA event capture',(out/'analysis.json').read_text())
            p=out/'dense_after/results.json';rows=json.loads(p.read_text());rows[1]['result']['timing']['graph_nodes']=30;p.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'Graph batch'):analyze.analyze(out)

    def test_launcher_packages_wrapper_failure_with_closed_log(self):
        with tempfile.TemporaryDirectory(prefix='clock audit ') as temp:
            root=Path(temp);package=root/'package';suite=root/'suite'
            package.mkdir();suite.mkdir()
            for p in audit.HERE.iterdir():
                if p.suffix in ('.py','.sh','.json'):shutil.copy2(p,package/p.name)
            wrapper=suite/'run_research_checked.sh';wrapper.write_text('#!/bin/bash\necho wrapper-error\nexit 9\n')
            (package/'baseline_v6.json').write_text(json.dumps({'code_hashes':{'run_research_checked.sh':audit.sha256_file(wrapper)}}))
            r=subprocess.run(['bash',str(package/'launch.sh'),str(suite)],capture_output=True,text=True,timeout=10)
            self.assertEqual(r.returncode,9,r.stdout+r.stderr)
            import zipfile
            archive=next((package/'runs').glob('*.zip'))
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read(archive.stem+'/exit_code.txt'),b'9\n')
                self.assertIn(b'wrapper-error',z.read(archive.stem+'/launcher.log'))


if __name__=='__main__':unittest.main()
