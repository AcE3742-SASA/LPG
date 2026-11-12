"""CPU-only checks of coverage, evidence gates, resume, locks and clock cleanup."""
import argparse
import copy
from collections import Counter, defaultdict
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run
import analyze_sweep


class SweepChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded=run.load()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)/'run'
        self.out.mkdir()
        self.manifest=run.manifest_for(self.out,'check',self.loaded)

    def synthetic_segment(self,tasks,session='s0',clock=780):
        suite=self.loaded['suite']; telemetry=self.loaded['telemetry']
        segment=self.out/'sessions'/session/'segment_00000'
        segment.mkdir(parents=True)
        suite.atomic_json(segment.parent/'environment.json',self.manifest['environment_signature'])
        origin=dt.datetime(2026,9,21,tzinfo=dt.timezone.utc)
        stamp=lambda seconds:(origin+dt.timedelta(seconds=seconds)).isoformat()
        def block(start,elapsed,repeats,factor):
            return dict(timestamp=stamp(start),repeats=repeats,operations_per_call=factor,
                elapsed_ms=elapsed,latency_ms=elapsed/repeats/factor,host_wall_ms=elapsed+.01,
                host_latency_ms=(elapsed+.01)/repeats/factor,
                host_envelope_start_utc=stamp(start),host_envelope_end_utc=stamp(start+.1))
        store=suite.Store(self.out);rows=[]
        try:
            for index,task in enumerate(tasks):
                factor=run.pinned_choices(self.loaded)[self.loaded['plan'].graph_key(task)] if task['mode']=='graph' else 1
                blocks=[block(index*2+i*.2,100,100,factor) for i in range(7)]
                timing=dict(setup=dict(settle_event_ms=3000,settle_blocks=[block(0,3000,3000,factor)],
                                       calibration=[block(0,100,100,factor)]),blocks=blocks,
                    **self.loaded['measure'].summarize(blocks,self.manifest['protocol']))
                if task['mode']=='graph': timing['graph_nodes']=factor
                m,n,_=task['shape'];a,b=self.manifest['protocol']['tolerances'][task['dtype']]
                correct=dict(passed=True,finite_all=True,max_tolerance_ratio=.01,atol=a,rtol=b,
                    scope='full' if m*n<=4096 else 'sampled',points=m*n if m*n<=4096 else min(m,16)*min(n,16))
                row=dict(task=task,session=session,segment=str(segment.relative_to(self.out)),timestamp=stamp(index*2),
                    status='PASS',input_seed=self.loaded['plan'].task_seed(task),timing=timing,
                    correctness=correct,task_wall_seconds=4.)
                store.append(row);rows.append(row)
        finally:store.close()
        suite.atomic_json(segment/'telemetry_metadata.json',dict(fields=list(telemetry.FIELDS),
            requested_interval_ms=50,timezone_offset_seconds=0))
        with (segment/'telemetry.csv').open('w') as f:
            for i in range(40*len(tasks)+1):
                when=(origin+dt.timedelta(seconds=i*.05)).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
                values=[when,self.manifest['environment_signature']['gpu_uuid'],'P0',60,99,clock,6801,60,70,
                        0,'Not Active','Not Active','Not Active',i*100000,0,0]
                f.write(','.join(map(str,values))+'\n')
        return segment,rows

    def finish_check(self):
        tasks=self.manifest['tasks']
        self.synthetic_segment(tasks[:30],'s0');self.synthetic_segment(tasks[30:],'s1')
        p=self.out/'launches'/'first';p.mkdir(parents=True)
        self.loaded['suite'].atomic_json(p/'status.json',dict(collection='COMPLETE',restoration='COMMANDS_ACCEPTED'))
        self.loaded['suite'].atomic_json(p/'restore.json',dict(commands_accepted=True,commands={
            k:dict(exit_code=0,command=['sudo','-n','nvidia-smi','-i',self.manifest['environment_signature']['gpu_uuid'],opt])
            for k,opt in [('graphics','-rgc'),('memory','-rmc')]}))

    def test_full_coverage_modes_rounds_and_no_new_auxiliary(self):
        plan=run.make_plan('full',self.loaded);tasks=plan['tasks']
        self.assertEqual(len(tasks),113382)
        self.assertEqual(Counter(t['phase'] for t in tasks),dict(main=108036,controls=3942,boundary=1350,compute_reference=54))
        self.assertEqual(len({t['id'] for t in tasks}),len(tasks))
        groups=defaultdict(set)
        for t in tasks:
            self.assertEqual(t['kind'],'gemm')
            key=(t['phase'],tuple(t['shape']),t['dtype'],t.get('group'))
            groups[key].add((t['mode'],t['round']))
        expected={(m,r) for m in run.MODES for r in range(3)}
        self.assertTrue(all(v==expected for v in groups.values()))
        legacy=set(self.loaded['plan'].read_shapes('legacy_shapes.csv'))
        old=[t for t in tasks if t['phase']=='main' and tuple(t['shape']) in legacy]
        self.assertEqual(len(old),90000)
        self.assertEqual(len({(tuple(t['shape']),t['dtype']) for t in old}),10000)
        self.assertEqual({t['round'] for t in tasks},{0,1,2})
        fresh={tuple(t['shape']) for t in tasks if t['split']=='fresh_holdout'}
        self.assertEqual(len(fresh),1000)
        self.assertFalse(fresh&legacy)

    def test_analysis_and_reviewed_full_admission(self):
        self.finish_check()
        summary=analyze_sweep.analyze(self.out,self.loaded)
        self.assertEqual(summary['recorded'],60)
        self.assertEqual(summary['timed_blocks'],420)
        self.assertEqual(summary['triplets_complete'],20)
        self.assertTrue(summary['timing_and_sampled_clocks_pass'])
        self.assertEqual(summary['gate'],'REVIEW')
        self.assertGreater(summary['counter_anomaly_blocks'],0)
        self.assertEqual(run.check_admission(self.out,'Reviewed timing/clocks; cumulative counter unresolved',self.loaded)['check_gate'],'REVIEW')
        with self.assertRaises(RuntimeError):run.check_admission(self.out,'',self.loaded)

    def test_clock_mismatch_cannot_enter_full(self):
        self.finish_check()
        p=self.out/'sessions/s1/segment_00000/telemetry.csv'
        p.write_text(p.read_text().replace(',780,6801,',',765,6801,'))
        s=analyze_sweep.analyze(self.out,self.loaded)
        self.assertFalse(s['timing_and_sampled_clocks_pass'])
        with self.assertRaises(RuntimeError):run.check_admission(self.out,'Reviewed',self.loaded)

    def test_incomplete_and_same_process_check_rejected(self):
        self.synthetic_segment(self.manifest['tasks'][:30])
        s=analyze_sweep.analyze(self.out,self.loaded)
        self.assertEqual((s['gate'],s['missing']),('INCOMPLETE',30))
        with self.assertRaises(RuntimeError):run.check_admission(self.out,'Reviewed',self.loaded)

    def test_denominator_or_accuracy_corruption_fails(self):
        directory,rows=self.synthetic_segment(self.manifest['tasks'][:3])
        rows[0]['timing']['blocks'][0]['latency_ms']*=2
        rows[1]['correctness']['passed']=False
        q=analyze_sweep.segment_quality(directory,rows,self.loaded)
        self.assertEqual([q['measurements'][r['task']['id']]['status'] for r in rows],['FAIL','FAIL','PASS'])

    def test_resume_preserves_attempts_and_graph_choices(self):
        _,rows=self.synthetic_segment(self.manifest['tasks'][:3])
        choices,done=run.choices_and_done(self.out,self.manifest,self.loaded)
        self.assertEqual(len(done),3)
        self.assertEqual(len(list(run.history(self.out))),3)
        graph=next(r for r in rows if r['task']['mode']=='graph')
        self.assertEqual(choices[self.loaded['plan'].graph_key(graph['task'])],graph['timing']['graph_nodes'])
        graph['timing']['graph_nodes']+=1
        store=self.loaded['suite'].Store(self.out)
        try:store.append(graph)
        finally:store.close()
        with self.assertRaises(ValueError):run.choices_and_done(self.out,self.manifest,self.loaded)
        self.assertEqual(len(list(run.history(self.out))),4)

    def test_manifest_change_and_output_lock_rejected(self):
        value=run.read(self.out/'manifest.json');value['config']['sm_mhz']=840
        self.loaded['suite'].atomic_json(self.out/'manifest.json',value)
        with self.assertRaises(RuntimeError):run.manifest_for(self.out,'check',self.loaded)
        with self.loaded['suite'].lock(self.out/'.run.lock'):
            with self.assertRaises(RuntimeError):
                with self.loaded['suite'].lock(self.out/'.run.lock'):pass

    def test_worker_process_exit_and_failure_logged(self):
        class Credentials:
            def check(self):pass
        run.run_worker([sys.executable,'-c','print("synthetic child")'],self.out,'ok',10,Credentials())
        self.assertEqual((self.out/'ok.exit_code.txt').read_text().strip(),'0')
        with self.assertRaises(RuntimeError):
            run.run_worker([sys.executable,'-c','raise SystemExit(7)'],self.out,'bad',10,Credentials())
        self.assertEqual((self.out/'bad.exit_code.txt').read_text().strip(),'7')

    def test_failed_credentials_stop_child_before_return(self):
        class Credentials:
            def check(self):raise RuntimeError('synthetic sudo expiry')
        with self.assertRaises(RuntimeError):
            run.run_worker([sys.executable,'-c','import time; time.sleep(60)'],self.out,'expired',10,Credentials())
        self.assertNotEqual((self.out/'expired.exit_code.txt').read_text().strip(),'None')

    def test_controller_restores_both_clocks_on_worker_failure(self):
        loaded=self.loaded;suite=loaded['suite'];calls=[]
        def command(cmd):
            calls.append(cmd)
            stdout=loaded['baseline']['environment_signature']['gpu_uuid'] if '--query-gpu=uuid' in cmd else '70' if '--query-gpu=power.limit' in cmd else ''
            return dict(command=cmd,exit_code=0,stdout=stdout,stderr='')
        args=argparse.Namespace(output=self.out,size='check',check=None,review_note=None,retry_failed=False)
        with patch.object(suite,'command_text',side_effect=command), patch.object(run.shutil,'which',return_value='/test/nvidia-smi'), \
             patch.object(run.shutil,'disk_usage',return_value=argparse.Namespace(free=30*1024**3)), \
             patch.object(run,'run_worker',side_effect=RuntimeError('synthetic worker failed')), \
             patch.object(suite,'lock',side_effect=lambda _:contextlib.nullcontext()):
            with self.assertRaisesRegex(RuntimeError,'synthetic worker failed'):run.run(args,loaded)
        options=[c[-1] for c in calls if c and c[0]=='sudo' and '-i' in c]
        self.assertEqual(options,['6801,6801','780,780','-rgc','-rmc'])
        self.assertFalse(any('-pl' in c for c in calls))
        state=run.read(next((self.out/'launches').glob('*/status.json')))
        self.assertEqual(state['restoration'],'COMMANDS_ACCEPTED')

    def test_preflight_power_mismatch_changes_no_clocks(self):
        suite=self.loaded['suite'];calls=[]
        def command(cmd):
            calls.append(cmd)
            stdout=self.manifest['environment_signature']['gpu_uuid'] if '--query-gpu=uuid' in cmd else '65' if '--query-gpu=power.limit' in cmd else ''
            return dict(exit_code=0,stdout=stdout,stderr='')
        args=argparse.Namespace(output=self.out,size='check',check=None,review_note=None,retry_failed=False)
        with patch.object(suite,'command_text',side_effect=command),patch.object(run.shutil,'which',return_value='/test/nvidia-smi'), \
             patch.object(run.shutil,'disk_usage',return_value=argparse.Namespace(free=30*1024**3)), \
             patch.object(suite,'lock',side_effect=lambda _:contextlib.nullcontext()):
            with self.assertRaises(RuntimeError):run.run(args,self.loaded)
        self.assertFalse(any(c[0]=='sudo' for c in calls))

    def test_worker_resume_collects_only_pending_tasks(self):
        _,fixtures=self.synthetic_segment(self.manifest['tasks'][:6])
        fixture_by_id={r['task']['id']:r for r in fixtures}
        target=Path(self.tmp.name)/'worker';target.mkdir()
        run.manifest_for(target,'check',self.loaded)
        args=argparse.Namespace(output=target,limit=3,retry_failed=False)
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda:(10*1024**3,20*1024**3)),
            inference_mode=contextlib.nullcontext,randn=lambda *a,**k:1.,empty_like=lambda _:0.)
        def measured(torch,task,protocol,graph_nodes=None):
            data=copy.deepcopy(fixture_by_id[task['id']])
            if task['mode']=='graph':self.assertEqual(graph_nodes,data['timing']['graph_nodes'])
            return {k:data[k] for k in ['timing','correctness','input_seed']}
        @contextlib.contextmanager
        def monitor(*args):yield SimpleNamespace(poll=lambda:None)
        ticks=iter(range(0,100000,1000))
        with patch.dict(sys.modules,torch=fake_torch), \
             patch.object(self.loaded['suite'],'environment',return_value=self.manifest['environment_signature']), \
             patch.object(self.loaded['audit'],'check_processes'), \
             patch.object(self.loaded['telemetry'],'collect',side_effect=monitor), \
             patch.object(self.loaded['measure'],'measure',side_effect=measured), \
             patch.object(analyze_sweep,'segment_quality',return_value={'timing_and_sampled_clocks_pass':True}), \
             patch.object(run.time,'monotonic',side_effect=lambda:next(ticks)):
            self.assertEqual(run.worker(args,self.loaded),0)
            self.assertEqual(run.worker(args,self.loaded),0)
        rows=list(run.history(target))
        self.assertEqual([r['task']['id'] for _,r in rows],[t['id'] for t in self.manifest['tasks'][:6]])
        self.assertEqual(len({r['session'] for _,r in rows}),2)

    def test_failed_measurement_without_seed_exports_as_fail(self):
        _,rows=self.synthetic_segment(self.manifest['tasks'][:3])
        bad=dict(rows[0]);bad.pop('timing');bad.pop('input_seed');bad['status']='FAIL';bad['error']='synthetic CUDA error'
        store=self.loaded['suite'].Store(self.out)
        try:store.append(bad)
        finally:store.close()
        summary=analyze_sweep.analyze(self.out,self.loaded)
        self.assertEqual(summary['gate'],'FAIL')
        self.assertEqual(summary['counts']['FAIL'],1)


if __name__=='__main__':
    unittest.main()
