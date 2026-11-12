"""Regression checks for separate paired targets and repeatable Graph batching."""
from collections import defaultdict
import contextlib
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import measure
import plan
import report
import suite


class PairedPlanTests(unittest.TestCase):
    def test_all_main_and_boundary_conditions_have_adjacent_modes(self):
        for size, count in [('pilot', 192), ('full', 78952)]:
            manifest = plan.make_plan(size)
            self.assertEqual(len(manifest['tasks']), count)
            groups = defaultdict(list)
            for i, task in enumerate(manifest['tasks']):
                if task.get('phase') in ('main', 'boundary'):
                    key = tuple((k, str(v)) for k,v in task.items() if k not in ('id','mode'))
                    groups[key].append((i,task))
            for pair in groups.values():
                self.assertEqual([t['mode'] for _,t in pair], ['prealloc','graph'] if pair[0][1]['round']%2 == 0 else ['graph','prealloc'])
                self.assertEqual(pair[1][0], pair[0][0]+1)
                self.assertEqual(plan.task_seed(pair[0][1]),plan.task_seed(pair[1][1]))
            if size == 'pilot':
                self.assertTrue(all(t.get('phase') == 'main' for t in manifest['tasks']))

    def test_batch_selection_and_restoration_across_rounds_and_phases(self):
        task = dict(id='g',kind='gemm',shape=[1024,512,256],dtype='fp16',mode='graph',round=0,phase='main')
        row = dict(task=task,timing={'graph_nodes':32},status='REVIEW')
        choices = suite.graph_node_choices([row,dict(row,task=dict(task,id='g2',round=1,phase='controls'))])
        self.assertEqual(choices[plan.graph_key(task)],32)
        with self.assertRaises(ValueError):
            suite.graph_node_choices([row,dict(row,timing={'graph_nodes':64})])
        for n in (0,65,1.5,True):
            with self.assertRaises(ValueError):
                suite.graph_node_choices([dict(row,timing={'graph_nodes':n})])
        self.assertEqual(measure.select_graph_nodes(.002,plan.PROTOCOL),64)
        self.assertEqual(measure.select_graph_nodes(1.,plan.PROTOCOL),2)
        self.assertEqual(measure.select_graph_nodes(.002,plan.PROTOCOL,fixed=32),32)
        for case in plan.PROTOCOL['verified_graph64_cases']:
            small = dict(task,shape=case['shape'],dtype=case['dtype'])
            self.assertEqual(suite.graph_nodes_for(small,{}),64)

    def test_fixed_capture_does_not_reprobe_or_change_count(self):
        captured=[]
        active=[False]
        @contextlib.contextmanager
        def context(graph):
            active[0]=True
            yield
            active[0]=False
        stream=SimpleNamespace(wait_stream=lambda _:None)
        graph=SimpleNamespace(replay=lambda:None)
        torch=SimpleNamespace(cuda=SimpleNamespace(Stream=lambda:stream,current_stream=lambda:stream,
            stream=lambda _:contextlib.nullcontext(),synchronize=lambda:None,CUDAGraph=lambda:graph,graph=context))
        timer=SimpleNamespace(p=plan.PROTOCOL,block=lambda *args: self.fail('fixed batch must not reprobe'))
        got,n=measure.capture(torch,lambda:captured.append(active[0]),timer,nodes=64)
        self.assertIs(got,graph)
        self.assertEqual(n,64)
        self.assertEqual(captured.count(True),64)


class PairedExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.out=Path(self.temp.name)
        self.tasks=[dict(id=m,kind='gemm',phase='main',shape=[64,64,64],dtype='fp16',mode=m,round=0,split='development') for m in ('prealloc','graph')]
        suite.atomic_json(self.out/'manifest.json',dict(size='pilot',protocol=plan.PROTOCOL,tasks=self.tasks,shape_splits=[]))

    def tearDown(self):
        self.temp.cleanup()

    def row(self,task,latency):
        timing=dict(median_ms=latency,relative_iqr=0,first_last_drift=0,flags=[],blocks=[dict(latency_ms=latency)]*7)
        if task['mode']=='graph':timing['graph_nodes']=64
        return dict(task=task,status='PASS',session='s',timestamp='t',input_seed=plan.task_seed(task),
                    timing=timing,correctness={'passed':True},tflops=1.)

    def test_two_targets_exported_without_averaging_and_partial_pair_visible(self):
        store=suite.Store(self.out)
        store.append(self.row(self.tasks[0],.004))
        result=report.export(self.out)
        self.assertEqual(result['paired_complete'],0)
        store.append(self.row(self.tasks[1],.002))
        store.close()
        result=report.export(self.out)
        self.assertEqual((result['paired_expected'],result['paired_complete']),(1,1))
        with (self.out/'paired_metrics.csv').open() as f:r=list(csv.DictReader(f))[0]
        self.assertEqual(float(r['prealloc_ms']),.004)
        self.assertEqual(float(r['graph_ms']),.002)
        self.assertEqual(float(r['graph_over_prealloc']),.5)
        self.assertEqual(int(r['graph_gemms_per_replay']),64)
        self.assertEqual(int(r['input_seed']),plan.task_seed(self.tasks[0]))

    def test_seed_mismatch_is_not_silently_paired(self):
        store=suite.Store(self.out)
        store.append(self.row(self.tasks[0],.004))
        bad=self.row(self.tasks[1],.002);bad['input_seed']+=1
        store.append(bad);store.close()
        with self.assertRaises(ValueError):report.export(self.out)

    def test_holdout_paired_latency_stays_in_locked_export(self):
        for t in self.tasks:t['split']='fresh_holdout'
        suite.atomic_json(self.out/'manifest.json',dict(size='full',protocol=plan.PROTOCOL,tasks=self.tasks,shape_splits=[]))
        store=suite.Store(self.out)
        for t in self.tasks:store.append(self.row(t,.123456))
        store.close();report.export(self.out)
        self.assertEqual((self.out/'paired_metrics.csv').read_text(),'')
        self.assertIn('0.123456',(self.out/'paired_fresh_holdout_LOCKED.csv').read_text())
        self.assertNotIn('0.123456',(self.out/'report.html').read_text())

    def test_pilot_gate_rejects_a_different_task_plan(self):
        env={k:'test' for k in ('gpu_uuid','device_driver','torch','cuda_runtime','precision','environment')}
        suite.atomic_json(self.out/'environment.json',env)
        m=json.loads((self.out/'manifest.json').read_text());m['code_hashes']=suite.code_fingerprint()
        suite.atomic_json(self.out/'manifest.json',m)
        store=suite.Store(self.out)
        for t in self.tasks:store.append(self.row(t,.002))
        store.close()
        with self.assertRaisesRegex(RuntimeError,'plan'):
            suite.verify_pilot(self.out,env,m['code_hashes'],'checked')

    def test_interrupted_acquisition_restores_graph_choice_and_skips_saved_rows(self):
        tasks=[]
        for rd in (0,1):
            for mode in (('prealloc','graph') if rd==0 else ('graph','prealloc')):
                tasks.append(dict(id=f'{rd}_{mode}',kind='gemm',phase='main',shape=[1024,512,256],dtype='fp16',
                                  mode=mode,round=rd,split='development'))
        manifest=dict(size='pilot',protocol=plan.PROTOCOL,tasks=tasks,shape_splits=[])
        suite.atomic_json(self.out/'manifest.json',manifest)
        class Tensor:
            def __mul__(self, other):return self
        torch=SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda:(8*1024**3,20*1024**3)),
             inference_mode=contextlib.nullcontext,randn=lambda *a,**k:Tensor(),empty_like=lambda x:Tensor())
        class Telemetry:
            def poll(self):return None
            def terminate(self):pass
            def wait(self,timeout=None):return 0
        received=[]
        def fake_measure(torch,task,protocol,graph_nodes=None):
            received.append((task['round'],task['mode'],graph_nodes))
            row=self.row(task,.002 if task['mode']=='graph' else .004)
            if task['mode']=='graph':row['timing']['graph_nodes']=graph_nodes or 8
            if len(received)==2:
                suite.signal.getsignal(suite.signal.SIGINT)(suite.signal.SIGINT,None)
            return {k:v for k,v in row.items() if k not in ('task','session','timestamp','status')}
        args=SimpleNamespace(size='pilot',retry_failed=False,pilot=None)
        with patch.dict(suite.PROTOCOL,{'preheat_seconds':0}), \
             patch('suite.subprocess.Popen',return_value=Telemetry()), \
             patch('suite.command_text',return_value={'exit_code':0,'stdout':''}), \
             patch('suite.time.sleep'),patch('measure.measure',side_effect=fake_measure):
            self.assertEqual(suite.acquire(args,self.out,manifest,{'gpu_uuid':'GPU-test'},torch),3)
            self.assertEqual(len(report.records(self.out)),2)
            self.assertEqual(suite.acquire(args,self.out,manifest,{'gpu_uuid':'GPU-test'},torch),0)
        self.assertEqual(received,[(0,'prealloc',None),(0,'graph',None),(1,'graph',8),(1,'prealloc',None)])
        self.assertEqual(len(report.records(self.out)),4)
        self.assertEqual(report.summary(self.out)[0]['paired_complete'],2)


class WrapperTests(unittest.TestCase):
    def test_only_current_complete_review_is_labelled_review(self):
        text=(plan.ROOT/'run_research_checked.sh').read_text()
        tail=text[text.index('if (( experiment_status == 0 )); then'):]
        prefix='test_python="$1"; export GEMM_RESEARCH_SUMMARY="$2"; experiment_status="$3"; experiment_started_ns="$4"; python() { "$test_python" "$@"; };\n'
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'summary.json'
            for gate,missing,failed,stale,expected in [('REVIEW',0,0,False,'EXPERIMENT: REVIEW'),
                 ('REVIEW',1,0,False,'EXPERIMENT: FAIL'),('REVIEW',0,1,False,'EXPERIMENT: FAIL'),
                 ('FAIL',0,1,False,'EXPERIMENT: FAIL'),('REVIEW',0,0,True,'EXPERIMENT: FAIL')]:
                path.write_text(json.dumps(dict(gate=gate,missing=missing,counts={'FAIL':failed})))
                started=path.stat().st_mtime_ns+(1 if stale else -1)
                p=subprocess.run(['bash','-c',prefix+tail,'test',sys.executable,str(path),'2',str(started)],capture_output=True,text=True)
                self.assertEqual(p.returncode,2)
                self.assertIn(expected,p.stdout+p.stderr)

    def test_portal_exception_is_exact_and_not_a_wildcard(self):
        rows='6470, /usr/libexec/xdg-desktop-portal-gnome, 9 MiB\n43, /tmp/xdg-desktop-portal-gnome, 9 MiB\n'
        self.assertEqual(suite.competing_processes({'exit_code':0,'stdout':rows},42),[rows.splitlines()[1]])


if __name__=='__main__':
    unittest.main()
