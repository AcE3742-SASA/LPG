"""CPU-only checks, reusing the frozen measurement/resume fixtures."""
import argparse
import ast
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run
import analyze_sweep

BASE = run.HERE.parent / 'abstract_sweep_750_v1'
spec = importlib.util.spec_from_file_location('previous_checks', BASE / 'test_subset.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


class Checks(previous.ReusedChecks):
    def synthetic_segment(self, tasks, session='s0', clock=None):
        return super().synthetic_segment(tasks, session, self.loaded['config']['sm_mhz'] if clock is None else clock)

    def finish_check(self):
        tasks = self.manifest['tasks']
        self.synthetic_segment(tasks[:24], 's0')
        self.synthetic_segment(tasks[24:], 's1')
        p = self.out / 'launches/first'
        p.mkdir(parents=True)
        suite = self.loaded['suite']
        suite.atomic_json(p / 'status.json', dict(collection='COMPLETE', restoration='COMMANDS_ACCEPTED'))
        suite.atomic_json(p / 'restore.json', dict(commands_accepted=True, commands={
            k: dict(exit_code=0, command=['sudo','-n','nvidia-smi','-i',self.manifest['environment_signature']['gpu_uuid'],opt])
            for k,opt in [('graphics','-rgc'),('memory','-rmc')]}))

    def test_exact_supported_selection(self):
        for text, expected in [('750\n735\n725\n720\n',725),('750\n735\n720\n705\n',720)]:
            self.assertEqual(run.choose_supported_clock(text), expected)
        self.assertEqual(run.choose_supported_clock('725\n720\n',720),720)
        for text, preferred in [('735\n',None),('N/A',None),('',None),('725 MHz',None),('725\n',720),('725\n',750)]:
            with self.assertRaises(RuntimeError): run.choose_supported_clock(text,preferred)

    def test_plan_and_measurement_reuse(self):
        plan = run.make_plan('full', self.loaded)
        self.assertEqual(len(plan['tasks']),3600)
        self.assertEqual(len(self.manifest['tasks']),48)
        self.assertEqual(self.manifest['tasks'][0]['shape'],[2329,3998,66])
        self.assertEqual(len(run.pinned_choices(self.loaded)),52)
        self.assertEqual((run.HERE/'selection.json').read_bytes(),(BASE/'selection.json').read_bytes())
        self.assertEqual((run.HERE/'analyze_sweep.py').read_bytes(),(BASE/'analyze_sweep.py').read_bytes())
        trees = [ast.parse((p/'run.py').read_text()) for p in (BASE,run.HERE)]
        funcs = [{n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef)} for tree in trees]
        for name in ('worker','run_worker','cleanup_signals','pack','require_clean_resume','choices_and_done'):
            self.assertEqual(funcs[0][name],funcs[1][name],name)
        self.assertEqual(run.make_plan('full',run.load(720))['tasks'],plan['tasks'])

    def test_reviewed_check_and_corruption(self):
        self.finish_check()
        result = analyze_sweep.analyze(self.out,self.loaded)
        self.assertEqual((result['recorded'],result['timed_blocks'],result['triplets_complete']),(48,336,16))
        self.assertTrue(result['timing_and_sampled_clocks_pass'])
        before={str(p.relative_to(self.out)):run.sha(p) for p in self.out.rglob('*') if p.is_file() and p.name!='.run.lock'}
        self.assertEqual(run.check_admission(self.out,'Reviewed diagnostic; counter unresolved',self.loaded)['check_gate'],'REVIEW')
        self.assertEqual(before,{str(p.relative_to(self.out)):run.sha(p) for p in self.out.rglob('*') if p.is_file() and p.name!='.run.lock'})
        with self.assertRaises(RuntimeError):run.check_admission(self.out,'',self.loaded)
        with self.assertRaises(RuntimeError):run.check_admission(self.out,'Reviewed',run.load(720))
        p=self.out/'sessions/s1/segment_00000/telemetry.csv'
        p.write_text(p.read_text().replace(',725,6801,',',720,6801,'))
        with self.assertRaisesRegex(RuntimeError,'resume refused'):run.check_admission(self.out,'Reviewed',self.loaded)

    def test_incomplete_check_blocks_full_without_settings(self):
        self.synthetic_segment(self.manifest['tasks'][:3])
        with self.assertRaises(RuntimeError):run.check_admission(self.out,'Reviewed',self.loaded)
        args=argparse.Namespace(output=Path(self.tmp.name)/'full',size='full',check=None,review_note=None,retry_failed=False)
        with patch.object(self.loaded['suite'],'command_text') as calls:
            with self.assertRaises(RuntimeError):run.run(args,self.loaded)
            calls.assert_not_called()

    def test_720_evidence_requires_720_analysis_and_dirty_resume_stops(self):
        self.loaded=run.load(720)
        self.out=Path(self.tmp.name)/'selected720'; self.out.mkdir()
        self.manifest=run.manifest_for(self.out,'check',self.loaded)
        directory,rows=self.synthetic_segment(self.manifest['tasks'][:3])
        run.require_clean_resume(self.out,self.loaded)
        q=analyze_sweep.segment_quality(directory,rows,run.load(725))
        self.assertFalse(q['collection_can_continue'])
        p=directory/'telemetry.csv'
        p.write_text(p.read_text().replace(',720,6801,',',705,6801,'))
        with self.assertRaisesRegex(RuntimeError,'resume refused'):run.require_clean_resume(self.out,self.loaded)
        args=argparse.Namespace(output=self.out,limit=None,retry_failed=False)
        with self.assertRaisesRegex(RuntimeError,'resume refused'):run.worker(args,self.loaded)

    def test_controller_restores_selected_720_after_failure(self):
        # Fresh output selects the authorized fallback from the GPU query.
        out=Path(self.tmp.name)/'new'; calls=[]
        def command(cmd):
            calls.append(cmd)
            stdout=('750\n735\n720\n705' if '--query-supported-clocks=graphics' in cmd else
                    self.manifest['environment_signature']['gpu_uuid'] if '--query-gpu=uuid' in cmd else
                    '70' if '--query-gpu=power.limit' in cmd else '')
            return dict(command=cmd,exit_code=0,stdout=stdout,stderr='')
        args=argparse.Namespace(output=out,size='check',check=None,review_note=None,retry_failed=False)
        with patch.object(self.loaded['suite'],'command_text',side_effect=command), \
             patch.object(run.shutil,'which',return_value='/test/nvidia-smi'), \
             patch.object(run.shutil,'disk_usage',return_value=argparse.Namespace(free=30*1024**3)), \
             patch.object(run,'run_worker',side_effect=RuntimeError('synthetic worker failed')), \
             patch.object(self.loaded['suite'],'lock',side_effect=lambda _:contextlib.nullcontext()):
            with self.assertRaisesRegex(RuntimeError,'synthetic worker failed'):run.run(args,self.loaded)
        self.assertEqual([c[-1] for c in calls if c and c[0]=='sudo' and '-i' in c],['6801,6801','720,720','-rgc','-rmc'])
        self.assertFalse(any('-pl' in c for c in calls))
        self.assertEqual(run.read(out/'manifest.json')['selected_clock_mhz'],720)
        self.assertEqual(run.read(next(out.glob('clock_selection_*.json')))['selected_clock_mhz'],720)
        self.assertTrue(run.read(next((out/'launches').glob('*/restore.json')))['commands_accepted'])

    def test_launcher_keeps_current_terminal_and_packs(self):
        with tempfile.TemporaryDirectory(prefix='low clock launcher ') as folder:
            root=Path(folder); package=root/'package'; package.mkdir(); bins=root/'bin'; bins.mkdir()
            shutil.copy2(run.HERE/'launch.sh',package/'launch.sh')
            (root/'gpu_sweep_v6').mkdir()
            for name,body in {'python3':'printf "%s\\n" "$*" >> "$CALLS"; exit 0','sudo':'exit 0',
                              'flock':'exit 0','tmux':'echo unexpected-tmux >> "$CALLS"; exit 99'}.items():
                p=bins/name;p.write_text('#!/usr/bin/env bash\n'+body+'\n');p.chmod(0o755)
            (root/'gpu_sweep_v6/run_research_checked.sh').write_text('printf "%s\\n" "$@" > "$ARGUMENTS"\nexit "$MEASURE_EXIT"\n')
            for size,inside,code in [('check',True,2),('check',False,1),('full',True,2)]:
                out=root/f'output {size} {inside}';calls=root/'calls';args=root/'args';calls.write_text('')
                env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ['PATH'],CALLS=str(calls),ARGUMENTS=str(args),MEASURE_EXIT=str(code))
                env.pop('TMUX',None)
                if inside:env['TMUX']='/tmp/existing,123,0'
                cmd=['bash',str(package/'launch.sh'),size,'--output',str(out)]
                if size=='full':cmd+=['--review-note','explicit review']
                result=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=5)
                self.assertEqual(result.returncode,code,result.stderr)
                self.assertNotIn('unexpected-tmux',calls.read_text())
                self.assertIn(' pack --output ',calls.read_text())
                actual=args.read_text().splitlines()
                self.assertEqual(actual[actual.index('--size')+1],size)
                self.assertEqual('--review-note' in actual,size=='full')


def load_tests(loader, tests, pattern):
    names=[name for name in Checks.__dict__ if name.startswith('test_')]
    names += ['test_denominator_or_accuracy_corruption_fails','test_resume_preserves_attempts_and_graph_choices',
              'test_manifest_change_and_output_lock_rejected','test_worker_process_exit_and_failure_logged',
              'test_failed_credentials_stop_child_before_return','test_worker_resume_collects_only_pending_tasks',
              'test_failed_measurement_without_seed_exports_as_fail','test_timing_review_is_retained_without_discarding_collection']
    suite=unittest.TestSuite(Checks(name) for name in names)
    for name in ('test_coverage_inputs_splits_mode_order_and_holdout','test_selection_reproducible_independent_of_order_or_results'):
        suite.addTest(previous.SubsetChecks(name))
    return suite


if __name__=='__main__':unittest.main(verbosity=2)
