"""CPU-only subset checks plus unchanged worker/cleanup checks from the frozen base."""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run
import subset
import analyze_sweep

BASE = run.HERE.parent/'fixed_sweep_780_v1'
spec = importlib.util.spec_from_file_location('verified_base_checks', BASE/'test_sweep.py')
base_checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_checks)


class SubsetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = run.load()
        cls.original = cls.loaded['plan'].make_plan('full')
        cls.selection = run.read(run.HERE/'selection.json')
        cls.plan = run.make_plan('full', cls.loaded)

    def test_coverage_inputs_splits_mode_order_and_holdout(self):
        tasks = self.plan['tasks']
        self.assertEqual(len(tasks), 3600)
        self.assertEqual(len({t['id'] for t in tasks}), 3600)
        groups = defaultdict(set)
        original = {t['id']: t for t in self.original['tasks'] if t['kind']=='gemm'}
        full_bases = {(tuple(t['shape']), t['dtype'], t['round']): t
                      for t in self.original['tasks'] if t['kind']=='gemm' and t.get('phase')=='main' and t['mode']=='prealloc'}
        for task in tasks:
            self.assertEqual(task['phase'], 'main')
            self.assertNotEqual(task['split'], 'fresh_holdout')
            groups[tuple(task['shape'])].add((task['dtype'],task['mode'],task['round']))
            base = full_bases[(tuple(task['shape']), task['dtype'], task['round'])]
            expected = {**base, 'mode': task['mode']}
            expected.pop('id')
            expected['id'] = self.loaded['plan'].digest(expected)[:24]
            self.assertEqual(task, json.loads(json.dumps(expected)))
            self.assertEqual(self.loaded['plan'].task_seed(task), self.loaded['plan'].task_seed(base))
        self.assertEqual(len(groups), 200)
        combinations = {(d,m,r) for d in ('fp16','fp32') for m in run.MODES for r in range(3)}
        self.assertTrue(all(g==combinations for g in groups.values()))
        for index in range(0,3600,3):
            self.assertEqual([t['mode'] for t in tasks[index:index+3]],
                             self.loaded['config']['mode_order'][tasks[index]['round']])
        counts = Counter(x['stratum'] for x in self.selection['shapes'])
        self.assertEqual(Counter(counts.values()), {6:24,7:8})
        self.assertEqual(len(self.plan['shape_splits']), 200)
        self.assertEqual(self.plan['protocol'], self.original['protocol'])
        self.assertEqual(len(subset.ANCHORS), 10)
        self.assertTrue(set(subset.ANCHORS)<=set(groups))

    def test_selection_reproducible_independent_of_order_or_results(self):
        changed = copy.deepcopy(self.original)
        random.Random(19).shuffle(changed['tasks'])
        for t in changed['tasks']:
            t.update(median_ms=99999, status='FAIL')
        self.assertEqual(subset.select(changed), self.selection)
        bad = copy.deepcopy(self.selection)
        bad['shapes'][0]['shape'] = [1,1,1]
        with self.assertRaises(ValueError): subset.validate(self.original,bad)

    def test_measurement_worker_controller_and_analyzer_unchanged(self):
        old = {n.name:ast.dump(n) for n in ast.parse((BASE/'run.py').read_text()).body if isinstance(n,ast.FunctionDef)}
        new = {n.name:ast.dump(n) for n in ast.parse((run.HERE/'run.py').read_text()).body if isinstance(n,ast.FunctionDef)}
        changed = {name for name in old if old[name]!=new[name]}
        self.assertEqual(changed, {'make_plan','manifest_for','check_admission','main'})
        self.assertEqual((BASE/'analyze_sweep.py').read_bytes(), (run.HERE/'analyze_sweep.py').read_bytes())

    def test_review_bridge_rejects_errors_and_unreviewed_digest(self):
        valid = dict(self.loaded['config']['reviewed_check'],check_gate='REVIEW')
        response = argparse.Namespace(returncode=0,stdout=json.dumps(valid),stderr='')
        with patch.object(run.subprocess,'run',return_value=response) as call:
            self.assertEqual(run.check_admission(Path('/example check'),'reviewed',self.loaded),valid)
            self.assertIn(str(BASE),call.call_args.args[0])
        response.stdout=json.dumps({**valid,'results_digest':'different'})
        with patch.object(run.subprocess,'run',return_value=response):
            with self.assertRaisesRegex(RuntimeError,'not the check'):run.check_admission(Path('/example'),'reviewed',self.loaded)
        response.returncode=1;response.stderr='clock mismatch'
        with patch.object(run.subprocess,'run',return_value=response):
            with self.assertRaisesRegex(RuntimeError,'clock mismatch'):run.check_admission(Path('/example'),'reviewed',self.loaded)
        with self.assertRaises(RuntimeError):run.check_admission(None,'',self.loaded)

    def test_rejected_check_prevents_gpu_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            args=argparse.Namespace(output=Path(folder),size='full',check=Path('/missing'),review_note='reviewed',retry_failed=False)
            with patch.object(run,'check_admission',side_effect=RuntimeError('check rejected')), \
                 patch.object(self.loaded['suite'],'command_text') as commands:
                with self.assertRaisesRegex(RuntimeError,'check rejected'):run.run(args,self.loaded)
                commands.assert_not_called()

    def test_subset_analysis_preserves_missing_and_resume(self):
        case=base_checks.SweepChecks('test_resume_preserves_attempts_and_graph_choices')
        case.loaded=self.loaded
        case.setUp()
        try:
            target=Path(case.tmp.name)/'subset';target.mkdir()
            case.out=target;case.manifest=run.manifest_for(target,'full',self.loaded)
            tasks=[t for t in self.plan['tasks'] if t['shape']==[64,64,64]]
            case.synthetic_segment(tasks,'subset_session')
            summary=analyze_sweep.analyze(target,self.loaded)
            self.assertEqual((summary['recorded'],summary['missing'],summary['triplets_complete']), (18,3582,6))
            self.assertEqual(summary['timed_blocks'],126)
            self.assertEqual(summary['gate'],'INCOMPLETE')
            self.assertFalse(summary['timing_and_sampled_clocks_pass'])
            choices,done=run.choices_and_done(target,case.manifest,self.loaded)
            self.assertEqual(len(done),18)
            self.assertEqual(sum(t['id'] not in done for t in self.plan['tasks']),3582)
        finally:
            case.doCleanups()

    def test_launcher_uses_current_terminal_and_preserves_exit_and_arguments(self):
        with tempfile.TemporaryDirectory(prefix='subset launcher ') as folder:
            root=Path(folder);package=root/'abstract_sweep_780_v1';package.mkdir()
            shutil.copy2(run.HERE/'launch.sh',package/'launch.sh')
            shutil.copy2(run.HERE/'review_note.md',package/'review_note.md')
            (root/'gpu_sweep_v6').mkdir();bins=root/'bin';bins.mkdir()
            for name,body in {'python3':'printf "%s\\n" "$*" >> "$CALLS"; exit 0',
                              'sudo':'exit 0','flock':'exit 0','tmux':'echo unexpected-tmux >> "$CALLS"; exit 99'}.items():
                p=bins/name;p.write_text('#!/usr/bin/env bash\n'+body+'\n');p.chmod(0o755)
            (root/'gpu_sweep_v6/run_research_checked.sh').write_text('printf "%s\\n" "$@" > "$ARGUMENTS"\nexit "$MEASURE_EXIT"\n')
            for inside,code in ((False,0),(True,0),(False,2),(True,2)):
                out=root/f'output {inside} {code}';calls=root/'calls';args=root/'args';calls.write_text('')
                env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ['PATH'],CALLS=str(calls),ARGUMENTS=str(args),MEASURE_EXIT=str(code))
                env.pop('TMUX',None)
                if inside:env['TMUX']='/tmp/existing,123,0'
                with tempfile.TemporaryFile() as log:
                    proc=subprocess.Popen(['bash',str(package/'launch.sh'),'--output',str(out)],env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
                    try:actual=proc.wait(timeout=5)
                    finally:
                        if proc.poll() is None:proc.kill();proc.wait()
                        proc.stdin.close()
                    log.seek(0);output=log.read().decode()
                self.assertEqual(actual,code,output)
                self.assertNotIn('unexpected-tmux',calls.read_text())
                self.assertIn(' pack --output ',calls.read_text())
                self.assertIn('현재 tmux 세션에서' if inside else 'SSH 연결을 유지',output)
                invocation,=(out/'launcher').iterdir()
                self.assertEqual((invocation/'exit_code.txt').read_text().strip(),str(code))
                self.assertEqual((invocation/'launcher_source.sh').read_bytes(),(run.HERE/'launch.sh').read_bytes())
                actual_args=args.read_text().splitlines()
                self.assertEqual(actual_args[actual_args.index('--size')+1],'full')
                self.assertEqual(actual_args[actual_args.index('--output')+1],str(out))
                self.assertEqual(actual_args[actual_args.index('--review-note')+1],(run.HERE/'review_note.md').read_text().strip())


def load_tests(loader, tests, pattern):
    for name in ('test_denominator_or_accuracy_corruption_fails','test_resume_preserves_attempts_and_graph_choices',
                 'test_manifest_change_and_output_lock_rejected','test_worker_process_exit_and_failure_logged',
                 'test_failed_credentials_stop_child_before_return','test_controller_restores_both_clocks_on_worker_failure',
                 'test_preflight_power_mismatch_changes_no_clocks','test_worker_resume_collects_only_pending_tasks',
                 'test_failed_measurement_without_seed_exports_as_fail','test_timing_review_is_retained_without_discarding_collection'):
        tests.addTest(base_checks.SweepChecks(name))
    return tests


if __name__=='__main__':unittest.main(verbosity=2)
