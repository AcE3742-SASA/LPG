"""CPU checks for the full plan and review bridge, reusing the existing worker tests."""
import argparse
import ast
import contextlib
import copy
import csv
import sys
from types import SimpleNamespace
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
import progress
import prepare

spec = importlib.util.spec_from_file_location('original_sweep_tests', run.HERE.parent / 'fixed_sweep_780_v1/test_sweep.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Checks(base.SweepChecks):
    def synthetic_segment(self, tasks, session='s0', clock=720):
        return super().synthetic_segment(tasks, session, clock)

    def finish_check(self):
        tasks = self.manifest['tasks']
        self.synthetic_segment(tasks[:24], 's0')
        self.synthetic_segment(tasks[24:], 's1')
        p = self.out / 'launches/first'
        p.mkdir(parents=True)
        self.loaded['suite'].atomic_json(p / 'status.json', dict(collection='COMPLETE', restoration='COMMANDS_ACCEPTED'))
        self.loaded['suite'].atomic_json(p / 'restore.json', dict(commands_accepted=True, commands={
            key: dict(exit_code=0, command=['sudo', '-n', 'nvidia-smi', '-i', self.manifest['environment_signature']['gpu_uuid'], option])
            for key, option in [('graphics', '-rgc'), ('memory', '-rmc')]}))

    def test_existing_plan_inputs_and_measurement_functions_unchanged(self):
        path = run.HERE.parent / 'fixed_sweep_780_v1/run.py'
        function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'make_plan')
        namespace = dict(json=json, MODES=run.MODES)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        expected = namespace['make_plan']('full', self.loaded)
        actual = run.make_plan('full', self.loaded)
        self.assertEqual(actual['tasks'], expected['tasks'])
        self.assertEqual(actual['shape_splits'], expected['shape_splits'])
        self.assertEqual(actual['protocol'], expected['protocol'])
        self.assertEqual(actual['selected_clock_mhz'], 720)
        self.assertEqual(len(run.pinned_choices(self.loaded)), 52)
        previous = run.HERE.parent / 'abstract_sweep_725_v1'
        trees = [{n.name: ast.dump(n) for n in ast.parse((folder / 'run.py').read_text()).body
                  if isinstance(n, ast.FunctionDef)} for folder in (previous, run.HERE)]
        for name in ('run_worker', 'cleanup_signals', 'pack',
                     'history', 'choices_and_done', 'pinned_choices'):
            self.assertEqual(trees[0][name], trees[1][name], name)

    def test_only_exact_720_supported(self):
        self.assertEqual(run.choose_supported_clock('780\n750\n735\n720\n705'), 720)
        for text, preferred in [('725\n735', None), ('N/A', None), ('', None), ('720 MHz', None), ('720', 780)]:
            with self.assertRaises(RuntimeError):
                run.choose_supported_clock(text, preferred)
        with self.assertRaises(ValueError):
            run.load(725)

    def test_reviewed_real_zip_and_wrong_evidence(self):
        archive = Path(os.environ.get('GEMM_TEST_PILOT_ZIP', str(run.HERE.parent /
            'analysis/abstract_200_720_recovery_result_2026-09-22/abstract_200_720_recovery_20260921_20260922_024735_e61123.zip')))
        before = run.sha(archive)
        result = run.check_admission(archive, 'User approved full collection; counter unresolved', self.loaded)
        self.assertEqual(result['reviewed_measurements'], 3600)
        self.assertEqual(result['check_gate'], 'REVIEW')
        self.assertEqual(run.sha(archive), before)
        with self.assertRaises(RuntimeError):
            run.check_admission(archive, '', self.loaded)
        wrong = Path(self.tmp.name) / 'wrong.zip'
        wrong.write_bytes(b'not the reviewed archive')
        args = argparse.Namespace(output=Path(self.tmp.name)/'full', size='full', check=wrong,
                                  review_note='approved', retry_failed=False)
        with patch.object(self.loaded['suite'], 'command_text') as commands:
            with self.assertRaisesRegex(RuntimeError, 'Not the reviewed'):
                run.run(args, self.loaded)
            commands.assert_not_called()

    def test_clock_flag_is_preserved_and_resume_allowed(self):
        directory, rows = self.synthetic_segment(self.manifest['tasks'][:3])
        self.assertEqual(run.require_clean_resume(self.out, self.loaded), 0)
        path = directory / 'telemetry.csv'
        path.write_text(path.read_text().replace(',720,6801,', ',720,6800,', 1))
        self.assertEqual(run.require_clean_resume(self.out, self.loaded), 3)
        quality = analyze_sweep.segment_quality(directory, rows, self.loaded)
        self.assertTrue(quality['collection_can_continue'])
        self.assertFalse(quality['timing_and_sampled_clocks_pass'])
        self.assertTrue(all(q['status'] == 'REVIEW' for q in quality['measurements'].values()))
        summary = analyze_sweep.analyze(self.out, self.loaded)
        self.assertEqual(summary['review_queue_tasks'], 3)
        self.assertEqual(summary['accepted_measurements'], 0)
        self.assertEqual(len(run.read(self.out/'remeasure_tasks.json')), 3)
        self.assertTrue(all(r['status'] == 'PASS' for _,r in run.history(self.out)))

    def test_controller_sets_720_and_restores_on_failure(self):
        calls = []
        def command(cmd):
            calls.append(cmd)
            stdout = ('750\n735\n720\n705' if '--query-supported-clocks=graphics' in cmd else
                      self.manifest['environment_signature']['gpu_uuid'] if '--query-gpu=uuid' in cmd else
                      '70' if '--query-gpu=power.limit' in cmd else '')
            return dict(command=cmd, exit_code=0, stdout=stdout, stderr='')
        args = argparse.Namespace(output=self.out, size='check', check=None, review_note=None, retry_failed=False)
        with patch.object(self.loaded['suite'], 'command_text', side_effect=command), \
             patch.object(run.shutil, 'which', return_value='/test/nvidia-smi'), \
             patch.object(run.shutil, 'disk_usage', return_value=argparse.Namespace(free=30*1024**3)), \
             patch.object(run, 'run_worker', side_effect=RuntimeError('synthetic worker failed')), \
             patch.object(self.loaded['suite'], 'lock', side_effect=lambda _: contextlib.nullcontext()):
            with self.assertRaisesRegex(RuntimeError, 'synthetic worker failed'):
                run.run(args, self.loaded)
        self.assertEqual([c[-1] for c in calls if c and c[0]=='sudo' and '-i' in c],
                         ['6801,6801', '720,720', '-rgc', '-rmc'])
        self.assertFalse(any('-pl' in c for c in calls))

    def test_progress_starts_at_zero_and_tracks_resume(self):
        self.assertIn('0/113,382 (0.0%)', progress.line(self.out))
        self.loaded['suite'].atomic_json(self.out / 'progress.json', dict(recorded=56691, expected=113382, state='RUNNING'))
        self.assertIn('56,691/113,382 (50.0%)', progress.line(self.out))
        self.assertIn('남음 56,691', progress.line(self.out))

    def test_launcher_default_full_current_tmux_and_closed_log_pack(self):
        with tempfile.TemporaryDirectory(prefix='full720 launcher ') as folder:
            root = Path(folder); package = root/'package'; bins = root/'bin'
            package.mkdir(); bins.mkdir(); (root/'gpu_sweep_v6').mkdir()
            previous=root/'fixed_sweep_720_v1/runs/full_720'
            previous.mkdir(parents=True); (previous/'manifest.json').write_text('{}')
            shutil.copy2(run.HERE/'launch.sh', package/'launch.sh')
            for name, body in {'python3': 'printf "%s\\n" "$*" >> "$CALLS"; exit 0',
                               'sudo': 'exit 0', 'flock': 'exit 0', 'tmux': 'echo unexpected-tmux >> "$CALLS"; exit 99'}.items():
                p=bins/name; p.write_text('#!/usr/bin/env bash\n'+body+'\n'); p.chmod(0o755)
            (root/'gpu_sweep_v6/run_research_checked.sh').write_text('printf "%s\\n" "$@" > "$ARGUMENTS"\nexit 2\n')
            env=dict(os.environ, PATH=str(bins)+os.pathsep+os.environ['PATH'], CALLS=str(root/'calls'),
                     ARGUMENTS=str(root/'arguments'), TMUX='/tmp/existing,123,0')
            result=subprocess.run(['bash', str(package/'launch.sh')], env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)
            args=(root/'arguments').read_text().splitlines()
            self.assertEqual(args[args.index('--size')+1], 'full')
            self.assertEqual(args[args.index('--output')+1], str(package/'runs/full_720'))
            self.assertTrue(args[args.index('--check')+1].endswith('_024735_e61123.zip'))
            self.assertNotIn('unexpected-tmux', (root/'calls').read_text())
            self.assertIn(' pack --output ', (root/'calls').read_text())
            self.assertIn('prepare.py --source ', (root/'calls').read_text())



    def test_only_clock_and_sw_power_cap_are_nonfatal(self):
        directory, rows = self.synthetic_segment(self.manifest['tasks'][:3])
        path = directory/'telemetry.csv'; original = path.read_text()
        cases = [(',720,6801,', ',705,6801,', True),
                 (',720,6801,', ',720,6800,', True),
                 ('Not Active,Not Active,Not Active', 'Active,Not Active,Not Active', True),
                 ('Not Active,Not Active,Not Active', 'Not Active,Active,Not Active', False),
                 ('Not Active,Not Active,Not Active', 'Not Active,Not Active,Active', False),
                 (',60,70,', ',60,65,', False),
                 (',720,6801,', ',N/A,6801,', False)]
        for before, after, allowed in cases:
            with self.subTest(after=after):
                path.write_text(original.replace(before, after, 1))
                quality = analyze_sweep.segment_quality(directory, rows, self.loaded)
                self.assertEqual(quality['collection_can_continue'], allowed)
                self.assertFalse(quality['timing_and_sampled_clocks_pass'])
                self.assertEqual({q['status'] for q in quality['measurements'].values()}, {'REVIEW'})
                if not allowed:
                    with self.assertRaisesRegex(RuntimeError, 'resume refused'):
                        run.require_clean_resume(self.out, self.loaded)
        path.write_text(original.splitlines()[0]+'\n')
        self.assertFalse(analyze_sweep.segment_quality(directory, rows, self.loaded)['collection_can_continue'])

    def test_complete_collection_separates_accepted_and_review_queue(self):
        self.finish_check()
        path = self.out/'sessions/s0/segment_00000/telemetry.csv'
        path.write_text(path.read_text().replace(',720,6801,', ',705,6801,', 1))
        summary = analyze_sweep.analyze(self.out, self.loaded)
        self.assertTrue(summary['collection_complete'])
        self.assertEqual(summary['counts'], {'REVIEW':24, 'PASS':24})
        self.assertEqual(summary['review_queue_tasks'], 24)
        self.assertEqual(summary['accepted_measurements'], 24)
        self.assertFalse(summary['timing_and_sampled_clocks_pass'])
        queue = run.read(self.out/'remeasure_tasks.json')
        self.assertEqual({r['task']['id'] for r in queue}, {t['id'] for t in self.manifest['tasks'][:24]})
        self.assertNotIn('median_ms', json.dumps(queue))
        self.assertTrue(all(r['status']=='PASS' for _,r in run.history(self.out)))

    def test_worker_continues_after_clock_flag_and_resumes_but_stops_on_fatal(self):
        directory, fixtures = self.synthetic_segment(self.manifest['tasks'][:9])
        by_id = {r['task']['id']:r for r in fixtures}
        # Use three-task chunks to exercise transitions without 120 fake measurements.
        loaded = {**self.loaded, 'config': {**self.loaded['config'], 'segment_tasks':3}}
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda:(10*1024**3,20*1024**3)),
            inference_mode=contextlib.nullcontext, randn=lambda *a,**k:1., empty_like=lambda _:0.)
        for fault, expected_count in [('clock',9), ('missing_sensor',3), ('correctness',1)]:
            with self.subTest(fault=fault):
                target = Path(self.tmp.name)/fault; target.mkdir()
                run.manifest_for(target, 'check', loaded)
                calls = []
                @contextlib.contextmanager
                def monitor(segment, *args):
                    calls.append(segment)
                    if fault != 'missing_sensor' or len(calls)>1:
                        shutil.copy2(directory/'telemetry_metadata.json',segment/'telemetry_metadata.json')
                        text=(directory/'telemetry.csv').read_text()
                        if fault=='clock' and len(calls)==1:
                            text=text.replace(',720,6801,', ',705,6801,',1)
                        (segment/'telemetry.csv').write_text(text)
                    yield SimpleNamespace(poll=lambda:None)
                def measured(torch, task, protocol, graph_nodes=None):
                    row = copy.deepcopy(by_id[task['id']])
                    if fault=='correctness': row['correctness']['passed']=False
                    return {k:row[k] for k in ['timing','correctness','input_seed']}
                ticks=iter(range(0,100000,1000))
                args=argparse.Namespace(output=target,limit=6,retry_failed=False)
                with patch.dict(sys.modules,torch=fake_torch), \
                     patch.object(loaded['suite'],'environment',return_value=self.manifest['environment_signature']), \
                     patch.object(loaded['audit'],'check_processes'), \
                     patch.object(loaded['telemetry'],'collect',side_effect=monitor), \
                     patch.object(loaded['measure'],'measure',side_effect=measured), \
                     patch.object(run.time,'monotonic',side_effect=lambda:next(ticks)):
                    if fault=='clock':
                        self.assertEqual(run.worker(args,loaded),0)
                        args.limit=3
                        self.assertEqual(run.worker(args,loaded),0)
                        self.assertEqual(run.read(target/'progress.json')['flagged'],3)
                    else:
                        with self.assertRaises(RuntimeError):run.worker(args,loaded)
                rows=list(run.history(target))
                self.assertEqual(len(rows),expected_count)
                self.assertEqual([r['task']['id'] for _,r in rows],
                                 [t['id'] for t in self.manifest['tasks'][:expected_count]])

    def test_stopped_v1_import_is_exact_idempotent_and_rejects_active_run(self):
        source = Path(self.tmp.name)/'old_full'; source.mkdir()
        old_root = run.HERE.parent/'fixed_sweep_720_v1'
        old_config = run.read(old_root/'protocol.json')
        old_hashes = {p.name:run.sha(p) for p in old_root.iterdir()
                      if p.is_file() and p.suffix in ('.py','.sh','.json','.md')}
        old_manifest = dict(run.make_plan('full',{**self.loaded,'config':old_config}), kind='fixed_sweep_720_v1',
            source_hashes=old_hashes, dependency_hashes=run.read(old_root/'dependency_hashes.json'),
            environment_signature=self.manifest['environment_signature'], pinned_graph_choices=run.pinned_choices(self.loaded))
        suite=self.loaded['suite'];suite.atomic_json(source/'manifest.json',old_manifest)
        for origin, folder, hashes in [(old_root,'source',old_hashes),
                                      (run.HERE.parent,'dependencies',old_manifest['dependency_hashes'])]:
            for name in hashes:
                dest=source/folder/name; dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin/name,dest)
        previous_out,previous_manifest = self.out,self.manifest
        self.out,self.manifest = source,old_manifest
        tasks=old_manifest['tasks'][:3]
        try: directory,fixtures=self.synthetic_segment(tasks)
        finally:self.out,self.manifest=previous_out,previous_manifest
        path=directory/'telemetry.csv';path.write_text(path.read_text().replace(',720,6801,',',705,6801,',1))
        launch=source/'launches/one';launch.mkdir(parents=True)
        suite.atomic_json(launch/'status.json',dict(collection='FAILED_OR_INTERRUPTED',restoration='COMMANDS_ACCEPTED'))
        suite.atomic_json(launch/'restore.json',dict(commands_accepted=True,commands={
            key:dict(exit_code=0,command=['sudo','-n','nvidia-smi','-i',old_manifest['environment_signature']['gpu_uuid'],opt])
            for key,opt in [('graphics','-rgc'),('memory','-rmc')]}))
        before={str(p.relative_to(source)):run.sha(p) for p in source.rglob('*') if p.is_file()}
        output=Path(self.tmp.name)/'upgraded'
        with suite.lock(source/'.launcher.lock'):
            with self.assertRaisesRegex(RuntimeError,'still running'):
                prepare.prepare(source,output,self.loaded)
        self.assertFalse(output.exists())
        with suite.lock(source/'.run.lock'):
            with self.assertRaisesRegex(RuntimeError,'still running'):
                prepare.prepare(source,output,self.loaded)
        result=prepare.prepare(source,output,self.loaded)
        self.assertEqual((result['recorded'],result['remaining'],result['flagged']),(3,113379,3))
        imported=list(run.history(output))
        self.assertEqual([r for _,r in imported],fixtures)
        self.assertEqual(run.read(output/'manifest.json')['kind'],'fixed_sweep_720_v2')
        self.assertEqual(run.read(output/'upgrade/source_manifest.json')['kind'],'fixed_sweep_720_v1')
        self.assertFalse(prepare.prepare(source,output,self.loaded)['imported'])
        for name,digest in before.items():self.assertEqual(run.sha(source/name),digest,name)
        self.assertEqual(len(list(run.history(output))),3)
        for q in (output/'sessions').glob('*/segment_*/quality.json'):
            self.assertTrue(run.read(q)['collection_can_continue'])
        # A failed reset or modified source must not yield a runnable upgraded folder.
        restore=launch/'restore.json'; original=restore.read_bytes()
        bad=run.read(restore); bad['commands']['graphics']['exit_code']=1
        suite.atomic_json(restore,bad)
        rejected=Path(self.tmp.name)/'rejected'
        with self.assertRaisesRegex(ValueError,'Reset evidence mismatch'):
            prepare.prepare(source,rejected,self.loaded)
        self.assertFalse(rejected.exists()); restore.write_bytes(original)
        (source/'source/run.py').write_text('changed source')
        with self.assertRaisesRegex(ValueError,'source/dependency mismatch'):
            prepare.prepare(source,rejected,self.loaded)
        self.assertFalse(rejected.exists())


def load_tests(loader, tests, pattern):
    names = [name for name in Checks.__dict__ if name.startswith('test_')]
    names += ['test_full_coverage_modes_rounds_and_no_new_auxiliary',
              'test_denominator_or_accuracy_corruption_fails', 'test_resume_preserves_attempts_and_graph_choices',
              'test_manifest_change_and_output_lock_rejected', 'test_worker_process_exit_and_failure_logged',
              'test_failed_credentials_stop_child_before_return', 'test_worker_resume_collects_only_pending_tasks',
              'test_failed_measurement_without_seed_exports_as_fail', 'test_timing_review_is_retained_without_discarding_collection']
    return unittest.TestSuite(Checks(name) for name in names)


if __name__ == '__main__':
    unittest.main(verbosity=2)
