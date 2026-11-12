"""CPU checks for the full plan and review bridge, reusing the existing worker tests."""
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
import progress

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
        self.assertEqual((run.HERE / 'analyze_sweep.py').read_bytes(), (previous / 'analyze_sweep.py').read_bytes())
        trees = [{n.name: ast.dump(n) for n in ast.parse((folder / 'run.py').read_text()).body
                  if isinstance(n, ast.FunctionDef)} for folder in (previous, run.HERE)]
        for name in ('worker', 'run_worker', 'cleanup_signals', 'pack', 'require_clean_resume',
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

    def test_dirty_resume_keeps_strict_memory_gate(self):
        directory, rows = self.synthetic_segment(self.manifest['tasks'][:3])
        run.require_clean_resume(self.out, self.loaded)
        path = directory / 'telemetry.csv'
        path.write_text(path.read_text().replace(',720,6801,', ',720,6800,', 1))
        with self.assertRaisesRegex(RuntimeError, 'resume refused'):
            run.require_clean_resume(self.out, self.loaded)
        with self.assertRaisesRegex(RuntimeError, 'resume refused'):
            run.worker(argparse.Namespace(output=self.out, limit=None, retry_failed=False), self.loaded)

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
