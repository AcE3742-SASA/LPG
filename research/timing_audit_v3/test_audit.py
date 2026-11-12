"""CPU-only checks of capture boundaries, correlation, and failure preservation."""
import json
import shutil
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import audit
import analyze


class AuditTests(unittest.TestCase):
    def test_focused_cases_preserve_v2_identities(self):
        self.assertEqual([(r['id'], r['shape'], r['dtype'], r['mode']) for r in audit.cases()],
                         [('c03', [3216,520,776], 'fp16', 'prealloc'),
                          ('c04', [4096,4096,4096], 'fp16', 'prealloc'),
                          ('c05', [4096,4096,4096], 'fp32', 'prealloc')])

    def test_prepare_failure_stops_collection_and_preserves_error(self):
        events = []
        class Timer:
            def prepare(self, fn, operations=1):
                events.append('prepare')
                raise ValueError('preparation failed')
        torch = SimpleNamespace(cuda=SimpleNamespace(
            cudart=lambda: SimpleNamespace(cudaProfilerStart=lambda: events.append('start'),
                                           cudaProfilerStop=lambda: events.append('stop')),
            nvtx=SimpleNamespace(range_push=lambda label: events.append(label),
                                 range_pop=lambda: events.append('pop'))))
        timer = audit.traced_timer(Timer, torch, 'c03')()
        with self.assertRaisesRegex(ValueError, 'preparation failed'):
            timer.prepare(None)
        self.assertEqual(events, ['start', 'prepare/c03', 'prepare', 'pop', 'stop'])

    def test_measured_failure_stops_collection_and_preserves_error(self):
        events = []
        class Timer:
            def block(self, fn, repeats, operations=1):
                raise ValueError('measurement failed')
        torch = SimpleNamespace(cuda=SimpleNamespace(
            cudart=lambda: SimpleNamespace(cudaProfilerStop=lambda: events.append('stop')),
            nvtx=SimpleNamespace(range_push=lambda label: events.append(label),
                                 range_pop=lambda: events.append('pop'))))
        timer = audit.traced_timer(Timer, torch, 'c03')()
        timer.collecting = True
        with self.assertRaisesRegex(ValueError, 'measurement failed'):
            timer.block(None, 1)
        self.assertEqual(events, ['audit/c03/block/0', 'pop', 'stop'])
        self.assertFalse(timer.collecting)

    def add_preparation_and_warning(self, path):
        con = sqlite3.connect(path)
        pid = 123 << 24
        con.execute("INSERT INTO NVTX_EVENTS VALUES (0,80,'prepare/c00',NULL,?)", (pid+7,))
        con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (5,7,?,99,2)', (pid+7,))
        con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10,20,?,99,0,1,7,1)', (pid,))
        con.execute('CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT)')
        con.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (2,'Not all CUDA events might have been collected.')")
        con.commit()
        con.close()

    def test_warning_and_prepare_metadata_preserved_and_bad_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'trace.sqlite'
            expected = self.fixture(p)
            self.add_preparation_and_warning(p)
            before = audit.sha256_file(p)
            metadata = analyze.read_trace_metadata(p)
            rows, _ = analyze.read_trace(p, expected)
            first = rows['audit/c00/block/0']
            self.assertEqual(first['kernel_count'], 2)  # preparation is not a measured block
            self.assertEqual(metadata['preparation']['prepare/c00']['kernel_count_in_envelope'], 1)
            self.assertEqual(len(metadata['warnings']), 1)
            self.assertAlmostEqual(analyze.validate_preparation(first, metadata, 'c00', .00005), .00002)
            with self.assertRaisesRegex(ValueError, 'Missing profiled preparation'):
                analyze.validate_preparation(first, metadata, 'c03', .00005)
            metadata['preparation']['prepare/c00']['end_ns'] = 110
            with self.assertRaisesRegex(ValueError, 'Invalid profiled preparation'):
                analyze.validate_preparation(first, metadata, 'c00', .00005)
            self.assertEqual(audit.sha256_file(p), before)

    def test_memset_before_first_kernel_is_counted_in_full_range(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'trace.sqlite'
            expected = self.fixture(p)
            con = sqlite3.connect(p)
            con.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,bytes INTEGER,value INTEGER)')
            con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (105,106,?,8,2)', ((123 << 24)+7,))
            con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (107,108,?,8,2048,0)', (123 << 24,))
            con.commit(); con.close()
            rows, _ = analyze.read_trace(p, expected)
            r = rows['audit/c00/block/0']
            self.assertEqual(r['recorded_transfer_counts_in_kernel_span'], {})
            self.assertEqual(r['recorded_transfers_correlated_to_nvtx'], {'CUPTI_ACTIVITY_KIND_MEMSET': 1})

    def test_exact_portal_allowed_and_original_snapshot_preserved(self):
        _, _, _, suite = audit.load_suite((audit.HERE.parent / 'gpu_sweep').resolve())
        snapshot = {'exit_code': 0, 'stdout':
                    '6470, /usr/libexec/xdg-desktop-portal-gnome, 9 MiB\n'
                    '11192, /usr/bin/ptyxis, 33 MiB\n'}
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'processes.json'
            with patch.object(suite, 'command_text', return_value=snapshot):
                audit.check_processes(suite, 'GPU-test', p)
            self.assertEqual(json.loads(p.read_text()), snapshot)
            decision = json.loads(p.with_name('processes_classification.json').read_text())
            self.assertEqual(len(decision['retained_portal']), 1)
            self.assertEqual(decision['blocked'], [])

    def test_portal_exception_does_not_hide_other_workloads(self):
        _, _, _, suite = audit.load_suite((audit.HERE.parent / 'gpu_sweep').resolve())
        for name in ('/tmp/xdg-desktop-portal-gnome',
                     '/usr/libexec/xdg-desktop-portal-gnome-worker',
                     '/home/eu/miniconda3/envs/ai_env/bin/python'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                p = Path(folder) / 'processes.json'
                snapshot = {'exit_code': 0, 'stdout':
                    '510, /usr/libexec/xdg-desktop-portal-gnome, 24 MiB\n'
                    f'511, {name}, 2000 MiB\n'}
                with patch.object(suite, 'command_text', return_value=snapshot):
                    with self.assertRaisesRegex(RuntimeError, 'Other GPU workloads'):
                        audit.check_processes(suite, 'GPU-test', p)
                decision = json.loads(p.with_name('processes_classification.json').read_text())
                self.assertIn(name, decision['blocked'][0])

    def test_portal_exception_retains_query_failure_and_malformed_row_checks(self):
        _, _, _, suite = audit.load_suite((audit.HERE.parent / 'gpu_sweep').resolve())
        for snapshot in ({'exit_code': 1, 'stdout': ''},
                         {'exit_code': 0, 'stdout': 'xdg-desktop-portal-gnome'}):
            with tempfile.TemporaryDirectory() as folder:
                with patch.object(suite, 'command_text', return_value=snapshot):
                    with self.assertRaises(RuntimeError):
                        audit.check_processes(suite, 'GPU-test', Path(folder)/'processes.json')

    def test_tracer_brackets_only_measured_blocks_and_preserves_calls(self):
        events = []
        class Timer:
            def __init__(self):
                self.p = {"blocks": 2}
            def block(self, fn, repeats, operations=1):
                events.append("original_block")
                for _ in range(repeats):
                    fn()
                return {"elapsed_ms": 10, "operations_per_call": operations}
            def prepare(self, fn, operations=1):
                self.block(fn, 3, operations)
                return 2, {"original_setup": True}
        torch = SimpleNamespace(cuda=SimpleNamespace(
            cudart=lambda: SimpleNamespace(cudaProfilerStart=lambda: events.append("start"),
                                           cudaProfilerStop=lambda: events.append("stop")),
            nvtx=SimpleNamespace(range_push=lambda label: events.append(label),
                                 range_pop=lambda: events.append("pop"))))
        timer = audit.traced_timer(Timer, torch, "c00")()
        repeats, setup = timer.prepare(lambda: events.append("gemm"))
        self.assertEqual(events, ["start", "prepare/c00", "original_block", "gemm", "gemm", "gemm", "pop"])
        self.assertIn("profiler_start_wall_ms", setup)
        rows = [timer.block(lambda: events.append("gemm"), repeats) for _ in range(2)]
        self.assertTrue(setup["original_setup"])
        self.assertEqual(events.count("gemm"), 7)
        self.assertEqual(events[-2:], ["pop", "stop"])
        self.assertEqual(rows[1]["trace_label"], "audit/c00/block/1")
        self.assertFalse(timer.collecting)

    def fixture(self, path, registered=False):
        con = sqlite3.connect(path)
        con.executescript('''
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER,demangledName INTEGER);
        ''')
        pid = 123 << 24
        con.executemany('INSERT INTO StringIds VALUES (?,?)',
                        [(1, 'gemm'), (2, 'cudaLaunchKernel'), (3, 'audit/c00/block/0')])
        con.execute('INSERT INTO NVTX_EVENTS VALUES (100,200,?,?,?)',
                    (None if registered else 'audit/c00/block/0', 3 if registered else None, pid + 7))
        con.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,2)',
                        [(110, 115, pid+7, 1), (160, 165, pid+7, 2)])
        con.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,0,1,7,1)',
                        [(120,130,pid,1), (170,180,pid,2), (122,129,456 << 24,1)])
        con.commit()
        con.close()
        return {'audit/c00/block/0': {'repeats': 2, 'operations_per_call': 1, 'elapsed_ms': .00008}}

    def test_registered_and_literal_nvtx_process_safe_correlation(self):
        with tempfile.TemporaryDirectory() as folder:
            for registered in (False, True):
                p = Path(folder) / f'{registered}.sqlite'
                expected = self.fixture(p, registered)
                before = audit.sha256_file(p)
                rows, digest = analyze.read_trace(p, expected)
                r = rows['audit/c00/block/0']
                self.assertEqual(r['kernel_count'], 2)
                self.assertEqual(r['kernel_union_ms'], .00002)
                self.assertAlmostEqual(r['inter_kernel_gap_fraction'], 2/3)
                self.assertEqual(r['largest_gaps'][0]['next_launch_start_minus_previous_end_us'], .03)
                self.assertEqual(digest, before)
                self.assertEqual(audit.sha256_file(p), before)

    def test_missing_kernels_rejected_not_passed(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'trace.sqlite'
            expected = self.fixture(p)
            expected['audit/c00/block/0']['repeats'] = 3
            with self.assertRaisesRegex(ValueError, 'Incomplete kernel trace'):
                analyze.read_trace(p, expected)

    def test_graph_one_launch_to_multiple_kernels(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'trace.sqlite'
            expected = self.fixture(p)
            con = sqlite3.connect(p)
            con.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=1')
            con.commit()
            con.close()
            expected['audit/c00/block/0'].update(repeats=1, operations_per_call=2)
            rows, _ = analyze.read_trace(p, expected)
            self.assertEqual(rows['audit/c00/block/0']['kernel_count'], 2)

    def test_union_does_not_double_count_concurrent_work(self):
        self.assertEqual(analyze.union_ns([(0,10), (5,15), (20,25)]), 20)

    def test_archive_preserves_final_exit_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'run'
            out.mkdir()
            (out / 'exit_code.txt').write_text('9\n')
            audit.package(out)
            archive = out.with_suffix('.zip')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read('run/exit_code.txt'), b'9\n')
            self.assertTrue(archive.with_suffix('.zip.sha256').read_text().startswith(audit.sha256_file(archive)))
            with self.assertRaises(FileExistsError):
                audit.package(out)

    def test_process_failure_is_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'exited 9'):
                audit.run_logged([sys.executable, '-c', 'print("diagnostic"); raise SystemExit(9)'], p, 'child', 5)
            self.assertEqual((p/'child.exit_code.txt').read_text(), '9\n')
            self.assertIn('diagnostic', (p/'child.log').read_text())

    def test_launcher_retains_wrapper_failure_and_closed_logs_in_zip(self):
        with tempfile.TemporaryDirectory(prefix='audit with spaces ') as folder:
            root = Path(folder)
            package = root / 'timing audit'
            suite = root / 'gpu suite'
            package.mkdir()
            suite.mkdir()
            for name in ('audit.py', 'launch.sh'):
                shutil.copy2(audit.HERE / name, package / name)
            wrapper = suite / 'run_research_checked.sh'
            wrapper.write_text('#!/bin/bash\necho mock-wrapper-failed\nexit 9\n')
            (package / 'baseline_v5.json').write_text(json.dumps(
                {'code_hashes': {'run_research_checked.sh': audit.sha256_file(wrapper)}}))
            result = subprocess.run(['bash', str(package/'launch.sh'), str(suite)],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
            archive = next((package/'runs').glob('*.zip'))
            with zipfile.ZipFile(archive) as z:
                name = archive.stem
                self.assertEqual(z.read(name+'/exit_code.txt'), b'9\n')
                self.assertIn(b'mock-wrapper-failed', z.read(name+'/launcher.log'))

    def test_end_to_end_analysis_and_missing_capture_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            traces = out / 'traces'
            traces.mkdir()
            self.fixture(traces/'capture.sqlite')
            self.add_preparation_and_warning(traces/'capture.sqlite')
            task = {'id': 'c00', 'shape': [64,64,64], 'dtype': 'fp16', 'mode': 'prealloc'}
            manifest = {'cases': [task], 'passes': list(audit.PASSES), 'diagnostic_version': 3,
                        'baseline': {'protocol': {'blocks': 1, 'settle_ms': .00005}}}
            (out/'manifest.json').write_text(json.dumps(manifest))
            for name in audit.PASSES:
                p = out / name
                p.mkdir()
                block = {'repeats': 2, 'operations_per_call': 1,
                         'elapsed_ms': .00008, 'latency_ms': .00004}
                if name == 'nsys':
                    block['trace_label'] = 'audit/c00/block/0'
                row = {'task': task, 'result': {'correctness': {'passed': True},
                    'timing': {'blocks': [block], 'median_ms': .00004, 'flags': [],
                               'setup': {'capture_started_before_prepare': True}, 'first_last_drift': 0}}}
                (p/'results.json').write_text(json.dumps([row]))
            analyze.analyze(out)
            summary = json.loads((out/'analysis.json').read_text())
            self.assertEqual(summary['traced_block_count'], 1)
            self.assertEqual(summary['conditions'][0]['nsys_to_plain_ratio'], 1)
            self.assertEqual(summary['collection_integrity'], 'PASS_WITH_WARNINGS')
            self.assertIn('Not all CUDA events', (out/'report.md').read_text())
            self.assertTrue((out/'report.md').is_file())
            (traces/'capture.sqlite').rename(traces/'preserved.sqlite.bak')
            with self.assertRaisesRegex(ValueError, 'Missing NVTX blocks'):
                analyze.analyze(out)


if __name__ == '__main__':
    unittest.main()
