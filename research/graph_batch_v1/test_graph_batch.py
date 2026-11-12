"""CPU checks for exact batching, normalization and conservative plateau decisions."""
import contextlib
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import graph_batch as gb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gpu_sweep'))
import measure
import suite
from plan import PROTOCOL, task_seed


def rows_for(times=None):
    times = times or {16: .002, 32: .002, 64: .002, 128: .002}
    rows = []
    for task in gb.tasks():
        batch = task['gemms_per_replay']
        latency = times[batch]
        blocks = [dict(repeats=4000, operations_per_call=batch,
                       elapsed_ms=latency * 4000 * batch, latency_ms=latency)
                  for _ in range(PROTOCOL['blocks'])]
        timing = dict(blocks=blocks, setup={'settle_event_ms': 3000},
                      **measure.summarize(blocks, PROTOCOL))
        rows.append(dict(task=task, result=dict(timing=timing,
                    input_seed=task_seed(task), correctness={'passed': True},
                    measured_gemm_calls=4000*batch*PROTOCOL['blocks'], tensor_addresses=[100, 200, 300])))
    return rows


class GraphBatchTests(unittest.TestCase):
    def test_balanced_positions_and_transitions(self):
        tasks = gb.tasks()
        self.assertEqual(len(tasks), 48)
        self.assertEqual(len({t['id'] for t in tasks}), 48)
        for case in gb.CASES:
            orders = [[t['gemms_per_replay'] for t in tasks
                       if t['round'] == rd and t['case_id'] == case['id']]
                      for rd in range(4)]
            for position in range(4):
                self.assertEqual({o[position] for o in orders}, set(gb.BATCHES))
            self.assertEqual(len({pair for o in orders for pair in zip(o, o[1:])}), 12)
        for case in gb.CASES:
            self.assertEqual(len({task_seed(t) for t in tasks if t['case_id'] == case['id']}), 1)

    def test_capture_contains_exact_number_of_gemms(self):
        calls = []
        state = {'capture': False}
        @contextlib.contextmanager
        def capture(graph):
            state['capture'] = True
            yield
            state['capture'] = False
        stream = SimpleNamespace(wait_stream=lambda _: None)
        graph = SimpleNamespace(replay=lambda: calls.append('replay'))
        torch = SimpleNamespace(cuda=SimpleNamespace(
            Stream=lambda: stream, current_stream=lambda: stream,
            stream=lambda _: contextlib.nullcontext(), synchronize=lambda: None,
            CUDAGraph=lambda: graph, graph=capture))
        for batch in gb.BATCHES:
            calls.clear()
            self.assertIs(gb.capture_fixed(torch, lambda: calls.append(state['capture']), batch), graph)
            self.assertEqual(calls.count(True), batch)
            self.assertEqual(calls.count(False), 3)
            self.assertEqual(calls[-1], 'replay')
        with self.assertRaises(ValueError):
            gb.capture_fixed(torch, lambda: None, 0)

    def test_all_pairs_must_converge_and_128_cannot_self_certify(self):
        analysis = gb.analyze_rows(rows_for(), measure, PROTOCOL)
        self.assertEqual([c['candidate_gemms_per_replay'] for c in analysis['cases']], [16]*3)
        # 64 and 128 agree; smaller batches still change by more than 1%.
        analysis = gb.analyze_rows(rows_for({16: .004, 32: .0021, 64: .002, 128: .002}), measure, PROTOCOL)
        self.assertEqual([c['candidate_gemms_per_replay'] for c in analysis['cases']], [64]*3)
        analysis = gb.analyze_rows(rows_for({16: .004, 32: .003, 64: .002, 128: .0015}), measure, PROTOCOL)
        self.assertTrue(all(c['candidate_gemms_per_replay'] is None for c in analysis['cases']))

    def test_one_round_outside_margin_blocks_candidate(self):
        rows = rows_for()
        for row in rows:
            if row['task']['round'] == 3 and row['task']['gemms_per_replay'] == 128:
                for block in row['result']['timing']['blocks']:
                    block['latency_ms'] *= 1.02
                    block['elapsed_ms'] *= 1.02
                row['result']['timing'].update(measure.summarize(row['result']['timing']['blocks'], PROTOCOL))
        a = gb.analyze_rows(rows, measure, PROTOCOL)
        self.assertTrue(all(c['candidate_gemms_per_replay'] is None for c in a['cases']))

    def test_incomplete_duplicate_or_bad_normalization_is_rejected(self):
        good = rows_for()
        variants = [good[:-1], good + [good[0]]]
        for field, value in [('operations_per_call', 1), ('latency_ms', .02), ('elapsed_ms', float('nan'))]:
            bad = copy.deepcopy(good)
            bad[0]['result']['timing']['blocks'][0][field] = value
            variants.append(bad)
        for rows in variants:
            with self.subTest():
                with self.assertRaises(ValueError):
                    gb.analyze_rows(rows, measure, PROTOCOL)

    def test_stability_review_is_not_hidden_by_plateau(self):
        rows = rows_for()
        for row in rows:
            if row['task']['round'] == 3:
                for block in row['result']['timing']['blocks']:
                    block['latency_ms'] *= 1.1
                    block['elapsed_ms'] *= 1.1
                row['result']['timing'].update(measure.summarize(row['result']['timing']['blocks'], PROTOCOL))
        a = gb.analyze_rows(rows, measure, PROTOCOL)
        self.assertEqual(a['stability'], 'REVIEW')
        self.assertTrue(all(c['candidate_gemms_per_replay'] is None for c in a['cases']))

    def test_wrong_seed_and_correctness_failure_rejected(self):
        for change in ('seed', 'correctness', 'count', 'addresses'):
            rows = rows_for()
            if change == 'seed':
                rows[0]['result']['input_seed'] += 1
            elif change == 'correctness':
                rows[0]['result']['correctness']['passed'] = False
            elif change == 'count':
                rows[0]['result']['measured_gemm_calls'] += 1
            else:
                rows[0]['result']['tensor_addresses'][0] += 1
            with self.assertRaises(ValueError):
                gb.analyze_rows(rows, measure, PROTOCOL)

    def test_report_generation_preserves_48_rows_and_336_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            rows = rows_for()
            for rd in range(4):
                directory = out / f'round_{rd}'
                directory.mkdir()
                (directory / 'results.json').write_text(json.dumps([r for r in rows if r['task']['round'] == rd]))
            a = gb.analyze(out, measure, SimpleNamespace(PROTOCOL=PROTOCOL), suite)
            self.assertEqual((a['measurements'], a['blocks']), (48, 336))
            self.assertEqual(json.loads((out / 'analysis.json').read_text()), a)
            self.assertEqual(len((out / 'summary.csv').read_text().splitlines()), 13)
            self.assertIn('신뢰구간', (out / 'REPORT.md').read_text())

    def test_launcher_source_failure_still_packages_logs_and_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / 'graph_batch_v1'
            dest.mkdir()
            for name in ('launch.sh', 'graph_batch.py', 'audit_support.py', 'baseline_v5.json'):
                shutil.copy2(Path(gb.__file__).parent / name, dest / name)
            baseline = Path(temp) / 'gpu_sweep'
            baseline.mkdir()
            completed = subprocess.run(['bash', str(dest / 'launch.sh'), str(baseline)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            archive = next((dest / 'runs').glob('*.zip'))
            self.assertIn('RESULT_ZIP=', completed.stdout)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read(archive.stem + '/exit_code.txt').strip(), b'1')
                self.assertIn(b'v5 source mismatch', z.read(archive.stem + '/source_check.log'))
            self.assertTrue(archive.with_suffix('.zip.sha256').is_file())


if __name__ == '__main__':
    unittest.main()
