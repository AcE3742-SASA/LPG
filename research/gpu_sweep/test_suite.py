import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from measure import Timer, summarize
from plan import PROTOCOL, ROOT, digest, make_plan, task_seed
from report import export, records, summary
from suite import Store, atomic_json, code_fingerprint, competing_processes, lock, verify_pilot
from profiling import collect, ncu_has_kernel_data, nsys_has_kernel_data


class ProfilingTests(unittest.TestCase):
    wide = ('==WARNING== unmodified clocks\n'
            '"ID","Kernel Name","gpu__time_duration.sum"\n'
            '"","","nsecond"\n'
            '"0","gemm<16, 16>","1,234"\n')
    narrow = ('"ID","Kernel Name","Metric Name","Metric Value"\n'
              '"0","gemm","gpu__time_duration.sum","1234"\n')

    def test_csv_requires_actual_kernel_metric_in_both_formats(self):
        self.assertTrue(ncu_has_kernel_data(self.wide))
        self.assertTrue(ncu_has_kernel_data(self.narrow))
        for text in ('', '\n'.join(self.wide.splitlines()[:-1]),
                     self.wide.replace('"1,234"', '"N/A"'),
                     self.wide.replace('"1,234"', '"nan"'),
                     self.wide.replace('"gemm<16, 16>"', '""'),
                     self.narrow.replace('"1234"', '"nan"')):
            with self.subTest(text=text):
                self.assertFalse(ncu_has_kernel_data(text))

    def test_collect_accepts_both_extensions_and_keeps_failure_checks(self):
        task = {'id': 'profile-test', 'tool': 'ncu', 'cache': 'all'}
        cases = [('.ncu-repz', self.wide, 0, b'report', 'PASS'),
                 ('.ncu-rep', self.narrow, 0, b'report', 'PASS'),
                 ('.ncu-repz', self.wide, 1, b'report', 'FAIL'),
                 ('.ncu-repz', '\n'.join(self.wide.splitlines()[:-1]), 0, b'report', 'FAIL'),
                 ('.ncu-repz', self.wide, 0, b'', 'FAIL')]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            for suffix, text, code, data, status in cases:
                def popen(cmd, **kwargs):
                    Path(cmd[cmd.index('--export')+1] + suffix).write_bytes(data)
                    kwargs['stdout'].write(text)
                    return type('Process', (), {'wait': lambda self, timeout=None: code})()
                with self.subTest(suffix=suffix, code=code, status=status), \
                        patch('profiling.shutil.which', return_value='/mock/ncu'), \
                        patch('profiling.subprocess.Popen', side_effect=popen):
                    result = collect(task, out)
                    self.assertEqual(result['status'], status)
                    self.assertEqual(result['exit_code'], code)
            self.assertEqual(len(list((out/'profiles/profile-test').glob('attempt-*'))), 5)

    def test_nsys_requires_nonempty_kernel_rows(self):
        header = 'Time (%),Total Time (ns),Instances,Name\n'
        self.assertTrue(nsys_has_kernel_data(header + '100,"524,166",80,"gemm<16,16>"\n'))
        for text in (header, 'SKIPPED: report does not contain CUDA kernel data.',
                     header + '100,nan,80,gemm\n', header + '100,1234,0,gemm\n',
                     header + '100,0,80,gemm\n', header + '100,1234,80,\n'):
            self.assertFalse(nsys_has_kernel_data(text))

    def test_nsys_node_trace_and_stats_failure_are_preserved(self):
        header = 'Time (%),Total Time (ns),Instances,Name\n'
        task = {'id': 'nsys-test', 'tool': 'nsys', 'mode': 'graph'}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            for text, code, expected in ((header + '100,1234,80,gemm\n', 0, 'PASS'),
                                         (header, 0, 'FAIL'),
                                         (header + '100,1234,80,gemm\n', 1, 'FAIL')):
                def popen(cmd, **kwargs):
                    self.assertIn('--cuda-graph-trace=node', cmd)
                    Path(cmd[cmd.index('--output')+1] + '.nsys-rep').write_bytes(b'report')
                    return type('Process', (), {'wait': lambda self, timeout=None: 0})()
                def run(cmd, **kwargs):
                    kwargs['stdout'].write(text)
                    return type('Result', (), {'returncode': code})()
                with patch('profiling.shutil.which', return_value='/mock/nsys'), \
                        patch('profiling.subprocess.Popen', side_effect=popen), \
                        patch('profiling.subprocess.run', side_effect=run):
                    result = collect(task, out)
                self.assertEqual(result['status'], expected)
                self.assertEqual(result['stats_exit_code'], code)
                self.assertEqual((out/result['directory']/'stats_exit_code.txt').read_text().strip(), str(code))
            self.assertEqual(len(list((out/'profiles/nsys-test').glob('attempt-*'))), 3)


class LauncherTests(unittest.TestCase):
    def test_checked_runner_is_the_original_and_both_scripts_are_fingerprinted(self):
        sha = hashlib.sha256((ROOT / "run_research_checked.sh").read_bytes()).hexdigest()
        self.assertEqual(sha, "640538b8f9fc7e640dbc09120ae2989d2e2d842720239a6020d766802c586048")
        self.assertIn("run_research_checked.sh", code_fingerprint())
        self.assertIn("launch.sh", code_fingerprint())

    def make_launcher(self, folder):
        bundle = folder / "bundle with spaces"
        bundle.mkdir()
        shutil.copy2(ROOT / "launch.sh", bundle / "launch.sh")
        script = "#!/usr/bin/env bash\nset -eu\n" + shlex.quote(sys.executable) + " - \"$@\" <<'PY'\n"
        script += """import json, os, sys
from pathlib import Path
p = Path(os.environ['GEMM_RESEARCH_ROOT'])
p.mkdir(parents=True, exist_ok=True)
os.chdir(p)
(p/'received.json').write_text(json.dumps({'argv':sys.argv[1:], 'cwd':os.getcwd()}))
print('mock checked runner called')
raise SystemExit(int(os.environ.get('MOCK_EXIT', '0')))
PY
"""
        (bundle / "run_research_checked.sh").write_text(script)
        return bundle

    def test_delegate_paths_with_spaces_review_text_and_real_exit_status(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp).resolve()
            bundle = self.make_launcher(folder)
            for code in (0, 2, 17):
                with self.subTest(code=code):
                    output = f"output {code}"
                    cmd = ["bash", str(bundle / "launch.sh"), "full", output,
                           "--pilot", "pilot with spaces", "--review-note", "--pilot"]
                    result = subprocess.run(cmd, cwd=folder, env=dict(os.environ, MOCK_EXIT=str(code)),
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, code, result.stderr)
                    record = json.loads((folder / output / "checked/received.json").read_text())
                    self.assertEqual(record['argv'], ['python', '-u', str(bundle / 'full.py'),
                        '--output', str(folder / output), '--pilot', str(folder / 'pilot with spaces'),
                        '--review-note', '--pilot'])
                    self.assertEqual(record['cwd'], str(folder / output / 'checked'))
                    exits = list((folder / output / 'launcher').glob('*.exit_code.txt'))
                    self.assertEqual(len(exits), 1)
                    self.assertEqual(exits[0].read_text().strip(), str(code))

    def test_default_pilot_and_equals_pilot_argument(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp).resolve()
            bundle = self.make_launcher(folder)
            result = subprocess.run(['bash', str(bundle / 'launch.sh')], cwd=folder,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((bundle / 'runs/pilot/checked/received.json').exists())
            result = subprocess.run(['bash', str(bundle / 'launch.sh'), 'full', 'full',
                                     '--pilot=relative pilot', '--review-note', 'checked'],
                                    cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads((folder / 'full/checked/received.json').read_text())['argv']
            self.assertEqual(args[args.index('--pilot')+1], str(folder / 'relative pilot'))


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot = make_plan("pilot")
        cls.full = make_plan("full")

    def test_repeatable_unique_and_counts(self):
        self.assertEqual(digest(self.full), digest(make_plan("full")))
        self.assertEqual(len(self.pilot["tasks"]), 253)
        self.assertEqual(len(self.full["tasks"]), 42490)
        for plan in (self.pilot, self.full):
            self.assertEqual(len({t["id"] for t in plan["tasks"]}), len(plan["tasks"]))

    def test_holdout_is_fresh_and_absent_from_all_other_experiments(self):
        tasks = self.full["tasks"]
        held = {tuple(t["shape"]) for t in tasks if t["split"] == "fresh_holdout"}
        self.assertEqual(len(held), 1000)
        others = {tuple(t["shape"]) for t in tasks if "shape" in t and t["split"] != "fresh_holdout"}
        for t in tasks:
            if "widths" in t:
                others.update((t["batch"], n, k) for k, n in zip(t["widths"], t["widths"][1:]))
        self.assertFalse(held & others)
        self.assertFalse(held & {tuple(t["shape"]) for t in self.pilot["tasks"] if "shape" in t})

    def test_split_is_by_shape_and_development_is_not_test(self):
        per_shape = {}
        for t in self.full["tasks"]:
            if t.get("phase") == "main":
                s = tuple(t["shape"])
                per_shape.setdefault(s, set()).add(t["split"])
        self.assertTrue(all(len(s) == 1 for s in per_shape.values()))
        for t in self.pilot["tasks"]:
            if t.get("phase") == "main":
                self.assertEqual(per_shape[tuple(t["shape"])], {"development"})

    def test_identical_inputs_and_counterbalanced_dtype_order(self):
        t = next(t for t in self.pilot["tasks"] if t.get("phase") == "main")
        u = dict(t, round=5, mode="graph")
        self.assertEqual(task_seed(t), task_seed(u))
        for rd, expected in ((0, "fp32"), (1, "fp16")):
            first = next(t for t in self.pilot["tasks"] if t.get("phase") == "main" and t["round"] == rd)
            self.assertEqual(first["dtype"], expected)


class TimerTests(unittest.TestCase):
    def test_reported_ubuntu_desktop_clients_do_not_block(self):
        rows = ("5883, /usr/libexec/gnome-remote-desktop-daemon, 162 MiB\n"
                "9408, /usr/bin/nautilus, 25 MiB\n11192, /usr/bin/ptyxis, 33 MiB\n")
        self.assertEqual(competing_processes({"exit_code": 0, "stdout": rows}, 42), [])
        self.assertEqual(competing_processes({"exit_code": 0, "stdout": ""}, 42), [])

    def test_desktop_allowlist_does_not_hide_real_or_similarly_named_jobs(self):
        rows = ("9408, /usr/bin/nautilus, 25 MiB\n"
                "43, /home/eu/miniconda3/envs/ai_env/bin/python, 4096 MiB\n"
                "44, /tmp/ptyxis, 33 MiB\n45, /usr/bin/ptyxis-worker, 30 MiB\n"
                '46, "/tmp/name,with-comma", 10 MiB\n')
        self.assertEqual(competing_processes({"exit_code": 0, "stdout": rows}, 42),
                         rows.splitlines()[1:])

    def test_malformed_gpu_process_output_is_not_accepted_as_idle(self):
        with self.assertRaises(RuntimeError):
            competing_processes({"exit_code": 0, "stdout": "unexpected output"}, 42)

    def test_other_gpu_workload_is_not_silently_accepted(self):
        result = {"exit_code": 0, "stdout": "42, python, 1 MiB\n43, python, 4 MiB\n"}
        self.assertEqual(competing_processes(result, 42), ["43, python, 4 MiB"])
        with self.assertRaises(RuntimeError):
            competing_processes({"exit_code": 1, "stdout": ""}, 42)

    def test_measured_settle_and_recalibration_after_warmup(self):
        class ChangingTimer(Timer):
            calls = 0

            def block(self, fn, repeats, operations=1):
                # A slow initial probe, then a 5x faster steady clock/kernel.
                self.calls += 1
                latency = .01 if self.calls == 1 else .002
                return {"elapsed_ms": repeats*latency, "latency_ms": latency,
                        "repeats": repeats, "operations_per_call": operations}
        timer = ChangingTimer(None, PROTOCOL)
        repeats, data = timer.prepare(lambda: None)
        self.assertGreaterEqual(data["settle_event_ms"], PROTOCOL["settle_ms"])
        self.assertEqual(repeats, 50000)
        self.assertEqual(data["calibration"][-1]["elapsed_ms"], 100)

    def test_iqr_and_directional_drift_are_separate(self):
        good = [{"latency_ms": x} for x in [1, 1.001, .999, 1, 1, 1, 1]]
        self.assertEqual(summarize(good, PROTOCOL)["flags"], [])
        drift = [{"latency_ms": x} for x in [1, 1, 1, 1, 1, 1, 1.1]]
        self.assertEqual(summarize(drift, PROTOCOL)["flags"], ["first_last_drift"])
        with self.assertRaises(ValueError):
            summarize([{"latency_ms": 0}], PROTOCOL)

    def test_calibration_keeps_margin_above_short_block_threshold(self):
        class FasterTimer(Timer):
            calls = 0

            def block(self, fn, repeats, operations=1):
                self.calls += 1
                latency = .01 if self.calls == 1 else .0084
                return {'elapsed_ms': repeats*latency, 'latency_ms': latency,
                        'repeats': repeats, 'operations_per_call': operations}
        repeats, data = FasterTimer(None, PROTOCOL).prepare(lambda: None)
        self.assertGreater(repeats, 10000)
        self.assertGreaterEqual(data['calibration'][-1]['elapsed_ms'], 90)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.tasks = [{"id": f"task-{i}", "kind": "gemm", "phase": "main",
                       "shape": [64, 64, 64], "dtype": "fp32", "mode": "prealloc",
                       "split": "development", "round": i} for i in range(2)]
        self.plan = {"size": "pilot", "protocol": PROTOCOL, "tasks": self.tasks,
                     "shape_splits": [], "code_hashes": code_fingerprint()}
        atomic_json(self.out / "manifest.json", self.plan)

    def tearDown(self):
        self.temp.cleanup()

    def row(self, i, latency=1, status="PASS"):
        return {"task": self.tasks[i], "session": "test", "timestamp": "test",
                "status": status, "correctness": {"passed": status != "FAIL"},
                "timing": {"median_ms": latency, "relative_iqr": 0, "first_last_drift": 0,
                           "blocks": [{"latency_ms": latency}]*7, "flags": []}}

    def test_resume_preserves_failure_and_latest_attempt(self):
        store = Store(self.out)
        store.append(self.row(0))
        store.close()
        self.assertEqual(summary(self.out)[0]["gate"], "INCOMPLETE")
        store = Store(self.out)
        store.append(self.row(1, status="FAIL"))
        self.assertEqual(summary(self.out)[0]["gate"], "FAIL")
        store.append(self.row(1))
        store.close()
        result, rows = summary(self.out)
        self.assertEqual(result["gate"], "PASS")
        self.assertEqual(len(rows), 2)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(records(self.out)[1]["status"], "FAIL")

    def test_round_review_is_exported_even_with_stable_blocks(self):
        store = Store(self.out)
        store.append(self.row(0))
        store.append(self.row(1, 1.2))
        store.close()
        self.assertEqual(export(self.out)["gate"], "REVIEW")
        with (self.out / "review_queue.csv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 2)
        self.assertTrue((self.out / "report.html").is_file())
        self.assertEqual(len((self.out / "raw_attempts.jsonl").read_text().splitlines()), 2)

    def test_uncommitted_transaction_does_not_become_a_completed_task(self):
        store = Store(self.out)
        store.append(self.row(0))
        store.con.execute("INSERT INTO attempts (task_id, record) VALUES (?,?)",
                          ("task-1", json.dumps(self.row(1))))
        self.assertEqual(len(records(self.out)), 1)
        store.con.rollback()
        store.close()
        self.assertEqual(summary(self.out)[0]["missing"], 1)

    def test_manifest_mismatch_and_manual_pilot_gate(self):
        env = {k: "test" for k in ("gpu_uuid", "device_driver", "torch", "cuda_runtime", "precision", "environment")}
        atomic_json(self.out / "environment.json", env)
        with self.assertRaises(RuntimeError):
            verify_pilot(self.out, env, self.plan["code_hashes"], "")
        with self.assertRaises(RuntimeError):
            verify_pilot(self.out, env, self.plan["code_hashes"], "checked")
        store = Store(self.out)
        store.append(self.row(0)); store.append(self.row(1)); store.close()
        review = verify_pilot(self.out, env, self.plan["code_hashes"], "checked graphs")
        self.assertEqual(review["pilot_gate"], "PASS")
        with self.assertRaises(RuntimeError):
            verify_pilot(self.out, env, {"changed": "code"}, "checked")
        with self.assertRaises(RuntimeError):
            verify_pilot(self.out, dict(env, torch="changed"), self.plan["code_hashes"], "checked")

    def test_second_process_lock_and_nonfinite_diagnostics(self):
        with lock(self.out / ".lock"):
            with self.assertRaises(RuntimeError):
                with lock(self.out / ".lock"):
                    pass
        store = Store(self.out)
        row = self.row(0, status="FAIL")
        row["diagnostic"] = [math.nan, math.inf]
        store.append(row); store.close()
        self.assertEqual(records(self.out)[0]["diagnostic"], ["nan", "inf"])


if __name__ == "__main__":
    unittest.main()
