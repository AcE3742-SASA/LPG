#!/usr/bin/env python3
"""Focused clock/timing diagnostic using unchanged v6 measurements."""
import argparse
import csv
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
PASSES = {"sparse_before": 1000, "dense_before": 100, "nsys": 100, "dense_after": 100, "sparse_after": 1000}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cases():
    specs = json.loads((HERE / "baseline_v6.json").read_text())["fixed_cases"]
    result = []
    for rd in range(2):
        indices = range(len(specs)) if rd == 0 else reversed(range(len(specs)))
        for i in indices:
            spec = specs[i]
            for mode in (("prealloc", "graph") if rd == 0 else ("graph", "prealloc")):
                result.append(dict(id=f"c{i}_{mode}_r{rd}", kind="gemm", shape=spec["shape"],
                                   dtype=spec["dtype"], mode=mode, round=rd,
                                   phase="clock_audit", split="diagnostic_only"))
    return result


def graph_nodes(task):
    if task["mode"] != "graph":
        return None
    for spec in json.loads((HERE / "baseline_v6.json").read_text())["fixed_cases"]:
        if spec["shape"] == task["shape"] and spec["dtype"] == task["dtype"]:
            return spec["graph_gemms_per_replay"]
    raise ValueError("Unknown Graph condition")


def verify_source(directory):
    expected = json.loads((HERE / "baseline_v6.json").read_text())
    for name, digest in expected["code_hashes"].items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"v6 source mismatch: {path}. Keep v6 unchanged.")
    return expected


def load_suite(directory):
    expected = verify_source(directory)
    sys.path.insert(0, str(directory))
    modules = [importlib.import_module(name) for name in ("plan", "measure", "suite")]
    for module in modules:
        if Path(module.__file__).resolve().parent != directory:
            raise RuntimeError("Imported a different suite: " + str(module.__file__))
    if modules[0].PROTOCOL != expected["protocol"]:
        raise RuntimeError("v6 protocol mismatch")
    return expected, *modules


def observed_timer(base):
    class ObservedTimer(base):
        def block(self, fn, repeats, operations=1):
            start = dt.datetime.now(dt.timezone.utc).isoformat()
            result = super().block(fn, repeats, operations)
            result['host_envelope_start_utc'] = start
            result['host_envelope_end_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            return result
    return ObservedTimer


def traced_timer(base, torch, case_id):
    """Only bracket the original Timer.block, never instrument each GEMM call."""
    class TracedTimer(base):
        collecting = False
        block_index = 0

        def prepare(self, fn, operations=1):
            started = time.perf_counter()
            torch.cuda.cudart().cudaProfilerStart()
            activation_ms = (time.perf_counter() - started) * 1000
            # Profiling must already be active throughout settling/calibration.
            torch.cuda.nvtx.range_push(f"prepare/{case_id}")
            try:
                try:
                    prepared = super().prepare(fn, operations)
                finally:
                    torch.cuda.nvtx.range_pop()
            except BaseException:
                torch.cuda.cudart().cudaProfilerStop()
                raise
            prepared[1]["profiler_start_wall_ms"] = activation_ms
            prepared[1]["capture_started_before_prepare"] = True
            self.collecting = True
            self.block_index = 0
            return prepared

        def block(self, fn, repeats, operations=1):
            if not self.collecting:
                return super().block(fn, repeats, operations)
            label = f"audit/{case_id}/block/{self.block_index}"
            torch.cuda.nvtx.range_push(label)
            try:
                try:
                    result = super().block(fn, repeats, operations)
                finally:
                    torch.cuda.nvtx.range_pop()
            except BaseException:
                self.collecting = False
                torch.cuda.cudart().cudaProfilerStop()
                raise
            result["trace_label"] = label
            self.block_index += 1
            if self.block_index == self.p["blocks"]:
                torch.cuda.cudart().cudaProfilerStop()
                self.collecting = False
            return result

    return TracedTimer


def check_processes(suite, gpu_uuid, path):
    snapshot = suite.command_text(["nvidia-smi", "-i", gpu_uuid,
                                   "--query-compute-apps=pid,process_name,used_memory",
                                   "--format=csv,noheader"])
    suite.atomic_json(path, dict(snapshot, diagnostic_pid=os.getpid()))
    others = suite.competing_processes(snapshot, os.getpid())
    if others:
        raise RuntimeError("Other GPU workloads are active: " + str(others))


def worker(args):
    import telemetry
    expected, plan, measure, suite = load_suite(args.suite)
    import torch
    out = args.output / args.pass_name
    out.mkdir()
    env = suite.environment(torch)
    suite.atomic_json(out / "environment.json", env)
    if suite.signature(env) != expected["environment_signature"]:
        raise RuntimeError("GPU/software/precision differs from v6; inspect environment.json.")
    if torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("Less than 4 GiB free GPU memory")
    check_processes(suite, env["gpu_uuid"], out / "processes_before.json")
    original_timer = measure.Timer
    base_timer = observed_timer(original_timer)
    rows = []
    with telemetry.collect(out, env["gpu_uuid"], PASSES[args.pass_name], suite) as monitor:
        with torch.inference_mode():
            x = torch.randn((1024, 1024), device="cuda") * .01
            y = torch.empty_like(x)
            timer = base_timer(torch, plan.PROTOCOL)
            started = time.perf_counter()
            print(f"{args.pass_name}: preheating {plan.PROTOCOL['preheat_seconds']} s", flush=True)
            while time.perf_counter() - started < plan.PROTOCOL["preheat_seconds"]:
                timer.block(lambda: torch.mm(x, x, out=y), 100)
            del x, y
            for index, task in enumerate(cases()):
                if monitor.poll() is not None:
                    raise RuntimeError("Telemetry process stopped")
                check_processes(suite, env["gpu_uuid"], out / f"processes_{task['id']}.json")
                measure.Timer = (traced_timer(base_timer, torch, task["id"])
                                 if args.pass_name == "nsys" else base_timer)
                print(f"{args.pass_name} [{index+1}/{len(cases())}] {task}", flush=True)
                start = suite.now()
                try:
                    result = measure.measure(torch, task, plan.PROTOCOL, graph_nodes=graph_nodes(task))
                finally:
                    measure.Timer = original_timer
                rows.append({"task": task, "result": result, "started_utc": start, "ended_utc": suite.now()})
                suite.atomic_json(out / "results.json", rows)
                if not result["correctness"]["passed"]:
                    raise RuntimeError("Correctness failed: " + task["id"])
        check_processes(suite, env["gpu_uuid"], out / "processes_after.json")
        suite.atomic_json(out / "gpu_after.json", suite.command_text(["nvidia-smi", "-q"]))


def run_logged(command, directory, name, timeout):
    """Own process group only; interruption never leaves a benchmark running."""
    (directory / f"{name}.command.json").write_text(json.dumps(command, indent=2) + "\n")
    code = 1
    with (directory / f"{name}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            (directory / f"{name}.exit_code.txt").write_text(str(process.returncode) + "\n")
    if code:
        raise RuntimeError(f"{name} exited {code}; inspect {directory / (name + '.log')}")


def package(out):
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, str(Path(out.name) / p.relative_to(out)))
    digest = sha256_file(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    print("RESULT_ZIP=" + str(archive), flush=True)


def worker_command(directory, out, pass_name):
    command = [sys.executable, "-u", "-B", str(HERE / "audit.py"), "worker",
               "--suite", str(directory), "--output", str(out), "--pass-name", pass_name]
    if pass_name == "nsys":
        command = ["nsys", "profile", "--sample=none", "--cpuctxsw=none",
                   "--trace=cuda,nvtx", "--cuda-graph-trace=node",
                   "--capture-range=cudaProfilerApi", f"--capture-range-end=repeat:{len(cases())}",
                   "--output", str(out / "traces/capture"), *command]
    return command


def run(args):
    expected, plan, measure, suite = load_suite(args.suite)
    out = args.output
    if (out / "manifest.json").exists():
        raise RuntimeError("Existing diagnostic output; choose a new directory.")
    out.mkdir(parents=True, exist_ok=True)
    snapshot = suite.command_text(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    lines = snapshot.get("stdout", "").strip().splitlines()
    if snapshot.get("exit_code") != 0 or len(lines) != 1 or lines[0].strip() != expected["environment_signature"]["gpu_uuid"]:
        raise RuntimeError("Cannot match target GPU UUID for the shared GPU lock")
    gpu_uuid = lines[0].strip()
    status = {"collection": "RUNNING", "interpretation": "NOT_REVIEWED"}
    files = [p for p in HERE.iterdir() if p.suffix in (".py", ".sh", ".json", ".md")]
    suite.atomic_json(out / "manifest.json", {"cases": cases(), "passes": PASSES,
                      "baseline": expected, "diagnostic_only": True, "diagnostic_version": 1,
                      "capture_order": "profiler_start_then_prepare_then_measure",
                      "source_hashes": {p.name: sha256_file(p) for p in files},
                      "note": "Separate five-process sparse/dense/NSYS comparison; no power or clock setting changes."})
    for src, names, destination in ((HERE, [p.name for p in files], out / "source"),
                                    (args.suite, expected["code_hashes"], out / "baseline_source")):
        for name in names:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, target)
    try:
        with suite.lock(Path("/tmp") / f"gemm-sweep-{gpu_uuid}.lock"):
            check_processes(suite, gpu_uuid, out / "processes_before.json")
            traces = out / "traces"
            traces.mkdir()
            for i, pass_name in enumerate(PASSES, 1):
                print(f"Starting {pass_name} ({i}/{len(PASSES)}); progress log: {out / (pass_name + '.log')}", flush=True)
                run_logged(worker_command(args.suite, out, pass_name), out, pass_name, timeout=1800)
                print(f"Finished {pass_name}: 8 conditions saved.", flush=True)
            reports = sorted(traces.glob("*.nsys-rep"))
            if len(reports) != len(cases()):
                raise RuntimeError(f"Expected {len(cases())} NSYS captures, found {len(reports)}")
            for i, report in enumerate(reports):
                run_logged(["nsys", "export", "--type=sqlite", "--output",
                            str(report.with_suffix(".sqlite")), str(report)],
                           out, f"export_{i:02d}", timeout=600)
            run_logged([sys.executable, "-B", str(HERE / "analyze.py"), str(out)], out, "analysis", timeout=300)
            verify_source(args.suite)
            status["collection"] = "COMPLETE"
            quality = json.loads((out / "summary.json").read_text())["gate"]
            status["measurement_quality"] = quality
            print(f"COLLECTION COMPLETE; MEASUREMENT QUALITY: {quality}; no automatic full sweep.", flush=True)
            return 0 if quality == "PASS" else 2
    except BaseException as error:
        status.update(collection="FAILED_OR_INTERRUPTED", error=str(error))
        raise
    finally:
        suite.atomic_json(out / "status.json", status)


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "run", "worker", "pack"))
    parser.add_argument("--suite", type=Path, default=HERE.parent / "gpu_sweep_v6")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pass-name", choices=PASSES)
    args = parser.parse_args()
    args.suite = args.suite.resolve()
    if args.output:
        args.output = args.output.resolve()
    if args.action == "verify":
        verify_source(args.suite)
        print("V6 SOURCE CHECK: PASS")
    elif args.action == "pack":
        package(args.output)
    elif args.action == "worker":
        worker(args)
    else:
        return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
