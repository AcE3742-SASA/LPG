"""Separate profiler subprocesses; reported times are never benchmark labels."""
import argparse
import csv
import json
import math
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path

from plan import PROTOCOL, ROOT


def ncu_has_kernel_data(text):
    """Accept raw (wide) and metric-per-row CSV, never a header alone."""
    header = None
    for row in csv.reader(text.splitlines()):
        if "ID" in row and "Kernel Name" in row:
            header = row
            continue
        if not header or len(row) != len(header):
            continue
        values = dict(zip(header, row))
        if not values["ID"].strip().isdigit() or not values["Kernel Name"].strip():
            continue
        if "Metric Name" in values:
            metrics = [values.get("Metric Value", "")] if values["Metric Name"] else []
        else:
            metrics = [v for k, v in values.items() if k.startswith(("gpu__", "sm__", "dram__"))]
        for value in metrics:
            try:
                if math.isfinite(float(value.replace(",", ""))):
                    return True
            except ValueError:
                pass
    return False


def nsys_has_kernel_data(text):
    """Require a positive kernel count and duration, not just CSV column names."""
    header = None
    for row in csv.reader(text.splitlines()):
        if all(name in row for name in ('Instances', 'Total Time (ns)', 'Name')):
            header = row
            continue
        if not header or len(row) != len(header):
            continue
        values = dict(zip(header, row))
        try:
            instances = int(values['Instances'].replace(',', ''))
            duration = float(values['Total Time (ns)'].replace(',', ''))
            if values['Name'].strip() and instances > 0 and math.isfinite(duration) and duration > 0:
                return True
        except ValueError:
            pass
    return False


def collect(task, out):
    directory = out / "profiles" / task["id"]
    directory.mkdir(parents=True, exist_ok=True)
    # Each retry gets its own directory; never overwrite an earlier failed report.
    attempt = len(list(directory.glob("attempt-*"))) + 1
    directory = directory / f"attempt-{attempt:03d}"
    directory.mkdir()
    spec = directory / "task.json"
    spec.write_text(json.dumps(task, indent=2))
    exe = shutil.which(task["tool"])
    if exe is None:
        return {"status": "FAIL", "error": f"{task['tool']} not found on PATH"}
    child = [sys.executable, str(ROOT / "profiling.py"), "--worker", str(spec)]
    if task["tool"] == "ncu":
        cmd = [exe, "--profile-from-start", "off", "--target-processes", "all",
               "--clock-control", "none", "--cache-control", task["cache"],
               "--section", "SpeedOfLight", "--section", "MemoryWorkloadAnalysis",
               "--section", "ComputeWorkloadAnalysis", "--section", "Occupancy",
               "--csv", "--page", "raw", "--export", str(directory / "capture"), *child]
        reports = [directory / name for name in ("capture.ncu-rep", "capture.ncu-repz")]
    else:
        cmd = [exe, "profile", "--sample=none", "--cpuctxsw=none", "--trace=cuda,nvtx",
               "--cuda-graph-trace=node",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
               "--output", str(directory / "capture"), *child]
        reports = [directory / "capture.nsys-rep"]
    (directory / "command.json").write_text(json.dumps(cmd, indent=2))
    with (directory / "output.log").open("w") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    (directory / "exit_code.txt").write_text(str(code) + "\n")
    report = next((p for p in reports if p.is_file() and p.stat().st_size > 0), None)
    complete = code == 0 and report is not None
    # A trace file alone is insufficient: require an actual profiled kernel as well.
    if complete and task["tool"] == "ncu":
        text = (directory / "output.log").read_text(errors="replace")
        complete = ncu_has_kernel_data(text)
    stats_code = None
    if complete and task["tool"] == "nsys":
        with (directory / "kernel_summary.csv").open("w") as log:
            result = subprocess.run([exe, "stats", "--report", "cuda_gpu_kern_sum",
                                     "--format", "csv", str(report)], stdout=log,
                                    stderr=subprocess.STDOUT, timeout=300)
        text = (directory / "kernel_summary.csv").read_text(errors="replace")
        stats_code = result.returncode
        (directory / "stats_exit_code.txt").write_text(str(stats_code) + "\n")
        complete = stats_code == 0 and nsys_has_kernel_data(text)
    return {"status": "PASS" if complete else "FAIL", "exit_code": code,
            "stats_exit_code": stats_code,
            "directory": str(directory.relative_to(out)),
            "report": str(report.relative_to(out)) if report else None,
            "error": "" if complete else "Profiler failed or no kernel data; inspect output.log and kernel_summary.csv (nsys)",
            "timing_use": "diagnostic_only"}


def worker(path):
    import torch
    from measure import Timer, capture, configure, make_chain, make_gemm
    configure(torch)
    task = json.loads(Path(path).read_text())
    with torch.inference_mode():
        if "shape" in task:
            a, b, out, fn = make_gemm(torch, task)
        else:
            xs, weights, parts, fn = make_chain(torch, task)
        timer = Timer(torch, PROTOCOL)
        timer.prepare(fn)
        if task["mode"] == "graph":
            graph, nodes = capture(torch, fn, timer)
            fn = graph.replay
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            for _ in range(1 if task["tool"] == "ncu" else 20):
                fn()
            torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    worker(parser.parse_args().worker)
