#!/usr/bin/env python3
"""Read-only v5 reuse; separate timing diagnostics, never training labels."""
import argparse
import csv
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
PASSES = ("plain_before", "nsys", "plain_after")
PORTAL_PATH = "/usr/libexec/xdg-desktop-portal-gnome"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cases():
    specs = [(64, 64, 64, "fp16", "prealloc"),
             (64, 64, 64, "fp32", "prealloc"),
             (128, 128, 128, "fp16", "prealloc"),
             (3216, 520, 776, "fp16", "prealloc"),
             (4096, 4096, 4096, "fp16", "prealloc"),
             (4096, 4096, 4096, "fp32", "prealloc"),
             (64, 64, 64, "fp16", "graph"),
             (64, 64, 64, "fp32", "graph")]
    return [dict(id=f"c{i:02d}", kind="gemm", shape=[m, n, k], dtype=dtype,
                 mode=mode, round=0, phase="timing_audit", split="diagnostic_only")
            for i, (m, n, k, dtype, mode) in enumerate(specs)]


def verify_source(directory):
    expected = json.loads((HERE / "baseline_v5.json").read_text())
    for name, digest in expected["code_hashes"].items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"v5 source mismatch: {path}. Keep v5 unchanged.")
    return expected


def load_suite(directory):
    expected = verify_source(directory)
    sys.path.insert(0, str(directory))
    modules = [importlib.import_module(name) for name in ("plan", "measure", "suite")]
    for module in modules:
        if Path(module.__file__).resolve().parent != directory:
            raise RuntimeError("Imported a different suite: " + str(module.__file__))
    if modules[0].PROTOCOL != expected["protocol"]:
        raise RuntimeError("v5 protocol mismatch")
    return expected, *modules


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
    suite.atomic_json(path, snapshot)
    others = suite.competing_processes(snapshot, os.getpid())
    # v5 already validates the CSV. Extend only this diagnostic's desktop policy.
    portal = [line for line in others
              if next(csv.reader([line], skipinitialspace=True))[1].strip() == PORTAL_PATH]
    blocked = [line for line in others if line not in portal]
    suite.atomic_json(path.with_name(path.stem + "_classification.json"), {
        "policy": "timing_audit_v2_exact_gnome_portal_path",
        "retained_portal": portal, "blocked": blocked,
        "meaning": "Desktop process retained, not evidence that GPU interference is zero."})
    if portal:
        print("NOTE: GNOME portal retained in process log: " + str(portal), flush=True)
    if blocked:
        raise RuntimeError("Other GPU workloads are active: " + str(blocked))


def worker(args):
    expected, plan, measure, suite = load_suite(args.suite)
    import torch
    out = args.output / args.pass_name
    out.mkdir()
    env = suite.environment(torch)
    suite.atomic_json(out / "environment.json", env)
    if suite.signature(env) != expected["environment_signature"]:
        raise RuntimeError("GPU/software/precision differs from v5; inspect environment.json.")
    if torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("Less than 4 GiB free GPU memory")
    check_processes(suite, env["gpu_uuid"], out / "processes_before.json")
    base_timer = measure.Timer
    rows = []
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
            check_processes(suite, env["gpu_uuid"], out / f"processes_{task['id']}.json")
            measure.Timer = (traced_timer(base_timer, torch, task["id"])
                             if args.pass_name == "nsys" else base_timer)
            print(f"{args.pass_name} [{index+1}/{len(cases())}] {task}", flush=True)
            try:
                result = measure.measure(torch, task, plan.PROTOCOL)
            finally:
                measure.Timer = base_timer
            rows.append({"task": task, "result": result})
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


def run(args):
    expected, plan, measure, suite = load_suite(args.suite)
    out = args.output
    if (out / "manifest.json").exists():
        raise RuntimeError("Existing diagnostic output; choose a new directory.")
    out.mkdir(parents=True, exist_ok=True)
    snapshot = suite.command_text(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    lines = snapshot.get("stdout", "").strip().splitlines()
    if snapshot.get("exit_code") != 0 or len(lines) != 1 or not lines[0].startswith("GPU-"):
        raise RuntimeError("Expected one physical GPU; cannot acquire v5 GPU lock")
    gpu_uuid = lines[0].strip()
    if gpu_uuid != expected["environment_signature"]["gpu_uuid"]:
        raise RuntimeError("GPU UUID differs from v5")
    status = {"collection": "RUNNING", "interpretation": "NOT_REVIEWED"}
    suite.atomic_json(out / "manifest.json", {"cases": cases(), "passes": PASSES,
                      "baseline": expected, "diagnostic_only": True,
                      "diagnostic_version": 4,
                      "capture_order": "profiler_start_then_prepare_then_measure",
                      "process_policy": {"diagnostic_version": 4,
                          "additional_logged_desktop_path": PORTAL_PATH},
                      "note": "Unchanged v5 Timer.block/run/prepare; profiling starts before prepare. "
                              "Original c00-c07 task identities and adaptive repeats are preserved. "
                              "Eight prepare ranges and 56 measured blocks are traced separately. "
                              "All internal GPU activity gaps are split by next CUDA API entry/return."})
    for src, destination in ((HERE, out / "source"), (args.suite, out / "baseline_source")):
        names = ([p.name for p in src.iterdir() if p.suffix in (".py", ".sh", ".json", ".md")]
                 if src == HERE else list(expected["code_hashes"]))
        for name in names:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, target)
    telemetry = None
    try:
        with suite.lock(Path("/tmp") / f"gemm-sweep-{gpu_uuid}.lock"):
            check_processes(suite, gpu_uuid, out / "processes_before.json")
            traces = out / "traces"
            traces.mkdir()
            with (out / "telemetry.csv").open("w") as log:
                telemetry = subprocess.Popen(["nvidia-smi", "-i", gpu_uuid,
                    "--query-gpu=timestamp,uuid,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,power.limit",
                    "--format=csv", "--loop-ms=1000"], stdout=log, stderr=subprocess.STDOUT)
                try:
                    for pass_name in PASSES:
                        if telemetry.poll() is not None:
                            raise RuntimeError("Telemetry stopped; inspect telemetry.csv")
                        command = [sys.executable, "-u", "-B", str(HERE / "audit.py"), "worker",
                                   "--suite", str(args.suite), "--output", str(out),
                                   "--pass-name", pass_name]
                        if pass_name == "nsys":
                            command = ["nsys", "profile", "--sample=none", "--cpuctxsw=none",
                                "--trace=cuda,nvtx", "--cuda-graph-trace=node",
                                "--capture-range=cudaProfilerApi",
                                f"--capture-range-end=repeat:{len(cases())}",
                                "--output", str(traces / "capture"), *command]
                        print(f"Starting {pass_name}; progress log: {out / (pass_name + '.log')}", flush=True)
                        run_logged(command, out, pass_name, timeout=1800)
                finally:
                    telemetry.terminate()
                    telemetry.wait(timeout=15)
            reports = sorted(traces.glob("*.nsys-rep"))
            if len(reports) != len(cases()):
                raise RuntimeError(f"Expected {len(cases())} NSYS captures, found {len(reports)}")
            for i, report in enumerate(reports):
                run_logged(["nsys", "export", "--type=sqlite", "--output",
                            str(report.with_suffix(".sqlite")), str(report)],
                           out, f"export_{i:02d}", timeout=600)
            run_logged([sys.executable, "-B", str(HERE / "analyze.py"), str(out)],
                       out, "analysis", timeout=300)
            verify_source(args.suite)
            status["collection"] = "COMPLETE"
            print("COLLECTION COMPLETE; measurement quality still requires review.", flush=True)
    except BaseException as error:
        status.update(collection="FAILED_OR_INTERRUPTED", error=str(error))
        raise
    finally:
        suite.atomic_json(out / "status.json", status)
        # The shell launcher packages after its wrapper and logs have closed.


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "run", "worker", "pack"))
    parser.add_argument("--suite", type=Path, default=HERE.parent / "gpu_sweep")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pass-name", choices=PASSES)
    args = parser.parse_args()
    args.suite = args.suite.resolve()
    if args.output:
        args.output = args.output.resolve()
    if args.action == "verify":
        verify_source(args.suite)
        print("V5 SOURCE CHECK: PASS")
    elif args.action == "pack":
        package(args.output)
    elif args.action == "worker":
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
