#!/usr/bin/env python3
"""Pilot-first data acquisition. Use pilot.py / full.py or the subcommands below."""
import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from pathlib import Path

from plan import PROTOCOL, ROOT, digest, make_plan
from report import export, records, summary


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def code_fingerprint():
    paths = sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("*.sh")) + sorted((ROOT / "inputs").glob("*"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


@contextlib.contextmanager
def lock(path):
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is using this run/GPU: {path}") from None
        yield


def command_text(command):
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"command": command, "exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": command, "error": str(e)}


def environment(torch):
    from measure import configure
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run on the RTX computer using ai_env.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one target GPU with CUDA_VISIBLE_DEVICES.")
    prop = torch.cuda.get_device_properties(0)
    if PROTOCOL["expected_gpu"] not in prop.name:
        raise RuntimeError(f"Unexpected GPU {prop.name}; revise protocol and rerun pilot first.")
    if list(torch.cuda.get_device_capability(0)) != PROTOCOL["expected_capability"]:
        raise RuntimeError("Unexpected GPU capability")
    devices = command_text(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"])
    if devices.get("exit_code") != 0:
        raise RuntimeError("nvidia-smi query failed")
    device_uuid = str(getattr(prop, "uuid", ""))
    if device_uuid.startswith("GPU-") and device_uuid in devices["stdout"]:
        device_line = next(line for line in devices["stdout"].splitlines() if device_uuid in line)
    elif len(devices["stdout"].strip().splitlines()) == 1:
        device_line = devices["stdout"].strip()
        device_uuid = device_line.split(",")[0].strip()
    else:
        raise RuntimeError("Cannot match PyTorch GPU to nvidia-smi UUID")
    settings = configure(torch)
    packages = {}
    for name in ("torch", "numpy", "pandas", "scikit-learn", "torchvision"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"timestamp": now(), "python": sys.version, "executable": sys.executable,
            "os": platform.platform(), "gpu": prop.name, "gpu_uuid": device_uuid,
            "total_memory": prop.total_memory, "device_driver": device_line,
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "precision": settings, "packages": packages,
            "environment": {k: os.environ.get(k) for k in (
                "CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES", "CUDA_HOME",
                "CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING", "NVIDIA_TF32_OVERRIDE")},
            "tools": {tool: command_text([tool, "--version"]) for tool in ("nvcc", "ncu", "nsys")},
            "nvidia_smi": command_text(["nvidia-smi", "-q"])}


def signature(env):
    return {k: env[k] for k in ("gpu_uuid", "device_driver", "torch", "cuda_runtime", "precision", "environment")}


def competing_processes(snapshot, own_pid):
    if snapshot.get("exit_code") != 0:
        raise RuntimeError("GPU process query failed; cannot check competing workloads")
    # Observed Ubuntu desktop clients also appear in --query-compute-apps.
    # Exact paths only: unknown programs and other Python/CUDA jobs still block.
    desktop_paths = {
        "/usr/libexec/gnome-remote-desktop-daemon",
        "/usr/bin/nautilus",
        "/usr/bin/ptyxis",
    }
    others = []
    desktop = []
    for line in snapshot["stdout"].splitlines():
        if not line.strip():
            continue
        fields = next(csv.reader([line], skipinitialspace=True))
        if len(fields) != 3 or not fields[0].strip().isdigit():
            raise RuntimeError("Unexpected GPU process query row: " + line)
        pid, name = fields[0].strip(), fields[1].strip()
        if pid.isdigit() and int(pid) != own_pid:
            (desktop if name in desktop_paths else others).append(line)
    if desktop:
        print("NOTE: Desktop GPU clients retained in process log: " + str(desktop), flush=True)
    return others


class Store:
    def __init__(self, out):
        self.con = sqlite3.connect(out / "results.sqlite3")
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=FULL")
        self.con.execute("CREATE TABLE IF NOT EXISTS attempts (seq INTEGER PRIMARY KEY, task_id TEXT NOT NULL, record TEXT NOT NULL)")
        self.con.commit()

    def append(self, record):
        # Preserve non-finite diagnostics as explicit strings; never emit invalid JSON.
        def clean(v):
            if isinstance(v, float) and not math.isfinite(v):
                return str(v)
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v
        value = json.dumps(clean(record), ensure_ascii=False, allow_nan=False)
        with self.con:
            self.con.execute("INSERT INTO attempts (task_id,record) VALUES (?,?)", (record["task"]["id"], value))

    def close(self):
        self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.con.close()


def verify_pilot(path, env, fingerprints, review_note):
    if not review_note or not review_note.strip():
        raise RuntimeError("Read pilot report.html first; full requires --review-note with your decision.")
    result, _ = summary(path)
    pilot_manifest = json.loads((path / "manifest.json").read_text())
    if result["size"] != "pilot" or result["gate"] not in ("PASS", "REVIEW"):
        raise RuntimeError(f"Pilot is not complete/correct: {result['gate']}")
    if pilot_manifest["protocol"] != PROTOCOL or pilot_manifest["code_hashes"] != fingerprints:
        raise RuntimeError("Code/protocol changed since pilot. Run a new pilot directory.")
    pilot_env = json.loads((path / "environment.json").read_text())
    if signature(env) != signature(pilot_env):
        raise RuntimeError("GPU, precision or software differs from pilot. Rerun pilot.")
    return {"pilot_directory": str(path), "pilot_digest": result["result_digest"],
            "pilot_gate": result["gate"], "review_note": review_note, "timestamp": now()}


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / ".run.lock"):
        plan = make_plan(args.size)
        fingerprints = code_fingerprint()
        plan["code_hashes"] = fingerprints
        path = out / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            if digest(old) != digest(plan):
                raise RuntimeError("Manifest/code mismatch. Keep existing data and choose a new output directory.")
        else:
            atomic_json(path, plan)
            source = out / "source"
            source.mkdir(exist_ok=True)
            for rel in fingerprints:
                dest = source / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / rel, dest)
            shutil.copy2(ROOT / "README.md", source / "README.md")
        import torch
        env = environment(torch)
        if env["environment"].get("CUDA_LAUNCH_BLOCKING") not in (None, "", "0"):
            raise RuntimeError("CUDA_LAUNCH_BLOCKING is enabled; unset it before benchmarking.")
        if (out / "environment.json").exists():
            if signature(env) != signature(json.loads((out / "environment.json").read_text())):
                raise RuntimeError("Environment differs from this run; use a new pilot/run directory.")
        else:
            atomic_json(out / "environment.json", env)
        if args.size == "full":
            if not args.pilot:
                raise RuntimeError("Full requires --pilot PATH and --review-note TEXT.")
            approval = verify_pilot(args.pilot.resolve(), env, fingerprints, args.review_note)
            atomic_json(out / "pilot_review.json", approval)
        gpu_lock = Path("/tmp") / ("gemm-sweep-" + env["gpu_uuid"] + ".lock")
        with lock(gpu_lock):
            return acquire(args, out, plan, env, torch)


def acquire(args, out, plan, env, torch):
    from measure import Timer, measure
    from profiling import collect
    session = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    sessions = out / "sessions" / session
    sessions.mkdir(parents=True)
    atomic_json(sessions / "environment.json", env)
    store = Store(out)
    latest = {r["task"]["id"]: r for r in records(out)}
    if any(r["status"] == "FAIL" for r in latest.values()) and not args.retry_failed:
        store.close()
        raise RuntimeError("Earlier task failed. Inspect it; use --retry-failed only after fixing the cause.")
    telemetry = None
    telemetry_file = None
    stop = False
    old_handlers = {}

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print("STOP requested: finishing this task and saving it; no next task.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, request_stop)
    started = time.perf_counter()
    done = 0
    session_state = "RUNNING"
    try:
        telemetry_file = (sessions / "telemetry.csv").open("w")
        telemetry = subprocess.Popen([
            "nvidia-smi", "-i", env["gpu_uuid"],
            "--query-gpu=timestamp,uuid,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,power.limit",
            "--format=csv", "--loop-ms=1000"], stdout=telemetry_file, stderr=subprocess.STDOUT)
        time.sleep(.2)
        if telemetry.poll() is not None:
            raise RuntimeError("Telemetry did not start; inspect sessions/*/telemetry.csv")
        # Save competing process information; do not kill or modify other users' workloads.
        processes = command_text(["nvidia-smi", "-i", env["gpu_uuid"],
                                  "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        atomic_json(sessions / "gpu_processes.json", processes)
        others = competing_processes(processes, os.getpid())
        if others:
            raise RuntimeError("Other GPU compute processes are active: " + str(others))
        free, total = torch.cuda.mem_get_info()
        if free < 4*1024**3:
            raise RuntimeError("Less than 4 GiB free GPU memory")
        with torch.inference_mode():
            x = torch.randn((1024, 1024), device="cuda")*.01
            y = torch.empty_like(x)
            timer = Timer(torch, PROTOCOL)
            before = time.perf_counter()
            print(f"Preheating GPU for {PROTOCOL['preheat_seconds']} seconds; telemetry is recording.", flush=True)
            while time.perf_counter()-before < PROTOCOL["preheat_seconds"]:
                timer.block(lambda: torch.mm(x, x, out=y), 100)
            del x, y
        for i, task in enumerate(plan["tasks"]):
            previous = latest.get(task["id"])
            if previous and not (args.retry_failed and previous["status"] == "FAIL"):
                continue
            if stop:
                break
            if telemetry.poll() is not None:
                raise RuntimeError("Telemetry stopped unexpectedly")
            if done and done % 100 == 0:
                snapshot = command_text(["nvidia-smi", "-i", env["gpu_uuid"],
                                         "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
                atomic_json(sessions / f"gpu_processes_{done:06d}.json", snapshot)
                others = competing_processes(snapshot, os.getpid())
                if others:
                    raise RuntimeError("New competing GPU process detected: " + str(others))
            print(f"[{i+1}/{len(plan['tasks'])}] {task['kind']} {task.get('shape', task.get('widths', task.get('mib')))} {task.get('dtype','')} {task['mode']} r{task.get('round','-')}", flush=True)
            atomic_json(out / "progress.json", {"session": session, "timestamp": now(), "current": task,
                                               "completed_this_session": done, "state": "RUNNING"})
            begin = time.perf_counter()
            row = {"task": task, "session": session, "timestamp": now()}
            try:
                if task["kind"] == "profile":
                    torch.cuda.empty_cache()
                    data = collect(task, out)
                else:
                    data = measure(torch, task, PROTOCOL)
                    data["status"] = ("FAIL" if not data["correctness"]["passed"] else
                                      "REVIEW" if data["timing"]["flags"] else "PASS")
                row.update(data)
            except Exception as exc:
                row.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
            row["task_wall_seconds"] = time.perf_counter()-begin
            store.append(row)
            latest[task["id"]] = row
            done += 1
            print(f"  {row['status']} ({row['task_wall_seconds']:.1f}s)", flush=True)
            # CUDA/accuracy errors need diagnosis; never let them poison a whole sweep.
            if row["status"] == "FAIL":
                print("  Error: " + row.get("error", "Numerical correctness check failed"), flush=True)
                if row.get("directory"):
                    print(f"  Profiler log: {out / row['directory'] / 'output.log'}", flush=True)
                stop = True
                break
        session_state = "STOPPED" if stop else "FINISHED"
    except BaseException:
        session_state = "ERROR"
        raise
    finally:
        if telemetry is not None and telemetry.poll() is None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill(); telemetry.wait()
        if telemetry_file:
            telemetry_file.close()
        atomic_json(sessions / "gpu_after.json", command_text(["nvidia-smi", "-q"]))
        store.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        atomic_json(out / "progress.json", {"session": session, "timestamp": now(),
                                           "state": session_state, "completed_this_session": done})
        atomic_json(sessions / "session.json", {"state": session_state,
                    "elapsed_seconds": time.perf_counter()-started, "new_records": done})
        export(out)
    result, _ = summary(out)
    code = 0 if result["gate"] == "PASS" else 2 if result["gate"] == "REVIEW" else 3
    (sessions / "exit_code.txt").write_text(str(code)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "between_round"}, indent=2))
    print(f"Report: {out / 'report.html'}\nNo automatic transition to full sweep.")
    return code


def pack(out):
    out = out.resolve()
    with lock(out / ".run.lock"):
        export(out)
        target = out.with_name(out.name + "_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".zip")
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file() and p.name != ".run.lock":
                    z.write(p, str(Path(out.name) / p.relative_to(out)))
        with target.open("rb") as f:
            checksum = hashlib.file_digest(f, "sha256").hexdigest()
        target.with_suffix(".zip.sha256").write_text(f"{checksum}  {target.name}\n")
        print(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("size", choices=("pilot", "full"))
    rp.add_argument("--output", type=Path, required=True)
    rp.add_argument("--pilot", type=Path)
    rp.add_argument("--review-note")
    rp.add_argument("--retry-failed", action="store_true")
    dp = sub.add_parser("plan")
    dp.add_argument("size", choices=("pilot", "full"))
    dp.add_argument("--save", type=Path)
    for name in ("status", "report", "pack"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "plan":
        from collections import Counter
        value = make_plan(args.size)
        if args.save:
            atomic_json(args.save, value)
        print(json.dumps({"size": args.size, "tasks": len(value["tasks"]),
                          "counts": dict(Counter(t.get("phase", t["kind"]) for t in value["tasks"])),
                          "protocol": PROTOCOL}, indent=2))
    elif args.command == "status":
        result, _ = summary(args.output.resolve())
        print(json.dumps({k: v for k, v in result.items() if k != "between_round"}, indent=2))
    elif args.command == "report":
        with lock(args.output / ".run.lock"):
            print(export(args.output.resolve())["gate"])
    elif args.command == "pack":
        pack(args.output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
