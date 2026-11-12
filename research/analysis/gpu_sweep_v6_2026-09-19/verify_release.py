"""Local release checks only; synthetic timings never represent GPU observations."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / "gpu_sweep_v6"
sys.path.insert(0, str(CODE))
import plan
import report
import suite


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = json.loads((ROOT / "timing_audit_v4/baseline_v5.json").read_text())
    for rel, wanted in baseline["code_hashes"].items():
        assert sha(ROOT / "gpu_sweep" / rel) == wanted, rel
    for path in CODE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    for path in CODE.glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    old = ast.parse((ROOT / "gpu_sweep/measure.py").read_text())
    new = ast.parse((CODE / "measure.py").read_text())
    def node(tree, name):
        return next(n for n in tree.body if getattr(n, "name", None) == name)
    unchanged = ("Timer", "check_mm", "make_gemm", "configure", "measure_chain", "measure_bandwidth")
    for name in unchanged:
        assert ast.dump(node(old, name)) == ast.dump(node(new, name)), name
    for rel in ("profiling.py", "full.py", "pilot.py", "inputs/legacy_shapes.csv",
                "inputs/review_shapes.csv", "inputs/provenance.json"):
        assert (ROOT / "gpu_sweep" / rel).read_bytes() == (CODE / rel).read_bytes(), rel

    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover",
                            "-s", str(CODE), "-p", "test_*.py"],
                           capture_output=True, text=True)
    (HERE / "unittest.log").write_text(tests.stdout + tests.stderr)
    assert tests.returncode == 0, (tests.stdout + tests.stderr)

    plans = {}
    for size in ("pilot", "full"):
        manifest = plan.make_plan(size)
        plans[size] = {
            "tasks": len(manifest["tasks"]),
            "task_digest": plan.digest(manifest["tasks"]),
            "phases": dict(Counter(t.get("phase", t["kind"]) for t in manifest["tasks"])),
            "paired_expected": len(report.paired_rows(manifest, [])),
            "splits": dict(Counter(r["split"] for r in manifest["shape_splits"])),
        }
        (HERE / (size + "_plan.json")).write_text(json.dumps(manifest, indent=2))

    # Positive integration check with the actual 192-task pilot plan, on temporary data.
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        manifest = plan.make_plan("pilot")
        manifest["code_hashes"] = suite.code_fingerprint()
        suite.atomic_json(out / "manifest.json", manifest)
        env = baseline["environment_signature"]
        suite.atomic_json(out / "environment.json", env)
        store = suite.Store(out)
        for task in manifest["tasks"]:
            latency = .004 if task["mode"] == "prealloc" else .002
            timing = dict(median_ms=latency, relative_iqr=0, first_last_drift=0, flags=[],
                          blocks=[dict(latency_ms=latency)] * 7)
            if task["mode"] == "graph":
                timing["graph_nodes"] = 64
            store.append(dict(task=task, timing=timing, input_seed=plan.task_seed(task),
                              status="PASS", session="SYNTHETIC", timestamp="SYNTHETIC",
                              correctness={"passed": True}))
        store.close()
        result = report.export(out)
        assert (result["gate"], result["recorded"], result["paired_complete"]) == ("PASS", 192, 96)
        approval = suite.verify_pilot(out, env, manifest["code_hashes"], "Synthetic gate test only")
        assert approval["pilot_gate"] == "PASS"
        choices = suite.graph_node_choices(report.records(out))
        common = [t for t in plan.make_plan("full")["tasks"]
                  if t["kind"] == "gemm" and t["mode"] == "graph" and plan.graph_key(t) in choices]
        assert all(suite.graph_nodes_for(t, choices) == 64 for t in common)

    result = {
        "status": "PASS", "python": sys.version, "gpu_executed": False,
        "unit_tests": 33, "bash_syntax": "PASS", "python_ast": "PASS",
        "v5_baseline_hashes_unchanged": baseline["code_hashes"],
        "unchanged_measure_nodes": unchanged,
        "v6_code_hashes": suite.code_fingerprint(), "plans": plans,
        "synthetic_pilot": {"recorded": 192, "pairs": 96, "full_pilot_gate": "PASS",
                            "matching_full_graph_tasks_reusing_choice": len(common)},
    }
    (HERE / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("v5_baseline_hashes_unchanged", "v6_code_hashes", "python")}, indent=2))


if __name__ == "__main__":
    main()
