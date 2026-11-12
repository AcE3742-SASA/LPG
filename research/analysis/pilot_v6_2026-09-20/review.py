"""Verify user-supplied v6 observations without running uploaded Python code."""
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import statistics as stats
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / "raw/pilot_v6"
ZIP = Path("/Users/studyhard/Desktop/Downloads/Unsorted:Unsure/pilot_v6_20260920_112624_b5ed79.zip")
sys.path.insert(0, str(ROOT / "gpu_sweep_v6"))
import plan
import report
import suite


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def main():
    assert sha(ZIP.read_bytes()) == "082d6f113aabf70391f79984673c50b988aa2e89ddca518c85d5bc6cefce9e4b"
    hashes = json.loads((HERE / "raw_sha256.json").read_text())
    def check_raw():
        actual = {"pilot_v6/" + str(p.relative_to(RAW)): sha(p.read_bytes())
                  for p in RAW.rglob("*") if p.is_file()}
        assert actual == hashes
    check_raw()
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        assert {p: sha(z.read(p)) for p in z.namelist()} == hashes
    manifest = json.loads((RAW / "manifest.json").read_text())
    expected = plan.make_plan("pilot")
    expected["code_hashes"] = suite.code_fingerprint()
    assert manifest == json.loads(json.dumps(expected))
    release = ROOT / "gpu_sweep_2026-09-19_v6.zip"
    assert sha(release.read_bytes()) == "210b9eb91820b0c16f12d118b7a27fcd3202de39837c37b1a3446f62a52916b0"
    with zipfile.ZipFile(release) as z:
        for name, value in manifest["code_hashes"].items():
            assert sha((RAW / "source" / name).read_bytes()) == value
            assert z.read("gpu_sweep_v6/" + name) == (RAW / "source" / name).read_bytes()
        assert z.read("gpu_sweep_v6/README.md") == (RAW / "source/README.md").read_bytes()
    assert (RAW / "results.sqlite3-wal").stat().st_size == 0
    con = sqlite3.connect((RAW / "results.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        rows = [json.loads(x[0]) for x in con.execute("SELECT record FROM attempts ORDER BY seq")]
    finally:
        con.close()
    assert rows == [json.loads(s) for s in (RAW / "raw_attempts.jsonl").read_text().splitlines()]
    saved = json.loads((RAW / "summary.json").read_text())
    assert saved["result_digest"] == plan.digest(rows)
    assert len(rows) == len({r["task"]["id"] for r in rows}) == 192
    assert [r["task"] for r in rows] == manifest["tasks"]
    env = json.loads((RAW / "environment.json").read_text())
    baseline = json.loads((ROOT / "timing_audit_v4/baseline_v5.json").read_text())
    assert suite.signature(env) == baseline["environment_signature"]
    for path in RAW.glob("sessions/*/environment.json"):
        assert json.loads(path.read_text()) == env
    choices = suite.graph_node_choices(rows)
    assert len(choices) == 48
    blocks, groups, correctness = [], defaultdict(list), Counter()
    operations = 0
    for row in rows:
        task, tm, cc = row["task"], row["timing"], row["correctness"]
        assert row["input_seed"] == plan.task_seed(task)
        assert cc["passed"] and cc["finite_all"] and cc["max_tolerance_ratio"] <= 1
        expected_full = task["shape"][0] * task["shape"][1] <= 4096
        assert cc["scope"] == ("full" if expected_full else "sampled")
        assert cc["points"] == len(cc["rows"]) * len(cc["cols"])
        assert [cc["atol"], cc["rtol"]] == manifest["protocol"]["tolerances"][task["dtype"]]
        correctness[cc["scope"]] += 1
        assert len(tm["blocks"]) == 7
        factor = tm["graph_nodes"] if task["mode"] == "graph" else 1
        if task["mode"] == "graph":
            assert suite.graph_nodes_for(task, choices) == factor
            sizing = tm["graph_batch_sizing"]
            probe = sizing["probe"]
            if tm["graph_nodes_selection"] == "initial_probe":
                assert probe is not None
                assert factor == max(1, min(64, math.ceil(2 / probe["latency_ms"])))
            else:
                assert tm["graph_nodes_selection"] == "fixed_or_reused" and probe is None
        values = []
        for phase, observations in [("settle", tm["setup"]["settle_blocks"]),
                                     ("calibration", tm["setup"]["calibration"]),
                                     ("timed", tm["blocks"])]:
            for i, block in enumerate(observations):
                assert block["operations_per_call"] == factor and block["repeats"] >= 1
                value = block["elapsed_ms"] / block["repeats"] / factor
                close(value, block["latency_ms"])
                close(block["host_latency_ms"], block["host_wall_ms"] / block["repeats"] / factor)
                assert math.isfinite(value) and value > 0
                blocks.append(dict(id=task["id"], phase=phase, block=i, **block))
                if phase == "timed":
                    values.append(value)
                    operations += block["repeats"] * factor
        close(sum(b["elapsed_ms"] for b in tm["setup"]["settle_blocks"]), tm["setup"]["settle_event_ms"])
        assert tm["setup"]["settle_event_ms"] >= 3000
        med = stats.median(values)
        q1, _, q3 = stats.quantiles(values, n=4, method="inclusive")
        iqr, drift = (q3 - q1) / med, (values[-1] - values[0]) / values[0]
        for name, value in dict(median_ms=med, relative_iqr=iqr, first_last_drift=drift,
                                min_ms=min(values), max_ms=max(values)).items():
            close(tm[name], value)
        flags = (["relative_iqr"] if iqr > .05 else []) + (["first_last_drift"] if abs(drift) > .05 else [])
        if any(b["elapsed_ms"] < 80 for b in tm["blocks"]):
            flags.append("short_block")
        assert tm["flags"] == flags
        assert row["status"] == ("REVIEW" if flags else "PASS")
        m, n, k = task["shape"]
        assert row["flops"] == 2 * m * n * k
        assert row["ideal_bytes"] == (m*k + k*n + m*n) * (4 if task["dtype"] == "fp32" else 2)
        close(row["ideal_oi"], row["flops"] / row["ideal_bytes"])
        close(row["tflops"], row["flops"] / (med * 1e9))
        groups[(tuple(task["shape"]), task["dtype"], task["mode"])].append(row)
    between = []
    for (shape, dtype, mode), group in groups.items():
        assert [r["task"]["round"] for r in group] == [0, 1]
        values = [r["timing"]["median_ms"] for r in group]
        spread = (max(values) - min(values)) / stats.median(values)
        between.append(dict(shape=shape, dtype=dtype, mode=mode, round0_us=values[0]*1000,
                            round1_us=values[1]*1000, relative_range_percent=spread*100,
                            review=spread > .05, ids=[r["task"]["id"] for r in group]))
    flagged = [g for g in between if g["review"]]
    assert len(flagged) == saved["between_round_flags"] == 2
    for actual, original in zip(flagged, saved["between_round"]):
        close(actual["relative_range_percent"], original["relative_range"] * 100)
        assert actual["ids"][0] == original["task"]["id"]
    recomputed = HERE / "recomputed"
    recomputed.mkdir(exist_ok=True)
    for name in ("manifest.json", "environment.json", "results.sqlite3"):
        shutil.copy2(RAW / name, recomputed / name)
    assert report.export(recomputed) == saved
    exported = ("summary.json", "raw_attempts.jsonl", "metrics.csv", "blocks.csv", "paired_metrics.csv",
                "paired_fresh_holdout_LOCKED.csv", "fresh_holdout_LOCKED.csv", "review_queue.csv",
                "shape_splits.csv", "report.html")
    for name in exported:
        assert (RAW / name).read_bytes() == (recomputed / name).read_bytes(), name
    mode_pairs = report.paired_rows(manifest, rows)
    assert len(mode_pairs) == 96 and all(p["pair_complete"] for p in mode_pairs)
    small = []
    for case in plan.PROTOCOL["verified_graph64_cases"]:
        selected = [p for p in mode_pairs if p["shape"] == case["shape"] and p["dtype"] == case["dtype"]]
        assert len(selected) == 2 and all(p["graph_gemms_per_replay"] == 64 for p in selected)
        pre, graph = [stats.median(p[mode + "_ms"] for p in selected) * 1000 for mode in ("prealloc", "graph")]
        small.append(dict(**case, prealloc_us=pre, graph_us=graph,
                          reduction_percent=(1-graph/pre)*100, graph_gemms_per_replay=64))
    telemetry = [{k.strip(): v.strip() for k, v in r.items()}
                 for r in csv.DictReader(next(RAW.glob("sessions/*/telemetry.csv")).open())]
    for row in telemetry:
        row["at"] = dt.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(
            tzinfo=dt.timezone(dt.timedelta(hours=9)))
    aligned = []
    for r in rows:
        tm, task = r["timing"], r["task"]
        start = dt.datetime.fromisoformat(tm["blocks"][0]["timestamp"])
        end = dt.datetime.fromisoformat(tm["blocks"][-1]["timestamp"]) + dt.timedelta(milliseconds=tm["blocks"][-1]["host_wall_ms"])
        samples = [x for x in telemetry if start <= x["at"] <= end]
        before = max((x for x in telemetry if x["at"] < start), key=lambda x:x["at"], default=None)
        after = min((x for x in telemetry if x["at"] > end), key=lambda x:x["at"], default=None)
        obs = dict(task=task, window_start_utc=start.isoformat(), window_end_utc=end.isoformat(),
                   samples_inside_window=len(samples))
        for name, points in (("inside", samples), ("before", [before] if before else []), ("after", [after] if after else [])):
            obs[name] = [{k: x[k] for k in ("timestamp", "temperature.gpu", "clocks.current.sm [MHz]", "power.draw [W]")}
                         for x in points]
        aligned.append(obs)
    before_text = env["nvidia_smi"]["stdout"]
    after_text = json.loads(next(RAW.glob("sessions/*/gpu_after.json")).read_text())["stdout"]
    counters = {}
    for name in ("SW Power Capping", "SW Thermal Slowdown", "HW Thermal Slowdown"):
        rx = re.escape(name) + r"\s*:\s*(\d+) us"
        counters[name] = (int(re.search(rx, after_text).group(1)) - int(re.search(rx, before_text).group(1))) / 1e6
    exits = {str(p.relative_to(RAW)): int(p.read_text()) for p in RAW.rglob("*exit_code.txt")}
    assert all(code == (0 if "tee_exit_code" in name else 2) for name, code in exits.items())
    assert "EXPERIMENT: REVIEW" in next(RAW.glob("launcher/*.log")).read_text()
    session = json.loads(next(RAW.glob("sessions/*/session.json")).read_text())
    assert session["state"] == "FINISHED" and session["new_records"] == 192
    flagged_ids = {i for b in flagged for i in b["ids"]} | {r["task"]["id"] for r in rows if r["status"] == "REVIEW"}
    output = dict(
        verification="PASS", quality_gate=saved["gate"], recommendation="HOLD_FULL_FOR_TARGETED_CLOCK_AND_TIMING_REVIEW",
        source_zip=str(ZIP), zip_sha256=sha(ZIP.read_bytes()), raw_file_count=len(hashes),
        source_hash_count=len(manifest["code_hashes"]), environment_matches_v5=True,
        task_count=len(rows), timed_blocks=1344, recorded_timed_gemm_operations=operations,
        counts=dict(Counter(r["status"] for r in rows)), paired_complete=96,
        saved_correctness=dict(correctness), max_saved_tolerance_ratio=max(r["correctness"]["max_tolerance_ratio"] for r in rows),
        relative_iqr_max_percent=max(r["timing"]["relative_iqr"] for r in rows)*100,
        abs_drift_max_percent=max(abs(r["timing"]["first_last_drift"]) for r in rows)*100,
        between_round_groups=len(between), stable_between_round_groups=len(between)-len(flagged),
        flagged_task_union_count=len(flagged_ids), flagged_conditions=flagged,
        graph_batch_stable_shape_dtype_count=len(choices), graph_batch_min_max=[min(choices.values()),max(choices.values())],
        small_cases=small, session=session, exit_codes=exits,
        telemetry_rows=len(telemetry), telemetry_interval_seconds_median=stats.median(
            (b["at"]-a["at"]).total_seconds() for a,b in zip(telemetry,telemetry[1:])),
        clock_event_counter_delta_seconds=counters, identical_regenerated_exports=list(exported),
        raw_preserved=True, benchmark_source_modified=False, gpu_rerun_performed=False,
    )
    dump("verification.json", output)
    dump("telemetry_alignment.json", aligned)
    report.csv_write(HERE / "between_round.csv", between)
    report.csv_write(HERE / "all_blocks_with_setup.csv", blocks)
    report.csv_write(HERE / "small_cases.csv", small)
    check_raw()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
