#!/usr/bin/env python3
"""Read-only supplemental review of the supplied timing audit; no CUDA execution."""
from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw/timing_audit_20260919_175307_zz6mlt"
REPO = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    hashes = read(HERE / "raw_sha256.json")
    for name, expected in hashes.items():
        assert digest(HERE / "raw" / name) == expected, name
    baseline = read(REPO / "timing_audit_v2/baseline_v5.json")
    manifest = read(RAW / "manifest.json")
    assert manifest["baseline"] == baseline
    for name, expected in baseline["code_hashes"].items():
        assert digest(RAW / "baseline_source" / name) == expected, name
    source_names = [p.name for p in (RAW / "source").iterdir() if p.is_file()]
    for name in source_names:
        assert digest(RAW / "source" / name) == digest(REPO / "timing_audit_v2" / name)
    exits = {str(p.relative_to(RAW)): int(p.read_text()) for p in RAW.rglob("*exit_code.txt")}
    assert exits and set(exits.values()) == {0}
    original = read(RAW / "analysis.json")
    assert original == read(HERE / "recomputed/analysis.json")
    data = {name: {r["task"]["id"]: r["result"] for r in read(RAW / name / "results.json")}
            for name in manifest["passes"]}
    assert sum(r["correctness"]["passed"] for rows in data.values() for r in rows.values()) == 24
    boundaries = []
    for name, rows in data.items():
        for case, result in rows.items():
            timing = result["timing"]
            last = timing["setup"]["calibration"][-1]
            first = timing["blocks"][0]
            gap = ((datetime.fromisoformat(first["timestamp"]) -
                    datetime.fromisoformat(last["timestamp"])).total_seconds() * 1000 - last["host_wall_ms"])
            boundaries.append({"pass": name, "case": case,
                "approx_prepare_to_first_block_ms": gap,
                "first_block_us": first["latency_ms"] * 1000,
                "last_block_us": timing["blocks"][-1]["latency_ms"] * 1000,
                "median_us": timing["median_ms"] * 1000,
                "first_last_drift": timing["first_last_drift"], "flags": timing["flags"]})
    traces = []
    details = []
    for path in sorted((RAW / "traces").glob("*.sqlite")):
        con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        con.row_factory = sqlite3.Row
        assert con.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        strings = dict(con.execute("SELECT id,value FROM StringIds"))
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        kernels = [dict(r) for r in con.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
        apis = [dict(r) for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
                if table in tables for r in con.execute(f"SELECT * FROM {table}")]
        transfers = {table: [dict(r) for r in con.execute(f"SELECT * FROM {table}")]
                     for table in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY") if table in tables}
        diagnostics = [dict(r) for r in con.execute("SELECT * FROM DIAGNOSTIC_EVENT")]
        ranges = [(r["text"] or strings.get(r["textId"], ""), dict(r))
                  for r in con.execute("SELECT * FROM NVTX_EVENTS")]
        ranges = [(label, r) for label, r in ranges if label.startswith("audit/")]
        assert len(ranges) == 7
        total_matched = 0
        for label, nvtx in ranges:
            case, block_index = label.split("/")[1], int(label.split("/")[-1])
            block = data["nsys"][case]["timing"]["blocks"][block_index]
            selected_apis = [a for a in apis if a["globalTid"] == nvtx["globalTid"] and
                             nvtx["start"] <= a["start"] and a["end"] <= nvtx["end"]]
            correlations = {(a["globalTid"] & ~0xFFFFFF, a["correlationId"]) for a in selected_apis}
            matched = [k for k in kernels if (k["globalPid"], k["correlationId"]) in correlations]
            calls = block["repeats"] * block["operations_per_call"]
            multiplicity = 2 if case in ("c01", "c07") else 1
            assert len(matched) == calls * multiplicity, label
            assert len(matched) == original["blocks"][label]["kernel_count"]
            names = Counter(strings[k["demangledName"]] for k in matched)
            assert len(names) == multiplicity and set(names.values()) == {calls}, label
            total_matched += len(matched)
            memory = {table: [r for r in values if (r["globalPid"], r["correlationId"]) in correlations]
                      for table, values in transfers.items()}
            if case == "c05":
                memset = memory["CUPTI_ACTIVITY_KIND_MEMSET"]
                assert len(memset) == calls
                assert {(r["bytes"], r["value"]) for r in memset} == {(2048, 0)}
            details.append({"label": label, "gemm_calls": calls, "kernel_count": len(matched),
                "kernel_names": dict(names), "kernel_mean_per_gemm_us": sum(k["end"]-k["start"] for k in matched)/calls/1000,
                "first_16_kernel_us": [(k["end"]-k["start"])/1000 for k in matched[:16]],
                "memory_counts_correlated_to_full_nvtx_range": {t: len(v) for t,v in memory.items()},
                "memory_gpu_us": {t: sum(r["end"]-r["start"] for r in v)/1000 for t,v in memory.items()},
                "api_counts": dict(Counter(strings[a["nameId"]] for a in selected_apis))})
        assert total_matched == len(kernels), path.name
        traces.append({"file": path.name, "kernel_count": len(kernels), "matched_block_count": len(ranges),
                       "diagnostics": diagnostics, "memory_counts": {t:len(v) for t,v in transfers.items()}})
        con.close()
        assert digest(path) == original["trace_sha256"][path.name]
    report = {"raw_file_count": len(hashes), "baseline_hashes_checked": len(baseline["code_hashes"]),
        "diagnostic_sources_checked": len(source_names), "exit_codes": exits,
        "trusted_analyzer_exact_reproduction": True, "correctness_passes": 24,
        "kernel_count": sum(t["kernel_count"] for t in traces), "traces": traces,
        "boundaries": boundaries, "blocks": details,
        "boundary_caveat": "Timestamp starts before CUDA event initialization; host_wall_ms starts after initialization. Difference includes that setup, Python bookkeeping, and profiler activation; it is not a direct measurement of cudaProfilerStart duration.",
        "decision": "REVIEW_REQUIRED; full remains on hold. Missing-event warnings retained despite exact expected target kernel counts."}
    (HERE / "supplemental.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (HERE / "boundaries.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(boundaries[0])); writer.writeheader(); writer.writerows(boundaries)
    for name, expected in hashes.items():
        assert digest(HERE / "raw" / name) == expected, name
    print(json.dumps({"verified_raw_files":len(hashes), "kernel_count":report["kernel_count"],
                      "blocks":len(details), "correctness":24, "decision":report["decision"]}))


if __name__ == "__main__":
    main()
