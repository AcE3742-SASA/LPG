"""Verify preserved files, independently expected operation counts, and comparisons."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw/timing_audit_20260919_201729_y43Okw"
REPO = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    hashes = read(HERE / "raw_sha256.json")
    for name, expected in hashes.items():
        assert sha(HERE / "raw" / name) == expected, name
    manifest = read(RAW / "manifest.json")
    baseline = read(REPO / "timing_audit_v4/baseline_v5.json")
    assert manifest["baseline"] == baseline and manifest["diagnostic_version"] == 4
    for name, expected in baseline["code_hashes"].items():
        assert sha(RAW / "baseline_source" / name) == expected
    sources = list((RAW / "source").iterdir())
    assert len(sources) == 6
    for p in sources:
        assert sha(p) == sha(REPO / "timing_audit_v4" / p.name)
    exits = {str(p.relative_to(RAW)): int(p.read_text()) for p in RAW.rglob("*exit_code.txt")}
    assert exits and set(exits.values()) == {0}
    result = read(RAW / "analysis.json")
    assert result == read(HERE / "recomputed/analysis.json")
    previous = read(REPO / "analysis/timing_audit_2026-09-19/wait_review_v2/analysis.json")
    data = {name: {r["task"]["id"]: r["result"] for r in read(RAW / name / "results.json")}
            for name in manifest["passes"]}
    comparisons, boundaries, stats = [], [], []
    for name, rows in data.items():
        environment = read(RAW / name / "environment.json")
        signature = {k:environment[k] for k in baseline["environment_signature"]}
        assert signature == baseline["environment_signature"], name
        for case, r in rows.items():
            assert r["correctness"]["passed"]
            t = r["timing"]
            assert len(t["blocks"]) == 7 and t["flags"] == []
            first, last = t["blocks"][0], t["setup"]["calibration"][-1]
            gap = ((datetime.fromisoformat(first["timestamp"]) -
                    datetime.fromisoformat(last["timestamp"])).total_seconds()*1000 - last["host_wall_ms"])
            boundaries.append({"pass":name, "case":case, "approx_calibration_to_measurement_ms":gap})
            stats.append({"pass":name, "case":case, "relative_iqr":t["relative_iqr"],
                          "first_last_drift":t["first_last_drift"]})
    total_kernels = total_preparation = total_memsets = 0
    for r in result["conditions"]:
        case = r["id"]
        blocks = [result["blocks"][f"audit/{case}/block/{i}"] for i in range(7)]
        multiplier = 2 if case in ("c01", "c07") else 1
        timing = data["nsys"][case]["timing"]
        operations = timing["blocks"][0]["operations_per_call"]
        prepare_calls = 3*operations + sum(b["repeats"]*b["operations_per_call"]
            for phase in ("settle_blocks", "calibration") for b in timing["setup"][phase])
        preparation = result["trace_metadata"][blocks[0]["trace"]]["preparation"]["prepare/"+case]
        assert preparation["kernel_count_in_envelope"] == multiplier*prepare_calls, case
        total_preparation += multiplier*prepare_calls
        for b in blocks:
            assert b["kernel_count"] == b["expected_gemm_calls"]*multiplier
            assert len(b["kernel_names"]) == multiplier
            assert set(b["kernel_names"].values()) == {b["expected_gemm_calls"]}
            assert b["unassociated_kernel_count_in_span"] == 0 and len(b["streams"]) == 1
            g = b["activity_gap_breakdown"]
            assert g["supported"] and g["total_gap_ns"] == sum(g["phase_ns"].values())
            assert g["gpu_activity_span_ns"] == g["gpu_activity_union_ns"] + g["total_gap_ns"]
            total_kernels += b["kernel_count"]
            memory = b["memory_activity_correlated_to_nvtx"]
            assert all(m["table"] == "CUPTI_ACTIVITY_KIND_MEMSET" for m in memory)
            if case == "c05":
                assert len(memory) == b["expected_gemm_calls"]
                assert {(m["bytes"], m["value"]) for m in memory} == {(2048,0)}
                total_memsets += len(memory)
            else:
                assert not memory
        old = next(v for v in previous["conditions"] if v["id"] == case)
        calls = sum(b["expected_gemm_calls"] for b in blocks)
        components = {key:sum(b["activity_gap_breakdown"]["phase_ns"][key] for b in blocks)/calls/1000
                      for key in ("before_api", "during_api", "after_api")}
        components["gpu_activity"] = sum(b["activity_gap_breakdown"]["gpu_activity_union_ns"] for b in blocks)/calls/1000
        comparisons.append({"id":case, "plain_reference_us":(r["plain_before_us"]+r["plain_after_us"])/2,
            "v4_nsys_change_pct":100*(r["nsys_to_plain_ratio"]-1),
            "v2_nsys_change_pct":100*(old["nsys_to_plain_ratio"]-1),
            "v2_nsys_drift_pct":100*old["first_last_drift_by_pass"]["nsys"],
            "v4_nsys_drift_pct":100*r["first_last_drift_by_pass"]["nsys"],
            "pooled_mean_activity_components_us":components})
    output = {"zip_sha256":"e311285d763d617f03b8d8b7974263306c3c3b3d26cca658b78286663107b4cf",
        "raw_files_verified":len(hashes), "source_files_verified":6, "v5_files_verified":13,
        "worker_environment_signatures_verified":3,
        "recomputed_exact_match":True, "exit_codes":exits, "correctness_passes":24,
        "all_24_timing_flags_empty":True, "measurement_kernel_count":total_kernels,
        "preparation_kernel_count":total_preparation, "memsets_in_measurement":total_memsets,
        "comparisons":comparisons, "boundaries":boundaries, "timing_quality":stats,
        "max_relative_iqr":max(r["relative_iqr"] for r in stats),
        "max_absolute_drift":max(abs(r["first_last_drift"]) for r in stats),
        "max_plain_session_difference":max(r["plain_session_relative_range"] for r in result["conditions"]),
        "interpretation":"Startup transient reduced below the retained gate in this run; small FP16 submission gaps persist under Nsight. Unprofiled gap magnitude and full protocol approval remain unresolved."}
    (HERE / "verification.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k not in ("comparisons","boundaries","timing_quality","exit_codes")},indent=2))


if __name__ == "__main__":
    main()
