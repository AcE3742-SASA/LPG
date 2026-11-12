#!/usr/bin/env python3
"""Analyze NVTX-bracketed v5 blocks; all trace databases are opened read-only."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sqlite3
import statistics
from audit import sha256_file


def union_ns(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError("Invalid GPU interval")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def finite_positive(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid timing value: " + str(value))
    return value


def read_trace(path, expected):
    before_hash = sha256_file(path)
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    try:
        results = parse_trace(con, path, expected)
    finally:
        con.close()
    if sha256_file(path) != before_hash:
        raise ValueError("Raw trace changed during read: " + str(path))
    return results, before_hash


def parse_trace(con, path, expected):
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_RUNTIME",
                "CUPTI_ACTIVITY_KIND_KERNEL"}
    if not required <= tables:
        raise ValueError(f"Missing trace tables in {path}: {required - tables}")
    strings = dict(con.execute("SELECT id,value FROM StringIds"))
    ranges = []
    for r in con.execute("SELECT * FROM NVTX_EVENTS"):
        row = dict(r)
        label = row.get("text") or strings.get(row.get("textId"), "")
        if label.startswith("audit/"):
            if label not in expected or row["end"] is None or row["end"] <= row["start"]:
                raise ValueError("Unexpected/incomplete NVTX range: " + label)
            ranges.append((label, row))
    kernels = [dict(r) for r in con.execute("SELECT rowid AS row_id,* FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    by_correlation = defaultdict(list)
    for k in kernels:
        by_correlation[k["globalPid"], k["correlationId"]].append(k)
    apis = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            apis.extend(dict(r) for r in con.execute(f'SELECT * FROM "{table}"'))
    transfers = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            transfers.extend((table, dict(r)) for r in con.execute(f'SELECT * FROM "{table}"'))
    results = {}
    for label, nvtx in ranges:
        if label in results:
            raise ValueError("Duplicate trace range: " + label)
        block = expected[label]
        calls = [a for a in apis if a["globalTid"] == nvtx["globalTid"] and
                 nvtx["start"] <= a["start"] and a["end"] <= nvtx["end"]]
        matched = {}
        launches = {}
        for a in calls:
            key = (a["globalTid"] & ~0xFFFFFF, a["correlationId"])
            for k in by_correlation.get(key, ()):
                matched[k["row_id"]] = k
                launches[k["row_id"]] = a
        ordered = sorted(matched.values(), key=lambda k: k["start"])
        if not ordered:
            raise ValueError("No correlated kernels for " + label)
        minimum = block["repeats"] * block["operations_per_call"]
        if len(ordered) < minimum:
            raise ValueError(f"Incomplete kernel trace {label}: {len(ordered)} < {minimum}")
        start = ordered[0]["start"]
        end = max(k["end"] for k in ordered)
        if start < nvtx["start"] or end > nvtx["end"]:
            raise ValueError("GPU work outside synchronized NVTX envelope: " + label)
        intervals = [(k["start"], k["end"]) for k in ordered]
        union = union_ns(intervals)
        span = end - start
        process = nvtx["globalTid"] & ~0xFFFFFF
        unmatched = [k for k in kernels if k["globalPid"] == process and
                     k["start"] < end and k["end"] > start and k["row_id"] not in matched]
        streams = sorted({(k["deviceId"], k["contextId"], k["streamId"]) for k in ordered})
        gaps = []
        if len(streams) == 1:
            for left, right in zip(ordered, ordered[1:]):
                if right["start"] <= left["end"]:
                    continue
                api = launches[right["row_id"]]
                gaps.append({"previous_end_ns": left["end"], "next_start_ns": right["start"],
                    "gap_us": (right["start"] - left["end"]) / 1000,
                    "next_launch_start_minus_previous_end_us": (api["start"] - left["end"]) / 1000,
                    "next_launch_end_minus_previous_end_us": (api["end"] - left["end"]) / 1000,
                    "next_launch_name": strings.get(api["nameId"], str(api["nameId"])),
                    "next_kernel_name": strings.get(right["demangledName"], "unknown")})
        elapsed_ns = finite_positive(block["elapsed_ms"]) * 1e6
        results[label] = {"trace": path.name, "nvtx_start_ns": nvtx["start"],
            "nvtx_end_ns": nvtx["end"], "kernel_count": len(ordered),
            "expected_gemm_calls": minimum, "streams": streams,
            "event_ms": block["elapsed_ms"], "kernel_union_ms": union / 1e6,
            "kernel_span_ms": span / 1e6, "inter_kernel_gap_ms": (span - union) / 1e6,
            "inter_kernel_gap_fraction": (span - union) / span,
            "event_minus_kernel_union_ms": (elapsed_ns - union) / 1e6,
            "event_minus_kernel_union_fraction": (elapsed_ns - union) / elapsed_ns,
            "unassociated_kernel_count_in_span": len(unmatched),
            "api_counts_in_cpu_envelope": dict(Counter(strings.get(a["nameId"], "unknown") for a in calls)),
            "recorded_transfer_counts_in_kernel_span": dict(Counter(table for table, r in transfers
                if r["start"] < end and r["end"] > start)),
            "kernel_names": dict(Counter(strings.get(k["demangledName"], "unknown") for k in ordered)),
            "largest_gaps": sorted(gaps, key=lambda g: g["gap_us"], reverse=True)[:20]}
    return results


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text())
    tasks = {t["id"]: t for t in manifest["cases"]}
    count = manifest["baseline"]["protocol"]["blocks"]
    data = {}
    expected = {}
    for name in manifest["passes"]:
        rows = json.loads((out / name / "results.json").read_text())
        if len(rows) != len(tasks) or {r["task"]["id"] for r in rows} != set(tasks):
            raise ValueError("Missing/duplicate condition in " + name)
        data[name] = {}
        for row in rows:
            case_id, result = row["task"]["id"], row["result"]
            if row["task"] != tasks[case_id] or not result["correctness"]["passed"]:
                raise ValueError("Task/correctness mismatch in " + name)
            timing = result["timing"]
            finite_positive(timing["median_ms"])
            if len(timing["blocks"]) != count:
                raise ValueError("Missing timing block")
            for i, block in enumerate(timing["blocks"]):
                finite_positive(block["elapsed_ms"])
                finite_positive(block["latency_ms"])
                for key in ("repeats", "operations_per_call"):
                    if type(block[key]) is not int or block[key] < 1:
                        raise ValueError("Invalid operation count")
                normalized = block["elapsed_ms"] / (block["repeats"] * block["operations_per_call"])
                if not math.isclose(normalized, block["latency_ms"], rel_tol=1e-10):
                    raise ValueError("Timing normalization mismatch")
                if name == "nsys":
                    label = f"audit/{case_id}/block/{i}"
                    if block.get("trace_label") != label:
                        raise ValueError("Wrong trace label: " + label)
                    expected[label] = block
            if not math.isclose(statistics.median(b["latency_ms"] for b in timing["blocks"]),
                                timing["median_ms"], rel_tol=1e-10):
                raise ValueError("Stored median mismatch")
            data[name][case_id] = timing
    traced, hashes = {}, {}
    for path in sorted((out / "traces").glob("*.sqlite")):
        rows, digest = read_trace(path, expected)
        if set(rows) & set(traced):
            raise ValueError("Duplicate NVTX range across trace files")
        traced.update(rows)
        hashes[path.name] = digest
    if set(traced) != set(expected):
        raise ValueError("Missing NVTX blocks: " + str(sorted(set(expected) - set(traced))))
    conditions = []
    for case_id, task in tasks.items():
        before, profile, after = (data[name][case_id] for name in manifest["passes"])
        base_ms = (before["median_ms"] + after["median_ms"]) / 2
        drift = abs(after["median_ms"] - before["median_ms"]) / base_ms
        rows = [traced[f"audit/{case_id}/block/{i}"] for i in range(count)]
        flags = [f"{name}:{flag}" for name in manifest["passes"]
                 for flag in data[name][case_id].get("flags", [])]
        if drift > .05:
            flags.append("unprofiled_session_difference_above_5pct")
        if any(r["unassociated_kernel_count_in_span"] for r in rows):
            flags.append("other_recorded_kernels_in_span")
        if any(len(r["streams"]) != 1 for r in rows):
            flags.append("multiple_cuda_streams")
        nodes = [data[name][case_id].get("graph_nodes", 1) for name in manifest["passes"]]
        if len(set(nodes)) > 1:
            flags.append("graph_node_counts_differ")
        conditions.append({"id": case_id, "shape": "x".join(map(str, task["shape"])),
            "dtype": task["dtype"], "mode": task["mode"],
            "plain_before_us": before["median_ms"] * 1000,
            "nsys_us": profile["median_ms"] * 1000,
            "plain_after_us": after["median_ms"] * 1000,
            "nsys_to_plain_ratio": profile["median_ms"] / base_ms,
            "plain_session_relative_range": drift,
            "median_inter_kernel_gap_fraction": statistics.median(r["inter_kernel_gap_fraction"] for r in rows),
            "median_event_minus_kernel_union_fraction": statistics.median(r["event_minus_kernel_union_fraction"] for r in rows),
            "graph_nodes_by_pass": nodes, "flags": flags})
    report = {"collection_integrity": "PASS", "measurement_quality": "REVIEW_REQUIRED",
        "condition_count": len(tasks), "traced_block_count": len(traced),
        "trace_sha256": hashes, "conditions": conditions, "blocks": traced,
        "interpretation": ["Profiled times are diagnostic only, never training labels.",
            "NVTX is a CPU envelope, not CUDA Event device boundaries. Event minus CUPTI union mixes boundary gaps and timer definitions.",
            "CUDA API setup/final synchronization is present in the NVTX envelope, not necessarily between the timed device events.",
            "Kernel-span gaps are not whole-GPU idle time; other processes may not be traced.",
            "The nsys/plain ratio includes session variation; compare both plain passes before attributing it to instrumentation.",
            "Adaptive repeats and graph node counts are saved in raw results; different counts are not identical workloads.",
            "Pressure controls and chain composition are outside this first focused audit."]}
    (out / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (out / "conditions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(conditions[0]))
        writer.writeheader()
        writer.writerows(conditions)
    lines = ["# 실제 측정 블록 진단", "", "수집 구조 검증: PASS. 측정 품질: 검토 필요.", "",
        "각 조건은 일반 실행, Nsight 실행, 일반 재실행으로 비교했다. 시간은 블록 평균의 중앙값이다.", "",
        "| 조건 | 일반 전 (μs) | Nsight (μs) | 일반 후 (μs) | Nsight/일반 | 커널 사이 간격 비중 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in conditions:
        lines.append(f"| {r['shape']} {r['dtype']} {r['mode']} | {r['plain_before_us']:.3g} | {r['nsys_us']:.3g} | {r['plain_after_us']:.3g} | {r['nsys_to_plain_ratio']:.3g} | {100*r['median_inter_kernel_gap_fraction']:.3g}% |")
    lines += ["", "간격 비중은 첫 커널 시작~마지막 커널 끝에서 기록된 커널 구간을 제외한 비중이다. 전체 GPU 유휴율 또는 원래 본측정의 오차율이 아니다.",
        "", "CPU NVTX 범위에는 event 초기화와 마지막 동기화도 포함된다. event 시간에서 CUPTI 커널 시간을 뺀 차이를 전부 CPU 제출 비용으로 해석하지 않는다.",
        "", "조건별 변동 표식과 블록별 API·커널 이름, 큰 간격의 다음 launch 시점은 analysis.json에 있다. 자동 판정만으로 full을 시작하지 않는다.", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"COLLECTION INTEGRITY: PASS ({len(tasks)} conditions, {len(traced)} traced blocks)")
    print("MEASUREMENT QUALITY: REVIEW REQUIRED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output.resolve())
