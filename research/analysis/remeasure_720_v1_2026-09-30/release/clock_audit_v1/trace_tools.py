#!/usr/bin/env python3
"""Analyze NVTX-bracketed timing blocks; all trace databases are opened read-only."""
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


def activity_gaps(activities, gemm_calls):
    """Partition gaps by API timestamps, without assigning an unobserved cause."""
    ordered = sorted(activities, key=lambda r: r["start"])
    if not ordered:
        return {"supported": False, "reason": "no GPU activities"}
    phases = dict(before_api=0, during_api=0, after_api=0)
    states = dict(not_entered=0, in_api=0, returned=0)
    gaps = []
    first = ordered[0]
    busy_end = first["end"]
    busy_name = first["name"]
    union = busy_end - first["start"]
    for right in ordered[1:]:
        if right["end"] <= right["start"]:
            raise ValueError("Invalid GPU activity duration")
        if right["start"] > busy_end:
            api = right.get("api")
            if api is None or api["end"] < api["start"] or api["start"] > right["start"]:
                return {"supported": False, "reason": "missing or inconsistent next API timestamps"}
            gap_end = right["start"]
            parts = {
                "before_api": max(0, min(gap_end, api["start"]) - busy_end),
                "during_api": max(0, min(gap_end, api["end"]) - max(busy_end, api["start"])),
                "after_api": max(0, gap_end - max(busy_end, api["end"]))}
            if sum(parts.values()) != gap_end - busy_end:
                raise ValueError("Gap partition does not conserve duration")
            for name, amount in parts.items():
                phases[name] += amount
            state = ("not_entered" if api["start"] > busy_end else
                     "returned" if api["end"] <= busy_end else "in_api")
            states[state] += 1
            gaps.append({"start_ns": busy_end, "end_ns": gap_end,
                         "gap_us": (gap_end-busy_end)/1000, "phase_ns": parts,
                         "previous_activity": busy_name, "next_activity": right["name"],
                         "next_api_start_ns": api["start"], "next_api_end_ns": api["end"],
                         "next_api_state_at_gap_start": state})
        if right["end"] > busy_end:
            union += right["end"] - max(right["start"], busy_end)
            busy_end = right["end"]
            busy_name = right["name"]
    span = busy_end-first["start"]
    total = span-union
    if total != sum(phases.values()):
        raise ValueError("GPU activity union and classified gaps disagree")
    return {"supported": True, "gpu_activity_count": len(ordered),
            "gpu_activity_union_ns": union, "gpu_activity_span_ns": span,
            "total_gap_ns": total, "gap_fraction_of_activity_span": total/span,
            "gap_us_per_gemm": total/gemm_calls/1000, "phase_ns": phases,
            "phase_us_per_gemm": {k:v/gemm_calls/1000 for k,v in phases.items()},
            "gap_count": len(gaps), "next_api_state_counts": states,
            "largest_gaps": sorted(gaps, key=lambda r:r["gap_us"], reverse=True)[:10]}


def read_trace_metadata(path):
    before = sha256_file(path)
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        diagnostics = ([dict(r) for r in con.execute("SELECT * FROM DIAGNOSTIC_EVENT")]
                       if "DIAGNOSTIC_EVENT" in tables else [])
        strings = dict(con.execute("SELECT id,value FROM StringIds"))
        preparation = {}
        for r in con.execute("SELECT * FROM NVTX_EVENTS"):
            label = r["text"] or strings.get(r["textId"], "")
            if not label.startswith("prepare/"):
                continue
            if label in preparation or r["end"] is None or r["end"] <= r["start"]:
                raise ValueError("Duplicate/incomplete preparation range: " + label)
            kernel_count = con.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE globalPid=? AND start>=? AND end<=?",
                (r["globalTid"] & ~0xFFFFFF, r["start"], r["end"])).fetchone()[0]
            preparation[label] = {"start_ns": r["start"], "end_ns": r["end"],
                "global_tid": r["globalTid"], "duration_ms": (r["end"]-r["start"])/1e6,
                "kernel_count_in_envelope": kernel_count}
        return {"diagnostic_table_present": "DIAGNOSTIC_EVENT" in tables,
                "diagnostics": diagnostics, "warnings": [r for r in diagnostics if r["severity"] >= 2],
                "preparation": preparation}
    finally:
        con.close()
        if sha256_file(path) != before:
            raise ValueError("Raw trace changed during metadata read")


def validate_preparation(first, metadata, case_id, settle_ms):
    label = "prepare/" + case_id
    preparation = metadata["preparation"].get(label)
    if preparation is None:
        raise ValueError("Missing profiled preparation range: " + label)
    if (preparation["global_tid"] != first["nvtx_global_tid"] or
            preparation["end_ns"] > first["nvtx_start_ns"] or
            preparation["duration_ms"] < settle_ms or
            preparation["kernel_count_in_envelope"] < 1):
        raise ValueError("Invalid profiled preparation: " + label)
    return (first["nvtx_start_ns"] - preparation["end_ns"]) / 1e6


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
    if not ranges:
        return {}
    # Keep preparation in the raw trace; only load the measurement envelope into Python.
    bounds = (min(r["start"] for _,r in ranges), max(r["end"] for _,r in ranges))
    kernels = [dict(r) for r in con.execute("SELECT rowid AS row_id,* FROM CUPTI_ACTIVITY_KIND_KERNEL "
                                          "WHERE end>? AND start<?", bounds)]
    by_correlation = defaultdict(list)
    for k in kernels:
        by_correlation[k["globalPid"], k["correlationId"]].append(k)
    apis = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            apis.extend(dict(r) for r in con.execute(f'SELECT * FROM "{table}" WHERE end>? AND start<?', bounds))
    transfers = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            transfers.extend((table, dict(r)) for r in con.execute(f'SELECT * FROM "{table}" WHERE end>? AND start<?', bounds))
    results = {}
    for label, nvtx in ranges:
        if label in results:
            raise ValueError("Duplicate trace range: " + label)
        block = expected[label]
        calls = [a for a in apis if a["globalTid"] == nvtx["globalTid"] and
                 nvtx["start"] <= a["start"] and a["end"] <= nvtx["end"]]
        call_keys = {(a["globalTid"] & ~0xFFFFFF, a["correlationId"]) for a in calls}
        api_by_key = {(a["globalTid"] & ~0xFFFFFF, a["correlationId"]): a for a in calls}
        memory = [(table, r) for table, r in transfers
                  if (r["globalPid"], r["correlationId"]) in call_keys]
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
        activities = [{"start": k["start"], "end": k["end"],
                       "name": strings.get(k["demangledName"], "kernel"), "api": launches[k["row_id"]]}
                      for k in ordered]
        activities.extend({"start": r["start"], "end": r["end"], "name": table,
                           "api": api_by_key.get((r["globalPid"], r["correlationId"]))}
                          for table,r in memory)
        memory_streams = {(r.get("deviceId"), r.get("contextId"), r.get("streamId"))
                          for _,r in memory if "streamId" in r}
        breakdown = (activity_gaps(activities, minimum)
                     if len(streams) == 1 and not unmatched and memory_streams <= set(streams)
                     else {"supported": False, "reason": "multiple streams or other associated-process kernels"})
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
            "nvtx_end_ns": nvtx["end"], "nvtx_global_tid": nvtx["globalTid"],
            "kernel_count": len(ordered),
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
            "recorded_transfers_correlated_to_nvtx": dict(Counter(table for table, r in memory)),
            "memory_activity_correlated_to_nvtx": [{"table": table, **r} for table, r in memory],
            "activity_gap_breakdown": breakdown,
            "kernel_names": dict(Counter(strings.get(k["demangledName"], "unknown") for k in ordered)),
            "largest_gaps": sorted(gaps, key=lambda g: g["gap_us"], reverse=True)[:20]}
    return results

