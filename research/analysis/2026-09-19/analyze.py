#!/usr/bin/env python3
"""Recompute the archived GEMM sweep on CPU. Requires NumPy and Matplotlib."""
import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
DTYPES = ("fp32", "fp16")
COLORS = {"fp32": "#245A81", "fp16": "#C96B27"}


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_csv(name, rows):
    if not rows:
        return
    with (TABLES / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(("min", "p05", "p25", "median", "p75", "p95", "p99", "max"),
                    map(float, np.percentile(a, [0, 5, 25, 50, 75, 95, 99, 100]))))


def key(row):
    return tuple(int(row[k]) for k in ("M", "N", "K")) + (row["dtype"],)


def measurements(row):
    blocks = np.asarray(json.loads(row["block_latencies_ms"]), dtype=float)
    assert len(blocks) == int(row["blocks"]) == 5
    assert np.all(np.isfinite(blocks)) and np.all(blocks > 0)
    q25, median, q75 = np.percentile(blocks, [25, 50, 75], method="linear")
    return blocks, float(median), float(q25), float(q75)


def flag_metrics(blocks):
    q25, median, q75 = np.percentile(blocks, [25, 50, 75], method="linear")
    return float(100 * (q75 - q25) / median), float(100 * (blocks[-1] / blocks[0] - 1))


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    # Small independent check of quartiles and directional drift.
    assert np.allclose(flag_metrics(np.array([1., 2., 3., 4., 5.])), [200 / 3, 400])
    manifest = json.loads((BASE / "source_manifest.json").read_text())
    for item in manifest:
        data = (BASE / item["local_path"]).read_bytes()
        assert len(data) == item["size"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]

    meta = json.loads((RAW / "full/full_5000x2.meta.json").read_text())
    source_shapes = {(int(s["M"]), int(s["N"]), int(s["K"])): s["source"] for s in meta["shapes"]}
    expected = {s + (dt,) for s in source_shapes for dt in DTYPES}
    raw_rows = load_csv(RAW / "full/full_5000x2.csv")
    assert len(source_shapes) == meta["num_shapes"] == 5000
    assert len(raw_rows) == len(expected) == meta["expected_conditions"] == 10000
    assert len(set(map(key, raw_rows))) == len(raw_rows)
    assert set(map(key, raw_rows)) == expected

    # Extract only the already-reviewed, CPU-only shape generator; never run the GPU entry point.
    tree = ast.parse((RAW / "scripts/gemm_pilot.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_shapes")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "archived_shape_generator", "exec"), namespace)
    regenerated = namespace["generate_shapes"](meta["num_shapes"], meta["seed"])
    assert regenerated == [(s["M"], s["N"], s["K"], s["source"]) for s in meta["shapes"]]
    assert list(map(key, raw_rows)) == [s[:3] + (dt,) for s in regenerated for dt in DTYPES]

    output = (RAW / "full/output.log").read_text()
    launcher = (RAW / "full/full_sweep_launcher.log").read_text()
    environment = (RAW / "full/environment.txt").read_text()
    assert output.endswith("completed_after_run=10000\nPILOT: PASS\n")
    assert launcher.endswith("EXPERIMENT: PASS\n") and output in launcher
    log_rows = re.findall(r"^PASS (\d+)/5000 (\d+)x(\d+)x(\d+) (fp32|fp16) median_ms=(\S+) iqr_ms=(\S+)$", output, re.M)
    assert len(log_rows) == 10000
    wrapper_hash = re.search(r"runner_sha256=(\w+)", environment).group(1)
    assert hashlib.sha256((RAW / "scripts/run_research_checked.sh").read_bytes()).hexdigest() == wrapper_hash

    rows = []
    for index, (r, log_row) in enumerate(zip(raw_rows, log_rows)):
        m, n, k, dtype = key(r)
        blocks, median, q25, q75 = measurements(r)
        assert r["status"] == "PASS" and r["finite_output"] == "True" and not r["error_message"]
        assert r["source"] == source_shapes[(m, n, k)]
        assert (r["aligned8"] == "True") == (m % 8 == n % 8 == k % 8 == 0)
        assert (r["square"] == "True") == (m == n == k)
        assert int(r["warmup"]) == meta["warmup_per_condition"] == 5
        assert float(r["settle_ms"]) == meta["settle_ms_per_condition"] == 500
        assert 3 <= int(r["repeats"]) <= 20000 and int(r["settle_repeats"]) > 0
        flops = 2 * m * n * k
        recomputed = [median, q25, q75, q75 - q25, flops / (median * 1e9)]
        recorded = [float(r[f]) for f in ("median_latency_ms", "q25_latency_ms", "q75_latency_ms", "iqr_latency_ms", "tflops")]
        assert np.allclose(recomputed, recorded, rtol=1e-12, atol=1e-14), key(r)
        _, lm, ln, lk, ldt, lmedian, liqr = log_row
        assert (int(lm), int(ln), int(lk), ldt) == key(r)
        assert float(lmedian) == recorded[0] and float(liqr) == recorded[3]
        rel_iqr, drift = flag_metrics(blocks)
        ideal_bytes = (4 if dtype == "fp32" else 2) * (m * k + k * n + m * n)
        rows.append({
            "M": m, "N": n, "K": k, "dtype": dtype, "source": r["source"],
            "aligned8": r["aligned8"] == "True", "square": m == n == k,
            "flops": flops, "ideal_bytes": ideal_bytes, "ideal_oi_flop_per_byte": flops / ideal_bytes,
            "latency_us": median * 1000, "tflops": recomputed[-1],
            "relative_iqr_pct": rel_iqr, "first_last_drift_pct": drift,
            "range_over_median_pct": float(100 * np.ptp(blocks) / median),
            "iqr_gt5": rel_iqr > 5, "abs_drift_gt5": abs(drift) > 5,
            "review_flag": rel_iqr > 5 or abs(drift) > 5,
            "repeats": int(r["repeats"]), "median_block_ms": median * int(r["repeats"]),
            "block1_us": blocks[0] * 1000, "block2_us": blocks[1] * 1000,
            "block3_us": blocks[2] * 1000, "block4_us": blocks[3] * 1000, "block5_us": blocks[4] * 1000,
            "timestamp": r["timestamp"], "run_fraction": index / len(raw_rows),
        })
    timestamps = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    assert timestamps == sorted(timestamps)
    save_csv("condition_metrics.csv", rows)
    flagged = sorted([r for r in rows if r["review_flag"]], key=lambda r: (r["iqr_gt5"], max(r["relative_iqr_pct"], abs(r["first_last_drift_pct"]))), reverse=True)
    save_csv("recheck_conditions.csv", flagged)
    save_csv("iqr_gt5_conditions.csv", [r for r in flagged if r["iqr_gt5"]])
    lookup = {key(r): r for r in rows}
    pairs = []
    for shape in source_shapes:
        a, b = [lookup[shape + (dt,)] for dt in DTYPES]
        pairs.append({"M": shape[0], "N": shape[1], "K": shape[2], "source": a["source"],
                      "aligned8": a["aligned8"], "square": a["square"], "flops": a["flops"],
                      "aspect_ratio": max(shape) / min(shape),
                      "fp32_latency_us": a["latency_us"], "fp16_latency_us": b["latency_us"],
                      "speedup_fp16": a["latency_us"] / b["latency_us"],
                      "both_iqr_le5": not (a["iqr_gt5"] or b["iqr_gt5"]),
                      "neither_review_flag": not (a["review_flag"] or b["review_flag"])})
    save_csv("precision_pairs.csv", pairs)
    save_csv("fp16_slower_pairs.csv", sorted([p for p in pairs if p["speedup_fp16"] < 1], key=lambda p: p["speedup_fp16"]))

    def pair_summary(group, ps):
        speed = [p["speedup_fp16"] for p in ps]
        return {"group": group, "shapes": len(ps), "speedup_median": float(np.median(speed)),
                "speedup_geomean": float(np.exp(np.mean(np.log(speed)))),
                "speedup_p05": float(np.percentile(speed, 5)), "speedup_p95": float(np.percentile(speed, 95)),
                "fp16_slower": sum(v < 1 for v in speed),
                "median_fp32_us": float(np.median([p["fp32_latency_us"] for p in ps])),
                "median_fp16_us": float(np.median([p["fp16_latency_us"] for p in ps]))}

    grouped = [pair_summary("all", pairs),
               pair_summary("both_iqr_le5", [p for p in pairs if p["both_iqr_le5"]]),
               pair_summary("neither_review_flag", [p for p in pairs if p["neither_review_flag"]])]
    for name, predicate in [("aligned8", lambda p: p["aligned8"]), ("unaligned8", lambda p: not p["aligned8"]),
                            ("aspect_le2", lambda p: p["aspect_ratio"] <= 2), ("aspect_gt8", lambda p: p["aspect_ratio"] > 8)]:
        grouped.append(pair_summary(name, [p for p in pairs if predicate(p)]))
    for power in range(5, 12):
        ps = [p for p in pairs if 10**power <= p["flops"] < 10**(power+1)]
        if ps:
            grouped.append(pair_summary(f"FLOP_[1e{power},1e{power+1})", ps))
    save_csv("group_comparisons.csv", grouped)

    boundary = []
    for r in rows:
        if not r["aligned8"]:
            continue
        shape = key(r)
        for axis in range(3):
            if shape[axis] not in (64, 128, 256, 512, 1024, 2048):
                continue
            neighbor = list(shape)
            neighbor[axis] += 1
            s = lookup.get(tuple(neighbor))
            if s:
                boundary.append({"M": r["M"], "N": r["N"], "K": r["K"], "dtype": r["dtype"],
                                 "axis_plus_one": "MNK"[axis], "base_latency_us": r["latency_us"],
                                 "neighbor_latency_us": s["latency_us"],
                                 "latency_ratio_neighbor_base": s["latency_us"] / r["latency_us"],
                                 "flop_ratio_neighbor_base": s["flops"] / r["flops"],
                                 "throughput_ratio_neighbor_base": s["tflops"] / r["tflops"],
                                 "both_iqr_le5": not (r["iqr_gt5"] or s["iqr_gt5"]),
                                 "neither_review_flag": not (r["review_flag"] or s["review_flag"])})
    save_csv("matched_boundary_pairs.csv", boundary)
    boundary_summary = []
    for dt in DTYPES:
        for axis in "MNK":
            bs = [r for r in boundary if r["dtype"] == dt and r["axis_plus_one"] == axis]
            good = [r for r in bs if r["neither_review_flag"]]
            boundary_summary.append({"dtype": dt, "axis": axis, "pairs": len(bs),
                                     "median_latency_ratio": float(np.median([r["latency_ratio_neighbor_base"] for r in bs])),
                                     "unflagged_pairs": len(good),
                                     "unflagged_median_latency_ratio": float(np.median([r["latency_ratio_neighbor_base"] for r in good]))})
    save_csv("matched_boundary_summary.csv", boundary_summary)

    pilots = []
    for name in ("pilot_100x2", "pilot_100x2_stable"):
        pr = load_csv(RAW / f"pilot/{name}.csv")
        vals = [flag_metrics(measurements(r)[0])[0] for r in pr]
        pilots.append({"dataset": name, "conditions": len(pr), "iqr_gt5": sum(v > 5 for v in vals), "relative_iqr_pct": distribution(vals)})
    stable_pilot = load_csv(RAW / "pilot/pilot_100x2_stable.csv")
    pilot_meta = json.loads((RAW / "pilot/pilot_100x2_stable.meta.json").read_text())
    for field in ("gpu", "pytorch", "cuda_runtime", "precision_settings", "preheat_seconds",
                  "warmup_per_condition", "blocks", "target_block_ms", "settle_ms_per_condition"):
        assert pilot_meta[field] == meta[field], field
    overlap = [{"M": int(r["M"]), "N": int(r["N"]), "K": int(r["K"]), "dtype": r["dtype"],
                "full_over_pilot_latency": lookup[key(r)]["latency_us"] / (1000 * float(r["median_latency_ms"]))}
               for r in stable_pilot if key(r) in lookup]
    save_csv("pilot_full_overlap.csv", overlap)
    cross_run_review = [r for r in overlap if abs(r["full_over_pilot_latency"] - 1) > .05]
    save_csv("cross_run_review.csv", cross_run_review)
    cross_lookup = {key(r): r["full_over_pilot_latency"] for r in cross_run_review}
    combined = []
    for r in rows:
        if r["review_flag"] or key(r) in cross_lookup:
            combined.append({**r, "cross_run_gt5": key(r) in cross_lookup,
                             "full_over_stable_pilot_latency": cross_lookup.get(key(r), ""),
                             "priority": 1 if r["iqr_gt5"] else (2 if r["abs_drift_gt5"] else 3)})
    combined.sort(key=lambda r: (r["priority"], -max(r["relative_iqr_pct"], abs(r["first_last_drift_pct"]))))
    save_csv("combined_review_queue.csv", combined)
    summary = {"analysis_date": "2026-09-19", "scope": "data validation and descriptive performance analysis only",
               "validation": {"rows": len(rows), "unique_shapes": len(source_shapes), "unique_conditions": len(expected),
                              "missing": 0, "duplicates": 0, "failed": 0, "recomputed_fields_match": True,
                              "regenerated_shapes_match": True, "log_rows_match": True, "wrapper_hash_matches": True,
                              "exit_code_file_uploaded": False, "launcher_experiment_pass": True},
               "source_shapes": dict(Counter(source_shapes.values())),
               "metadata": {k: v for k, v in meta.items() if k != "shapes"},
               "first_condition_start": rows[0]["timestamp"], "last_condition_start": rows[-1]["timestamp"],
               "condition_start_span_seconds": (timestamps[-1] - timestamps[0]).total_seconds(),
               "relative_iqr_pct": distribution([r["relative_iqr_pct"] for r in rows]),
               "iqr_gt5": len([r for r in rows if r["iqr_gt5"]]),
               "abs_drift_gt5": len([r for r in rows if r["abs_drift_gt5"]]),
               "review_union": len(flagged), "review_shapes": len(set(key(r)[:3] for r in flagged)),
               "dtype": {}, "groups": grouped, "boundary": boundary_summary, "pilots": pilots,
               "pilot_full_overlap_conditions": len(overlap),
               "cross_run_gt5": len(cross_run_review), "combined_review_conditions": len(combined),
               "combined_review_shapes": len(set(key(r)[:3] for r in combined)),
               "pilot_full_latency_ratio": distribution([p["full_over_pilot_latency"] for p in overlap]),
               "analysis_runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    for dt in DTYPES:
        rs = [r for r in rows if r["dtype"] == dt]
        summary["dtype"][dt] = {"conditions": len(rs), "iqr_gt5": sum(r["iqr_gt5"] for r in rs),
                                "abs_drift_gt5": sum(r["abs_drift_gt5"] for r in rs),
                                "review_union": sum(r["review_flag"] for r in rs),
                                "relative_iqr_pct": distribution([r["relative_iqr_pct"] for r in rs]),
                                "latency_us": distribution([r["latency_us"] for r in rs]),
                                "tflops": distribution([r["tflops"] for r in rs]),
                                "max_tflops_condition": max(rs, key=lambda r: r["tflops"]),
                                "median_block_ms": distribution([r["median_block_ms"] for r in rs])}
    (BASE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    make_figures(rows, pairs, boundary)
    print(json.dumps({k: summary[k] for k in ("validation", "relative_iqr_pct", "iqr_gt5", "abs_drift_gt5", "review_union", "groups", "boundary", "pilot_full_latency_ratio")}, indent=2))


def make_figures(rows, pairs, boundary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                         "figure.dpi": 140, "savefig.dpi": 180, "pdf.fonttype": 42})

    def save(fig, name):
        fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
        fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for dt in DTYPES:
        rs = [r for r in rows if r["dtype"] == dt]
        x = np.sort([r["relative_iqr_pct"] for r in rs])
        axes[0].plot(x, np.arange(1, len(x)+1)/len(x)*100, color=COLORS[dt], label=dt.upper())
        axes[1].scatter([r["run_fraction"]*100 for r in rs], [r["relative_iqr_pct"] for r in rs], s=5,
                        alpha=.5, color=COLORS[dt], rasterized=True)
    axes[0].set(xscale="symlog", xlabel="Relative IQR (%)", ylabel="Cumulative conditions (%)", title="Five-block variability")
    axes[0].axvline(5, ls="--", color="black", lw=1); axes[0].legend()
    axes[1].set(xlabel="Position in sweep (%)", ylabel="Relative IQR (%)", title="Flagged conditions occur throughout the run")
    axes[1].axhline(5, ls="--", color="black", lw=1)
    save(fig, "01_quality")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for dt in DTYPES:
        rs = [r for r in rows if r["dtype"] == dt]
        axes[0].scatter([r["flops"] for r in rs], [r["latency_us"] for r in rs], s=4, alpha=.22,
                        color=COLORS[dt], label=dt.upper(), rasterized=True)
        axes[1].scatter([r["ideal_oi_flop_per_byte"] for r in rs], [r["tflops"] for r in rs], s=4,
                        alpha=.25, color=COLORS[dt], rasterized=True)
    axes[0].set(xscale="log", yscale="log", xlabel="Useful operations, 2MNK (FLOP)", ylabel="Latency per call (us)", title="Repeated torch.mm calls timed with CUDA Events")
    axes[0].legend(markerscale=3)
    axes[1].set(xscale="log", yscale="log", xlabel="Ideal arithmetic intensity (FLOP/byte)", ylabel="Achieved throughput (TFLOP/s)", title="Ideal bytes; no measured bandwidth ceiling")
    save(fig, "02_performance")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for aligned, color, label in [(True, "#245A81", "All dimensions divisible by 8"), (False, "#C96B27", "Other shapes")]:
        ps = [p for p in pairs if p["aligned8"] == aligned]
        axes[0].scatter([p["flops"] for p in ps], [p["speedup_fp16"] for p in ps], s=6, alpha=.3,
                        color=color, label=label, rasterized=True)
    axes[0].set(xscale="log", xlabel="Useful operations, 2MNK (FLOP)", ylabel="FP32 latency / FP16 latency", title="Matched-shape precision comparison")
    axes[0].axhline(1, color="black", ls="--", lw=1); axes[0].legend(fontsize=8)
    anchors = [p for p in pairs if p["M"] == p["N"] == p["K"] and p["M"] in [64, 128, 256, 512, 1024, 2048, 4096]]
    anchors.sort(key=lambda p: p["M"])
    for dt in DTYPES:
        axes[1].plot([p["M"] for p in anchors], [p["flops"]/(p[f"{dt}_latency_us"]*1e6) for p in anchors],
                     "o-", color=COLORS[dt], label=dt.upper())
    axes[1].set(xscale="log", yscale="log", xlabel="M = N = K", ylabel="Achieved throughput (TFLOP/s)", title="Square anchors")
    axes[1].legend()
    save(fig, "03_precision")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, dt in zip(axes, DTYPES):
        for i, axis in enumerate("MNK"):
            bs = [b for b in boundary if b["dtype"] == dt and b["axis_plus_one"] == axis]
            values = [b["latency_ratio_neighbor_base"] for b in bs]
            jitter = np.linspace(-.12, .12, len(bs))
            for x, b in zip(i+jitter, bs):
                ax.scatter(x, b["latency_ratio_neighbor_base"], color=COLORS[dt],
                           marker="o" if b["neither_review_flag"] else "x", s=36)
            ax.plot([i-.2, i+.2], [np.median(values)]*2, color="black", lw=2)
        ax.axhline(1, color="black", ls="--", lw=1)
        ax.set(xticks=[0,1,2], xticklabels=["M + 1 (n=6)", "N + 1 (n=5)", "K + 1 (n=10)"],
               ylabel="Latency at +1 / latency at boundary", title=f"{dt.upper()}: matched neighboring shapes", ylim=(.7,2.1))
    fig.supxlabel("Other dimensions fixed and divisible by 8. Black bar: median; x: quality review flag.", fontsize=9)
    save(fig, "04_boundary")


if __name__ == "__main__":
    main()
