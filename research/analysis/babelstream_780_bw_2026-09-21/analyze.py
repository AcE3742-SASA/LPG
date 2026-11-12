"""Audit a BabelStream result archive without executing its contents.

Run: python3 analyze.py /path/to/roofline_780_bw.eAk4tz.tar.gz
Original bytes and hashes are retained; r0 is excluded from measurement statistics.
"""
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics as st
import sys
import tarfile


ROOT = Path(__file__).resolve().parent
SIZES = (67108864, 134217728, 268435456)
WEIGHTS = {"Copy": 2, "Mul": 2, "Add": 3, "Triad": 3, "Dot": 2}
PIN = "17ab377b0e919e14fd3df2b67268761fdac8abb3"


def preserve(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f"Existing original differs: {path}"
    else:
        with path.open("xb") as stream:
            stream.write(data)


def main(archive):
    blob = archive.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    data = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts, member.name
            if member.isdir() and str(path) == ".":
                continue
            assert member.isfile() and len(path.parts) == 1, member.name
            assert member.size < 2_000_000 and path.name not in data, member.name
            data[path.name] = bundle.extractfile(member).read()
    runs = [f"r{r}_n{n}" for r in range(4) for n in SIZES]
    expected = {"results.csv", "device.txt", "clocks.csv", "before.csv", "binary.sha256",
                "timeline.tsv", "commit.txt", "monitor.err"}
    expected |= {r + suffix for r in runs for suffix in (".log", ".exit_code")}
    assert set(data) == expected, (set(data) - expected, expected - set(data))
    preserve(ROOT / "raw" / archive.name, blob)
    for name, contents in data.items():
        preserve(ROOT / "raw" / "extracted" / name, contents)
    manifest = {name: {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
                for name, contents in sorted(data.items())}
    (ROOT / "manifest.json").write_text(json.dumps({"archive": archive.name,
        "sha256": digest, "files": manifest}, indent=2) + "\n")
    text = {name: contents.decode("utf-8") for name, contents in data.items()}
    assert text["commit.txt"].strip() == PIN
    assert re.fullmatch(r"[0-9a-f]{64}\s+\S+\s*", text["binary.sha256"])
    assert not text["monitor.err"].strip()
    rows = list(csv.DictReader(io.StringIO(text["results.csv"])))
    assert len(rows) == 60
    assert len({(r["run"], r["function"]) for r in rows}) == 60
    errors = []
    for run in runs:
        assert text[run + ".exit_code"].strip() == "0", run
        log = text[run + ".log"]
        assert not re.search(r"FAILED|CUDA Error|NVML Error|exception", log, re.I), run
        assert "Memory Allocation: DEFAULT" in log and '6801 MHz' in log
        csv_start = log.index("function,num_times,")
        original = list(csv.DictReader(io.StringIO(log[csv_start:])))
        selected = [r for r in rows if r["run"] == run]
        assert len(original) == len(selected) == 5
        assert {r["function"] for r in selected} == set(WEIGHTS)
        assert original == [{k: v for k, v in r.items() if k != "run"} for r in selected]
        for row in selected:
            assert row["num_times"] == "100" and row["sizeof"] == "4"
            assert int(row["n_elements"]) == int(run.split("_n")[1])
            low, high, avg = (float(row[k]) for k in ("min_runtime", "max_runtime", "avg_runtime"))
            bw = float(row["max_GB_per_sec"])
            assert all(math.isfinite(v) and v > 0 for v in (low, high, avg, bw))
            assert low <= avg <= high
            expected_bw = WEIGHTS[row["function"]] * 4 * int(row["n_elements"]) / low / 1e9
            errors.append(abs(expected_bw / bw - 1))
    # Six-significant-digit CSV values are rounded; this is an arithmetic consistency check.
    assert max(errors) < 2e-5

    windows = {}
    for run, event, value in csv.reader(io.StringIO(text["timeline.tsv"]), delimiter="\t"):
        assert run in runs and event in ("start", "end")
        assert event not in windows.setdefault(run, {})
        windows[run][event] = dt.datetime.fromisoformat(value.replace(",", "."))
    assert set(windows) == set(runs)
    timezone = windows[runs[0]]["start"].tzinfo
    assert timezone is not None
    clocks = []
    for row in csv.DictReader(io.StringIO(text["clocks.csv"]), skipinitialspace=True):
        assert None not in row and all(v is not None for v in row.values())
        when = dt.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone)
        numeric = [float(row[k].split()[0]) for k in list(row)[1:]]
        assert len(numeric) == 6 and all(math.isfinite(v) for v in numeric)
        util, sm, mem, power, limit, temp = numeric
        assert 0 <= util <= 100 and power >= 0
        clocks.append(dict(time=when, util=util, sm=sm, mem=mem, power=power, limit=limit, temp=temp))
    assert len(clocks) > 1
    gaps = [(b["time"] - a["time"]).total_seconds() for a, b in zip(clocks, clocks[1:])]
    assert min(gaps) > 0
    per_run = {}
    for i, run in enumerate(runs):
        window = windows[run]
        assert window["start"] < window["end"]
        if i:
            assert windows[runs[i-1]]["end"] <= window["start"]
        inside = [c for c in clocks if window["start"] <= c["time"] <= window["end"]]
        active = [c for c in inside if c["util"] > 0]
        assert active, run
        per_run[run] = {"start": window["start"].isoformat(), "end": window["end"].isoformat(),
            "wall_seconds": (window["end"] - window["start"]).total_seconds(),
            "samples": len(inside), "active_samples": len(active)}
    mismatches = [c for c in clocks if (c["sm"], c["mem"], c["limit"]) != (780, 6801, 70)]

    summary = []
    for n in SIZES:
        for function, weight in WEIGHTS.items():
            selected = [r for r in rows if r["run"] in {f"r{k}_n{n}" for k in (1, 2, 3)}
                        and r["function"] == function]
            assert len(selected) == 3
            values = [float(r["max_GB_per_sec"]) for r in selected]
            median = st.median(values)
            average_based = [weight * 4 * n / float(r["avg_runtime"]) / 1e9 for r in selected]
            summary.append(dict(n_elements=n, total_GiB=3*4*n/2**30, function=function,
                median_GB_s=median, min_GB_s=min(values), max_GB_s=max(values),
                range_over_median_pct=100*(max(values)-min(values))/median,
                first_to_last_pct=100*(values[-1]/values[0]-1),
                median_average_time_based_GB_s=st.median(average_based)))
    candidate = max(summary, key=lambda r: r["median_GB_s"])
    telemetry = dict(samples=len(clocks), active_samples=sum(c["util"] > 0 for c in clocks),
        fully_busy_samples=sum(c["util"] == 100 for c in clocks),
        mismatches=len(mismatches), sm_values=sorted({c["sm"] for c in clocks}),
        memory_values=sorted({c["mem"] for c in clocks}), power_limit_values=sorted({c["limit"] for c in clocks}),
        min_temperature_C=min(c["temp"] for c in clocks), max_temperature_C=max(c["temp"] for c in clocks),
        max_power_W=max(c["power"] for c in clocks), max_poll_gap_s=max(gaps),
        first=clocks[0]["time"].isoformat(), last=clocks[-1]["time"].isoformat())
    result = dict(archive_sha256=digest, file_count=len(data), status="PASS" if not mismatches else "REVIEW",
        total_runs=12, warmup_runs=3, measured_runs=9, kernel_rows=60, measured_kernel_rows=45,
        arithmetic_max_relative_error=max(errors), telemetry=telemetry, per_run=per_run,
        summary=summary, bandwidth_candidate=candidate,
        max_repeat_range_pct=max(r["range_over_median_pct"] for r in summary),
        scope="Within-session short-run audit; not uninterrupted clock proof or an application-independent hardware maximum.")
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(json.dumps({k:v for k,v in result.items() if k not in ("per_run", "summary")}, indent=2))
    for r in summary:
        print(f'{r["total_GiB"]:4g} GiB {r["function"]:5} median={r["median_GB_s"]:.1f} GB/s range/median={r["range_over_median_pct"]:.2f}%')


if __name__ == "__main__":
    main(Path(sys.argv[1]))
