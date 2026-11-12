"""Read-only SQLite analysis; export never changes raw observations."""
import csv
import contextlib
import html
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from plan import digest, task_seed


def records(out):
    db = out / "results.sqlite3"
    if not db.exists():
        return []
    with contextlib.closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as con:
        return [json.loads(r[0]) for r in con.execute("SELECT record FROM attempts ORDER BY seq")]


def paired_rows(manifest, latest):
    planned = defaultdict(dict)
    for task in manifest["tasks"]:
        if task["kind"] == "gemm" and task["mode"] in ("prealloc", "graph"):
            key = digest({k: v for k, v in task.items() if k not in ("id", "mode")})
            planned[key][task["mode"]] = task
    lookup = {row["task"]["id"]: row for row in latest}
    output = []
    for pair in planned.values():
        if set(pair) != {"prealloc", "graph"}:
            continue
        task = pair["prealloc"]
        row = {k: v for k, v in task.items() if k not in ("id", "mode")}
        row.update(prealloc_id=task["id"], graph_id=pair["graph"]["id"],
                   input_seed=task_seed(task), prealloc_status="MISSING", graph_status="MISSING",
                   prealloc_ms=None, graph_ms=None, graph_gemms_per_replay=None,
                   graph_over_prealloc=None, pair_complete=False)
        for mode, member in pair.items():
            data = lookup.get(member["id"])
            if data is None:
                continue
            row[mode + "_status"] = data["status"]
            if data["status"] not in ("PASS", "REVIEW") or "timing" not in data:
                continue
            if data.get("input_seed") != row["input_seed"]:
                raise ValueError("Paired GEMM input seed mismatch")
            row[mode + "_ms"] = data["timing"]["median_ms"]
            row[mode + "_tflops"] = data.get("tflops")
            if mode == "graph":
                row["graph_gemms_per_replay"] = data["timing"]["graph_nodes"]
        if row["prealloc_ms"] is not None and row["graph_ms"] is not None:
            row["pair_complete"] = True
            row["graph_over_prealloc"] = row["graph_ms"] / row["prealloc_ms"]
        output.append(row)
    return output


def summary(out):
    manifest = json.loads((out / "manifest.json").read_text())
    rows = records(out)
    latest = {r["task"]["id"]: r for r in rows}
    planned = {t["id"] for t in manifest["tasks"]}
    unknown = set(latest) - planned
    if unknown:
        raise ValueError(f"Unplanned records: {unknown}")
    counts = Counter(r["status"] for r in latest.values())
    missing = len(planned - set(latest))
    pairs = defaultdict(list)
    for r in latest.values():
        if r["status"] not in ("PASS", "REVIEW") or "timing" not in r:
            continue
        task = {k: v for k, v in r["task"].items() if k not in ("round", "id")}
        pairs[digest(task)].append(r)
    between = []
    for group in pairs.values():
        if len(group) < 2:
            continue
        values = [r["timing"]["median_ms"] for r in group]
        spread = (max(values)-min(values))/statistics.median(values)
        if spread > manifest["protocol"]["between_round_limit"]:
            between.append({"task": group[0]["task"], "relative_range": spread,
                            "round_median_ms": values})
    failed = counts["FAIL"] > 0
    gate = "FAIL" if failed else ("INCOMPLETE" if missing else (
        "REVIEW" if counts["REVIEW"] or between else "PASS"))
    mode_pairs = paired_rows(manifest, list(latest.values()))
    return {"gate": gate, "expected": len(planned), "recorded": len(latest),
            "missing": missing, "attempts": len(rows), "counts": dict(counts),
            "between_round_flags": len(between), "between_round": between,
            "result_digest": digest(rows), "size": manifest["size"],
            "paired_expected": len(mode_pairs),
            "paired_complete": sum(p["pair_complete"] for p in mode_pairs)}, list(latest.values())


def csv_write(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(out):
    result, latest = summary(out)
    between_keys = {digest({k: v for k, v in b["task"].items() if k not in ("round", "id")})
                    for b in result["between_round"]}
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with (out / "raw_attempts.jsonl").open("w") as f:
        for row in records(out):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    table, holdout, blockrows = [], [], []
    for r in latest:
        t = r["task"]
        flat = {**t, "status": r["status"], "session": r["session"], "timestamp": r["timestamp"]}
        flat.update({k: v for k, v in r.get("timing", {}).items()
                     if k not in ("blocks", "setup")})
        for key in ("tflops", "flops", "ideal_bytes", "ideal_oi", "effective_copy_gbps", "input_seed"):
            if key in r:
                flat[key] = r[key]
        flat["correctness_passed"] = r.get("correctness", {}).get("passed")
        flat["error"] = r.get("error", "")
        flat["between_round_review"] = digest({k: v for k, v in t.items()
                                               if k not in ("round", "id")}) in between_keys
        (holdout if t["split"] == "fresh_holdout" else table).append(flat)
        for i, block in enumerate(r.get("timing", {}).get("blocks", [])):
            blockrows.append({"id": t["id"], "split": t["split"], "block": i, **block})
    csv_write(out / "metrics.csv", table)
    csv_write(out / "fresh_holdout_LOCKED.csv", holdout)
    csv_write(out / "blocks.csv", blockrows)
    manifest = json.loads((out / "manifest.json").read_text())
    mode_pairs = paired_rows(manifest, latest)
    for pair in mode_pairs:
        for mode in ("prealloc", "graph"):
            key_task = {k: v for k, v in pair.items() if k in ("kind", "phase", "shape", "dtype", "split", "group")}
            pair[mode + "_between_round_review"] = digest(dict(key_task, mode=mode)) in between_keys
    csv_write(out / "paired_metrics.csv", [p for p in mode_pairs if p["split"] != "fresh_holdout"])
    csv_write(out / "paired_fresh_holdout_LOCKED.csv", [p for p in mode_pairs if p["split"] == "fresh_holdout"])
    csv_write(out / "shape_splits.csv", manifest["shape_splits"])
    csv_write(out / "review_queue.csv", [r for r in table if r["status"] != "PASS" or r["between_round_review"]])
    # Render every pilot condition; limit full-run charts to flagged development data.
    charts = []
    for r in latest:
        if r["task"]["split"] == "fresh_holdout" or "timing" not in r:
            continue
        if result["size"] == "full" and r["status"] == "PASS":
            continue
        vals = [b["latency_ms"] for b in r["timing"]["blocks"]]
        med = statistics.median(vals)
        scaled = [100*(v/med-1) for v in vals]
        span = max(6, max(abs(v) for v in scaled))
        points = " ".join(f"{20+i*260/max(1,len(vals)-1):.1f},{70-v/span*50:.1f}"
                          for i, v in enumerate(scaled))
        t = r["task"]
        label = f"{t.get('phase', t['kind'])} {t.get('shape', t.get('widths', t.get('mib')))} {t.get('dtype','')} {t['mode']} round={t.get('round','')}"
        charts.append(f'<article><b>{html.escape(label)}</b><p>{med*1000:.4g} µs · '
                      f'IQR {r["timing"]["relative_iqr"]*100:.3g}% · {r["status"]}</p>'
                      f'<svg viewBox="0 0 300 140"><path d="M20 70H280" stroke="#aaa"/>'
                      f'<polyline points="{points}" fill="none" stroke="#1262b7" stroke-width="2"/>'
                      f'<text x="20" y="135">block order; median ±{span:.2g}%</text></svg></article>')
    between = [{**b, "round_median_ms": "withheld"} if b["task"]["split"] == "fresh_holdout"
               else b for b in result["between_round"]]
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>GEMM 파일럿/스윕 결과</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px;color:#222}}
section{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px}}
article{{border:1px solid #ccc;padding:12px}}svg{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>{result['size']}: {result['gate']}</h1>
<p>기록 {result['recorded']}/{result['expected']}, 누락 {result['missing']}, 상태 {html.escape(str(result['counts']))}.</p>
<p>prealloc·Graph 쌍 {result['paired_complete']}/{result['paired_expected']}. 두 시간은 paired_metrics.csv의 별도 열에 저장한다.
Graph도 내부 GPU 간격을 포함하며 두 시간의 차이를 순수 CPU 비용으로 해석하지 않는다.</p>
<p>실행 간 5% 초과 {result['between_round_flags']}조건. PASS도 사람이 결과를 확인한 뒤 풀 스윕을 시작한다.
REVIEW는 오류가 아니라 변동 검토 대상이다. 반복 round는 같은 프로세스 안의 반복이며 독립 세션이 아니다.</p>
<p>짧은 GEMM의 event 구간에는 CPU 제출 간격이 포함될 수 있다. Graph, allocate, pressure는 서로 다른 측정 조건이다.
Q_ideal은 실제 DRAM 전송량이 아니며 copy GB/s와 GEMM 최대값은 실험 기준값이다. 프로파일링 시간은 학습 정답으로 쓰지 않는다.</p>
<h2>블록별 양상</h2><section>{''.join(charts)}</section>
<h2>round 간 차이</h2><pre>{html.escape(json.dumps(between, indent=2, ensure_ascii=False))}</pre>
</html>'''
    (out / "report.html").write_text(page)
    return result
