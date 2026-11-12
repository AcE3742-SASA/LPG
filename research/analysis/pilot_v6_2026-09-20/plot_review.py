"""Standalone figure of all seven timed blocks in the two flagged conditions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = [json.loads(x) for x in (HERE / "raw/pilot_v6/raw_attempts.jsonl").read_text().splitlines()]
cases = [([1888, 832, 632], "fp16", "graph"), ([392, 2776, 80], "fp32", "prealloc")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
for ax, (shape, dtype, mode) in zip(axes, cases):
    part = [r for r in rows if (r["task"]["shape"], r["task"]["dtype"], r["task"]["mode"]) == (shape, dtype, mode)]
    assert len(part) == 2
    for r, color in zip(part, ("#176b9c", "#c2522c")):
        values = [b["latency_ms"] * 1000 for b in r["timing"]["blocks"]]
        assert len(values) == 7
        ax.plot(range(1, 8), values, "o-", color=color, linewidth=1.7,
                label=f"Round {r['task']['round']}: median {r['timing']['median_ms']*1000:.1f} us")
    ax.set(title=f"{shape[0]} x {shape[1]} x {shape[2]} | {dtype.upper()} {mode}",
           xlabel="Timed block (all blocks retained)", ylabel="GEMM time (us)", xticks=range(1,8))
    ax.grid(axis="y", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=9)
fig.suptitle("v6 pilot: a steady block series can still shift between rounds", fontsize=13)
fig.savefig(HERE / "review_blocks.png", dpi=180)
fig.savefig(HERE / "review_blocks.pdf")
