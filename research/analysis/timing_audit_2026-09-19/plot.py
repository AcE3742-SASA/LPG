"""Plot all seven blocks without dropping the first observation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = Path(__file__).resolve().parent
raw = p / "raw/timing_audit_20260919_175307_zz6mlt"
data = {name: {r["task"]["id"]: r["result"]["timing"]
              for r in json.loads((raw / name / "results.json").read_text())}
        for name in ("plain_before", "nsys", "plain_after")}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (case, title) in zip(axes, [("c03", "3216 x 520 x 776 / FP16"),
                                   ("c04", "4096 cubed / FP16"), ("c05", "4096 cubed / FP32")]):
    baseline = (data["plain_before"][case]["median_ms"] + data["plain_after"][case]["median_ms"])/2
    for name, label, color in [("plain_before", "Plain before", "#177e89"),
                               ("nsys", "Nsight Systems", "#c75b12"),
                               ("plain_after", "Plain after", "#5352a6")]:
        ax.plot(range(1, 8), [100*b["latency_ms"]/baseline for b in data[name][case]["blocks"]],
                marker="o", linewidth=1.7, markersize=4, label=label, color=color)
    ax.axhline(100, color="gray", linewidth=.7, linestyle=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Measurement block")
    ax.set_xticks(range(1, 8))
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Latency / plain reference (%)")
axes[1].legend(loc="lower right", frameon=False, fontsize=9)
fig.suptitle("First-block transient in the profiled run", fontsize=14)
fig.text(.5, .01, "Reference: mean of the two plain-run medians. All blocks retained; three separate sessions.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .045, 1, .95))
fig.savefig(p / "first_block_transient.png", dpi=180)
fig.savefig(p / "first_block_transient.pdf")
