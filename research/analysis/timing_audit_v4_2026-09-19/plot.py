"""Scientific plots from verified measurements, without dropping any blocks."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
v = json.loads((HERE / "verification.json").read_text())
rows = {r["id"]:r for r in v["comparisons"]}
ids = ["c00", "c06", "c01", "c07", "c02"]
labels = ["64 cubed FP16 / prealloc", "64 cubed FP16 / Graph",
          "64 cubed FP32 / prealloc", "64 cubed FP32 / Graph", "128 cubed FP16 / prealloc"]
fig, ax = plt.subplots(figsize=(11,4.5))
left = [0.]*len(ids)
for key, label, color in [("gpu_activity","GPU work (all kernels + memory)","#246a90"),
                           ("before_api","Gap before next API entry","#d56526"),
                           ("during_api","Gap during next API","#e8ba4a"),
                           ("after_api","Gap after next API return","#8d91a0")]:
    values = [rows[i]["pooled_mean_activity_components_us"][key] for i in ids]
    ax.barh(labels, values, left=left, color=color, label=label, height=.65)
    left = [a+b for a,b in zip(left,values)]
ax.invert_yaxis()
ax.set_xlabel("Profiled GPU activity span per GEMM (microseconds)")
ax.set_title("Small GEMM: GPU work and internal gaps in the v4 trace", pad=18)
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(.43,-.17), ncol=2, frameon=False, fontsize=9)
fig.text(.5,.01,"Pooled means across all 7 blocks. Nsight is ON; excludes event-boundary time. API phases do not identify exclusive causes.",
         ha="center",fontsize=8)
fig.tight_layout(rect=(0,.06,1,1))
fig.savefig(HERE / "profiled_gaps.png",dpi=180)
fig.savefig(HERE / "profiled_gaps.pdf")
plt.close(fig)

v2 = HERE.parent / "timing_audit_2026-09-19/raw/timing_audit_20260919_175307_zz6mlt/nsys/results.json"
v4 = HERE / "raw/timing_audit_20260919_201729_y43Okw/nsys/results.json"
data = {name:{r["task"]["id"]:r["result"]["timing"] for r in json.loads(path.read_text())}
        for name,path in [("v2",v2),("v4",v4)]}
fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
for ax,case,title in zip(axes,["c03","c04","c05"],["3216 x 520 x 776 / FP16","4096 cubed / FP16","4096 cubed / FP32"]):
    for name,color in [("v2","#a76540"),("v4","#246a90")]:
        t=data[name][case]
        ax.plot(range(1,8),[100*b["latency_ms"]/t["median_ms"] for b in t["blocks"]],
                marker="o",markersize=3,linewidth=1.5,label=name,color=color)
    ax.axhline(100,color="gray",linestyle=":",linewidth=.7)
    ax.set_title(title,fontsize=10);ax.set_xlabel("Measurement block");ax.set_xticks(range(1,8))
    ax.grid(axis="y",alpha=.2);ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Latency / own run median (%)")
axes[1].legend(frameon=False,loc="lower right")
fig.suptitle("First-block transient after changing capture order")
fig.text(.5,.01,"All 7 blocks retained. Separate runs; normalized to each run's own median.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.035,1,.95))
fig.savefig(HERE / "first_block_comparison.png",dpi=180)
fig.savefig(HERE / "first_block_comparison.pdf")
