"""Plot paired, observed batch-size sensitivity; shaded band is not uncertainty."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'recomputed/analysis.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
for ax, case in zip(axes, data['cases']):
    sizes = case['sizes']
    batches = [s['gemms_per_replay'] for s in sizes]
    ref = sizes[-1]['round_medians_ms']
    curves = [[100 * (1 - ref[rd] / s['round_medians_ms'][rd]) for s in sizes] for rd in range(4)]
    ax.axhspan(0, 1, color='#e6f1ed', zorder=0)
    ax.axhline(1, color='#577b6c', linestyle='--', linewidth=1)
    for rd, curve in enumerate(curves):
        ax.plot(batches, curve, 'o-', color='#7793ac', alpha=.55, linewidth=.8,
                markersize=3, label='Each process run' if rd == 0 else None)
    medians = [statistics.median(v) for v in zip(*curves)]
    ax.plot(batches, medians, 'o-', color='#172f48', linewidth=1.8, markersize=4, label='Paired median')
    ax.set_xscale('log', base=2)
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.set_ylim(-.07, 2.13)
    ax.set_xlabel('GEMMs per Graph replay (B)')
    shape = case['case']['shape'][0]
    ax.set_title(f"{shape} x {shape} x {shape}  {case['case']['dtype'].upper()}", fontweight='bold')
    ax.grid(axis='y', alpha=.16)
axes[0].set_ylabel('Time reduction when increasing B to 128 (%)')
axes[0].legend(frameon=False, loc='upper right', fontsize=8)
fig.suptitle('Unprofiled CUDA Graph batch-size comparison', fontsize=14, fontweight='bold')
fig.text(.5, .025, '4 process runs; 7 timed blocks per point. Green band: predeclared 1% comparison margin, not a confidence interval.',
         ha='center', fontsize=8)
fig.tight_layout(rect=(0, .055, 1, .93))
fig.savefig(HERE / 'batch_convergence.png', dpi=190)
fig.savefig(HERE / 'batch_convergence.pdf')
