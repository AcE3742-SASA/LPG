#!/usr/bin/env python3
"""Plot the quality-approved paired ratios; retain the REVIEW point separately."""
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'layer_compare_plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
values = json.loads((HERE / 'recomputed/summary.json').read_text())['comparisons']
groups = json.loads((HERE / 'group_summary.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), layout='constrained')
colors = {'fp16': '#2768ad', 'fp32': '#de7843'}
rng = np.random.default_rng(20261007)
for ax, layers, labels in zip(axes, [('conv1', 'conv3'), ('linear', 'ffn')],
                             [('Conv 1x1', 'Conv 3x3'), ('Linear', 'GELU FFN')]):
    for index, layer in enumerate(layers):
        for dtype, offset in [('fp16', -.18), ('fp32', .18)]:
            group = next(g for g in groups if g['layer'] == layer and g['dtype'] == dtype)
            ratios = np.array([v['ratio'] for v in values if v['layer'] == layer and
                               v['dtype'] == dtype and v['status'] == 'PASS'])
            x = index + offset
            ax.scatter(x + rng.uniform(-.09, .09, len(ratios)), ratios, s=21,
                       color=colors[dtype], alpha=.53, linewidths=0, label=dtype.upper() if index == 0 else None)
            lo, median, hi = np.quantile(ratios, [.25, .5, .75])
            ax.plot([x, x], [lo, hi], color='black', linewidth=3, solid_capstyle='round')
            ax.plot([x-.09, x+.09], [median, median], color='black', linewidth=2)
            ax.text(x, median * (1.13 if ax is axes[0] else .87), f'{median:.2f}x\nn={len(ratios)}',
                    ha='center', va='top' if ax is axes[1] else 'bottom', fontsize=9)
            for value in values:
                if value['layer'] == layer and value['dtype'] == dtype and value['status'] == 'REVIEW':
                    ax.scatter([x+.12], [value['ratio']], marker='D', s=64, facecolors='none',
                               edgecolors='#a32634', linewidths=1.6, label='REVIEW (excluded)')
    ax.axhline(1, color='#777777', linestyle='--', linewidth=1)
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-.5, 1.5)
    ax.grid(axis='y', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Decomposed time / standard time')
    ax.legend(loc='upper right', fontsize=9, frameon=False)
axes[0].set_ylim(.75, 10)
axes[1].set_ylim(.65, 1.95)
axes[0].set_title('Standard Conv faster in all 255 PASS comparisons', fontsize=11)
axes[1].set_title('Linear / FFN: smaller, shape-dependent differences', fontsize=11)
fig.suptitle('PyTorch layer comparison: 160 development shapes, 3 rounds', fontsize=14)
fig.supxlabel('Points: PASS comparisons; black line: median and IQR. Above 1: standard faster.\nNCHW/2D; FP32 IEEE; ordinary forward; RTX 4000 SFF Ada at 720 MHz.', fontsize=10)
for suffix in ('png', 'pdf'):
    fig.savefig(HERE / ('latency_ratio.' + suffix), dpi=180)
print('PLOT PASS: latency_ratio.png and latency_ratio.pdf')
