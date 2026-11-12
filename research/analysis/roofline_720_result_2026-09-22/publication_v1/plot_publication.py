"""Render publication figures at their final physical sizes; python3 -B plot_publication.py."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/roofline-720-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
for name, digest in json.loads((ROOT/'source_hashes.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
p = json.loads((ROOT/'roofline_parameters.json').read_text())
s = json.loads((ROOT/'analysis.json').read_text())
c = s['ceiling_candidates']
assert s['status'] == 'PASS' and s['collection_complete']
assert [p['sm_MHz'], p['memory_MHz'], p['power_limit_W']] == [720, 6801, 70]
assert all(r['repeats'] == 5 for r in s['summaries'])
assert [c[k]['median'] for k in ('simt', 'tensorop', 'bw')] == [
    p['fp32_TFLOP_s'], p['fp16_tensor_fp32_accum_TFLOP_s'], p['bandwidth_GB_s']]
bw = p['bandwidth_GB_s']
peaks = [p['fp32_TFLOP_s'], p['fp16_tensor_fp32_accum_TFLOP_s']]
knees = np.array(peaks)*1000/bw
curve = np.loadtxt(ROOT/'roofline_curve.csv', delimiter=',', skiprows=1)
for i, peak in enumerate(peaks):
    assert np.allclose(curve[:, i+1], np.minimum(peak, bw*curve[:, 0]/1000), rtol=1e-14)
intensity = np.unique(np.r_[curve[:, 0], knees])
BLUE, ORANGE, GRAY = '#0072B2', '#D55E00', '#777777'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
    'axes.titlesize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'axes.linewidth': .65, 'lines.linewidth': 1.25,
    'xtick.major.width': .65, 'ytick.major.width': .65,
    'xtick.minor.width': .45, 'ytick.minor.width': .45,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 4, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.bbox': None,
})
checks = []


def export(fig, name, width_mm, height_mm):
    """Keep exact page dimensions; reject clipped text or smaller-than-intended labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outer = fig.bbox
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    for t in texts:
        assert t.get_fontsize() >= 10, (name, t.get_text(), t.get_fontsize())
        box = t.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= outer.x1+1 and box.y1 <= outer.y1+1, (name, t.get_text(), box.bounds)
    assert np.allclose(fig.get_size_inches()*25.4, [width_mm, height_mm], atol=1e-10)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(ROOT/f'{name}.{ext}', dpi=600)
    checks.append(dict(figure=name, width_mm=width_mm, height_mm=height_mm,
                       min_font_pt=min(t.get_fontsize() for t in texts),
                       text_bounds='PASS', source_values='unchanged', png_dpi=600))
    plt.close(fig)


def roofline(width):
    height = 80 if width == 89 else 92
    fig, ax = plt.subplots(figsize=(width/25.4, height/25.4))
    fig.subplots_adjust(left=16/width, right=1-4/width, bottom=14/height, top=1-4/height)
    for peak, color, ls, label in zip(peaks, (BLUE, ORANGE), ('-', (0, (5, 2.5))),
                                     ('FP32 SIMT', 'FP16 Tensor Core')):
        ax.loglog(intensity, np.minimum(peak, bw*intensity/1000), color=color,
                  ls=ls, lw=1.6, label=label, zorder=3)
        ax.text(3300, peak*1.13, f'{peak:.2f}', ha='right', va='bottom', color='black')
    ax.loglog(intensity, bw*intensity/1000, color=GRAY, ls=':', lw=1,
              label=f'B = {bw:.1f} GB/s', zorder=1)
    for knee, peak, color in zip(knees, peaks, (BLUE, ORANGE)):
        ax.plot(knee, peak, 'o', ms=3.5, color=color, zorder=4)
    ax.set(xlim=(.1, 4096), ylim=(.02, 100), xlabel='Operational intensity (FLOP/byte)',
           ylabel='Performance (TFLOP/s)')
    ax.set_xticks([.1, 1, 10, 100, 1000])
    ax.set_yticks([.1, 1, 10, 100])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which='major', color='#DDDDDD', lw=.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False, borderpad=.1, labelspacing=.35,
              handlelength=1.7, handletextpad=.5)
    export(fig, f'figure1_roofline_{width}mm', width, height)


def scaling(width):
    height = 180 if width == 89 else 80
    fig, axes = plt.subplots(3 if width == 89 else 1, 1 if width == 89 else 3,
                             figsize=(width/25.4, height/25.4))
    fig.subplots_adjust(left=15/width, right=1-4/width, bottom=14/height,
                        top=1-8/height, hspace=.70, wspace=.52)
    for ax, path, title, color, maximum in zip(
            axes[:2], ('simt', 'tensorop'), ('(a) FP32 SIMT', '(b) FP16 Tensor Core'),
            (BLUE, ORANGE), (7, 40)):
        rows = [r for r in s['summaries'] if r['path'] == path]
        for kernel in sorted({r['function'] for r in rows}):
            group = sorted((r for r in rows if r['function'] == kernel), key=lambda r:r['size'])
            x = [r['size'] for r in group]
            y = [r['median'] for r in group]
            chosen = kernel == c[path]['function']
            ax.plot(x, y, color=color if chosen else '#999999',
                    lw=1.35 if chosen else .65, marker='o' if chosen else '.',
                    ms=3.7 if chosen else 2.4, ls='-' if chosen else '--',
                    zorder=3 if chosen else 1)
            if chosen:
                err = [[r['median']-r['min'] for r in group], [r['max']-r['median'] for r in group]]
                ax.errorbar(x, y, yerr=err, fmt='none', ecolor=color, capsize=2,
                            elinewidth=.8, capthick=.8, zorder=4)
                ax.annotate(f'{y[-1]:.2f}', (x[-1], y[-1]), xytext=(0, 7),
                            textcoords='offset points', ha='center')
        ax.set(xlim=(3000, 18400), ylim=(0, maximum), xlabel='M = N = K', ylabel='TFLOP/s')
        ax.set_xticks([4096, 8192, 16384])
        ax.set_yticks([0, 2, 4, 6] if path == 'simt' else [0, 10, 20, 30, 40])
        ax.set_title(title, loc='left', pad=8)
    ax = axes[2]
    styles = [('Copy', BLUE, 'o', '-'), ('Mul', '#555555', 's', '--'),
              ('Add', '#009E73', '^', '-.'), ('Triad', '#9E508C', 'D', ':'),
              ('Dot', ORANGE, 'v', '-')]
    for function, color, marker, ls in styles:
        group = sorted((r for r in s['summaries'] if r['path'] == 'bw' and r['function'] == function),
                       key=lambda r:r['size'])
        x = [3*4*r['size']/2**30 for r in group]
        y = [r['median'] for r in group]
        err = [[r['median']-r['min'] for r in group], [r['max']-r['median'] for r in group]]
        ax.errorbar(x, y, yerr=err, color=color, marker=marker, ls=ls, ms=3.5,
                    capsize=2, lw=1.15, elinewidth=.8, capthick=.8, label=function)
        if function == 'Dot':
            ax.annotate(f'{y[-1]:.1f}', (x[-1], y[-1]), xytext=(0, 7),
                        textcoords='offset points', ha='center')
    ax.set(xlim=(.5, 3.4), ylim=(0, 310), xlabel='Array footprint (GiB)', ylabel='Bandwidth (GB/s)')
    ax.set_xticks([.75, 1.5, 3])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _:f'{x:g}'))
    ax.set_yticks([0, 100, 200, 300])
    ax.set_title('(c) Memory bandwidth', loc='left', pad=8)
    ax.legend(loc='lower center', ncol=2, frameon=False, borderpad=.1,
              columnspacing=.7, handlelength=1.2, handletextpad=.35, labelspacing=.25)
    for ax in axes:
        ax.grid(axis='y', color='#DDDDDD', lw=.4)
        ax.set_axisbelow(True)
    export(fig, f'figure2_scaling_{width}mm', width, height)


for width in (89, 182):
    roofline(width)
    scaling(width)
(ROOT/'layout_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
print(json.dumps(checks, indent=2))
