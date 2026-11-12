"""Reproduce the Roofline and size-scaling figures from audited saved measurements."""
from pathlib import Path
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/roofline-720-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
s = json.loads((ROOT/'recomputed/analysis.json').read_text())
audit = json.loads((ROOT/'audit.json').read_text())
assert s['status']=='PASS' and audit['status']=='AUDIT_PASS'
c = s['ceiling_candidates']
bw = c['bw']['median']
p32, p16 = c['simt']['median'], c['tensorop']['median']
k32, k16 = p32*1000/bw, p16*1000/bw
I = np.geomspace(.1, 4096, 900)
y32 = np.minimum(p32, bw*I/1000)
y16 = np.minimum(p16, bw*I/1000)
assert np.all(np.diff(y32)>=0) and np.all(np.diff(y16)>=0)
assert y32[-1]==p32 and y16[-1]==p16 and np.isclose(y32[0]/I[0],bw/1000)
parameters = dict(sm_MHz=720,memory_MHz=6801,power_limit_W=70,
    fp32_TFLOP_s=p32,fp16_tensor_fp32_accum_TFLOP_s=p16,bandwidth_GB_s=bw,
    fp32_ridge_FLOP_per_byte=k32,fp16_ridge_FLOP_per_byte=k16,
    formula='P_TFLOP_s(I) = min(P_compute_TFLOP_s, B_GB_s * I_FLOP_per_byte / 1000)',
    aggregation='Maximum of five-run medians among tested kernel/size groups; round 0 excluded',
    archive_sha256=audit['archive_sha256'])
(ROOT/'roofline_parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
with (ROOT/'roofline_curve.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['OI_FLOP_per_byte','FP32_TFLOP_s','FP16_tensor_TFLOP_s'])
    w.writerows(zip(I,y32,y16))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
                     'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
                     'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white',
                     'axes.spines.top':False,'axes.spines.right':False})
BLUE, ORANGE = '#0072B2', '#D55E00'

def export(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(ROOT/(name+'.'+ext),dpi=300,facecolor='white')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(10.4,6.8))
fig.subplots_adjust(left=.10,right=.97,bottom=.19,top=.81)
fig.text(.10,.94,'Empirical Roofline at 720 MHz',fontsize=21,fontweight='bold')
fig.text(.10,.891,'NVIDIA RTX 4000 SFF Ada | SM 720 MHz | Memory 6801 MHz | Power limit 70 W',fontsize=11,color='#444444')
ax.loglog(I,bw*I/1000,color='#8A8A8A',ls=':',lw=1.5,label=f'Streaming bandwidth: {bw:.1f} GB/s')
ax.loglog(I,y32,color=BLUE,lw=2.8,label='FP32 SIMT')
ax.loglog(I,y16,color=ORANGE,lw=2.5,ls=(0,(5,2.5)),label='FP16 Tensor Core / FP32 accumulation')
for knee,peak,color in ((k32,p32,BLUE),(k16,p16,ORANGE)):
    ax.plot(knee,peak,'o',ms=6,color=color)
    ax.vlines(knee,.02,peak,color=color,ls=':',lw=1,alpha=.5)
ax.annotate(f'{k32:.1f} FLOP/byte',xy=(k32,p32),xytext=(11,12.3),fontsize=10,color=BLUE,
            arrowprops=dict(arrowstyle='-',color=BLUE,lw=.8))
ax.annotate(f'{k16:.1f} FLOP/byte',xy=(k16,p16),xytext=(55,61),fontsize=10,color=ORANGE,
            arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.8))
ax.text(470,p32*1.16,f'{p32:.2f} TFLOP/s',color=BLUE,fontweight='bold',fontsize=12)
ax.text(470,p16*1.15,f'{p16:.2f} TFLOP/s',color=ORANGE,fontweight='bold',fontsize=12)
ax.set(xlim=(.1,4096),ylim=(.02,90),xlabel='Operational intensity (FLOP/byte)',ylabel='Performance (TFLOP/s)')
ax.set_xticks([.1,1,10,100,1000]);ax.set_yticks([.03,.1,1,10,50])
fmt=FuncFormatter(lambda x,_:f'{x:g}')
ax.xaxis.set_major_formatter(fmt);ax.yaxis.set_major_formatter(fmt)
ax.grid(which='major',color='#E5E5E5',lw=.7);ax.set_axisbelow(True)
ax.legend(loc='upper left',frameon=False,fontsize=10)
fig.text(.10,.088,'Ceilings: highest five-run median over the tested kernels and sizes; warmup round excluded.',fontsize=9.5,color='#444444')
fig.text(.10,.052,'Bandwidth: BabelStream logical bytes. FP16 Tensor Core output is FP16. Telemetry diagnostics are in the report.',fontsize=9.5,color='#444444')
export(fig,'roofline_720')

fig,axes=plt.subplots(1,3,figsize=(14.5,5.5))
fig.subplots_adjust(left=.055,right=.985,bottom=.22,top=.76,wspace=.27)
fig.text(.055,.93,'Size scaling and repeatability',fontsize=20,fontweight='bold')
fig.text(.055,.872,'720/6801 MHz, 70 W | Median of five runs; error bars show the observed minimum and maximum',fontsize=11,color='#444444')
for ax,path,title,color,limit in zip(axes[:2],('simt','tensorop'),('FP32 SIMT','FP16 Tensor Core / FP32 accumulation'),(BLUE,ORANGE),(8,45)):
    rows=[r for r in s['summaries'] if r['path']==path]
    for kernel in sorted({r['function'] for r in rows}):
        group=sorted([r for r in rows if r['function']==kernel],key=lambda r:r['size'])
        x=np.array([r['size'] for r in group]);y=np.array([r['median'] for r in group])
        selected=kernel==c[path]['function']
        ax.plot(x,y,color=color if selected else '#B8BFC7',lw=2 if selected else .9,
                marker='o' if selected else '.',ms=5 if selected else 3,zorder=3 if selected else 1)
        if selected:
            err=np.array([[r['median']-r['min'] for r in group],[r['max']-r['median'] for r in group]])
            ax.errorbar(x,y,yerr=err,fmt='none',ecolor=color,capsize=3,zorder=4)
            for xx,yy in zip(x,y):ax.annotate(f'{yy:.2f}',(xx,yy),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color=color)
    ax.set(title=title,xlim=(2500,18100),ylim=(0,limit),xlabel='Matrix dimension (M = N = K)',ylabel='TFLOP/s')
    ax.set_xticks([4096,8192,16384]);ax.tick_params(axis='x',labelsize=9)
    ax.text(.03,.08,'Color: selected ceiling kernel\nGray: other seven kernels',transform=ax.transAxes,fontsize=9,color='#555555')
ax=axes[2]
colors=['#0072B2','#777777','#009E73','#CC79A7','#D55E00']
for function,color,marker in zip(('Copy','Mul','Add','Triad','Dot'),colors,('o','s','^','D','v')):
    group=sorted([r for r in s['summaries'] if r['path']=='bw' and r['function']==function],key=lambda r:r['size'])
    x=np.array([3*4*r['size']/2**30 for r in group]);y=np.array([r['median'] for r in group])
    err=np.array([[r['median']-r['min'] for r in group],[r['max']-r['median'] for r in group]])
    ax.errorbar(x,y,yerr=err,color=color,marker=marker,ms=5,capsize=3,lw=1.6,label=function)
    if function=='Dot':
        for xx,yy in zip(x,y):ax.annotate(f'{yy:.1f}',(xx,yy),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9,color=color)
ax.set(title='BabelStream bandwidth',xlim=(.55,3.25),ylim=(0,295),xlabel='Total size of three arrays (GiB)',ylabel='GB/s (logical bytes)')
ax.set_xticks([.75,1.5,3]);ax.legend(loc='lower center',ncol=3,frameon=False,fontsize=9)
for ax in axes:ax.grid(axis='y',color='#E5E5E5',lw=.7);ax.set_axisbelow(True)
fig.text(.055,.087,'Repeat ranges are not confidence intervals. The largest tested matrix is 16384 cubed; larger-size saturation is unmeasured.',fontsize=10,color='#444444')
export(fig,'size_scaling_720')
print(json.dumps(parameters,indent=2))
print('Saved two figures in PNG/PDF/SVG plus exact curve data and parameters.')
