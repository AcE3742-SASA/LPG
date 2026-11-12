"""Plot preserved telemetry for the first FP32 incident-shape triplet in each stage."""
from datetime import datetime
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'clock_recheck_v1'))
import lower
loaded=lower.load(); telemetry=loaded[4]
RAW=next((HERE/'raw').iterdir())
fig,axes=plt.subplots(3,1,figsize=(10,7.1),sharex=True,sharey=True)
for ax,stage in zip(axes,loaded[-1]['stages']):
    name=stage['name'];meta=lower.read(RAW/name/'telemetry_metadata.json')
    samples,_=telemetry.read(RAW/name/'telemetry.csv',meta['timezone_offset_seconds'],50)
    records=[r for r in lower.read(RAW/name/'results.json') if r['task']['shape']==[3128,680,488] and r['task']['dtype']=='fp32' and r['task']['round']==0]
    assert len(records)==3
    start=datetime.fromisoformat(records[0]['result']['timing']['setup']['settle_blocks'][0]['host_envelope_start_utc']).timestamp()
    seconds=lambda utc:datetime.fromisoformat(utc).timestamp()-start
    found=[s for s in samples if -.1<=seconds(s['utc'])<=13]
    x=[seconds(s['utc']) for s in found]
    ax.scatter(x,[s['clocks.sm'] for s in found],s=10,color='#425b76',label='Sampled SM clock',zorder=3)
    caps=[s for s in found if s[telemetry.REASONS[0]] is True]
    ax.scatter([seconds(s['utc']) for s in caps],[s['clocks.sm'] for s in caps],s=25,color='#c4492d',label='Power-cap flag active',zorder=4)
    for r in records:
        timing=r['result']['timing'];a=seconds(timing['setup']['settle_blocks'][0]['host_envelope_start_utc']);b=seconds(timing['blocks'][-1]['host_envelope_end_utc'])
        ax.text((a+b)/2,789,r['task']['mode'].replace('graph','Graph'),ha='center',fontsize=10)
        for block in timing['blocks']:
            ax.axvspan(seconds(block['host_envelope_start_utc']),seconds(block['host_envelope_end_utc']),color='#b7d8ef',alpha=.55,lw=0)
    title={'s0_780_before':'First 780 MHz','s1_750':'750 MHz','s2_780_after':'Return to 780 MHz'}[name]
    ax.text(.015,.29,title,transform=ax.transAxes,ha='left',fontsize=11,fontweight='bold')
    ax.set_ylabel('SM clock (MHz)');ax.set_ylim(742,798);ax.set_yticks([750,765,780]);ax.set_xlim(-.1,13)
    ax.grid(axis='y',alpha=.25)
    ax.spines[['top','right']].set_visible(False)
axes[-1].set_xlabel('Seconds from the first FP32 task stabilization start in each stage')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.914),ncol=2,frameon=False)
fig.suptitle('The same GEMM shape reproduces the 780 MHz clock issue',fontsize=14,y=.993)
fig.text(.5,.944,'M=3128, N=680, K=488 | FP32 | round 0 | memory 6801 MHz | power limit 70 W',ha='center',fontsize=10)
fig.text(.08,.027,'Blue bands: timed-block host windows. Other intervals include calibration and stabilization.\nDots are telemetry samples (~52 ms cadence); they do not establish continuous frequency or cap duration.',fontsize=9,color='#444444')
fig.subplots_adjust(top=.84,bottom=.15,hspace=.20,left=.10,right=.97)
for suffix in ('png','pdf'):fig.savefig(HERE/('clock_incident_comparison.'+suffix),dpi=180)
print('Saved PNG/PDF')
