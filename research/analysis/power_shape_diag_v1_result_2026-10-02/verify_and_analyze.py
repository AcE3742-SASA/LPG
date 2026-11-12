#!/usr/bin/env python3
"""Reproduce the returned diagnostic and compare only temperature-matched rounds."""
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0,str(PROJECT/'power_shape_diag_v1'))
import diagnose as diagnostic


def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def epoch(value):
    return dt.datetime.fromisoformat(value).timestamp()


def parse_ncu(path):
    header=units=None
    result=[]
    for row in csv.reader(path.read_text().splitlines()):
        if 'ID' in row and 'Kernel Name' in row:
            header=row
        elif header and len(row)==len(header):
            if row[0].isdigit():
                assert units is not None
                result.append(dict(values=dict(zip(header,row)),units=dict(zip(header,units))))
            else:
                units=row
    assert len(result)==1
    return result[0]


def main():
    raw=next((HERE/'raw').iterdir())
    archive=next(HERE.glob('*.zip'))
    assert diagnostic.sha(archive)=='719cb872d36026494713637f447c79bd609fb4d40de064ca356476ac75246b29'
    before={str(p.relative_to(raw)):diagnostic.sha(p) for p in raw.rglob('*') if p.is_file()}
    manifest=diagnostic.read(raw/'manifest.json')
    for folder,key in [('source','source_hashes'),('dependencies','dependency_hashes')]:
        for name,h in manifest[key].items():
            assert diagnostic.sha(raw/folder/name)==h
            local=PROJECT/'power_shape_diag_v1'/('dependencies' if folder=='dependencies' else '')/name
            assert diagnostic.sha(local)==h
    loaded,config=diagnostic.load()
    assert manifest['protocol']==config and manifest['timing_protocol']==loaded['plan'].PROTOCOL
    assert manifest['input_seeds']==[loaded['plan'].task_seed(t) for t in config['tasks']]
    recomputed=HERE/'recomputed'/raw.name
    if not recomputed.exists():
        shutil.copytree(raw,recomputed)
    summary=diagnostic.analyze(recomputed,loaded,config)
    assert summary==diagnostic.read(raw/'summary.json')
    telemetry=loaded['telemetry']
    telemetry.FIELDS=tuple(manifest['telemetry_fields'])
    stages=[]
    for directory in sorted(p for p in raw.iterdir() if (p/'worker_complete.json').exists()):
        env=diagnostic.read(directory/'environment.json')
        assert loaded['suite'].signature(env)==manifest['environment_signature']
        assert int((directory/'worker.exit_code.txt').read_text())==0
        meta=diagnostic.read(directory/'telemetry_metadata.json')
        rows,cadence=telemetry.read(directory/'telemetry.csv',meta['timezone_offset_seconds'],50)
        assert not cadence['unavailable_fields'] and cadence['observed_interval_seconds_max']<5
        for row in rows:
            diagnostic.safe_row(row,env['gpu_uuid'])
        correctness=list(directory.glob('correctness*.json'))
        for p in correctness:
            c=diagnostic.read(p)
            assert c['passed'] and c['finite_all'] and c['max_tolerance_ratio']<=1 and c['points']==256
        recovery=diagnostic.read(directory/'recovery.json')
        last=epoch(recovery['start_sample']['utc'])
        cool=[r for r in rows if last-10<=epoch(r['utc'])<=last]
        assert cool and max(r['temperature.gpu'] for r in cool)<=55
        trial_files=sorted(directory.glob('trial_*.json'))
        if trial_files:
            assert len(trial_files)==16
            first=diagnostic.read(trial_files[0])['start_utc']
            last_trial=diagnostic.read(trial_files[-1])['end_utc']
            active=[r for r in rows if epoch(first)<=epoch(r['utc'])<=epoch(last_trial)]
            for p in trial_files:
                row=diagnostic.read(p); timing=row['timing']; blocks=timing['blocks']
                stats=loaded['measure'].summarize(blocks,loaded['plan'].PROTOCOL)
                assert timing['median_ms']==stats['median_ms'] and timing['flags']==stats['flags']
                assert len(blocks)==7 and min(b['elapsed_ms'] for b in blocks)>=80
                assert math.isclose(sum(b['elapsed_ms'] for b in timing['setup']['settle_blocks']),timing['setup']['settle_event_ms'])
                assert timing['setup']['settle_event_ms']>=3000
                inside=[r for r in rows if any(epoch(b['host_envelope_start_utc'])<=epoch(r['utc'])<=epoch(b['host_envelope_end_utc']) for b in blocks)]
                reference=next(r for r in summary['trials'] if r['round']==int(directory.name[1]) and r['k']==row['task']['shape'][2] and r['trial']==row['trial'])
                assert len(inside)==reference['timing']['samples']
                assert sum(r['clocks_event_reasons.sw_power_cap'] for r in inside)==reference['timing']['sw_power_cap_samples']
                assert sum(r['clocks.sm']<720 for r in inside)==reference['timing']['below_target_samples']
                assert all(r['clocks.mem']==6801 for r in inside)
                assert min(r['clocks.sm'] for r in inside)==reference['timing']['clocks.sm']['min']
        else:
            window=diagnostic.read(directory/'capture_window.json')
            first,last_trial=window['start_utc'],window['end_utc']
            active=[r for r in rows if epoch(first)<=epoch(r['utc'])<=epoch(last_trial)]
        capped=[r for r in active if r['clocks_event_reasons.sw_power_cap']]
        dropped=[r for r in active if r['clocks.sm']<720]
        keys=('utc','temperature.gpu','clocks.sm','power.draw','power.draw.instant','clocks_event_reasons.active')
        stages.append(dict(stage=directory.name,rows=len(rows),correctness_checks=len(correctness),
            maximum_gap_s=cadence['observed_interval_seconds_max'],recovery_wait_s=recovery['wait_seconds'],
            starting_temperature_c=recovery['start_sample']['temperature.gpu'],active_start_utc=first,active_end_utc=last_trial,
            thermal_samples=sum(r['clocks_event_reasons.sw_thermal_slowdown'] or r['clocks_event_reasons.hw_thermal_slowdown'] for r in rows),
            flag_masks=sorted(set(r['clocks_event_reasons.active'] for r in rows)),
            first_cap={k:capped[0][k] for k in keys} if capped else None,
            first_drop={k:dropped[0][k] for k in keys} if dropped else None,
            cap_after_start_s=epoch(capped[0]['utc'])-epoch(first) if capped else None,
            drop_after_start_s=epoch(dropped[0]['utc'])-epoch(first) if dropped else None,
            raw_memory_clocks=sorted(set(r['clocks.mem'] for r in rows)),
            cap_samples_in_selected_interval=len(capped),drop_samples_in_selected_interval=len(dropped),
            all_event_blocks=len(diagnostic.read(directory/'all_event_blocks.json'))))
    groups=[]
    for rd in range(1,5):
        for k in (72,127):
            rs=[r for r in summary['trials'] if r['round']==rd and r['k']==k]
            groups.append(dict(round=rd,k=k,trials=len(rs),start_temperature_c=rs[0]['start_temperature_c'],
                power_w_median=st.median(r['timing']['power.draw']['median'] for r in rs),
                power_instant_w_median=st.median(r['timing']['power.draw.instant']['median'] for r in rs),
                timing_cap_trials=sum(r['timing']['sw_power_cap_samples']>0 for r in rs),
                timing_drop_trials=sum(r['timing']['below_target_samples']>0 for r in rs),
                minimum_sm_mhz=min(r['timing']['clocks.sm']['min'] for r in rs),
                temperature_median_c=st.median(r['timing']['temperature.gpu']['median'] for r in rs),
                latency_ms_median=st.median(r['latency_ms'] for r in rs),
                intertrial_gap_ms_median=st.median(r['gap_from_previous_trial_ms'] for r in rs if r['trial']>0)))
    matched={}
    for k in (72,127):
        rs=[r for r in summary['trials'] if r['round']>=2 and r['k']==k]
        matched[k]=dict(trials=len(rs),power_w_median=st.median(r['timing']['power.draw']['median'] for r in rs),
            cap_trials=sum(r['timing']['sw_power_cap_samples']>0 for r in rs),
            drop_trials=sum(r['timing']['below_target_samples']>0 for r in rs),
            min_sm_mhz=min(r['timing']['clocks.sm']['min'] for r in rs),
            latency_ms_median=st.median(r['latency_ms'] for r in rs),
            timing_flags=sum(bool(r['timing_flags']) for r in rs))
    deltas=[a['timing']['power.draw']['median']-next(b['timing']['power.draw']['median'] for b in summary['trials']
               if b['round']==a['round'] and b['trial']==a['trial'] and b['k']==127)
            for a in summary['trials'] if a['round']>=2 and a['k']==72]
    ncu={k:parse_ncu(raw/f'ncu_k{k}/worker.log') for k in (72,127)}
    save('ncu_metrics.json',ncu)
    traces={}
    for k in (72,127):
        directory=raw/f'nsys_k{k}'
        with sqlite3.connect((directory/'capture.sqlite').as_uri()+'?mode=ro',uri=True) as con:
            assert con.execute('PRAGMA quick_check').fetchone()==('ok',)
            con.row_factory=sqlite3.Row
            kernels=[dict(r) for r in con.execute('SELECT k.*,s.value AS kernel_name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start')]
        assert len(kernels)==20
        assert set(r['kernel_name'] for r in kernels)=={ncu[k]['values']['Kernel Name']}
        csv_rows=list(csv.reader((directory/'kernel_summary.csv').read_text().splitlines()))
        header_index=next(i for i,r in enumerate(csv_rows) if 'Instances' in r and 'Name' in r)
        values=dict(zip(csv_rows[header_index],csv_rows[header_index+1]))
        assert sum(r['end']-r['start'] for r in kernels)==int(values['Total Time (ns)'])
        traces[k]=dict(kernels=len(kernels),kernel_name=kernels[0]['kernel_name'],
            resources={key:sorted(set(r[key] for r in kernels)) for key in ('gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread','staticSharedMemory','dynamicSharedMemory')},
            mean_gpu_duration_us=sum(r['end']-r['start'] for r in kernels)/len(kernels)/1000,
            mean_gap_us=st.mean((b['start']-a['end'])/1000 for a,b in zip(kernels,kernels[1:])),
            kernel_duty_fraction_in_trace=sum(r['end']-r['start'] for r in kernels)/(kernels[-1]['end']-kernels[0]['start']))
    assert traces[72]['resources']==traces[127]['resources']
    assert sum(g['timing_cap_trials'] for g in groups)==48
    assert sum(g['timing_drop_trials'] for g in groups)==19
    assert sum(x['thermal_samples'] for x in stages)==0
    assert sum(x['correctness_checks'] for x in stages)==144
    assert diagnostic.read(raw/'restore.json')['commands_accepted']
    assert int((raw/'launcher/exit_code.txt').read_text())==2
    assert all(diagnostic.sha(raw/p)==h for p,h in before.items())
    findings=dict(source_archive_sha256=diagnostic.sha(archive),raw_files=len(before),source_files=7,dependency_files=110,
        summary_exact_reproduction=True,independent_timing_window_checks=128,raw_preserved=True,
        collection='COMPLETE',correctness_checks=144,groups=groups,matched_rounds=[2,3,4],matched=matched,
        paired_trial_power_delta_w=dict(median=st.median(deltas),min=min(deltas),max=max(deltas)),
        stages=stages,nsys=traces,ncu_selected_capture_count=2,ncu_passes_per_capture=10,
        limitations=['Round 1 starts differ by 17 C and is excluded from paired inference.',
         'Normal trials are repeated measurements within 8 bouts, not 128 independent sessions.',
         'Kernel identity and activity were profiled in separate runs, not throughout every ordinary bout.',
         'NCU used uncontrolled caches with 10 replay passes; K127 L2 hit rate 100.11 percent is not a valid literal probability.',
         'Voltage query was unavailable. Power attribution to individual GPU units remains unmeasured.',
         'A 50 ms polling request is not the power sensor update interval.'])
    save('findings.json',findings)
    save('raw_hashes.json',before)
    with (HERE/'trial_comparison.csv').open('w',newline='') as f:
        keys=['round','k','trial','start_temperature_c','latency_ms','gap_from_previous_trial_ms',
              'power_w','sm_mhz','temperature_c','sw_power_cap_samples','below_target_samples']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in summary['trials']:
            writer.writerow(dict({k:r[k] for k in keys[:6]},
                power_w=r['timing']['power.draw']['median'],sm_mhz=r['timing']['clocks.sm']['median'],
                temperature_c=r['timing']['temperature.gpu']['median'],sw_power_cap_samples=r['timing']['sw_power_cap_samples'],
                below_target_samples=r['timing']['below_target_samples']))
    plot(summary)
    print(json.dumps({k:v for k,v in findings.items() if k not in ('stages','limitations','nsys')},indent=2))


def plot(summary):
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/power-shape-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained')
    fields=[('power.draw','Power (W)'),('clocks.sm','SM clock (MHz)'),('temperature.gpu','GPU temperature (C)')]
    for ax,(field,label) in zip(axes,fields):
        for k,color in [(72,'#0072B2'),(127,'#D55E00')]:
            values=[[r['timing'][field]['median'] for r in summary['trials'] if r['k']==k and r['trial']==i and r['round']>=2] for i in range(16)]
            ax.plot(range(1,17),[st.median(v) for v in values],'-o',markersize=3,color=color,label=f'K={k}, matched R2-R4')
            ax.fill_between(range(1,17),[min(v) for v in values],[max(v) for v in values],color=color,alpha=.15)
        cold=[r['timing'][field]['median'] for r in summary['trials'] if r['k']==72 and r['round']==1]
        ax.plot(range(1,17),cold,'--',color='#747474',label='K=72, cold R1 (34 C start)')
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].axhline(70,ls=':',color='black',linewidth=1)
    axes[0].text(15.9,70.15,'70 W limit',ha='right',fontsize=9)
    axes[0].set_ylim(61,72);axes[1].set_ylim(700,725)
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.27),ncol=2,frameon=False,fontsize=9)
    axes[-1].set_xticks(range(1,17));axes[-1].set_xlabel('Trial within a 16-trial bout (each includes 3 s settling + 7 timing blocks)')
    fig.suptitle('Matched starts at 54 C: power cap in K=72 only\nM=N=4096, FP32 prealloc, 720/6801 MHz, 70 W',fontsize=13)
    fig.text(.13,-.035,'Lines: medians of trial timing-window medians; bands: min-max across R2-R4.\nCold R1 is descriptive only; its K=127 counterpart started at 51 C.',fontsize=9)
    fig.savefig(HERE/'power_shape_comparison.png',dpi=180,bbox_inches='tight')
    fig.savefig(HERE/'power_shape_comparison.pdf',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
