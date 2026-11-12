"""Reproduce the received 720 MHz check without changing raw evidence or gates."""
from collections import Counter
import csv
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import statistics as st
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'abstract_sweep_725_v1'))
import run
import analyze_sweep

RAW=HERE/'raw/check_low_clock'
ZIP=HERE/'check_low_clock_20260921_190318_9219a0.zip'
ZIP_SHA='1eb542db853db7010cd85e49cd269bf11bf11ec11cfbac981d620b73aa63bda6'
KST=dt.timezone(dt.timedelta(hours=9))


def verify_raw():
    assert run.sha(ZIP)==ZIP_SHA
    assert {str(p.relative_to(HERE/'raw')):run.sha(p) for p in (HERE/'raw').rglob('*') if p.is_file()}==run.read(HERE/'raw_sha256.json')


def main():
    verify_raw()
    manifest=run.read(RAW/'manifest.json')
    loaded=run.load(manifest['selected_clock_mhz'])
    assert manifest['selected_clock_mhz']==720
    assert run.manifest_for(RAW,'check',loaded)==manifest
    selection=run.read(next(RAW.glob('clock_selection_*.json')))
    supported=[int(s) for s in selection['query']['stdout'].splitlines()]
    assert 725 not in supported and 720 in supported
    assert selection['selected_clock_mhz']==run.choose_supported_clock(selection['query']['stdout'])==720
    derived=HERE/'recomputed'
    if not derived.exists():shutil.copytree(RAW,derived)
    analyze_sweep.analyze(derived,loaded)
    assert run.read(derived/'summary.json')==run.read(RAW/'summary.json')
    names=['raw_attempts.jsonl','metrics.csv','fresh_holdout_LOCKED.csv','blocks.csv',
           'blocks_fresh_holdout_LOCKED.csv','between_round.json','review_queue.csv',
           'review_fresh_holdout_LOCKED.csv','triplets.csv','triplets_fresh_holdout_LOCKED.csv',
           'shape_splits.csv','reference_780_comparison.csv','summary.json','report.md']
    names += [str(p.relative_to(RAW)) for p in (RAW/'sessions').glob('*/segment_*/quality.json')]
    assert all(run.sha(derived/name)==run.sha(RAW/name) for name in names)
    rows=[r for _,r in run.history(RAW)]
    assert len(rows)==24==len({r['task']['id'] for r in rows})
    assert [r['task'] for r in rows]==manifest['tasks'][:24]
    assert all(r['status']=='PASS' and r['correctness']['passed'] and r['correctness']['finite_all'] for r in rows)
    assert {r['task']['round'] for r in rows}=={0}
    assert sum(len(r['timing']['blocks']) for r in rows)==168
    envelopes=[]
    for r in rows:
        timing=r['timing']
        for b in timing['blocks']:
            assert math.isclose(b['latency_ms'],b['elapsed_ms']/(b['repeats']*b['operations_per_call']),rel_tol=1e-12)
        assert math.isclose(st.median(b['latency_ms'] for b in timing['blocks']),timing['median_ms'],rel_tol=1e-12)
        for phase,blocks in [('settle',timing['setup']['settle_blocks']),('calibration',timing['setup']['calibration']),('timed',timing['blocks'])]:
            envelopes.extend((phase,dt.datetime.fromisoformat(b['host_envelope_start_utc']),
                              dt.datetime.fromisoformat(b['host_envelope_end_utc'])) for b in blocks)
    folder,=list((RAW/'sessions').glob('*/segment_*'))
    meta=run.read(folder/'telemetry_metadata.json')
    samples,cadence=loaded['telemetry'].read(folder/'telemetry.csv',meta['timezone_offset_seconds'],50)
    bad=[s for s in samples if s['clocks.mem']!=6801]
    assert len(bad)==1 and bad[0]['clocks.mem']==6800
    assert all(s['clocks.sm']==720 and s['power.limit']==70 for s in samples)
    active={k:sum(s[k] is True for s in samples) for k in loaded['telemetry'].REASONS}
    assert not any(active.values())
    instant=dt.datetime.fromisoformat(bad[0]['utc'])
    assert not any(start<=instant<=end for _,start,end in envelopes)
    first_setup=min(start for _,start,_ in envelopes)
    assert instant<first_setup
    index=samples.index(bad[0]);context=samples[max(0,index-3):index+4]
    with (HERE/'memory_sample_context.csv').open('w',newline='') as f:
        fields=['utc','kst','clocks.sm','clocks.mem','power.draw','temperature.gpu','clocks_event_reasons.sw_power_cap']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for s in context:writer.writerow(dict(s,kst=dt.datetime.fromisoformat(s['utc']).astimezone(KST).isoformat()))
    session=run.read(folder.parent/'session.json')
    assert session['state']=='ERROR' and session['new_records']==24
    launch,=list((RAW/'launches').iterdir())
    restore=run.read(launch/'restore.json')
    assert restore['commands_accepted'] and all(restore['commands'][k]['exit_code']==0 for k in ('graphics','memory'))
    assert run.read(launch/'set_graphics.json')['command'][-1]=='720,720'
    assert run.read(launch/'set_memory.json')['command'][-1]=='6801,6801'
    errors={}
    for name,fn in [('resume',lambda:run.require_clean_resume(RAW,loaded)),
                    ('full_admission',lambda:run.check_admission(RAW,'Review retained; incomplete check is not admitted',loaded))]:
        try:fn()
        except RuntimeError as e:
            errors[name]=str(e)
            assert 'resume refused' in str(e)
        else:raise AssertionError(name+' incorrectly accepted')
    summary=run.read(RAW/'summary.json')
    assert summary['counts']=={'PASS':24} and summary['max_between_round_range'] is None
    quality=run.read(folder/'quality.json')
    assert quality['flags']==['worker_mismatch:clocks.mem']
    assert all(q['status']=='PASS' and not q['flags'] for q in quality['measurements'].values())
    report=dict(verification='PASS',decision='INCOMPLETE_PREHEAT_MEMORY_SAMPLE_REVIEW_FULL_HOLD',
        zip_sha256=ZIP_SHA,raw_files=len(run.read(HERE/'raw_sha256.json')),
        source_files_verified=len(manifest['source_hashes']),dependencies_verified=len(manifest['dependency_hashes']),
        reproduced_identical_outputs=len(names),gpu_execution_on_mac=False,result=summary,
        requested_clock_mhz=725,selected_clock_mhz=720,supports_requested_725=False,supports_selected_720=True,
        supported_graphics_clocks_count=len(supported),supported_neighbors=[x for x in supported if 690<=x<=765],
        correctness_scopes=dict(Counter(r['correctness']['scope'] for r in rows)),
        telemetry_samples=len(samples),cadence=cadence,core_counts=dict(Counter(s['clocks.sm'] for s in samples)),
        memory_counts=dict(Counter(s['clocks.mem'] for s in samples)),instantaneous_active_samples=active,
        ranges={k:[min(s[k] for s in samples),max(s[k] for s in samples)] for k in ('clocks.sm','clocks.mem','power.limit','power.draw','temperature.gpu')},
        anomalous_memory_sample=bad[0],anomaly_kst=instant.astimezone(KST).isoformat(),
        first_task_setup_kst=first_setup.astimezone(KST).isoformat(),
        seconds_before_first_task_setup=(first_setup-instant).total_seconds(),
        overlapping_task_setup_calibration_or_timed_blocks=0,affected_timed_measurements=0,
        session=session,restoration_commands_accepted=True,rejections=errors,
        exit_codes={str(p.relative_to(RAW)):int(p.read_text()) for p in RAW.rglob('*exit_code.txt')},
        interpretation='One 6800 MHz sample during preheating triggered the unchanged worker-wide exact-match gate; the cause of that 1 MHz reporting difference is not established. No power-cap/core-clock failure observed; round 1 remains missing.')
    (HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    verify_raw()
    print(json.dumps({k:report[k] for k in ('verification','decision','raw_files','source_files_verified',
        'dependencies_verified','reproduced_identical_outputs','core_counts','memory_counts',
        'seconds_before_first_task_setup','correctness_scopes','rejections')},indent=2))


if __name__=='__main__':main()
