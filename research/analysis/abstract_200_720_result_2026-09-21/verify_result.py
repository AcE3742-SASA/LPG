"""Reproduce the partial 720 MHz sweep and identify a conservative recovery scope."""
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

RAW=HERE/'raw/abstract_200_720_20260921'
ZIP=HERE/'abstract_200_720_20260921_20260921_213639_262a4a.zip'
ZIP_SHA='fec775a7d658ac4f2e65df76cbef573b94781dd0e1e7483e34d9c2ec110593fb'
KST=dt.timezone(dt.timedelta(hours=9))


def verify_raw():
    assert run.sha(ZIP)==ZIP_SHA
    assert {str(p.relative_to(HERE/'raw')):run.sha(p) for p in (HERE/'raw').rglob('*') if p.is_file()}==run.read(HERE/'raw_sha256.json')


def table(name,rows):
    with (HERE/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def main():
    verify_raw()
    loaded=run.load(720);manifest=run.manifest_for(RAW,'full',loaded)
    assert manifest['selected_clock_mhz']==720
    derived=HERE/'recomputed'
    if not derived.exists():shutil.copytree(RAW,derived)
    analyze_sweep.analyze(derived,loaded)
    assert run.read(derived/'summary.json')==run.read(RAW/'summary.json')
    names=['raw_attempts.jsonl','metrics.csv','fresh_holdout_LOCKED.csv','blocks.csv',
           'blocks_fresh_holdout_LOCKED.csv','between_round.json','review_queue.csv',
           'review_fresh_holdout_LOCKED.csv','triplets.csv','triplets_fresh_holdout_LOCKED.csv',
           'shape_splits.csv','summary.json','report.md']
    names += [str(p.relative_to(RAW)) for p in (RAW/'sessions').glob('*/segment_*/quality.json')]
    assert all(run.sha(derived/name)==run.sha(RAW/name) for name in names)
    entries=list(run.history(RAW));rows=[r for _,r in entries]
    assert len(rows)==1800==len({r['task']['id'] for r in rows})
    assert [r['task'] for r in rows]==manifest['tasks'][:1800]
    assert Counter(r['task']['round'] for r in rows)=={0:1200,1:600}
    assert all(r['correctness']['passed'] and r['correctness']['finite_all'] for r in rows)
    assert sum(len(r['timing']['blocks']) for r in rows)==12600
    with (derived/'metrics.csv').open() as f:metrics={r['id']:r for r in csv.DictReader(f)}
    assert Counter(r['status'] for r in metrics.values())=={'PASS':1799,'REVIEW':1}
    assert len({tuple(r['task']['shape']) for r in rows[:1200]})==200
    assert len({tuple(r['task']['shape']) for r in rows[1200:]})==100
    samples=[];segments=[]
    for folder in sorted((RAW/'sessions').glob('*/segment_*')):
        meta=run.read(folder/'telemetry_metadata.json')
        found,cadence=loaded['telemetry'].read(folder/'telemetry.csv',meta['timezone_offset_seconds'],50)
        samples.extend(found);q=run.read(folder/'quality.json')
        segments.append(dict(segment=str(folder.relative_to(RAW)),cadence=cadence,flags=q['flags'],
                             collection_can_continue=q['collection_can_continue']))
    assert len(segments)==15
    dirty={s['segment'] for s in segments if not s['collection_can_continue']}
    assert len(dirty)==1 and next(iter(dirty)).endswith('segment_00014')
    assert all(s['clocks.sm']==720 and s['power.limit']==70 for s in samples)
    bad=[s for s in samples if s['clocks.mem']!=6801]
    assert len(bad)==1 and bad[0]['clocks.mem']==6800
    instant=dt.datetime.fromisoformat(bad[0]['utc']);hits=[]
    for seq,r in entries:
        timing=r['timing'];blocks=timing['blocks']
        for b in blocks:
            assert math.isclose(b['latency_ms'],b['elapsed_ms']/(b['repeats']*b['operations_per_call']),rel_tol=1e-12)
        assert math.isclose(st.median(b['latency_ms'] for b in blocks),timing['median_ms'],rel_tol=1e-12)
        for phase,bs in [('settle',timing['setup']['settle_blocks']),('calibration',timing['setup']['calibration']),('timed',blocks)]:
            for i,b in enumerate(bs):
                start=dt.datetime.fromisoformat(b['host_envelope_start_utc']);end=dt.datetime.fromisoformat(b['host_envelope_end_utc'])
                if start<=instant<=end:
                    hits.append(dict(seq=seq,**r['task'],block_phase=phase,block=i,latency_ms=b['latency_ms'],
                        median_ms=timing['median_ms'],block_vs_median_pct=100*(b['latency_ms']/timing['median_ms']-1),
                        relative_iqr=timing['relative_iqr'],first_last_drift=timing['first_last_drift'],
                        start_kst=start.astimezone(KST).isoformat(),end_kst=end.astimezone(KST).isoformat(),
                        status=metrics[r['task']['id']]['status'],flags=metrics[r['task']['id']]['flags']))
    assert len(hits)==1 and (hits[0]['seq'],hits[0]['block_phase'],hits[0]['block'])==(1777,'timed',4)
    table('affected_block.csv',hits)
    index=samples.index(bad[0])
    table('memory_sample_context.csv',[dict(utc=s['utc'],kst=dt.datetime.fromisoformat(s['utc']).astimezone(KST).isoformat(),
        core_mhz=s['clocks.sm'],memory_mhz=s['clocks.mem'],power_w=s['power.draw'],temperature_c=s['temperature.gpu'],
        power_cap=s['clocks_event_reasons.sw_power_cap']) for s in samples[index-3:index+4]])
    active={k:sum(s[k] is True for s in samples) for k in loaded['telemetry'].REASONS};assert not any(active.values())
    launch,=list((RAW/'launches').iterdir());restore=run.read(launch/'restore.json')
    assert restore['commands_accepted'] and all(restore['commands'][k]['exit_code']==0 for k in ('graphics','memory'))
    session_path,=list((RAW/'sessions').glob('*/session.json'));session=run.read(session_path)
    assert session['state']=='ERROR' and session['new_records']==1800
    admission=run.read(launch/'check_review.json')
    previous=run.read(ROOT/'analysis/check_720_repeat_result_2026-09-21/validation.json')
    assert admission['results_digest']==previous['result']['results_digest']
    check=ROOT/'analysis/check_720_repeat_result_2026-09-21/raw/check_720_repeat_20260921_190318'
    assert run.sha(check/'manifest.json')==admission['manifest_sha256']
    try:run.require_clean_resume(RAW,loaded)
    except RuntimeError as e:
        rejection=str(e);assert 'resume refused' in rejection
    else:raise AssertionError('Original dirty run was accepted')
    # Plan only: retain entire clean segments; remeasure the entire flagged segment.
    retained=[r['task'] for r in rows if r['segment'] not in dirty]
    redo=[r['task'] for r in rows if r['segment'] in dirty]
    missing=manifest['tasks'][1800:]
    assert len(retained)==1680 and len(redo)==120 and len(missing)==1800
    assert retained+redo+missing==manifest['tasks']
    table('recovery_queue_proposed.csv',[dict(reason='flagged_segment_remeasurement',**t) for t in redo]+
          [dict(reason='not_collected',**t) for t in missing])
    result=dict(verification='PASS',decision='INCOMPLETE_TIMED_MEMORY_SAMPLE_REVIEW',zip_sha256=ZIP_SHA,
        raw_files=len(run.read(HERE/'raw_sha256.json')),source_files_verified=len(manifest['source_hashes']),
        dependencies_verified=len(manifest['dependency_hashes']),reproduced_identical_outputs=len(names),
        new_gpu_execution=False,original_summary=run.read(RAW/'summary.json'),
        round_record_counts=dict(Counter(r['task']['round'] for r in rows)),round_shape_counts={'0':200,'1':100},
        correctness_scopes=dict(Counter(r['correctness']['scope'] for r in rows)),raw_status_counts=dict(Counter(r['status'] for r in rows)),
        telemetry_samples=len(samples),core_counts=dict(Counter(s['clocks.sm'] for s in samples)),
        memory_counts=dict(Counter(s['clocks.mem'] for s in samples)),instantaneous_active_samples=active,
        ranges={k:[min(s[k] for s in samples),max(s[k] for s in samples)] for k in ('power.draw','temperature.gpu','clocks.sm','clocks.mem','power.limit')},
        anomaly=bad[0],affected_blocks=hits,segments=segments,session=session,admission=admission,
        resume_rejection=rejection,restoration_commands_accepted=True,
        recovery_proposal=dict(retain_clean_segment_records=1680,remeasure_flagged_segment_records=120,
                              collect_missing_records=1800,total_pending=1920,implemented=False,
                              note='Preserve original failed run. New import/continuation requires a separate reviewed recovery step. No result or gate has been changed.'),
        interpretation='A single 6800 MHz sample overlaps a timed block. Timing variation and instantaneous cap/thermal checks pass; the cause of the 1 MHz memory report is unproven.')
    (HERE/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    verify_raw()
    print(json.dumps({k:result[k] for k in ('verification','raw_files','source_files_verified','dependencies_verified',
        'reproduced_identical_outputs','core_counts','memory_counts','affected_blocks','recovery_proposal')},indent=2))


if __name__=='__main__':main()
