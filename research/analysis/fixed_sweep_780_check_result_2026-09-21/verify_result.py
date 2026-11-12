"""Recheck the received GPU result without modifying its preserved raw files."""
from collections import Counter,defaultdict
import csv,hashlib,json,math,shutil,statistics as st,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'fixed_sweep_780_v1'))
import run,analyze_sweep
raw=HERE/'raw/check_780'
expected_hashes=run.read(HERE/'raw_sha256.json')
def verify_raw():
    actual={str(p.relative_to(HERE/'raw')):run.sha(p) for p in (HERE/'raw').rglob('*') if p.is_file()}
    assert actual==expected_hashes,'Raw evidence changed'
verify_raw()
manifest=run.read(raw/'manifest.json')
assert manifest['source_hashes']==run.own_hashes()
for folder,base,hashes in [('source',ROOT/'fixed_sweep_780_v1',manifest['source_hashes']),
                          ('dependencies',ROOT,manifest['dependency_hashes'])]:
    for name,wanted in hashes.items():
        assert run.sha(raw/folder/name)==run.sha(base/name)==wanted
loaded=run.load()
derived=HERE/'recomputed'
if derived.exists():raise SystemExit('Keep previous evidence; select a new verification output')
shutil.copytree(raw,derived)
original_files={str(p.relative_to(raw)):run.sha(p) for p in raw.rglob('*') if p.is_file()}
result=analyze_sweep.analyze(derived,loaded)
assert result==run.read(raw/'summary.json')
outputs=['raw_attempts.jsonl','metrics.csv','fresh_holdout_LOCKED.csv','blocks.csv',
         'blocks_fresh_holdout_LOCKED.csv','between_round.json','review_queue.csv',
         'review_fresh_holdout_LOCKED.csv','triplets.csv','triplets_fresh_holdout_LOCKED.csv',
         'shape_splits.csv','reference_780_comparison.csv','summary.json','report.md']
outputs += [str(p.relative_to(raw)) for p in (raw/'sessions').glob('*/segment_*/quality.json')]
assert all(run.sha(derived/name)==original_files[name] for name in outputs)
rows=[row for _,row in run.history(derived)]
assert len(rows)==60==len({r['task']['id'] for r in rows})
assert {r['task']['id'] for r in rows}=={t['id'] for t in manifest['tasks']}
assert sum(len(r['timing']['blocks']) for r in rows)==420
assert all(r['correctness']['passed'] for r in rows)
sessions=Counter(r['session'] for r in rows)
assert sorted(sessions.values())==[30,30]
for session in sessions:
    info=run.read(raw/'sessions'/session/'session.json')
    assert info['state']=='FINISHED' and info['new_records']==30
pairs=defaultdict(list)
for r in rows:pairs[(tuple(r['task']['shape']),r['task']['dtype'],r['task']['mode'])].append(r)
assert len(pairs)==30
assert all({r['task']['round'] for r in group}=={0,1} and len({r['session'] for r in group})==2 for group in pairs.values())
rounds=[dict(shape=list(k[0]),dtype=k[1],mode=k[2],relative_range=(max(v['timing']['median_ms'] for v in g)-min(v['timing']['median_ms'] for v in g))/st.median(v['timing']['median_ms'] for v in g)) for k,g in pairs.items()]
assert math.isclose(max(r['relative_range'] for r in rounds),result['max_between_round_range'],rel_tol=1e-12)
samples=[];quality=[]
for folder in sorted((raw/'sessions').glob('*/segment_*')):
    meta=run.read(folder/'telemetry_metadata.json')
    found,cadence=loaded['telemetry'].read(folder/'telemetry.csv',meta['timezone_offset_seconds'],50)
    samples.extend(found)
    quality.append(run.read(folder/'quality.json'))
assert all((s['clocks.sm'],s['clocks.mem'],s['power.limit'])==(780,6801,70) for s in samples)
active={name:sum(s[name] is True for s in samples) for name in loaded['telemetry'].REASONS}
assert not any(active.values())
exit_codes={str(p.relative_to(raw)):int(p.read_text()) for p in raw.rglob('*exit_code.txt')}
assert all(code==(0 if p.endswith(('worker_0.exit_code.txt','worker_1.exit_code.txt','tee_exit_code.txt')) else 2) for p,code in exit_codes.items())
launches=list((raw/'launches').iterdir())
assert len(launches)==1
assert run.read(launches[0]/'status.json')==dict(collection='COMPLETE',restoration='COMMANDS_ACCEPTED')
reference=list(csv.DictReader((raw/'reference_780_comparison.csv').open()))
changes=[float(r['change_pct']) for r in reference]
wall=[r['task_wall_seconds'] for r in rows]
elapsed=sum(run.read(raw/'sessions'/s/'session.json')['elapsed_seconds'] for s in sessions)
full=run.make_plan('full',loaded)
count=len(full['tasks'])
assert count==113382
note=('Reviewed check ZIP 655d794d706dd340a3276c651ff4eb43276880fe61077c829d3877e6996e99b6: '
      '60/60 PASS, 420 blocks, 20 triplets; IQR 0.05273%, drift 0.11602%, round range 0.57559%; '
      'two workers and both clock resets verified. 780/6801 MHz, 70 W sampled clocks pass. '
      'Cumulative SWPowerCap ambiguity remains UNRESOLVED (23/420 blocks); '
      'historical timing shift up to 2.09662% retained without correction. '
      'Proceed with collection and quality review only; no model training.')
admission=run.check_admission(derived,note,loaded)
(HERE/'review_note.txt').write_text(note+'\n')
(HERE/'admission_verified.json').write_text(json.dumps(admission,indent=2)+'\n')
summary=dict(verification='PASS',decision='FULL_COLLECTION_ELIGIBLE_AFTER_REVIEW',
 zip_sha256=run.sha(HERE/'check_780_20260921_033834_065a88.zip'),raw_files=len(expected_hashes),
 source_files_verified=len(manifest['source_hashes']),dependencies_verified=len(manifest['dependency_hashes']),
 reproduced_identical_outputs=len(outputs),gpu_execution_on_mac=False,result=result,
 session_counts=dict(sessions),exit_codes=exit_codes,correctness_scopes=dict(Counter(r['correctness']['scope'] for r in rows)),
 telemetry_samples=len(samples),instantaneous_active_samples=active,
 telemetry_ranges={k:[min(s[k] for s in samples),max(s[k] for s in samples)] for k in ['clocks.sm','clocks.mem','power.limit','power.draw','temperature.gpu']},
 max_round_condition=max(rounds,key=lambda r:r['relative_range']),
 historical_comparison=dict(pairs=len(changes),min_pct=min(changes),median_pct=st.median(changes),max_pct=max(changes)),
 time_estimate=dict(worker_elapsed_seconds=elapsed,task_wall_seconds_sum=sum(wall),task_wall_seconds_mean=st.mean(wall),
   full_tasks=count,full_mean_task_hours=count*st.mean(wall)/3600,
   full_worker_excluding_preheat_hours=count*(elapsed-2*180)/60/3600,
   caveat='Representative check extrapolation, not a completion-time guarantee; shape mix and restarts differ.'),
 launcher_source_present=any(raw.glob('launcher/*/launcher_source.sh')))
(HERE/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
verify_raw()
print(json.dumps(summary,ensure_ascii=False,indent=2))
