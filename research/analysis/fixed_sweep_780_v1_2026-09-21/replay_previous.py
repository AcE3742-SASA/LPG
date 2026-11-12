"""Offline compatibility check; replays old files, never runs a GPU worker."""
from pathlib import Path
import json,sys,shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT/'fixed_sweep_780_v1'))
import run,analyze_sweep
loaded=run.load()
out=HERE/'offline_replay_release'
if out.exists(): raise SystemExit('Use a fresh validation output; do not mix source versions')
out.mkdir()
manifest=run.manifest_for(out,'check',loaded)
raw=ROOT/'analysis/clock_range_result_2026-09-21/raw/clock_range_20260921_011219_0bnxY6'
store=loaded['suite'].Store(out)
source_hashes={}
try:
 for rd,stage in [(0,'s0_780_before'),(1,'s8_780_after')]:
  segment=out/'sessions'/stage/'segment_00000';segment.mkdir(parents=True)
  for name in ['telemetry.csv','telemetry_metadata.json']:
   shutil.copy2(raw/stage/name,segment/name)
   source_hashes[str((raw/stage/name).relative_to(ROOT))]=run.sha(raw/stage/name)
  shutil.copy2(raw/stage/'environment.json',segment.parent/'environment.json')
  rows=run.read(raw/stage/'results.json')
  for task in [t for t in manifest['tasks'] if t['round']==rd]:
   old=next(r for r in rows if all(r['task'][k]==task[k] for k in ['shape','dtype','mode','round']))
   store.append(dict(task=task,session=stage,segment=str(segment.relative_to(out)),timestamp=old['started_utc'],
    status='PASS',**old['result'],offline_replay=True,original_task=old['task']))
finally:store.close()
summary=analyze_sweep.analyze(out,loaded)
assert summary['recorded']==60 and summary['timed_blocks']==420 and summary['triplets_complete']==20
assert summary['timing_and_sampled_clocks_pass'] and summary['counts']=={'PASS':60}
assert not summary['all_restorations_accepted'] # This is not a newly executed check.
(HERE/'offline_replay_validation.json').write_text(json.dumps(dict(status='PASS',kind='old_data_offline_replay_not_gpu_execution',summary=summary,source_hashes=source_hashes),indent=2)+'\n')
print(json.dumps(summary,indent=2))
