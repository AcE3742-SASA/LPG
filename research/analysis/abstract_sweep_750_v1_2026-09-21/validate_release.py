"""Verify the 750 MHz delivery ZIP and replay existing 750 records offline."""
from pathlib import Path
import ast
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'abstract_sweep_750_v1'
sys.path.insert(0,str(PACKAGE))
import run
import analyze_sweep

loaded=run.load();plan=run.make_plan('full',loaded)
old_raw=ROOT/'analysis/abstract_200_780_result_2026-09-21/raw/abstract_200_780'
reviewed=next((ROOT/'analysis/clock_recheck_result_2026-09-21/raw').iterdir())
raw_hashes={str(p):run.sha(p) for base in (old_raw,reviewed) for p in base.rglob('*') if p.is_file()}
assert plan['tasks']==run.read(old_raw/'manifest.json')['tasks']
assert (PACKAGE/'selection.json').read_bytes()==(ROOT/'abstract_sweep_780_v1/selection.json').read_bytes()
assert (PACKAGE/'subset.py').read_bytes()==(ROOT/'abstract_sweep_780_v1/subset.py').read_bytes()
source_hashes=run.own_hashes();assert len(source_hashes)==11
for name in source_hashes:
    if name.endswith('.py'):ast.parse((PACKAGE/name).read_text())
subprocess.run(['bash','-n',str(PACKAGE/'launch.sh')],check=True)
archive=ROOT/'abstract_sweep_750_2026-09-21_v1.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for name in source_hashes:z.write(PACKAGE/name,str(Path(PACKAGE.name)/name))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==11
    assert all(z.read(str(Path(PACKAGE.name)/name))==(PACKAGE/name).read_bytes() for name in source_hashes)
archive.with_suffix('.zip.sha256').write_text(run.sha(archive)+'  '+archive.name+'\n')
with tempfile.TemporaryDirectory(prefix='750 deployment ') as temp:
    temp=Path(temp)
    with zipfile.ZipFile(archive) as z:z.extractall(temp)
    deps=run.read(PACKAGE/'dependency_hashes.json')
    for name,wanted in deps.items():
        assert run.sha(ROOT/name)==wanted
        dest=temp/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    package=temp/PACKAGE.name
    for label,args in [('verify',['run.py','verify']),('tests',['test_subset.py'])]:
        result=subprocess.run([sys.executable,'-B',str(package/args[0]),*args[1:]],capture_output=True,text=True,timeout=90)
        (HERE/('extracted_'+label+'.log')).write_text(result.stdout+result.stderr)
        assert result.returncode==0,(label,result.stdout+result.stderr)
    check=temp/'clock_recheck_v1/runs'/reviewed.name
    check.parent.mkdir(parents=True);shutil.copytree(reviewed,check)
    code=("import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import run;"
          "print(json.dumps(run.check_admission(Path(sys.argv[2]),(run.HERE/'review_note.md').read_text(),run.load())))")
    args=[sys.executable,'-B','-c',code,str(package),str(check)]
    result=subprocess.run(args,capture_output=True,text=True,timeout=60)
    (HERE/'extracted_reviewed_750_bridge.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    admission=json.loads(result.stdout)
    assert all(admission[k]==v for k,v in loaded['config']['reviewed_check'].items())
    assert all(run.sha(check/Path(p).relative_to(reviewed))==wanted for p,wanted in raw_hashes.items() if Path(p).is_relative_to(reviewed))
    # A changed evidence record must be rejected before reaching any GPU code.
    evidence=check/'s1_750/results.json';evidence.write_text(evidence.read_text()+'\n')
    rejected=subprocess.run(args,capture_output=True,text=True,timeout=60)
    assert rejected.returncode!=0 and 'Not the reviewed 750 MHz evidence' in rejected.stderr
    (HERE/'tampered_evidence_rejection.log').write_text(rejected.stderr)

# Real 750 measurements are mapped to the unchanged selected task IDs only for offline compatibility.
replay=HERE/'offline_replay';replay.mkdir(exist_ok=False)
manifest=run.manifest_for(replay,'full',loaded)
session=replay/'sessions/recorded_750';segment=session/'segment_00000';segment.mkdir(parents=True)
for name in ('telemetry.csv','telemetry_metadata.json'):
    shutil.copy2(reviewed/'s1_750'/name,segment/name)
shutil.copy2(reviewed/'s1_750/environment.json',session/'environment.json')
indexed={(tuple(t['shape']),t['dtype'],t['mode'],t['round']):t for t in manifest['tasks']}
store=loaded['suite'].Store(replay)
try:
    for record in run.read(reviewed/'s1_750/results.json'):
        t=record['task'];task=indexed[(tuple(t['shape']),t['dtype'],t['mode'],t['round'])]
        result=record['result']
        store.append(dict(**result,task=task,session='recorded_750',segment=str(segment.relative_to(replay)),
            timestamp=result['timing']['blocks'][0]['timestamp'],status='PASS'))
finally:store.close()
run.require_clean_resume(replay,loaded)
summary=analyze_sweep.analyze(replay,loaded)
assert (summary['recorded'],summary['missing'],summary['timed_blocks'],summary['triplets_complete'])==(36,3564,252,12)
assert summary['counts']=={'PASS':36} and summary['gate']=='INCOMPLETE'
assert summary['round_groups_complete']==0 and summary['round_group_sizes']=={2:18}
assert not summary['all_restorations_accepted']  # This mapping was never a new GPU run.
assert raw_hashes=={str(p):run.sha(p) for base in (old_raw,reviewed) for p in base.rglob('*') if p.is_file()}
assert source_hashes==run.own_hashes()
validation=dict(status='PASS',zip=str(archive),zip_sha256=run.sha(archive),zip_files=len(source_hashes),
    source_hashes=source_hashes,dependencies_verified=len(deps),local_tests=20,extracted_zip_tests=20,
    identical_200_shape_selection=True,identical_task_ids_inputs_order_and_splits=True,
    task_digest=loaded['plan'].digest(plan['tasks']),selected_clock_mhz=750,memory_mhz=6801,power_limit_w=70,
    measurements=3600,blocks=25200,mode_triplets=1200,rounds=3,
    correctness_full_measurements=36,correctness_sampled_measurements=3564,
    graph_pin_conditions=len(run.pinned_choices(loaded)),reviewed_750_stage=admission,
    tampered_reviewed_evidence_rejected=True,raw_files_unchanged=len(raw_hashes),
    offline_750_replay=dict(records=36,blocks=252,triplets=12,missing=3564,gate='INCOMPLETE',new_gpu_run=False),
    worker_controller_reuse='AST identical after removing shared resume guard calls and 780-to-750 setting literal',
    resume_policy='Recompute saved sensor quality before controller/worker GPU calls; reject clock/telemetry/data issues',
    partial_round_policy='Report null when no group has at least two rounds; expose group coverage',
    full_113382_plan='unchanged and deferred; old 780 results separate',new_gpu_measurements=0)
(HERE/'validation.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:validation[k] for k in ('status','zip','zip_sha256','dependencies_verified','local_tests','extracted_zip_tests','raw_files_unchanged','measurements','new_gpu_measurements')},indent=2,ensure_ascii=False))
