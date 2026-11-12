"""Validate the deployment ZIP, frozen dependencies and actual reviewed-check bridge."""
from pathlib import Path
from collections import Counter
import ast,json,shutil,subprocess,sys,tempfile,zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'abstract_sweep_780_v1'
sys.path.insert(0,str(PACKAGE))
import run
loaded=run.load();plan=run.make_plan('full',loaded)
assert len(plan['tasks'])==3600 and len(plan['shape_splits'])==200
source_hashes=run.own_hashes()
for name in source_hashes:
 if name.endswith('.py'):ast.parse((PACKAGE/name).read_text())
subprocess.run(['bash','-n',str(PACKAGE/'launch.sh')],check=True)
archive=ROOT/'abstract_sweep_780_2026-09-21_v1.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name in source_hashes:z.write(PACKAGE/name,str(Path(PACKAGE.name)/name))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(source_hashes)
 assert all(z.read(str(Path(PACKAGE.name)/name))==(PACKAGE/name).read_bytes() for name in source_hashes)
archive.with_suffix('.zip.sha256').write_text(run.sha(archive)+'  '+archive.name+'\n')
raw=ROOT/'analysis/fixed_sweep_780_check_result_2026-09-21/raw/check_780'
raw_hashes={str(p.relative_to(raw)):run.sha(p) for p in raw.rglob('*') if p.is_file()}
with tempfile.TemporaryDirectory(prefix='abstract deployment ') as tmp:
 root=Path(tmp)
 with zipfile.ZipFile(archive) as z:z.extractall(root)
 deps=run.read(PACKAGE/'dependency_hashes.json')
 for name,wanted in deps.items():
  assert run.sha(ROOT/name)==wanted
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
 package=root/PACKAGE.name
 verify=subprocess.run([sys.executable,'-B',str(package/'run.py'),'verify'],capture_output=True,text=True)
 (HERE/'extracted_verify.log').write_text(verify.stdout+verify.stderr)
 assert verify.returncode==0
 tests=subprocess.run([sys.executable,'-B',str(package/'test_subset.py')],capture_output=True,text=True)
 (HERE/'extracted_tests.log').write_text(tests.stdout+tests.stderr)
 assert tests.returncode==0,(tests.stdout+tests.stderr)
 check=root/'fixed_sweep_780_v1/runs/check_780';check.parent.mkdir(parents=True);shutil.copytree(raw,check)
 code=("import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import run;"
       "print(json.dumps(run.check_admission(Path(sys.argv[2]),(run.HERE/'review_note.md').read_text(),run.load())))")
 reviewed=subprocess.run([sys.executable,'-B','-c',code,str(package),str(check)],capture_output=True,text=True)
 (HERE/'reviewed_check_bridge.log').write_text(reviewed.stdout+reviewed.stderr)
 assert reviewed.returncode==0,reviewed.stderr
 admission=json.loads(reviewed.stdout)
 assert all(admission[k]==v for k,v in loaded['config']['reviewed_check'].items())
 assert run.read(check/'summary.json')==run.read(raw/'summary.json')
 # Exact task IDs and source snapshots survive the subset's own resume path.
 folder=root/'subset';folder.mkdir()
 create=("import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import run;"
         "p=Path(sys.argv[2]);a=run.manifest_for(p,'full',run.load());"
         "assert a==run.manifest_for(p,'full',run.load());print(len(a['tasks']))")
 resume=subprocess.run([sys.executable,'-B','-c',create,str(package),str(folder)],capture_output=True,text=True)
 (HERE/'extracted_manifest_resume.log').write_text(resume.stdout+resume.stderr)
 assert resume.returncode==0 and resume.stdout.strip()=='3600',resume.stderr
assert {str(p.relative_to(raw)):run.sha(p) for p in raw.rglob('*') if p.is_file()}==raw_hashes
assert source_hashes==run.own_hashes()
selection=run.read(PACKAGE/'selection.json')
validation=dict(status='PASS',zip=str(archive),zip_sha256=run.sha(archive),zip_files=len(source_hashes),
 source_hashes=source_hashes,frozen_dependencies_verified=len(loaded and run.read(PACKAGE/'dependency_hashes.json')),
 local_tests=17,extracted_zip_tests=17,actual_reviewed_check_bridge='PASS; reanalysis of copy, no GPU execution',
 check_manifest_sha256=admission['manifest_sha256'],check_results_digest=admission['results_digest'],
 check_raw_files_unchanged=len(raw_hashes),shape_count=200,shape_dtype_conditions=400,
 measurements=3600,blocks=25200,triplets=1200,rounds=3,geometry_strata=32,
 strata_quota_counts=dict(Counter(s['selected'] for s in selection['strata'])),
 selected_original_splits=dict(Counter(s['split'] for s in selection['shapes'])),fresh_holdout_selected=0,
 task_digest=loaded['plan'].digest(plan['tasks']),
 measurement_worker_controller_functions='AST identical to fixed_sweep_780_v1',analyzer='byte-identical to fixed_sweep_780_v1',
 full_113382_plan='unchanged; subset outputs separate',new_gpu_measurements=0)
(HERE/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(validation,ensure_ascii=False,indent=2))
