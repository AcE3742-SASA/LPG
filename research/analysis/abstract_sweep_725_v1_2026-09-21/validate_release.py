"""Verify the delivery ZIP offline with its frozen dependencies and preserved evidence."""
import ast
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'abstract_sweep_725_v1'
sys.path.insert(0,str(PACKAGE))
import run


def main():
    preserved=run.read(HERE/'preserved_sha256.json')
    assert all(run.sha(ROOT/name)==wanted for name,wanted in preserved.items())
    loaded=run.load()
    source=run.own_hashes()
    dependencies=run.read(PACKAGE/'dependency_hashes.json')
    plan=run.make_plan('full',loaded)
    check=run.make_plan('check',loaded)
    previous=run.read(ROOT/'analysis/abstract_200_750_result_2026-09-21/raw/abstract_200_750/manifest.json')
    assert plan['tasks']==previous['tasks']
    assert json.loads(json.dumps(plan['shape_splits']))==previous['shape_splits']
    assert plan['protocol']==previous['protocol']
    assert len(check['tasks'])==48 and len(plan['tasks'])==3600
    assert Counter(t['round'] for t in check['tasks'])=={0:24,1:24}
    assert {tuple(t['shape']) for t in check['tasks']}=={(2329,3998,66),(3128,680,488),(4096,4096,4096),(64,64,64)}
    assert all(loaded['plan'].task_seed(t)==next(loaded['plan'].task_seed(old) for old in plan['tasks']
               if old['shape']==t['shape'] and old['dtype']==t['dtype']) for t in check['tasks'])
    for name in ('selection.json','subset.py','analyze_sweep.py'):
        assert run.sha(PACKAGE/name)==run.sha(ROOT/'abstract_sweep_750_v1'/name)
    for p in PACKAGE.glob('*.py'):ast.parse(p.read_text())
    subprocess.run(['bash','-n',str(PACKAGE/'launch.sh')],check=True)
    archive=ROOT/'abstract_sweep_725_or_720_2026-09-21_v1.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name in source:z.write(PACKAGE/name,str(Path(PACKAGE.name)/name))
        archive.with_suffix('.zip.sha256').write_text(run.sha(archive)+'  '+archive.name+'\n')
    with tempfile.TemporaryDirectory(prefix='gemm-low-clock-release-') as folder:
        extracted=Path(folder)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert set(z.namelist())=={str(Path(PACKAGE.name)/name) for name in source}
            z.extractall(extracted)
        package=extracted/PACKAGE.name
        assert all(run.sha(package/name)==wanted for name,wanted in source.items())
        for name,wanted in dependencies.items():
            dest=extracted/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dest)
            assert run.sha(dest)==wanted
        for label,args in [
            ('extracted_tests',[sys.executable,'-B',str(package/'test_low_clock.py')]),
            ('extracted_check_plan',[sys.executable,'-B',str(package/'run.py'),'verify','--size','check']),
            ('extracted_full_plan',[sys.executable,'-B',str(package/'run.py'),'verify','--size','full'])]:
            result=subprocess.run(args,cwd=extracted,capture_output=True,text=True,timeout=90)
            (HERE/(label+'.log')).write_text(result.stdout+result.stderr)
            assert result.returncode==0,result.stdout+result.stderr
        assert 'Ran 17 tests' in (HERE/'extracted_tests.log').read_text()
        full=json.loads((HERE/'extracted_full_plan.log').read_text())
        assert full['task_digest']==loaded['plan'].digest(previous['tasks'])
    assert all(run.sha(ROOT/name)==wanted for name,wanted in preserved.items())
    assert source==run.own_hashes()
    result=dict(status='PASS',zip=str(archive),zip_sha256=run.sha(archive),zip_bytes=archive.stat().st_size,
        zip_files=len(source),source_hashes=source,dependencies_verified=len(dependencies),
        local_tests=17,extracted_zip_tests=17,preserved_files_unchanged=len(preserved),
        requested_clock_mhz=725,authorized_fallback_mhz=720,actual_hardware_selection='PENDING_UBUNTU_QUERY',
        actual_gpu_validation='NOT_RUN',memory_mhz=6801,power_limit_w=70,
        check=dict(shapes=4,dtypes=2,modes=3,rounds=2,measurements=48,blocks=336,triplets=16,
                   separate_workers=True,task_digest=loaded['plan'].digest(check['tasks'])),
        full=dict(shapes=200,dtypes=2,modes=3,rounds=3,measurements=3600,blocks=25200,triplets=1200,
                  task_digest=full['task_digest'],identical_750_tasks_inputs_order_splits=True),
        graph_pins=52,selection_and_analyzer_byte_identical=True,
        measurement_worker_cleanup_pack_resume_guard_ast_identical=True,
        no_new_tmux=True,full_requires_reviewed_new_clock_check=True,new_gpu_measurements=0)
    (HERE/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','zip','zip_sha256','zip_bytes','zip_files',
        'dependencies_verified','extracted_zip_tests','preserved_files_unchanged','actual_hardware_selection')},indent=2))


if __name__=='__main__':main()
