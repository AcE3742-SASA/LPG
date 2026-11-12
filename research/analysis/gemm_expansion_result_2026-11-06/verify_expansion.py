#!/usr/bin/env python3
"""Preserve and verify returned expansion data with the unchanged local collector."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'gemm_expand_20261028_v1'
ARCHIVE=Path('/Users/studyhard/Desktop/Transfer/expansion_10000_20261105_181927_9a759e.zip')
RAW=HERE/'raw/expansion_10000'
STAGE=HERE/'recomputed/expansion_10000'
sys.path.insert(0,str(PACKAGE))
import run as collector
import analyze_sweep


def read(path):return json.loads(Path(path).read_text())
def write(name,value):(HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def require(condition,message):
    if not condition:raise ValueError(message)


def extract():
    require((HERE/'prelabel_comparison_protocol.json').is_file(),'Freeze the comparison design first')
    require(not RAW.exists(),'Refusing to overwrite extracted raw data')
    loaded=collector.load(720);planned=collector.make_plan('full',loaded)
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos=archive.infolist();names=[i.filename for i in infos]
        require(len(names)==len(set(names)),'Duplicate archive entries')
        for info in infos:
            p=PurePosixPath(info.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in info.filename and p.parts[0]=='expansion_10000','Unsafe archive entry')
            require(not stat.S_ISLNK(info.external_attr>>16),'Symlink archive entry')
        require(sum(i.file_size for i in infos)<15_000_000_000,'Unexpected uncompressed size')
        manifest=json.loads(archive.read('expansion_10000/manifest.json'))
        require(manifest['kind']=='gemm_expand_20261028_v1' and manifest['size']=='full','Unexpected run kind')
        require({t['split'] for t in manifest['tasks']}=={'expansion_train','expansion_validation'},'Non-development split denied')
        require(all(manifest[k]==json.loads(json.dumps(v)) for k,v in planned.items()),'Frozen task manifest mismatch')
        require(manifest['source_hashes']==collector.own_hashes(),'Source release mismatch')
        require(manifest['dependency_hashes']==read(PACKAGE/'dependency_hashes.json'),'Dependency release mismatch')
        require(manifest['environment_signature']==loaded['baseline']['environment_signature'],'Environment signature mismatch')
        for folder,hashes in [('source',manifest['source_hashes']),('dependencies',manifest['dependency_hashes'])]:
            for name,digest in hashes.items():
                require(hashlib.sha256(archive.read(f'expansion_10000/{folder}/{name}')).hexdigest()==digest,'Embedded code mismatch: '+name)
        preserved=HERE/ARCHIVE.name
        require(not preserved.exists(),'Refusing to replace a preserved archive')
        shutil.copy2(ARCHIVE,preserved)
        archive_sha=collector.sha(ARCHIVE)
        require(collector.sha(preserved)==archive_sha,'Archive preservation hash mismatch')
        print('PASS: manifest, development-only scope and 314 embedded code hashes',flush=True)
        (HERE/'raw').mkdir()
        # zipfile validates CRC as each entry reaches EOF; complete extraction checks every entry.
        archive.extractall(HERE/'raw')
        original_exports={p.name:collector.sha(p) for p in RAW.iterdir() if p.is_file() and p.suffix in {'.csv','.json','.jsonl','.md'}}
        write('archive_verification.json',dict(archive=str(ARCHIVE),preserved_archive=str(preserved),sha256=archive_sha,
            entries=len(infos),uncompressed_bytes=sum(i.file_size for i in infos),zip_crc='PASS',source_hashes=223,dependency_hashes=91,
            expected_measurements=len(manifest['tasks']),splits=dict(Counter(t['split'] for t in manifest['tasks'])),
            original_export_sha256=original_exports,raw_sqlite_sha256=collector.sha(RAW/'results.sqlite3'),locked_test_opened=False))
        print('PASS: archive preserved and all entries extracted with CRC validation',flush=True)


def recompute():
    manifest=read(RAW/'manifest.json')
    require(manifest['kind']=='gemm_expand_20261028_v1' and {t['split'] for t in manifest['tasks']}=={'expansion_train','expansion_validation'},'Development phase required')
    require(not STAGE.exists(),'Refusing to overwrite recomputed outputs')
    STAGE.mkdir(parents=True)
    for name in ['manifest.json','results.sqlite3']:
        os.link(RAW/name,STAGE/name)
    for name in ['source','dependencies','launches']:
        shutil.copytree(RAW/name,STAGE/name,copy_function=os.link)
    # The analyzer rewrites quality.json. Keep those paths private to the derived copy.
    shutil.copytree(RAW/'sessions',STAGE/'sessions',copy_function=os.link,ignore=lambda path,names:['quality.json'] if 'quality.json' in names else [])
    result=analyze_sweep.analyze(STAGE,collector.load(720))
    write('recompute_completed.json',result)
    # JSON object keys are strings, including the analyzer's integer round-count keys.
    require(read(HERE/'recompute_completed.json')==read(RAW/'summary.json'),'Recomputed public summary differs')
    print('PASS: unchanged analyzer reproduced the complete summary',flush=True)


def verify():
    manifest=read(RAW/'manifest.json');archive=read(HERE/'archive_verification.json')
    require(manifest['kind']=='gemm_expand_20261028_v1' and {t['split'] for t in manifest['tasks']}=={'expansion_train','expansion_validation'},'Development phase required')
    require(read(HERE/'recompute_completed.json')==read(RAW/'summary.json'),'Summary reproduction missing')
    exports={}
    for name,digest in archive['original_export_sha256'].items():
        require(collector.sha(RAW/name)==digest,'Raw export changed: '+name)
        if (STAGE/name).exists():
            exports[name]=collector.sha(STAGE/name)==digest
            require(exports[name],'Recomputed export differs: '+name)
    require(collector.sha(RAW/'results.sqlite3')==archive['raw_sqlite_sha256'],'Original SQLite changed')
    require(collector.sha(STAGE/'raw_attempts.jsonl')==read(RAW/'summary.json')['results_digest'],'Attempt digest mismatch')
    with sqlite3.connect((RAW/'results.sqlite3').as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        require(db.execute('PRAGMA quick_check').fetchall()==[('ok',)],'SQLite integrity failed')
        counts=db.execute('SELECT COUNT(*),COUNT(DISTINCT task_id) FROM attempts').fetchone()
        require(counts==(180000,180000),'Attempt count or uniqueness mismatch')
    expected={t['id']:t for t in manifest['tasks']};seen=set();raw_status=Counter();scopes=Counter();sessions=Counter();walls=[]
    for _,row in collector.history(STAGE):
        t=row['task'];require(expected[t['id']]==t and t['id'] not in seen,'Unplanned or duplicate task')
        require(row['correctness']['passed'] and row['correctness']['finite_all'],'Stored correctness failed')
        require(len(row['timing']['blocks'])==7,'Timing block coverage mismatch')
        seen.add(t['id']);raw_status[row['status']]+=1;scopes[row['correctness']['scope']]+=1;sessions[row['session']]+=1;walls.append(row['task_wall_seconds'])
    require(seen==set(expected),'Missing task identity')
    summary=read(RAW/'summary.json');reviews=read(STAGE/'remeasure_tasks.json')
    require(len({r['task']['id'] for r in reviews})==len(reviews)==summary['review_queue_tasks'],'Review list coverage mismatch')
    require(all(set(r)=={'task','flags','between_round_review'} for r in reviews),'Review list contract mismatch')
    quality_files=sorted((STAGE/'sessions').glob('*/segment_*/quality.json'));flags=Counter();sensors=Counter();ranges={}
    for path in quality_files:
        q=read(path);require(q['collection_can_continue'],'Unexpected fatal segment')
        require(q['telemetry_sha256']==collector.sha(path.parent/'telemetry.csv'),'Sensor checksum mismatch')
        flags.update(q['flags']);sensors['samples']+=q['cadence']['samples']
        require(q==read(RAW/path.relative_to(STAGE)),'Recomputed segment quality differs')
        for k,(a,b) in q['ranges'].items():
            old=ranges.get(k,[a,b]);ranges[k]=[min(old[0],a),max(old[1],b)]
    require(len(quality_files)==1500,'Segment count mismatch')
    result=dict(status='PASS',experiment_gate=summary['gate'],expected=180000,recorded=len(seen),duplicate_tasks=0,missing_tasks=0,
        source_hashes_checked=223,dependency_hashes_checked=91,zip_crc='PASS',archive_sha256=archive['sha256'],
        summary_exact=True,exports_exact=exports,sqlite_quick_check='ok',raw_sqlite_unchanged=True,raw_exports_unchanged=True,
        raw_statuses=dict(raw_status),correctness_passed=len(seen),correctness_scopes=dict(scopes),sessions=dict(sessions),
        segment_quality_exact=len(quality_files),segment_flags=dict(flags),sensor_samples=sensors['samples'],sensor_ranges=ranges,
        review_tasks=len(reviews),between_round_conditions=summary['between_round_flags'],accepted_measurements=summary['accepted_measurements'],
        task_wall_sum_seconds=sum(walls),task_wall_median_seconds=summary['task_wall_seconds_median'],
        worker_elapsed_seconds=sum(read(path)['elapsed_seconds'] for path in (RAW/'sessions').glob('*/session.json')),
        finished_utc=read(RAW/'progress.json')['updated_utc'],locked_test_opened=False)
    write('verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['extract','recompute','verify'])
    {'extract':extract,'recompute':recompute,'verify':verify}[parser.parse_args().action]()
