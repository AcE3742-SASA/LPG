#!/usr/bin/env python3
"""Four fixed GEMM learners on verified development-only expansion data."""
import argparse
from collections import Counter,defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

import joblib
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'analysis/learner_comparison_2026-10-28'
RAW=HERE/'raw/expansion_10000'
OUT=HERE/'fit_v1'
spec=importlib.util.spec_from_file_location('matched_learners',BASE/'compare_learners.py')
learners=importlib.util.module_from_spec(spec);spec.loader.exec_module(learners)
old=learners.old
SPLITS={'expansion_train','expansion_validation'}


def read(path):return json.loads(Path(path).read_text())
def csv_rows(path):
    with Path(path).open(newline='') as stream:yield from csv.DictReader(stream)


def gate(manifest):
    if manifest.get('kind')!='gemm_expand_20261028_v1' or manifest.get('size')!='full':
        raise ValueError('Only full expansion development data is allowed')
    if {t['split'] for t in manifest['tasks']}!=SPLITS:
        raise ValueError('Non-development task denied before timing access')


def adopt(manifest,rows,review_ids):
    gate(manifest)
    expected={t['id']:t for t in manifest['tasks']};seen=set();accepted=defaultdict(list)
    shape_meta={tuple(t['shape']):(t['split'],t['group']) for t in manifest['tasks']}
    for row in rows:
        rid=row['id']
        if rid not in expected or rid in seen:raise ValueError('Unplanned or duplicate metric')
        seen.add(rid);task=expected[rid];shape=tuple(json.loads(row['shape']))
        if (shape,row['split'],row['dtype'],row['mode'],int(row['round']))!=(tuple(task['shape']),task['split'],task['dtype'],task['mode'],task['round']):
            raise ValueError('Metric identity differs from manifest')
        if row['status']!='PASS' or rid in review_ids:continue
        value=float(row['median_ms'])
        if not math.isfinite(value) or value<=0:raise ValueError('Invalid eligible latency')
        accepted[(shape,row['dtype'],row['mode'])].append((int(row['round']),value))
    if seen!=set(expected):raise ValueError('Incomplete metric coverage')
    complete={};counts=Counter();eligibility=[]
    for shape,(split,stratum) in sorted(shape_meta.items()):
        condition_values={};raw_count=0
        for dtype in old.DTYPES:
            for mode in old.MODES:
                values=accepted[(shape,dtype,mode)];raw_count+=len(values)
                if len(values)==3 and {r for r,_ in values}=={0,1,2}:
                    yy=[v for _,v in sorted(values)];median=statistics.median(yy)
                    if (max(yy)-min(yy))/median>.05:raise ValueError('Eligible condition violates the frozen round limit')
                    condition_values[(shape,dtype,mode)]=median
        common=len(condition_values)==6
        eligibility.append(dict(shape=old.shape_id(shape),split=split,stratum=stratum,accepted_raw_records=raw_count,
            complete_conditions=len(condition_values),adopted_common_shape=common))
        counts[(split,'planned_shapes')]+=1;counts[(split,'accepted_raw_records')]+=raw_count
        counts[(split,'complete_conditions')]+=len(condition_values)
        if common:
            complete.update(condition_values);counts[(split,'common_shapes')]+=1
    return complete,eligibility,counts


def prepare():
    if OUT.exists():raise ValueError('Refusing to overwrite a prepared comparison')
    prelabel=read(HERE/'prelabel_comparison_protocol.json')
    if any(old.sha(ROOT/n)!=h for n,h in prelabel['inputs'].items()):raise ValueError('Prelabel design input changed')
    verified=read(HERE/'verification.json')
    if verified['status']!='PASS' or not verified['summary_exact']:raise ValueError('Full offline verification required')
    manifest=read(RAW/'manifest.json');gate(manifest)
    reviews=read(RAW/'remeasure_tasks.json');review_ids={r['task']['id'] for r in reviews}
    labels,eligibility,counts=adopt(manifest,csv_rows(RAW/'metrics.csv'),review_ids)
    if sum(r['accepted_raw_records'] for r in eligibility)!=verified['accepted_measurements']:raise ValueError('Accepted raw count mismatch')
    shapes={s:sorted(tuple(map(int,r['shape'].split('x'))) for r in eligibility if r['split']==s and r['adopted_common_shape']) for s in sorted(SPLITS)}
    if len(shapes['expansion_train'])<=3000 or not shapes['expansion_validation']:raise ValueError('Insufficient adopted expansion cohort')
    a,c,v=old.metadata();assert (len(a),len(c),len(v))==(2652,693,716)
    if (set(a)|set(c)|set(v)) & set(shapes['expansion_train']+shapes['expansion_validation']):raise ValueError('Exact overlap with legacy cohorts')
    previous=read(BASE/'runs/v1/frozen_protocol.json');index=read(BASE/'runs/v1/artifact_hashes_final.json')
    weights={}
    for family in learners.LEARNERS:
        for target in learners.TARGETS:
            for dtype in old.DTYPES:
                for mode in old.MODES:
                    for seed in old.SEEDS:
                        name=f'models/rq1_{family}_{dtype}_{mode}_{target}_3345_{seed}.joblib';path=BASE/'runs/v1'/name
                        if old.sha(path)!=index[name]:raise ValueError('Original baseline weight checksum mismatch')
                        weights[str(path.relative_to(ROOT))]=index[name]
    OUT.mkdir();(OUT/'models').mkdir()
    adopted=[]
    for r in eligibility:
        if r['adopted_common_shape']:
            shape=tuple(map(int,r['shape'].split('x')))
            for dtype in old.DTYPES:
                for mode in old.MODES:
                    adopted.append(dict(shape=r['shape'],split=r['split'],stratum=r['stratum'],dtype=dtype,mode=mode,
                        accepted_rounds=3,median_ms=labels[(shape,dtype,mode)]))
    old.save_csv(OUT/'adopted_conditions.csv',adopted);old.save_csv(OUT/'shape_eligibility.csv',eligibility)
    old.save_json(OUT/'dataset_summary.json',dict(quality=verified['experiment_gate'],planned_shapes=10000,planned_raw_records=180000,
        held_review_raw_records=len(review_ids),eligible_raw_records=verified['accepted_measurements'],
        common_shape_counts={s:len(vv) for s,vv in shapes.items()},common_conditions=len(adopted),
        common_raw_records=len(adopted)*3,legacy_train_shapes=3345,legacy_validation_shapes=716,
        augmented_train_shapes=3345+len(shapes['expansion_train']),
        split_counts=[dict(split=s,measure=k,count=n) for (s,k),n in sorted(counts.items())],locked_test_opened=False))
    new_orders={str(seed):old.ordered(shapes['expansion_train'],f'gemm-expansion-20261106-{seed}') for seed in old.SEEDS}
    paths=[Path(__file__),HERE/'verify_expansion.py',HERE/'prelabel_comparison_protocol.json',HERE/'verification.json',RAW/'manifest.json',RAW/'metrics.csv',
           RAW/'remeasure_tasks.json',OUT/'adopted_conditions.csv',OUT/'shape_eligibility.csv',BASE/'compare_learners.py',old.DATA,old.ROLES,
           BASE/'runs/v1/artifact_hashes_final.json',BASE/'runs/v1/frozen_protocol.json']
    p=dict(version='gemm_expansion_comparison_v1',created_utc=old.stamp(),script_sha256=old.sha(__file__),
        prelabel_protocol_sha256=old.sha(HERE/'prelabel_comparison_protocol.json'),
        inputs={str(path.relative_to(ROOT)):old.sha(path) for path in paths},baseline_weights=weights,
        learners=learners.LEARNERS,targets=learners.TARGETS,seeds=old.SEEDS,features=old.FEATURES,
        rf=old.RF,ridge=dict(alpha=1),mlp=learners.MLP,widths=[32,64],
        added_sizes=[0,1000,3000,len(shapes['expansion_train'])],
        old_train_orders={str(seed):previous['a_orders'][str(seed)]+previous['c_orders'][str(seed)] for seed in old.SEEDS},
        new_train_orders=new_orders,legacy_validation_shapes=previous['validation_shapes'],new_validation_shapes=shapes['expansion_validation'],
        selection=prelabel['primary'],quality_policy=prelabel['quality'],locked_test_status='CLOSED',
        runtime=dict(python=sys.version,numpy=np.__version__,sklearn=old.sklearn.__version__))
    old.save_json(OUT/'frozen_protocol.json',p)
    print('PASS: adopted common training shapes='+str(p['added_sizes'][-1])+' validation shapes='+str(len(p['new_validation_shapes'])),flush=True)


def source_labels(p):
    access=[];old_shapes=list(map(tuple,p['old_train_orders'][str(old.SEEDS[0])]))
    legacy_validation=list(map(tuple,p['legacy_validation_shapes']))
    train=old.load_labels('legacy_train',old_shapes,old.MODES,access)
    val=old.load_labels('legacy_validation',legacy_validation,old.MODES,access)
    expanded={};new_validation={}
    for row in csv_rows(OUT/'adopted_conditions.csv'):
        shape=tuple(map(int,row['shape'].split('x')))
        if row['split'] not in SPLITS:raise ValueError('Non-development derived label denied')
        destination=expanded if row['split']=='expansion_train' else new_validation
        destination[(shape,row['dtype'],row['mode'])]=float(row['median_ms'])
    if set(train)&set(expanded):raise ValueError('Overlapping old and new training labels')
    train.update(expanded)
    access.extend([dict(split='expansion_train',shapes=p['added_sizes'][-1],conditions=len(expanded)),
                   dict(split='expansion_validation',shapes=len(p['new_validation_shapes']),conditions=len(new_validation))])
    return train,{'legacy':val,'expansion':new_validation},access


def run():
    p=read(OUT/'frozen_protocol.json')
    if (OUT/'execution_started.json').exists():raise ValueError('Refusing to overwrite an execution')
    for name,digest in {**p['inputs'],**p['baseline_weights']}.items():
        if old.sha(ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)
    train,validation,access=source_labels(p)
    old.save_json(OUT/'label_access_log.json',access)
    old.save_json(OUT/'execution_started.json',dict(time=old.stamp(),protocol_sha256=old.sha(OUT/'frozen_protocol.json')))
    valid_shapes={'legacy':list(map(tuple,p['legacy_validation_shapes'])),'expansion':list(map(tuple,p['new_validation_shapes']))}
    started=time.perf_counter();scores=[];fits=[];subgroups=[];weights=[];configs=0
    fields=['learner','target','dtype','mode','added','n','seed','validation','shape','actual_ms','predicted_ms']
    with (OUT/'validation_predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for added in p['added_sizes']:
            for family in p['learners']:
                for dtype in old.DTYPES:
                    designs={split:old.design(ss,dtype) for split,ss in valid_shapes.items()}
                    for mode in old.MODES:
                        yy={split:old.target(validation[split],ss,dtype,mode) for split,ss in valid_shapes.items()}
                        for target in p['targets']:
                            for seed in old.SEEDS:
                                shapes=list(map(tuple,p['old_train_orders'][str(seed)]+p['new_train_orders'][str(seed)][:added]))
                                x,lower=old.design(shapes,dtype);y=old.target(train,shapes,dtype,mode)
                                info=dict(learner=family,target=target,dtype=dtype,mode=mode,added=added,n=len(shapes),seed=seed)
                                if added==0:
                                    ref=f'analysis/learner_comparison_2026-10-28/runs/v1/models/rq1_{family}_{dtype}_{mode}_{target}_3345_{seed}.joblib'
                                    envelope=joblib.load(ROOT/ref)
                                    if (envelope['learner'],envelope['dtype'],envelope['mode'],envelope['target'],envelope['seed'],envelope['n'])!=(family,dtype,mode,target,seed,3345):raise ValueError('Baseline model identity mismatch')
                                    model=envelope['model'];detail=dict(fit_seconds=0,epochs='',final_scaled_loss='',convergence_warnings=0,warning_messages='',parameter_count='')
                                else:
                                    response=np.log(y)-(np.log(lower) if target=='normalized' else 0)
                                    model,detail=learners.fit(family,x,response,seed)
                                train_pred=np.exp(model.predict(x))*(lower if target=='normalized' else 1)
                                fits.append({**info,**detail,'baseline_model_reused':added==0,'train_mape_pct':old.metrics(y,train_pred)['mape_pct']})
                                pooled_y=[];pooled_pred=[];cache={}
                                for split,ss in valid_shapes.items():
                                    xv,lv=designs[split]
                                    with np.errstate(over='ignore',invalid='ignore'):
                                        pred=np.exp(model.predict(xv))*(lv if target=='normalized' else 1)
                                    cache[split]=pred
                                    scores.append({**info,'validation':split,**old.metrics(yy[split],pred)})
                                    pooled_y.extend(yy[split]);pooled_pred.extend(pred)
                                    writer.writerows({**info,'validation':split,'shape':old.shape_id(shape),'actual_ms':a,'predicted_ms':b} for shape,a,b in zip(ss,yy[split],pred))
                                    for name,mask in old.subgroups(ss).items():
                                        if mask.any():subgroups.append({**info,'validation':split,'subgroup':name,**old.metrics(yy[split][mask],pred[mask])})
                                scores.append({**info,'validation':'pooled',**old.metrics(np.asarray(pooled_y),np.asarray(pooled_pred))})
                                if added==p['added_sizes'][-1]:
                                    path=OUT/'models'/f'{family}_{dtype}_{mode}_{target}_{seed}.joblib'
                                    joblib.dump(dict(model=model,**info,features=old.FEATURES,protocol_sha256=old.sha(OUT/'frozen_protocol.json'),
                                        training_order_ref='frozen_protocol.json old_train_orders + new_train_orders',
                                        scope='RQ1 full legacy_train including original C; preserve the independent A/C source split for RQ4',
                                        validation_only=True,locked_test_opened=False),path)
                                    loaded=joblib.load(path)['model']
                                    for split in valid_shapes:
                                        xv,lv=designs[split];pred=np.exp(loaded.predict(xv))*(lv if target=='normalized' else 1)
                                        if not np.array_equal(pred,cache[split]):raise ValueError('Saved full model prediction mismatch')
                                    weights.append(dict(file=path.name,sha256=old.sha(path),validation_predictions_exact=True))
                                configs+=1
                                old.save_json(OUT/'progress.json',dict(configurations=configs,expected=576,current=info,wall_seconds=time.perf_counter()-started))
                            stream.flush()
                        print(f'PASS: added={added} {family} {dtype} {mode}',flush=True)
    for filename,rows in [('metrics.csv',scores),('fit_diagnostics.csv',fits),('subgroup_metrics.csv',subgroups),('model_weight_verification.csv',weights)]:old.save_csv(OUT/filename,rows)
    old.save_json(OUT/'execution_completed.json',dict(time=old.stamp(),configurations=configs,actual_fits=sum(not r['baseline_model_reused'] for r in fits),
        saved_full_models=len(weights),convergence_warning_fits=sum(bool(r['convergence_warnings']) for r in fits),
        prediction_failures=sum(r['failures'] for r in scores if r['validation']!='pooled'),
        wall_seconds=time.perf_counter()-started,total_fit_seconds=sum(r['fit_seconds'] for r in fits),
        frozen_inputs_unchanged=all(old.sha(ROOT/n)==h for n,h in p['inputs'].items()),locked_test_opened=False))
    print('PASS: expansion learner comparison complete',flush=True)


def selfcheck():
    tasks=[];rows=[]
    for shape,split in [([64,64,64],'expansion_train'),([128,64,64],'expansion_validation')]:
        for dtype in old.DTYPES:
            for mode in old.MODES:
                for rd in range(3):
                    rid=f'{shape[0]}_{dtype}_{mode}_{rd}'
                    tasks.append(dict(id=rid,shape=shape,split=split,group='synthetic',dtype=dtype,mode=mode,round=rd))
                    rows.append(dict(id=rid,shape=json.dumps(shape),split=split,dtype=dtype,mode=mode,round=rd,status='PASS',median_ms=1+.001*rd))
    m=dict(kind='gemm_expand_20261028_v1',size='full',tasks=tasks)
    labels,eligibility,_=adopt(m,rows,set());assert len(labels)==12 and all(r['adopted_common_shape'] for r in eligibility)
    labels,eligibility,_=adopt(m,rows,{rows[0]['id']});assert len(labels)==6 and not eligibility[0]['adopted_common_shape']
    denied=json.loads(json.dumps(m));denied['tasks'][0]['split']='fresh_holdout'
    def forbidden_rows():raise AssertionError('Labels read before split gate');yield
    try:adopt(denied,forbidden_rows(),set())
    except ValueError:pass
    else:raise AssertionError('Non-development phase accepted')
    try:adopt(m,rows+[rows[0]],set())
    except ValueError:pass
    else:raise AssertionError('Duplicate metric accepted')
    print('PASS: synthetic common-shape aggregation, REVIEW exclusion, duplicate rejection and locked-before-label gate')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['selfcheck','prepare','run'])
    {'selfcheck':selfcheck,'prepare':prepare,'run':run}[parser.parse_args().action]()
