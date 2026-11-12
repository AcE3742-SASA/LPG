"""Separate 10000-epoch audit for MLP fits that hit the primary 3000-epoch cap."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np

import compare_learners as c

OUT=c.HERE/'runs/v1'
AUDIT=c.HERE/'runs/convergence_audit_v1'


def read(name):
    with (OUT/name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    if not (OUT/'execution_completed.json').exists():
        raise RuntimeError('Primary comparison must finish first')
    AUDIT.mkdir(exist_ok=False)
    p=json.loads((OUT/'frozen_protocol.json').read_text())
    if any(c.old.sha(c.ROOT/name)!=digest for name,digest in p['sources'].items()):
        raise ValueError('Primary frozen source or data changed')
    hashes=json.loads((OUT/'artifact_hashes.json').read_text())
    flagged=[r for r in read('fit_diagnostics.csv') if int(r['convergence_warnings'])>0]
    if any(not r['learner'].startswith('mlp') for r in flagged):
        raise ValueError('Unexpected non-MLP convergence warning')
    c.old.save_json(AUDIT/'frozen_protocol.json',dict(primary_protocol_sha256=c.old.sha(OUT/'frozen_protocol.json'),
        script_sha256=c.old.sha(__file__),selection='All primary MLP fits with convergence warnings, regardless of validation error',
        change='Same initialization, data, scalers, optimizer, patience and train-loss tolerance; max_iter=10000 instead of 3000; refit from scratch',
        primary_results_replaced=False,extra_outer_validation_tuning=False,locked_test='NOT_OPENED',flagged=flagged))
    access=[]
    a=list(map(tuple,p['a_shapes'])); cc=list(map(tuple,p['c_shapes']))
    v=list(map(tuple,p['validation_shapes'])); v30=list(map(tuple,p['rq4_validation_shapes']))
    train=c.old.load_labels('legacy_train',a+cc,c.old.MODES,access)
    validation=c.old.load_labels('legacy_validation',v,c.old.MODES,access)
    selected=json.loads((OUT/'source_selection_before_target_access.json').read_text())['selection']
    c.MLP=dict(c.MLP,max_iter=10000)
    scores=[]; predictions=[]
    for row in flagged:
        learner,dtype,mode,method=row['learner'],row['dtype'],row['mode'],row['method']
        n,seed=int(row['n']),int(row['seed'])
        if row['rq']=='RQ1':
            shapes=list(map(tuple,(p['a_orders'][str(seed)]+p['c_orders'][str(seed)])[:n])); vs=v
        else:
            shapes=list(map(tuple,p['c_orders'][str(seed)][:n]));vs=v30
        x,lower=c.old.design(shapes,dtype); xv,lv=c.old.design(vs,dtype)
        y=c.old.target(train,shapes,dtype,mode); yv=c.old.target(validation,vs,dtype,mode)
        response=np.log(y);factor=1
        if row['rq']=='RQ1' and method=='normalized':
            response-=np.log(lower);factor=lv
        if row['rq']=='RQ4' and method!='direct':
            target=next(r['target'] for r in selected if r['learner']==learner and r['dtype']==dtype)
            path=OUT/'models'/f'rq1_{learner}_{dtype}_allocate_{target}_2652_{seed}.joblib'
            assert c.old.sha(path)==hashes[str(path.relative_to(OUT))]
            source=joblib.load(path)['model']
            prior=np.exp(source.predict(x))*(lower if target=='normalized' else 1)
            pv=np.exp(source.predict(xv))*(lv if target=='normalized' else 1)
            x=np.column_stack((x,np.log(prior)));xv=np.column_stack((xv,np.log(pv)))
            if method=='residual':
                response-=np.log(prior)
                factor=pv
        model,detail=c.fit(learner,x,response,seed)
        pred=np.exp(model.predict(xv))*factor
        metric=c.old.metrics(yv,pred)
        info={k:row[k] for k in ['rq','learner','dtype','mode','method','n','seed']}
        before=next(r for r in read('metrics.csv') if all(r[k]==value for k,value in info.items()))
        scores.append({**info,'primary_mape_pct':before['mape_pct'],'primary_p90_pct':before['p90_pct'],
            **metric,**detail})
        for shape,actual,predicted in zip(vs,yv,pred):
            predictions.append({**info,'shape':c.old.shape_id(shape),'actual_ms':actual,'predicted_ms':predicted})
        print(f'PASS: convergence audit {len(scores)}/{len(flagged)} {learner} {row["rq"]} {dtype} {mode} {method} n={n} epochs={detail["epochs"]}',flush=True)
    c.old.save_csv(AUDIT/'metrics.csv',scores)
    c.old.save_csv(AUDIT/'validation_predictions.csv',predictions)
    c.old.save_json(AUDIT/'label_access_log.json',access)
    c.old.save_json(AUDIT/'completed.json',dict(audited=len(scores),remaining_warning_fits=sum(bool(r['convergence_warnings']) for r in scores),
        primary_numerical_artifacts_unchanged=all(c.old.sha(OUT/name)==digest for name,digest in hashes.items() if name!='console.log'),locked_test='NOT_OPENED'))


if __name__=='__main__':
    main()
